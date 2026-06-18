"""
HybridBrain: Kelas utama yang menggabungkan Hebbian Learning dan Kohonen SOM.
- Hebbian Learning -> belajar dari kebiasaan user (waktu & prioritas)
- Kohonen SOM -> mengelompokkan pola produktivitas dan prioritas
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from matplotlib.lines import Line2D

from config.config import TIMES, PRIORITY_LEVELS, STATE_FILE


# ============================
# FUNGSI SIMPAN/MUAT STATE
# ============================

def save_state():
    """Simpan state aplikasi ke file JSON untuk persistensi data."""
    state = {
        "tasks": st.session_state.tasks,
        "time_weights": st.session_state.time_weights.tolist() if len(st.session_state.time_weights) > 0 else [],
        "priority_weights": st.session_state.priority_weights.tolist() if len(st.session_state.priority_weights) > 0 else [],
        "history": st.session_state.history
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Muat state aplikasi dari file JSON jika ada."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None


# ============================
# HYBRID BRAIN CLASS
# ============================

class HybridBrain:
    """Kelas utama yang menggabungkan Hebbian Learning dan Kohonen SOM."""

    def __init__(self, times, priority_levels):
        self.times = times
        self.priority_levels = priority_levels
        self.scaler_time = MinMaxScaler()
        self.scaler_priority = MinMaxScaler()
        self.scaler_cluster = MinMaxScaler()
        self.grid_size = 5
        self.ensure_weight_shapes()

    def ensure_weight_shapes(self):
        """Pastikan kedua matriks bobot memiliki bentuk yang benar."""
        time_shape = st.session_state.time_weights.shape
        required_time_shape = (len(st.session_state.tasks), len(self.times))

        if time_shape != required_time_shape:
            new_time_weights = np.zeros(required_time_shape)
            min_rows = min(time_shape[0], required_time_shape[0])
            if min_rows > 0:
                new_time_weights[:min_rows] = st.session_state.time_weights[:min_rows]
            st.session_state.time_weights = new_time_weights

        priority_shape = st.session_state.priority_weights.shape
        required_priority_shape = (len(st.session_state.tasks), len(self.priority_levels))

        if priority_shape != required_priority_shape:
            new_priority_weights = np.zeros(required_priority_shape)
            min_rows = min(priority_shape[0], required_priority_shape[0])
            if min_rows > 0:
                new_priority_weights[:min_rows] = st.session_state.priority_weights[:min_rows]
            st.session_state.priority_weights = new_priority_weights

    def train_som(self, X_scaled, num_iteration=200):
        """Latih SOM dengan MiniSOM untuk menemukan pola."""
        som = MiniSom(
            x=self.grid_size, y=self.grid_size,
            input_len=X_scaled.shape[1],
            sigma=1.0,
            learning_rate=0.5,
            random_seed=42
        )
        som.random_weights_init(X_scaled)
        som.train_random(X_scaled, num_iteration=num_iteration)
        return som

    # ============================
    # SOM CLUSTERING - WAKTU
    # ============================

    def run_som_time(self):
        """Jalankan SOM clustering berdasarkan pola waktu."""
        if len(st.session_state.history) < 4:
            return None, None, None, None

        df = pd.DataFrame(st.session_state.history)

        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        time_map = {
            "Pagi (05-12)": 0, "Siang (12-15)": 1,
            "Sore (15-18)": 2, "Malam (18-21)": 3
        }

        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        X = df[["task_num", "time_num", "duration", "energy", "difficulty", "deadline"]]
        X_scaled = self.scaler_time.fit_transform(X)

        som = self.train_som(X_scaled, num_iteration=200)

        bmu_positions = [som.winner(x) for x in X_scaled]
        df["cluster"] = [pos[0] * self.grid_size + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        st.session_state.clusters_time = df
        return df, som, X_scaled, som.quantization_error(X_scaled)

    # ============================
    # SOM CLUSTERING - PRIORITAS
    # ============================

    def _calc_priority_score(self, cluster_data):
        """Hitung priority score untuk satu cluster."""
        if len(cluster_data) == 0:
            return None
        return (
            cluster_data["deadline"].mean() / 10 * 0.3 +
            cluster_data["importance"].mean() / 10 * 0.3 +
            cluster_data["effort"].mean() / 10 * 0.2 +
            cluster_data["energy"].mean() / 10 * 0.1 +
            cluster_data["difficulty"].mean() / 10 * 0.1
        )

    def _assign_label(self, score):
        """Assign label berdasarkan threshold ABSOLUT."""
        if score is None:
            return "No Data"
        if score > 0.6:
            return "High"
        elif score > 0.4:
            return "Medium"
        return "Low"

    def run_som_priority(self):
        """Jalankan SOM clustering berdasarkan pola prioritas."""
        if len(st.session_state.history) < 4:
            return None, None, None, None

        df = pd.DataFrame(st.session_state.history)

        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        df["task_num"] = df["task"].map(task_map)

        X = df[["task_num", "deadline", "importance", "effort", "energy", "difficulty"]]
        X_scaled = self.scaler_priority.fit_transform(X)

        som = self.train_som(X_scaled, num_iteration=200)

        bmu_positions = [som.winner(x) for x in X_scaled]
        df["priority_cluster"] = [pos[0] * self.grid_size + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        # Bangun debug info untuk SEMUA 25 cluster
        cluster_debug_info = {}

        for col in range(self.grid_size):
            for row in range(self.grid_size):
                cluster_id = col * self.grid_size + row
                cluster_data = df[df["priority_cluster"] == cluster_id]

                if len(cluster_data) > 0:
                    priority_score = self._calc_priority_score(cluster_data)
                    label = self._assign_label(priority_score)
                    dominant_task = cluster_data["task"].mode()[0]
                    avg_deadline = cluster_data["deadline"].mean()
                    avg_importance = cluster_data["importance"].mean()
                    avg_effort = cluster_data["effort"].mean()
                else:
                    priority_score = None
                    label = "No Data"
                    dominant_task = None
                    avg_deadline = avg_importance = avg_effort = 0

                cluster_debug_info[cluster_id] = {
                    "x": col, "y": row,
                    "samples": len(cluster_data),
                    "priority_score": priority_score,
                    "label": label,
                    "dominant_task": dominant_task,
                    "avg_deadline": avg_deadline,
                    "avg_importance": avg_importance,
                    "avg_effort": avg_effort
                }

        df["priority_label"] = df["priority_cluster"].map(
            lambda c: cluster_debug_info[c]["label"]
        )
        df["priority_score"] = df["priority_cluster"].map(
            lambda c: cluster_debug_info[c]["priority_score"]
        )

        st.session_state.clusters_priority = df
        st.session_state.cluster_debug_info = cluster_debug_info

        return df, som, X_scaled, cluster_debug_info

    def get_cluster_debug_df(self):
        """Get debug dataframe untuk UI."""
        if not hasattr(st.session_state, 'cluster_debug_info'):
            return None
        if not isinstance(st.session_state.cluster_debug_info, dict):
            return None

        rows = []
        for cid, info in sorted(st.session_state.cluster_debug_info.items()):
            if not isinstance(info, dict):
                continue
            score = info.get("priority_score")
            rows.append({
                "Cluster": cid,
                "X": info.get("x", 0),
                "Y": info.get("y", 0),
                "Samples": info.get("samples", 0),
                "Score": f"{score:.2f}" if score else "N/A",
                "Label": info.get("label", "N/A"),
                "Dominant Task": info.get("dominant_task") or "N/A"
            })
        return pd.DataFrame(rows)

    # ============================
    # SOM CLUSTERING - UNIFIED
    # ============================

    def run_som_unified(self):
        """Jalankan SOM clustering umum untuk semua pola."""
        if len(st.session_state.history) < 4:
            return None, None, None, None

        df = pd.DataFrame(st.session_state.history)

        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        time_map = {
            "Pagi (05-12)": 0, "Siang (12-15)": 1,
            "Sore (15-18)": 2, "Malam (18-21)": 3
        }

        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        X = df[["task_num", "time_num", "duration", "energy", "difficulty",
                "deadline", "importance", "effort"]]
        X_scaled = self.scaler_cluster.fit_transform(X)

        som = self.train_som(X_scaled, num_iteration=200)

        bmu_positions = [som.winner(x) for x in X_scaled]
        df["cluster"] = [pos[0] * self.grid_size + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        for cluster_id in df["cluster"].unique():
            subset = df[df["cluster"] == cluster_id]
            combined_score = (
                subset["deadline"].mean() / 10 * 0.3 +
                subset["importance"].mean() / 10 * 0.3 +
                subset["effort"].mean() / 10 * 0.2 +
                subset["energy"].mean() / 10 * 0.1 +
                subset["difficulty"].mean() / 10 * 0.1
            )
            if combined_score > 0.6:
                priority_label = "High"
            elif combined_score > 0.4:
                priority_label = "Medium"
            else:
                priority_label = "Low"
            df.loc[df["cluster"] == cluster_id, "priority_label"] = priority_label

        st.session_state.clusters = df
        return df, som, X_scaled, som.quantization_error(X_scaled)

    # ============================
    # HEBBIAN LEARNING - WAKTU
    # ============================

    def learn_time(self, task_idx, time_idx, duration_minutes, energy_level, difficulty_level, deadline_score):
        """Update bobot asosiasi task-waktu menggunakan Hebbian learning."""
        x = np.array([
            energy_level / 10,
            difficulty_level / 10,
            deadline_score / 10,
            min(duration_minutes / 120, 1.0)
        ])

        y = np.zeros(len(self.times))
        y[time_idx] = 1.0

        lr = 0.1
        decay = 0.005
        st.session_state.time_weights *= (1 - decay)

        for i, xi in enumerate(x):
            st.session_state.time_weights[task_idx] += lr * xi * y

        st.session_state.time_weights = np.clip(st.session_state.time_weights, 0, 10)

    def learn_priority(self, task_idx, urgency_level, importance_level, effort_level):
        """Update bobot asosiasi task-prioritas menggunakan Hebbian learning."""
        x = np.array([
            urgency_level / 10,
            importance_level / 10,
            effort_level / 10,
        ])

        priority_score = (urgency_level * 0.4 + importance_level * 0.4 + (10 - effort_level) * 0.2)
        priority_score_norm = priority_score / 10

        if priority_score_norm > 0.7:
            priority_idx = 2
        elif priority_score_norm > 0.4:
            priority_idx = 1
        else:
            priority_idx = 0

        y = np.zeros(len(self.priority_levels))
        y[priority_idx] = 1.0

        lr = 0.1
        decay = 0.005
        st.session_state.priority_weights *= (1 - decay)

        for i, xi in enumerate(x):
            st.session_state.priority_weights[task_idx] += lr * xi * y

        st.session_state.priority_weights = np.clip(st.session_state.priority_weights, 0, 10)

    def learn(self, task_idx, time_idx, duration_minutes, energy_level,
              difficulty_level, deadline_score, importance_level, effort_level):
        """Jalankan kedua Hebbian learning sekaligus dan simpan histori."""
        self.learn_time(task_idx, time_idx, duration_minutes, energy_level, difficulty_level, deadline_score)
        self.learn_priority(task_idx, deadline_score, importance_level, effort_level)

        st.session_state.history.append({
            "task": st.session_state.tasks[task_idx],
            "time": self.times[time_idx],
            "duration": duration_minutes,
            "energy": energy_level,
            "difficulty": difficulty_level,
            "deadline": deadline_score,
            "importance": importance_level,
            "effort": effort_level,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        save_state()

    # ============================
    # RECOMMENDATION
    # ============================

    def recommend_time(self, task_idx):
        """Rekomendasikan waktu terbaik berdasarkan Hebbian weights."""
        weights = st.session_state.time_weights[task_idx]
        if np.all(weights == 0):
            return None, 0
        best_idx = np.argmax(weights)
        return self.times[best_idx], weights[best_idx]

    def recommend_priority(self, task_idx):
        """Rekomendasikan prioritas berdasarkan Hebbian weights."""
        weights = st.session_state.priority_weights[task_idx]
        if np.all(weights == 0):
            return None, 0
        best_idx = np.argmax(weights)
        return self.priority_levels[best_idx], weights[best_idx]

    def get_priority_ranking(self):
        """Dapatkan ranking prioritas untuk semua task."""
        rankings = []

        for idx, task in enumerate(st.session_state.tasks):
            priority, score = self.recommend_priority(idx)

            if priority is None:
                task_history = [h for h in st.session_state.history if h["task"] == task]
                if task_history:
                    avg_deadline = np.mean([h["deadline"] for h in task_history])
                    avg_importance = np.mean([h["importance"] for h in task_history])
                    avg_effort = np.mean([h["effort"] for h in task_history])

                    priority_score = avg_deadline * 0.4 + avg_importance * 0.4 + (10 - avg_effort) * 0.2

                    if priority_score > 7:
                        priority = "High"
                    elif priority_score > 4:
                        priority = "Medium"
                    else:
                        priority = "Low"
                    score = priority_score / 10
                else:
                    priority = "Medium"
                    score = 0.5

            rankings.append({"task": task, "priority": priority, "score": score})

        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        rankings.sort(key=lambda x: (priority_order[x["priority"]], -x["score"]))

        return rankings


# ============================
# VISUALIZATION
# ============================

def plot_som_grid(som, X_scaled, labels, clusters_df, color_by="task", title="SOM Grid"):
    """Plot SOM Grid 2D dengan U-Matrix dan nomor cluster di setiap kotak."""
    fig, ax = plt.subplots(figsize=(12, 10))

    umatrix = som.distance_map()
    im = ax.pcolor(umatrix, cmap='Blues_r')

    grid_size = umatrix.shape[0]
    occupied_clusters = set()
    for x in X_scaled:
        bmu = som.winner(x)
        cluster_num = bmu[0] * grid_size + bmu[1]
        occupied_clusters.add(cluster_num)

    for col in range(grid_size):
        for row in range(grid_size):
            cluster_num = col * grid_size + row
            is_occupied = cluster_num in occupied_clusters
            text_color = 'white' if not is_occupied else 'black'
            fontweight = 'normal' if not is_occupied else 'bold'

            ax.text(col + 0.5, row + 0.5, str(cluster_num),
                   ha='center', va='center', fontsize=10, fontweight=fontweight, color=text_color)

            if not is_occupied:
                rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax.add_patch(rect)

    if color_by == "task":
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
        color_map = {label: colors[i] for i, label in enumerate(labels)}
    elif color_by == "priority":
        color_map = {"High": "red", "Medium": "orange", "Low": "green"}

    for i, x in enumerate(X_scaled):
        bmu = som.winner(x)
        if color_by == "task":
            task_name = clusters_df.iloc[i]["task"]
            color = color_map.get(task_name, 'gray')
        else:
            color = color_map.get(clusters_df.iloc[i].get("priority_label", "Low"), "gray")

        ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o', markersize=15,
               markerfacecolor=color, markeredgecolor='black', markeredgewidth=2)

    if color_by == "task":
        for idx, label in enumerate(labels):
            ax.plot([], [], 'o', markersize=10, markerfacecolor=colors[idx], markeredgecolor='black', label=label)
    else:
        for priority, color in color_map.items():
            ax.plot([], [], 'o', markersize=10, markerfacecolor=color, markeredgecolor='black', label=priority + " Priority")

    ax.plot([], [], '--', color='red', linewidth=2, label='Kosong (tidak ada data)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title(title)
    ax.set_xlabel("SOM X (Column)")
    ax.set_ylabel("SOM Y (Row)")
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label="Jarak (U-Matrix)")

    return fig


def plot_som_priority_heatmap(som, X_scaled, clusters_df, cluster_debug_info, title="SOM Grid - Pola Prioritas"):
    """Plot SOM Grid dengan Priority Score Heatmap yang jelas."""
    fig, ax = plt.subplots(figsize=(14, 11))

    grid_size = 5
    priority_matrix = np.full((grid_size, grid_size), np.nan)

    # Convert to dict if needed
    if isinstance(cluster_debug_info, list):
        info_dict = {}
        for item in cluster_debug_info:
            if isinstance(item, dict) and 'x' in item and 'y' in item:
                cid = item['x'] * grid_size + item['y']
                info_dict[cid] = item
        cluster_debug_info = info_dict

    if not isinstance(cluster_debug_info, dict):
        cluster_debug_info = {}

    for cid, info in cluster_debug_info.items():
        if isinstance(info, dict) and info.get("priority_score") is not None:
            priority_matrix[info["y"], info["x"]] = info["priority_score"]

    im = ax.pcolor(priority_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, edgecolors='black', linewidths=1)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0.0', '0.2', '0.4 (Low)', '0.6 (Med)', '0.8', '1.0 (High)'])
    cbar.set_label("Priority Score", fontsize=11)

    for cid, info in cluster_debug_info.items():
        if not isinstance(info, dict):
            continue

        x, y = info["x"], info["y"]

        if info.get("priority_score") is not None:
            score = info["priority_score"]
            label = info["label"]
            samples = info["samples"]

            if label == "High":
                bg = (1, 0.85, 0.85)
                tc = 'darkred'
            elif label == "Medium":
                bg = (1, 0.95, 0.8)
                tc = 'darkorange'
            else:
                bg = (0.88, 1, 0.88)
                tc = 'darkgreen'

            rect = plt.Rectangle((x + 0.02, y + 0.02), 0.96, 0.96, facecolor=bg, edgecolor='none', alpha=0.8)
            ax.add_patch(rect)

            ax.text(x + 0.5, y + 0.72, f"#{cid}", ha='center', va='center', fontsize=12, fontweight='bold', color=tc)
            ax.text(x + 0.5, y + 0.50, f"[{label}]", ha='center', va='center', fontsize=10, fontweight='bold', color=tc)
            ax.text(x + 0.5, y + 0.25, f"{score:.2f}", ha='center', va='center', fontsize=16, fontweight='bold', color=tc)
            ax.text(x + 0.5, y + 0.08, f"n={samples}", ha='center', va='center', fontsize=9, color=tc)
        else:
            rect = plt.Rectangle((x, y), 1, 1, facecolor='#E8E8E8', edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.65, f"#{cid}", ha='center', va='center', fontsize=11, fontweight='bold', color='gray')
            ax.text(x + 0.5, y + 0.35, "No Data", ha='center', va='center', fontsize=9, color='red')

    for i, x_vec in enumerate(X_scaled):
        bmu = som.winner(x_vec)
        plabel = clusters_df.iloc[i].get("priority_label", "Low")
        mc = {'High': 'darkred', 'Medium': 'darkorange', 'Low': 'darkgreen'}.get(plabel, 'gray')

        ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o', markersize=22, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)
        ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o', markersize=16, markerfacecolor=mc, markeredgecolor='white', markeredgewidth=2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=12, label='High Priority'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=12, label='Medium Priority'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=12, label='Low Priority'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', markersize=12, label='BMU Marker'),
        Line2D([0], [0], linestyle='--', color='red', linewidth=2, label='No Data')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("SOM Column (X)", fontsize=11)
    ax.set_ylabel("SOM Row (Y)", fontsize=11)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig
