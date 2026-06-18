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
        self.scaler_time = MinMaxScaler()      # Scaler untuk fitur waktu
        self.scaler_priority = MinMaxScaler()  # Scaler untuk fitur prioritas
        self.scaler_cluster = MinMaxScaler()   # Scaler untuk clustering umum
        self.ensure_weight_shapes()

    # ============================
    # RESIZE WEIGHT MATRICES
    # ============================

    def ensure_weight_shapes(self):
        """Pastikan kedua matriks bobot memiliki bentuk yang benar."""
        # Matriks bobot untuk waktu: (jumlah_task x jumlah_waktu)
        time_shape = st.session_state.time_weights.shape
        required_time_shape = (
            len(st.session_state.tasks),
            len(self.times)
        )

        if time_shape != required_time_shape:
            new_time_weights = np.zeros(required_time_shape)
            min_rows = min(time_shape[0], required_time_shape[0])
            if min_rows > 0:
                new_time_weights[:min_rows] = st.session_state.time_weights[:min_rows]
            st.session_state.time_weights = new_time_weights

        # Matriks bobot untuk prioritas: (jumlah_task x jumlah_priority_level)
        priority_shape = st.session_state.priority_weights.shape
        required_priority_shape = (
            len(st.session_state.tasks),
            len(self.priority_levels)
        )

        if priority_shape != required_priority_shape:
            new_priority_weights = np.zeros(required_priority_shape)
            min_rows = min(priority_shape[0], required_priority_shape[0])
            if min_rows > 0:
                new_priority_weights[:min_rows] = st.session_state.priority_weights[:min_rows]
            st.session_state.priority_weights = new_priority_weights

    # ============================
    # SOM TRAINING (UNIVERSAL)
    # ============================

    def train_som(self, X_scaled, grid_size=5, num_iteration=200):
        """Latih SOM dengan MiniSOM untuk menemukan pola."""
        som = MiniSom(
            x=grid_size, y=grid_size,
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
            return None, None, None

        df = pd.DataFrame(st.session_state.history)

        # Mapping nama task ke indeks numerik
        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        time_map = {
            "Pagi (05-12)": 0, "Siang (12-15)": 1,
            "Sore (15-18)": 2, "Malam (18-21)": 3
        }

        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        # Fitur untuk clustering waktu
        X = df[["task_num", "time_num", "duration", "energy", "difficulty", "deadline"]]
        X_scaled = self.scaler_time.fit_transform(X)

        som = self.train_som(X_scaled, num_iteration=200)

        bmu_positions = [som.winner(x) for x in X_scaled]
        df["cluster"] = [pos[0] * 5 + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        st.session_state.clusters_time = df
        return df, som, X_scaled

    # ============================
    # SOM CLUSTERING - PRIORITAS
    # ============================

    def run_som_priority(self):
        """Jalankan SOM clustering berdasarkan pola prioritas."""
        if len(st.session_state.history) < 4:
            return None, None, None

        df = pd.DataFrame(st.session_state.history)

        # Mapping nama task ke indeks numerik
        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}

        df["task_num"] = df["task"].map(task_map)

        # Fitur untuk clustering prioritas
        X = df[["task_num", "deadline", "importance", "effort", "energy", "difficulty"]]
        X_scaled = self.scaler_priority.fit_transform(X)

        som = self.train_som(X_scaled, num_iteration=200)

        bmu_positions = [som.winner(x) for x in X_scaled]
        df["priority_cluster"] = [pos[0] * 5 + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        # Hitung priority score untuk setiap cluster
        priority_scores = []
        for cluster_id in df["priority_cluster"].unique():
            subset = df[df["priority_cluster"] == cluster_id]
            score = (
                subset["deadline"].mean() / 10 * 0.3 +
                subset["importance"].mean() / 10 * 0.3 +
                subset["effort"].mean() / 10 * 0.2 +
                subset["energy"].mean() / 10 * 0.1 +
                subset["difficulty"].mean() / 10 * 0.1
            )
            priority_scores.append((cluster_id, score))

        # Sort berdasarkan score (tertinggi = High Priority)
        priority_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign priority labels
        cluster_priority_map = {}
        for idx, (cluster_id, score) in enumerate(priority_scores):
            if idx == 0:
                cluster_priority_map[cluster_id] = "High"
            elif idx == 1:
                cluster_priority_map[cluster_id] = "Medium"
            else:
                cluster_priority_map[cluster_id] = "Low"

        df["priority_label"] = df["priority_cluster"].map(cluster_priority_map)

        st.session_state.clusters_priority = df
        return df, som, X_scaled, cluster_priority_map

    # ============================
    # SOM CLUSTERING - UNIFIED
    # ============================

    def run_som_unified(self):
        """Jalankan SOM clustering umum untuk semua pola."""
        if len(st.session_state.history) < 4:
            return None, None, None

        df = pd.DataFrame(st.session_state.history)

        # Mapping
        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        time_map = {
            "Pagi (05-12)": 0, "Siang (12-15)": 1,
            "Sore (15-18)": 2, "Malam (18-21)": 3
        }

        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        # Fitur lengkap
        X = df[["task_num", "time_num", "duration", "energy", "difficulty",
                "deadline", "importance", "effort"]]
        X_scaled = self.scaler_cluster.fit_transform(X)

        som = self.train_som(X_scaled, num_iteration=200)

        bmu_positions = [som.winner(x) for x in X_scaled]
        df["cluster"] = [pos[0] * 5 + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        # Hitung priority score per cluster
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
        return df, som, X_scaled

    # ============================
    # HEBBIAN LEARNING - WAKTU
    # ============================

    def learn_time(self, task_idx, time_idx, duration_minutes, energy_level, difficulty_level, deadline_score):
        """Update bobot asosiasi task-waktu menggunakan Hebbian learning."""
        # Pre-synaptic activations
        x = np.array([
            energy_level / 10,
            difficulty_level / 10,
            deadline_score / 10,
            min(duration_minutes / 120, 1.0)
        ])

        # Post-synaptic: one-hot untuk waktu
        y = np.zeros(len(self.times))
        y[time_idx] = 1.0

        # Hebbian update untuk waktu
        lr = 0.1
        decay = 0.005
        st.session_state.time_weights *= (1 - decay)

        for i, xi in enumerate(x):
            st.session_state.time_weights[task_idx] += lr * xi * y

        st.session_state.time_weights = np.clip(
            st.session_state.time_weights, 0, 10
        )

    # ============================
    # HEBBIAN LEARNING - PRIORITAS
    # ============================

    def learn_priority(self, task_idx, urgency_level, importance_level, effort_level):
        """Update bobot asosiasi task-prioritas menggunakan Hebbian learning."""
        # Pre-synaptic activations
        x = np.array([
            urgency_level / 10,
            importance_level / 10,
            effort_level / 10,
        ])

        # Hitung priority score
        priority_score = (
            urgency_level * 0.4 +
            importance_level * 0.4 +
            (10 - effort_level) * 0.2
        )

        # Normalisasi ke 0-1
        priority_score_norm = priority_score / 10

        # Assign priority level
        if priority_score_norm > 0.7:
            priority_idx = 2  # High
        elif priority_score_norm > 0.4:
            priority_idx = 1  # Medium
        else:
            priority_idx = 0  # Low

        # One-hot encoding untuk priority
        y = np.zeros(len(self.priority_levels))
        y[priority_idx] = 1.0

        # Hebbian update untuk prioritas
        lr = 0.1
        decay = 0.005
        st.session_state.priority_weights *= (1 - decay)

        for i, xi in enumerate(x):
            st.session_state.priority_weights[task_idx] += lr * xi * y

        st.session_state.priority_weights = np.clip(
            st.session_state.priority_weights, 0, 10
        )

    # ============================
    # LEARN (COMBINED)
    # ============================

    def learn(self, task_idx, time_idx, duration_minutes, energy_level,
              difficulty_level, deadline_score, importance_level, effort_level):
        """Jalankan kedua Hebbian learning sekaligus dan simpan histori."""
        # Update bobot waktu
        self.learn_time(
            task_idx, time_idx, duration_minutes,
            energy_level, difficulty_level, deadline_score
        )

        # Update bobot prioritas
        self.learn_priority(
            task_idx, deadline_score, importance_level, effort_level
        )

        # Simpan histori aktivitas
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
    # RECOMMEND TIME
    # ============================

    def recommend_time(self, task_idx):
        """Rekomendasikan waktu terbaik berdasarkan Hebbian weights."""
        weights = st.session_state.time_weights[task_idx]

        if np.all(weights == 0):
            return None, 0

        best_idx = np.argmax(weights)
        return (
            self.times[best_idx],
            weights[best_idx]
        )

    # ============================
    # RECOMMEND PRIORITY
    # ============================

    def recommend_priority(self, task_idx):
        """Rekomendasikan prioritas berdasarkan Hebbian weights."""
        weights = st.session_state.priority_weights[task_idx]

        if np.all(weights == 0):
            return None, 0

        best_idx = np.argmax(weights)
        return (
            self.priority_levels[best_idx],
            weights[best_idx]
        )

    # ============================
    # GET ALL PRIORITIES
    # ============================

    def get_priority_ranking(self):
        """Dapatkan ranking prioritas untuk semua task."""
        rankings = []

        for idx, task in enumerate(st.session_state.tasks):
            priority, score = self.recommend_priority(idx)

            # Jika belum ada data, hitung dari history
            if priority is None:
                task_history = [h for h in st.session_state.history if h["task"] == task]
                if task_history:
                    avg_deadline = np.mean([h["deadline"] for h in task_history])
                    avg_importance = np.mean([h["importance"] for h in task_history])
                    avg_effort = np.mean([h["effort"] for h in task_history])

                    priority_score = (
                        avg_deadline * 0.4 +
                        avg_importance * 0.4 +
                        (10 - avg_effort) * 0.2
                    )

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

            rankings.append({
                "task": task,
                "priority": priority,
                "score": score
            })

        # Sort berdasarkan priority level dan score
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        rankings.sort(key=lambda x: (priority_order[x["priority"]], -x["score"]))

        return rankings


# ============================
# VISUALIZATION
# ============================

def plot_som_grid(som, X_scaled, labels, clusters_df, color_by="task", title="SOM Grid"):
    """Plot SOM Grid 2D dengan U-Matrix."""
    fig, ax = plt.subplots(figsize=(8, 8))

    umatrix = som.distance_map()
    im = ax.pcolor(umatrix.T, cmap='Blues_r')

    # Buat warna
    if color_by == "task":
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
        color_map = {label: colors[i] for i, label in enumerate(labels)}
        legend_labels = labels
    elif color_by == "priority":
        priority_colors = {"High": "red", "Medium": "orange", "Low": "green"}
        color_map = priority_colors
        legend_labels = ["High Priority", "Medium Priority", "Low Priority"]

    # Plot BMU untuk setiap data point
    for i, x in enumerate(X_scaled):
        bmu = som.winner(x)

        if color_by == "task":
            task_name = clusters_df.iloc[i]["task"]
            color = color_map.get(task_name, 'gray')
        else:
            color = color_map.get(clusters_df.iloc[i].get("priority_label", "Low"), "gray")

        ax.plot(
            bmu[0] + 0.5,
            bmu[1] + 0.5,
            'o',
            markersize=15,
            markerfacecolor=color,
            markeredgecolor='black',
            markeredgewidth=2
        )

    # Legend
    if color_by == "task":
        for idx, label in enumerate(labels):
            ax.plot([], [], 'o', markersize=10,
                    markerfacecolor=colors[idx],
                    markeredgecolor='black',
                    label=label)
    else:
        for priority, color in priority_colors.items():
            ax.plot([], [], 'o', markersize=10,
                    markerfacecolor=color,
                    markeredgecolor='black',
                    label=priority + " Priority")

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title(title)
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    plt.colorbar(im, ax=ax, label="Jarak (U-Matrix)")

    return fig
