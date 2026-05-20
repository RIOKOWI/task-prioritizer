import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt



# STREAMLIT CONFIG

st.set_page_config(
    page_title="kohonen SOM + Hebbian Task Prioritizer",
    layout="wide"
)



# CONSTANT

TIMES = [
    "Pagi (05-12)",
    "Siang (12-15)",
    "Sore (15-18)",
    "Malam (18-21)"
]

STATE_FILE = "app_state.json"





def save_state():
    """Simpan state ke file JSON"""
    state = {
        "tasks": st.session_state.tasks,
        "weights": st.session_state.weights.tolist() if len(st.session_state.weights) > 0 else [],
        "history": st.session_state.history
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Muat state dari file JSON"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None



# LOAD SAVED STATE

saved_state = load_state()



# HYBRID BRAIN (DEFINED BEFORE SESSION STATE INIT)

class HybridBrain:

    def __init__(self, times):
        self.times = times
        self.scaler = MinMaxScaler()
        self.ensure_weight_shape()


    
    # RESIZE WEIGHT MATRIX

    def ensure_weight_shape(self):
        current_shape = st.session_state.weights.shape
        required_shape = (
            len(st.session_state.tasks),
            len(self.times)
        )

        if current_shape != required_shape:
            new_weights = np.zeros(required_shape)

            min_rows = min(
                current_shape[0],
                required_shape[0]
            )

            if min_rows > 0:
                new_weights[:min_rows] = \
                    st.session_state.weights[:min_rows]

            st.session_state.weights = new_weights


    
    # SOM TRAINING

    def train_som(self, X_scaled, num_iteration=200):
        """Latih SOM dengan MiniSOM"""
        som = MiniSom(
            x=5, y=5,
            input_len=X_scaled.shape[1],
            sigma=1.0,
            learning_rate=0.5,
            random_seed=42
        )
        som.random_weights_init(X_scaled)
        som.train_random(X_scaled, num_iteration=num_iteration)
        return som


    
    # SOM CLUSTERING (RETURN X_SCALED FOR CONSISTENCY)

    def run_som(self):
        if len(st.session_state.history) < 4:
            return None, None, None

        df = pd.DataFrame(st.session_state.history)

        # Encode task dan time
        task_map = {
            task: idx
            for idx, task in enumerate(st.session_state.tasks)
        }

        time_map = {
            "Pagi (05-12)": 0,
            "Siang (12-15)": 1,
            "Sore (15-18)": 2,
            "Malam (18-21)": 3
        }

        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        # Features untuk SOM
        X = df[[
            "task_num",
            "time_num",
            "duration",
            "energy",
            "difficulty",
            "deadline"
        ]]

        X_scaled = self.scaler.fit_transform(X)

        # Latih SOM
        som = self.train_som(X_scaled, num_iteration=200)

        # Get BMU untuk setiap data point
        bmu_positions = [som.winner(x) for x in X_scaled]

        # Convert BMU positions ke cluster labels (x*5 + y)
        df["cluster"] = [pos[0] * 5 + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        st.session_state.clusters = df

        # Return X_scaled for consistent plotting
        return df, som, X_scaled


    
    # HEBBIAN LEARNING (FORMULA BENAR)

    def learn(
        self,
        task_idx,
        time_idx,
        duration_minutes,
        energy_level,
        difficulty_level,
        deadline_score
    ):
        # Pre-synaptic activations (input features)
        x = np.array([
            energy_level / 10,
            difficulty_level / 10,
            deadline_score / 10,
            min(duration_minutes / 120, 1.0)  # normalized duration
        ])

        # Post-synaptic activation (output: time slot yang dipilih - one-hot)
        y = np.zeros(len(self.times))
        y[time_idx] = 1.0

        # Learning parameters
        lr = 0.1
        decay = 0.005

        # Decay semua weight (prevent unbounded growth)
        st.session_state.weights *= (1 - decay)

        # Hebbian update: w += lr * x * y
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                st.session_state.weights[task_idx, j] += lr * xi * yj

        # Clip to prevent overflow
        st.session_state.weights = np.clip(
            st.session_state.weights,
            0,
            10
        )

        # Simpan histori
        st.session_state.history.append({
            "task": st.session_state.tasks[task_idx],
            "time": self.times[time_idx],
            "duration": duration_minutes,
            "energy": energy_level,
            "difficulty": difficulty_level,
            "deadline": deadline_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Save state to JSON
        save_state()


    
    # RECOMMENDATION

    def recommend(self, task_idx):
        weights = st.session_state.weights[task_idx]

        if np.all(weights == 0):
            return None, 0

        best_idx = np.argmax(weights)
        return (
            self.times[best_idx],
            weights[best_idx]
        )





if "tasks" not in st.session_state:
    st.session_state.tasks = saved_state["tasks"] if saved_state else []

if "weights" not in st.session_state:
    weights_data = saved_state["weights"] if saved_state else []
    st.session_state.weights = np.array(weights_data) if weights_data else np.zeros((0, 4))

if "history" not in st.session_state:
    st.session_state.history = saved_state["history"] if saved_state else []

if "clusters" not in st.session_state:
    st.session_state.clusters = pd.DataFrame()

if "brain" not in st.session_state:
    st.session_state.brain = HybridBrain(TIMES)



# INIT BRAIN


brain = st.session_state.brain



# PLOT SOM GRID 2D


def plot_som_grid(som, X_scaled, labels, clusters_df):
    """Plot SOM Grid 2D dengan U-Matrix dan BMU markers"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Distance map (U-Matrix)
    umatrix = som.distance_map()
    im = ax.pcolor(umatrix.T, cmap='Blues_r')

    # Plot BMU untuk setiap data point
    colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}

    for i, x in enumerate(X_scaled):
        bmu = som.winner(x)
        task_name = clusters_df.iloc[i]["task"]
        ax.plot(
            bmu[0] + 0.5,
            bmu[1] + 0.5,
            'o',
            markersize=15,
            markerfacecolor=color_map.get(task_name, 'gray'),
            markeredgecolor='black',
            markeredgewidth=2
        )

    # Add legend
    for idx, label in enumerate(labels):
        ax.plot([], [], 'o', markersize=10,
                markerfacecolor=colors[idx],
                markeredgecolor='black',
                label=label)

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title("SOM Grid - Peta Produktivitas (5x5)")
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    plt.colorbar(im, ax=ax, label="Jarak (U-Matrix)")

    return fig



# HEADER


st.title("Hybrid SOM + Hebbian Task Prioritizer")

st.write("""
Sistem menggunakan:

- **Hebbian Learning** -> mempelajari kebiasaan user berdasarkan asosiasi fitur-waktu

- **Kohonen SOM** -> mengelompokkan pola produktivitas dalam grid 2D (5x5)
""")



# SIDEBAR


st.sidebar.header("Tambah Task Baru")

new_task = st.sidebar.text_input("Nama Task")

if st.sidebar.button("Tambah Task"):
    cleaned_task = new_task.strip().title()

    # Validasi input
    if not cleaned_task or len(cleaned_task) < 2:
        st.sidebar.error("Nama task minimal 2 karakter.")
    elif cleaned_task in st.session_state.tasks:
        st.sidebar.warning("Task sudah ada.")
    else:
        st.session_state.tasks.append(cleaned_task)
        brain.ensure_weight_shape()
        save_state()
        st.sidebar.success(f"Task '{cleaned_task}' ditambahkan.")



# STOP IF NO TASK


if len(st.session_state.tasks) == 0:
    st.warning("Tambahkan minimal 1 task.")
    st.stop()



# INPUT ACTIVITY


st.sidebar.header("Input Aktivitas")

selected_task = st.sidebar.selectbox(
    "Jenis Task",
    st.session_state.tasks
)

selected_time = st.sidebar.selectbox(
    "Waktu",
    TIMES
)

focus_duration = st.sidebar.slider(
    "Durasi Fokus (menit)",
    10, 240, 60, 10
)

energy_level = st.sidebar.slider(
    "Level Energi",
    1, 10, 5
)

difficulty_level = st.sidebar.slider(
    "Tingkat Kesulitan",
    1, 10, 5
)

deadline_score = st.sidebar.slider(
    "Tingkat Deadline",
    1, 10, 5
)



# SAVE ACTIVITY (NO DUPLICATE run_som CALL)


if st.sidebar.button("Simpan Aktivitas"):
    task_idx = st.session_state.tasks.index(selected_task)
    time_idx = TIMES.index(selected_time)

    brain.learn(
        task_idx,
        time_idx,
        focus_duration,
        energy_level,
        difficulty_level,
        deadline_score
    )

    # NOTE: run_som() dipanggil hanya di section SOM VISUALIZATION
    # Tidak perlu dipanggil di sini untuk menghindari duplicate call

    st.sidebar.success("AI memory diperbarui.")



# DASHBOARD


col1, col2 = st.columns(2)



# WEIGHT MATRIX


with col1:
    st.subheader("Hebbian Weight Matrix")

    if len(st.session_state.weights) > 0:
        df_weights = pd.DataFrame(
            st.session_state.weights,
            index=st.session_state.tasks,
            columns=TIMES
        )

        st.dataframe(
            df_weights.style.background_gradient(
                cmap="Blues",
                axis=None
            )
        )
    else:
        st.info("Belum ada data.")



# RECOMMENDATION


with col2:
    st.subheader("Rekomendasi Waktu")

    target_task = st.selectbox(
        "Pilih Task",
        st.session_state.tasks
    )

    best_time, score = brain.recommend(
        st.session_state.tasks.index(target_task)
    )

    if best_time:
        st.success(f"Waktu terbaik: {best_time}")
        st.write(f"Skor asosiasi: {score:.2f}")
        st.progress(min(score / 10, 1.0))
    else:
        st.warning("Belum ada pembelajaran.")



# SOM VISUALIZATION (USE RETURNED X_SCALED)


st.subheader("SOM Grid 2D (5x5)")

if len(st.session_state.history) >= 4:
    # Get clusters, SOM, dan X_scaled (consistent, no re-fit)
    df, som, X_scaled = brain.run_som()

    if df is not None:
        fig = plot_som_grid(som, X_scaled, st.session_state.tasks, df)
        st.pyplot(fig)
else:
    st.info("Minimal 4 aktivitas diperlukan untuk visualisasi SOM.")



# HISTORY


st.subheader("Histori Aktivitas")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.info("Belum ada histori.")



# CLUSTER INTERPRETATION (IMPROVED)


st.subheader("Interpretasi Cluster Productivity")

if len(st.session_state.clusters) > 0:
    cluster_df = st.session_state.clusters

    for cluster_id in sorted(cluster_df["cluster"].unique()):
        subset = cluster_df[cluster_df["cluster"] == cluster_id]

        dominant_task = subset["task"].mode()[0]
        avg_duration = subset["duration"].mean()
        avg_energy = subset["energy"].mean()
        avg_deadline = subset["deadline"].mean()
        dominant_time = subset["time"].mode()[0]

        # Priority label based on combined score
        combined_score = (
            avg_energy / 10 * 0.3 +
            avg_deadline / 10 * 0.4 +
            avg_duration / 240 * 0.3
        )

        if combined_score > 0.6:
            priority_label = "High Priority"
        elif combined_score > 0.4:
            priority_label = "Medium Priority"
        else:
            priority_label = "Low Priority"

        st.markdown(f"""
        ### Cluster {cluster_id}: {priority_label}

        | Metrik | Nilai |
        |--------|-------|
        | Task Dominan | {dominant_task} |
        | Waktu Paling Produktif | {dominant_time} |
        | Rata-rata Durasi | {avg_duration:.1f} menit |
        | Rata-rata Energi | {avg_energy:.1f} |
        | Rata-rata Deadline | {avg_deadline:.1f} |
        """)

        # Tampilkan tabel cluster
        st.dataframe(subset[["task", "time", "duration", "energy", "difficulty", "deadline"]])
else:
    st.warning("Belum cukup data untuk clustering.")



# DELETE TASK


st.subheader("Manajemen Task")

task_to_delete = st.selectbox(
    "Hapus Task",
    st.session_state.tasks
)

if st.button("Hapus Task"):
    idx = st.session_state.tasks.index(task_to_delete)
    st.session_state.tasks.pop(idx)

    if len(st.session_state.tasks) > 0:
        st.session_state.weights = np.delete(
            st.session_state.weights, idx, axis=0
        )
    else:
        st.session_state.weights = np.zeros((0, 4))

    save_state()
    st.success(f"Task '{task_to_delete}' dihapus.")
    st.rerun()



# RESET SYSTEM


if st.button("Reset Semua Memori"):
    st.session_state.tasks = []
    st.session_state.weights = np.zeros((0, 4))
    st.session_state.history = []
    st.session_state.clusters = pd.DataFrame()

    # Hapus file state
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    st.rerun()