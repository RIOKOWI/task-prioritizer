import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime


# HEBBIAN BRAIN
class HebbianBrain:

    def __init__(self, tasks, times):

        self.tasks = tasks
        self.times = times

        # Membuat memori AI jika belum ada
        if "weights" not in st.session_state:
            st.session_state.weights = np.zeros((len(tasks), len(times)))

        # Menyimpan histori aktivitas
        if "history" not in st.session_state:
            st.session_state.history = []


    # UBAH DURASI MENJADI AKTIVASI NEURON
    def activation(self, duration_minutes):

        """
        Aktivasi neuron:
        0 menit  = 0.0
        120 menit = 1.0 (maksimal)
        """

        return min(duration_minutes / 120, 1.0)


    # HEBBIAN UPDATE
    def learn(self, task_idx, time_idx, duration_minutes, lr=0.3):

        # Aktivasi neuron berdasarkan durasi fokus
        act = self.activation(duration_minutes)

        # DECAY
        st.session_state.weights *= 0.995

        # HEBBIAN LEARNING
        # "Neuron that fires together wires together"
        st.session_state.weights[task_idx, time_idx] += lr * act

        # Batasi bobot
        st.session_state.weights = np.clip(
            st.session_state.weights,
            0,
            10
        )

        # Simpan histori
        st.session_state.history.append({
            "task": self.tasks[task_idx],
            "time": self.times[time_idx],
            "duration": duration_minutes,
            "activation": round(act, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })



    # REKOMENDASI
    def recommend(self, task_idx):

        weights = st.session_state.weights[task_idx]

        if np.all(weights == 0):
            return None, 0

        best_idx = np.argmax(weights)

        return self.times[best_idx], weights[best_idx]



# STREAMLIT CONFIG
st.set_page_config(
    page_title="Hebbian Task Prioritizer System",
    layout="wide"
)

# DATASET
TASKS = [
    "Coding",
    "Exercise",
    "Research",
    "Reading"
]

TIMES = [
    "Pagi (05-12)",
    "Siang (12-15)",
    "Sore (15-18)",
    "Malam (18-21)"
]

brain = HebbianBrain(TASKS, TIMES)


# HEADER
st.title("Hebbian Task Prioritizer System")

st.write("""
Sistem belajar dari hubungan antara:

- Jenis tugas
- Waktu pengerjaan
- Durasi fokus

Semakin sering suatu tugas dilakukan secara fokus pada waktu tertentu,
semakin kuat koneksi memorinya.
""")


# SIDEBAR INPUT
st.sidebar.header("Log Aktivitas")

selected_task = st.sidebar.selectbox(
    "Jenis Tugas",
    TASKS
)

selected_time = st.sidebar.selectbox(
    "Waktu Pengerjaan",
    TIMES
)

focus_duration = st.sidebar.slider(
    "Durasi Fokus (menit)",
    min_value=10,
    max_value=240,
    value=60,
    step=10
)


# BUTTON LEARN
if st.sidebar.button("Simpan Aktivitas"):

    task_idx = TASKS.index(selected_task)
    time_idx = TIMES.index(selected_time)

    brain.learn(
        task_idx,
        time_idx,
        focus_duration
    )

    st.sidebar.success("Memori Hebbian diperbarui.")


# DASHBOARD
col1, col2 = st.columns(2)


# KOLOM 1 - WEIGHTS
with col1:

    st.subheader("Hebbian Weight Matrix")

    df = pd.DataFrame(
        st.session_state.weights,
        index=TASKS,
        columns=TIMES
    )

    st.dataframe(
        df.style.background_gradient(
            cmap="Blues",
            axis=None
        )
    )

    st.caption("""
    Semakin biru:
    semakin kuat hubungan antara tugas dan waktu.
    """)


# KOLOM 2 - REKOMENDASI
with col2:

    st.subheader("Rekomendasi Waktu Terbaik")

    target_task = st.selectbox(
        "Pilih Tugas",
        TASKS
    )

    best_time, score = brain.recommend(
        TASKS.index(target_task)
    )

    if best_time:

        st.success(
            f"Waktu terbaik untuk '{target_task}' adalah '{best_time}'"
        )

        st.write(f"Kekuatan asosiasi: {score:.2f}")

        st.progress(min(score / 10, 1.0))

    else:

        st.warning(
            "Belum ada data pembelajaran."
        )


# HISTORI AKTIVITAS
st.subheader("Histori Aktivitas")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(history_df)

else:

    st.info("Belum ada histori aktivitas.")


# RESET MEMORY
if st.button("Reset Semua Memori"):

    st.session_state.weights = np.zeros(
        (len(TASKS), len(TIMES))
    )

    st.session_state.history = []

    st.rerun()