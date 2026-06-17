import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt



# ============================
# KONFIGURASI STREAMLIT
# ============================

# Mengatur judul halaman dan layout menjadi wide (lebar)
st.set_page_config(
    page_title="kohonen SOM + Hebbian Task Prioritizer",
    layout="wide"
)



# ============================
# KONSTANTA
# ============================

# Daftar periode waktu dalam sehari untuk merepresentasikan
# kapan user biasanya produktif melakukan tugas
TIMES = [
    "Pagi (05-12)",      # Waktu produktivitas pagi hari
    "Siang (12-15)",     # Waktu produktivitas siang hari
    "Sore (15-18)",      # Waktu produktivitas sore hari
    "Malam (18-21)"      # Waktu produktivitas malam hari
]

# Nama file untuk menyimpan state aplikasi secara persistence
STATE_FILE = "app_state.json"




# ============================
# FUNGSI SIMPAN/MUAT STATE
# ============================

def save_state():
    """Simpan state aplikasi ke file JSON untuk persistensi data"""
    state = {
        "tasks": st.session_state.tasks,                                  # Daftar nama task
        "weights": st.session_state.weights.tolist() if len(st.session_state.weights) > 0 else [],  # Matriks bobot Hebbian
        "history": st.session_state.history                               # Riwayat aktivitas
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Muat state aplikasi dari file JSON jika ada"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return None



# ============================
# LOAD STATE AWAL
# ============================

# Cek apakah ada state tersimpan sebelumnya
saved_state = load_state()



# ============================
# HYBRID BRAIN CLASS
# ============================
# Kelas utama yang menggabungkan:
# 1. Hebbian Learning - belajar dari kebiasaan user
# 2. Kohonen SOM - mengelompokkan pola produktivitas

class HybridBrain:

    def __init__(self, times):
        self.times = times
        self.scaler = MinMaxScaler()        # Scaler untuk normalisasi fitur
        self.ensure_weight_shape()



    # ============================
    # RESIZE WEIGHT MATRIX
    # ============================
    # Menyesuaikan bentuk matriks bobot saat jumlah task berubah

    def ensure_weight_shape(self):
        # Bentuk yang dibutuhkan: (jumlah_task x jumlah_waktu)
        current_shape = st.session_state.weights.shape
        required_shape = (
            len(st.session_state.tasks),
            len(self.times)
        )

        # Jika bentuk tidak cocok, perlu resize matriks
        if current_shape != required_shape:
            new_weights = np.zeros(required_shape)  # Matriks baru dengan bentuk benar

            # Salin bobot lama yang masih valid (baris yang masih ada)
            min_rows = min(
                current_shape[0],
                required_shape[0]
            )

            if min_rows > 0:
                new_weights[:min_rows] = \
                    st.session_state.weights[:min_rows]

            # Simpan matriks bobot yang sudah disesuaikan
            st.session_state.weights = new_weights



    # ============================
    # SOM TRAINING
    # ============================
    # Melatih Self-Organizing Map (SOM) untuk clustering pola

    def train_som(self, X_scaled, num_iteration=200):
        """Latih SOM dengan MiniSOM untuk menemukan pola produktivitas"""
        # Inisialisasi SOM dengan grid 5x5 neuron
        som = MiniSom(
            x=5, y=5,                           # Ukuran grid SOM 5x5
            input_len=X_scaled.shape[1],         # Jumlah fitur input
            sigma=1.0,                           # Radius neighborhood awal
            learning_rate=0.5,                   # Kecepatan pembelajaran
            random_seed=42                       # Seed untuk reproducibility
        )
        som.random_weights_init(X_scaled)       # Inisialisasi bobot random
        som.train_random(X_scaled, num_iteration=num_iteration)  # Training SOM
        return som



    # ============================
    # SOM CLUSTERING
    # ============================
    # Menjalankan SOM untuk clustering data histori aktivitas

    def run_som(self):
        """Jalankan SOM clustering dan return hasil"""
        # Minimal butuh 4 data untuk clustering yang bermakna
        if len(st.session_state.history) < 4:
            return None, None, None

        # Konversi histori ke DataFrame
        df = pd.DataFrame(st.session_state.history)

        # Mapping nama task ke indeks numerik
        task_map = {
            task: idx
            for idx, task in enumerate(st.session_state.tasks)
        }

        # Mapping nama waktu ke indeks numerik
        time_map = {
            "Pagi (05-12)": 0,
            "Siang (12-15)": 1,
            "Sore (15-18)": 2,
            "Malam (18-21)": 3
        }

        # Konversi kolom kategorik ke numerik
        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        # Pilih fitur untuk clustering SOM
        # Fitur: task, waktu, durasi, energi, kesulitan, deadline
        X = df[[
            "task_num",
            "time_num",
            "duration",
            "energy",
            "difficulty",
            "deadline"
        ]]

        # Normalisasi fitur ke range 0-1 menggunakan MinMaxScaler
        X_scaled = self.scaler.fit_transform(X)

        # Latih SOM dengan data yang sudah dinormalisasi
        som = self.train_som(X_scaled, num_iteration=200)

        # Dapatkan Best Matching Unit (BMU) untuk setiap data point
        # BMU adalah neuron SOM yang paling mirip dengan input
        bmu_positions = [som.winner(x) for x in X_scaled]

        # Konversi posisi BMU (x,y) ke label cluster (0-24 untuk grid 5x5)
        df["cluster"] = [pos[0] * 5 + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]  # Koordinat X BMU
        df["bmu_y"] = [pos[1] for pos in bmu_positions]  # Koordinat Y BMU

        # Simpan hasil clustering ke session state
        st.session_state.clusters = df

        # Return X_scaled untuk plotting yang konsisten
        return df, som, X_scaled



    # ============================
    # HEBBIAN LEARNING
    # ============================
    # Mengupdate bobot asosiasi berdasarkan aktivitas yang dilakukan

    def learn(
        self,
        task_idx,           # Indeks task dalam daftar tasks
        time_idx,           # Indeks waktu yang dipilih
        duration_minutes,   # Durasi fokus dalam menit
        energy_level,       # Level energi user (1-10)
        difficulty_level,   # Tingkat kesulitan task (1-10)
        deadline_score      # Skor urgensi deadline (1-10)
    ):
        # Pre-synaptic activations (input features)
        # Menggambarkan kondisi saat aktivitas dilakukan
        x = np.array([
            energy_level / 10,                    # Energi dinormalisasi (0-1)
            difficulty_level / 10,               # Kesulitan dinormalisasi (0-1)
            deadline_score / 10,                  # Urgensi dinormalisasi (0-1)
            min(duration_minutes / 120, 1.0)      # Durasi dinormalisasi (max 120 menit)
        ])

        # Post-synaptic activation (output: time slot yang dipilih)
        # Representasi one-hot untuk waktu yang digunakan
        y = np.zeros(len(self.times))
        y[time_idx] = 1.0

        # Hyperparameters untuk Hebbian Learning
        lr = 0.1      # Learning rate - kecepatan pembelajaran
        decay = 0.005 # Decay rate -防止 bobot grows terlalu besar

        # Decay semua bobot untuk mencegah unbounded growth
        # Ini memastikan bobot tetap dalam batas yang stabil
        st.session_state.weights *= (1 - decay)

        # Hebbian update: w += lr * x * y
        # "Neuron yang aktif bersama, saling terhubung"
        # Hanya bobot untuk waktu yang dipilih (y[j]=1) yang diupdate
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                st.session_state.weights[task_idx, j] += lr * xi * yj

        # Clip bobot ke range [0, 10] untuk stabilitas numerik
        st.session_state.weights = np.clip(
            st.session_state.weights,
            0,
            10
        )

        # Simpan histori aktivitas untuk analisis dan visualisasi SOM
        st.session_state.history.append({
            "task": st.session_state.tasks[task_idx],
            "time": self.times[time_idx],
            "duration": duration_minutes,
            "energy": energy_level,
            "difficulty": difficulty_level,
            "deadline": deadline_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Simpan state ke JSON untuk persistensi
        save_state()



    # ============================
    # RECOMMENDATION
    # ============================
    # Memberikan rekomendasi waktu berdasarkan bobot yang sudah dipelajari

    def recommend(self, task_idx):
        """Rekomendasikan waktu terbaik berdasarkan Hebbian weights"""
        weights = st.session_state.weights[task_idx]

        # Jika belum ada pembelajaran (semua bobot 0)
        if np.all(weights == 0):
            return None, 0

        # Pilih waktu dengan bobot tertinggi (paling sering diasosiasikan)
        best_idx = np.argmax(weights)
        return (
            self.times[best_idx],    # Nama waktu yang direkomendasikan
            weights[best_idx]        # Skor kepercayaan rekomendasi
        )




# ============================
# INISIALISASI SESSION STATE
# ============================
# Session state digunakan untuk menyimpan data antar refresh halaman

if "tasks" not in st.session_state:
    # Load tasks dari state tersimpan, atau kosong jika tidak ada
    st.session_state.tasks = saved_state["tasks"] if saved_state else []

if "weights" not in st.session_state:
    # Load matriks bobot dari state tersimpan
    # Default: matriks kosong dengan shape (0, 4)
    weights_data = saved_state["weights"] if saved_state else []
    st.session_state.weights = np.array(weights_data) if weights_data else np.zeros((0, 4))

if "history" not in st.session_state:
    # Load histori aktivitas dari state tersimpan
    st.session_state.history = saved_state["history"] if saved_state else []

if "clusters" not in st.session_state:
    # DataFrame kosong untuk hasil clustering SOM
    st.session_state.clusters = pd.DataFrame()

if "brain" not in st.session_state:
    # Inisialisasi HybridBrain untuk managing SOM dan Hebbian Learning
    st.session_state.brain = HybridBrain(TIMES)



# ============================
# AMBIL INSTANCE BRAIN
# ============================


brain = st.session_state.brain



# ============================
# FUNGSI PLOTTING SOM GRID 2D
# ============================


def plot_som_grid(som, X_scaled, labels, clusters_df):
    """Plot SOM Grid 2D dengan U-Matrix dan BMU markers"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # U-Matrix (Unified Distance Matrix)
    # Menampilkan jarak antar neuron - area gelap = cluster, area terang = boundary
    umatrix = som.distance_map()
    im = ax.pcolor(umatrix.T, cmap='Blues_r')

    # Buat warna untuk setiap task
    colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}

    # Plot BMU untuk setiap data point
    for i, x in enumerate(X_scaled):
        bmu = som.winner(x)  # Dapatkan posisi BMU
        task_name = clusters_df.iloc[i]["task"]
        ax.plot(
            bmu[0] + 0.5,      # Offset ke tengah sel neuron
            bmu[1] + 0.5,
            'o',               # Bentuk marker circle
            markersize=15,     # Ukuran marker
            markerfacecolor=color_map.get(task_name, 'gray'),  # Warna berdasarkan task
            markeredgecolor='black',  # Border hitam
            markeredgewidth=2
        )

    # Tambahkan legend untuk identifikasi task
    for idx, label in enumerate(labels):
        ax.plot([], [], 'o', markersize=10,
                markerfacecolor=colors[idx],
                markeredgecolor='black',
                label=label)

    # Konfigurasi plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title("SOM Grid - Peta Produktivitas (5x5)")
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    plt.colorbar(im, ax=ax, label="Jarak (U-Matrix)")

    return fig



# ============================
# TAMPILAN UTAMA APLIKASI
# ============================


st.title("Hybrid Kohonen SOM + Hebbian Task Prioritizer")

st.write("""
Sistem menggunakan:

- **Hebbian Learning** -> mempelajari kebiasaan user berdasarkan asosiasi fitur-waktu

- **Kohonen SOM** -> mengelompokkan pola produktivitas dalam grid 2D (5x5)
""")



# ============================
# SIDEBAR - INPUT DATA
# ============================


st.sidebar.header("Tambah Task Baru")

# Input nama task baru
new_task = st.sidebar.text_input("Nama Task")

# Tombol untuk menambahkan task
if st.sidebar.button("Tambah Task"):
    # Normalisasi nama task: capitalize setiap kata, hapus spasi extra
    cleaned_task = new_task.strip().title()

    # Validasi input
    if not cleaned_task or len(cleaned_task) < 2:
        st.sidebar.error("Nama task minimal 2 karakter.")
    elif cleaned_task in st.session_state.tasks:
        st.sidebar.warning("Task sudah ada.")
    else:
        # Tambahkan task dan update matriks bobot
        st.session_state.tasks.append(cleaned_task)
        brain.ensure_weight_shape()  # Sesuaikan shape matriks bobot
        save_state()
        st.sidebar.success(f"Task '{cleaned_task}' ditambahkan.")



# ============================
# CEK JIKA BELUM ADA TASK
# ============================


if len(st.session_state.tasks) == 0:
    st.warning("Tambahkan minimal 1 task.")
    st.stop()  # Stop eksekusi jika tidak ada task



# ============================
# INPUT AKTIVITAS
# ============================


st.sidebar.header("Input Aktivitas")

# Dropdown untuk pilih task
selected_task = st.sidebar.selectbox(
    "Jenis Task",
    st.session_state.tasks
)

# Dropdown untuk pilih waktu
selected_time = st.sidebar.selectbox(
    "Waktu",
    TIMES
)

# Slider untuk durasi fokus (10-240 menit)
focus_duration = st.sidebar.slider(
    "Durasi Fokus (menit)",
    10, 240, 60, 10
)

# Slider untuk level energi (1-10)
energy_level = st.sidebar.slider(
    "Level Energi",
    1, 10, 5
)

# Slider untuk tingkat kesulitan (1-10)
difficulty_level = st.sidebar.slider(
    "Tingkat Kesulitan",
    1, 10, 5
)

# Slider untuk tingkat deadline (1-10)
deadline_score = st.sidebar.slider(
    "Tingkat Deadline",
    1, 10, 5
)



# ============================
# SIMPAN AKTIVITAS
# ============================


if st.sidebar.button("Simpan Aktivitas"):
    # Konversi pilihan ke indeks
    task_idx = st.session_state.tasks.index(selected_task)
    time_idx = TIMES.index(selected_time)

    # Jalankan Hebbian learning untuk update bobot
    brain.learn(
        task_idx,
        time_idx,
        focus_duration,
        energy_level,
        difficulty_level,
        deadline_score
    )

    st.sidebar.success("AI memory diperbarui.")



# ============================
# DASHBOARD - TAMPILAN UTAMA
# ============================


col1, col2 = st.columns(2)  # Dua kolom untuk layout



# ============================
# WEIGHT MATRIX
# ============================
# Menampilkan matriks bobot Hebbian

with col1:
    st.subheader("Hebbian Weight Matrix")

    if len(st.session_state.weights) > 0:
        # Konversi matriks bobot ke DataFrame untuk tampilan
        df_weights = pd.DataFrame(
            st.session_state.weights,
            index=st.session_state.tasks,
            columns=TIMES
        )

        # Tampilkan dengan gradient warna (lebih gelap = bobot lebih tinggi)
        st.dataframe(
            df_weights.style.background_gradient(
                cmap="Blues",
                axis=None
            )
        )
    else:
        st.info("Belum ada data.")



# ============================
# RECOMMENDATION
# ============================
# Panel rekomendasi waktu

with col2:
    st.subheader("Rekomendasi Waktu")

    # Dropdown untuk pilih task yang ingin direkomendasikan
    target_task = st.selectbox(
        "Pilih Task",
        st.session_state.tasks
    )

    # Dapatkan rekomendasi dari brain
    best_time, score = brain.recommend(
        st.session_state.tasks.index(target_task)
    )

    # Tampilkan hasil rekomendasi
    if best_time:
        st.success(f"Waktu terbaik: {best_time}")
        st.write(f"Skor asosiasi: {score:.2f}")  # Skor kepercayaan
        st.progress(min(score / 10, 1.0))        # Progress bar
    else:
        st.warning("Belum ada pembelajaran.")



# ============================
# SOM VISUALIZATION
# ============================
# Tampilkan grid SOM 2D dengan clustering

st.subheader("SOM Grid 2D (5x5)")

if len(st.session_state.history) >= 4:
    # Jalankan SOM clustering
    df, som, X_scaled = brain.run_som()

    if df is not None:
        # Plot dan tampilkan SOM grid
        fig = plot_som_grid(som, X_scaled, st.session_state.tasks, df)
        st.pyplot(fig)
else:
    st.info("Minimal 4 aktivitas diperlukan untuk visualisasi SOM.")



# ============================
# HISTORY
# ============================
# Tampilkan semua aktivitas yang sudah direkam

st.subheader("Histori Aktivitas")

if len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.info("Belum ada histori.")



# ============================
# CLUSTER INTERPRETATION
# ============================
# Interpretasi hasil clustering SOM

st.subheader("Interpretasi Cluster Productivity")

if len(st.session_state.clusters) > 0:
    cluster_df = st.session_state.clusters

    # Proses setiap cluster yang unik
    for cluster_id in sorted(cluster_df["cluster"].unique()):
        subset = cluster_df[cluster_df["cluster"] == cluster_id]

        # Hitung statistik untuk cluster ini
        dominant_task = subset["task"].mode()[0]     # Task paling sering
        avg_duration = subset["duration"].mean()     # Rata-rata durasi
        avg_energy = subset["energy"].mean()          # Rata-rata energi
        avg_deadline = subset["deadline"].mean()     # Rata-rata deadline
        dominant_time = subset["time"].mode()[0]    # Waktu paling produktif

        # Hitung combined score untuk menentukan priority
        # Formula: berbobot pada energi, deadline, dan durasi
        combined_score = (
            avg_energy / 10 * 0.3 +     # 30% bobot energi
            avg_deadline / 10 * 0.4 +   # 40% bobot deadline (paling penting)
            avg_duration / 240 * 0.3   # 30% bobot durasi
        )

        # Tentukan label priority berdasarkan combined score
        if combined_score > 0.6:
            priority_label = "High Priority"    # Skor > 0.6
        elif combined_score > 0.4:
            priority_label = "Medium Priority"  # Skor > 0.4
        else:
            priority_label = "Low Priority"     # Skor <= 0.4

        # Tampilkan hasil interpretasi dalam markdown
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

        # Tampilkan detail data dalam cluster
        st.dataframe(subset[["task", "time", "duration", "energy", "difficulty", "deadline"]])
else:
    st.warning("Belum cukup data untuk clustering.")



# ============================
# DELETE TASK
# ============================
# Manajemen hapus task

st.subheader("Manajemen Task")

# Dropdown untuk pilih task yang akan dihapus
task_to_delete = st.selectbox(
    "Hapus Task",
    st.session_state.tasks
)

# Tombol hapus task
if st.button("Hapus Task"):
    idx = st.session_state.tasks.index(task_to_delete)
    st.session_state.tasks.pop(idx)

    # Hapus baris bobot yang sesuai
    if len(st.session_state.tasks) > 0:
        st.session_state.weights = np.delete(
            st.session_state.weights, idx, axis=0
        )
    else:
        st.session_state.weights = np.zeros((0, 4))

    save_state()
    st.success(f"Task '{task_to_delete}' dihapus.")
    st.rerun()  # Refresh halaman



# ============================
# RESET SYSTEM
# ============================
# Hapus semua data dan reset sistem

if st.button("Reset Semua Memori"):
    # Reset semua session state ke nilai awal
    st.session_state.tasks = []
    st.session_state.weights = np.zeros((0, 4))
    st.session_state.history = []
    st.session_state.clusters = pd.DataFrame()

    # Hapus file state jika ada
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    st.rerun()
