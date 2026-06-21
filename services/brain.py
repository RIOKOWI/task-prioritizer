"""
HybridBrain: Kelas utama yang menggabungkan Hebbian Learning dan Kohonen SOM.
- Hebbian Learning -> belajar dari kebiasaan user (waktu & prioritas)
- Kohonen SOM -> mengelompokkan pola produktivitas dan prioritas
"""

# Import library standar untuk manipulasi data dan file
import json      # Untuk menyimpan dan memuat data dalam format JSON
import os       # Untuk memeriksa keberadaan file state
import numpy as np              # Untuk operasi array dan perhitungan numerik
import pandas as pd              # Untuk manipulasi data tabular (DataFrame)
import streamlit as st           # Untuk mengakses session_state Streamlit
import matplotlib.pyplot as plt  # Untuk membuat visualisasi grafik
from datetime import datetime    # Untuk mencatat timestamp saat menyimpan histori
from sklearn.preprocessing import MinMaxScaler  # Untuk normalisasi data ke rentang 0-1
from minisom import MiniSom       # Library MiniSOM untuk Self-Organizing Map
from matplotlib.lines import Line2D   # Untuk membuat legenda visualisasi kustom

# Import konstanta dari file config
from config.config import TIMES, PRIORITY_LEVELS, STATE_FILE


# ============================
# FUNGSI SIMPAN/MUAT STATE
# ============================

def save_state():
    """Simpan state aplikasi ke file JSON untuk persistensi data."""
    # Buat dictionary yang berisi semua data state aplikasi
    state = {
        "tasks": st.session_state.tasks,   # Daftar semua task yang ada
        "time_weights": st.session_state.time_weights.tolist() if len(st.session_state.time_weights) > 0 else [],  # Bobot asosiasi task-waktu
        "priority_weights": st.session_state.priority_weights.tolist() if len(st.session_state.priority_weights) > 0 else [],  # Bobot asosiasi task-prioritas
        "history": st.session_state.history  # Riwayat semua pembelajaran
    }
    # Tulis semua data ke file JSON dengan indentasi untuk keterbacaan
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    """Muat state aplikasi dari file JSON jika ada."""
    # Periksa apakah file state sudah ada
    if os.path.exists(STATE_FILE):
        # Baca dan parse file JSON
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    # Kembalikan None jika file tidak ditemukan
    return None


# ============================
# HYBRID BRAIN CLASS
# ============================

class HybridBrain:
    """Kelas utama yang menggabungkan Hebbian Learning dan Kohonen SOM."""

    def __init__(self, times, priority_levels):
        # Simpan daftar waktu yang digunakan aplikasi
        self.times = times
        # Simpan daftar level prioritas
        self.priority_levels = priority_levels
        # Inisialisasi scaler untuk menormalisasi fitur waktu (rentang 0-1)
        self.scaler_time = MinMaxScaler()
        # Inisialisasi scaler untuk menormalisasi fitur prioritas
        self.scaler_priority = MinMaxScaler()
        # Inisialisasi scaler untuk clustering unified (gabungan)
        self.scaler_cluster = MinMaxScaler()
        # Ukuran grid SOM: 5x5 = 25 neuron/cluster
        self.grid_size = 5
        # Pastikan matriks bobot memiliki dimensi yang benar
        self.ensure_weight_shapes()

    def ensure_weight_shapes(self):
        """Pastikan kedua matriks bobot memiliki bentuk yang benar."""
        # Ambil bentuk matriks bobot waktu saat ini
        time_shape = st.session_state.time_weights.shape
        # Hitung bentuk yang diperlukan berdasarkan jumlah task dan waktu
        required_time_shape = (len(st.session_state.tasks), len(self.times))

        # Jika bentuk tidak sesuai, perlu dilakukan resize
        if time_shape != required_time_shape:
            # Buat matriks nol dengan dimensi yang benar
            new_time_weights = np.zeros(required_time_shape)
            # Hitung jumlah baris minimum yang bisa disalin
            min_rows = min(time_shape[0], required_time_shape[0])
            # Salin data lama jika ada
            if min_rows > 0:
                new_time_weights[:min_rows] = st.session_state.time_weights[:min_rows]
            # Update session state dengan matriks baru
            st.session_state.time_weights = new_time_weights

        # Ambil bentuk matriks bobot prioritas saat ini
        priority_shape = st.session_state.priority_weights.shape
        # Hitung bentuk yang diperlukan
        required_priority_shape = (len(st.session_state.tasks), len(self.priority_levels))

        # Jika bentuk tidak sesuai, perlu dilakukan resize
        if priority_shape != required_priority_shape:
            # Buat matriks nol dengan dimensi yang benar
            new_priority_weights = np.zeros(required_priority_shape)
            # Hitung jumlah baris minimum yang bisa disalin
            min_rows = min(priority_shape[0], required_priority_shape[0])
            # Salin data lama jika ada
            if min_rows > 0:
                new_priority_weights[:min_rows] = st.session_state.priority_weights[:min_rows]
            # Update session state dengan matriks baru
            st.session_state.priority_weights = new_priority_weights

    def train_som(self, X_scaled, num_iteration=200):
        """Latih SOM dengan MiniSOM untuk menemukan pola."""
        # Inisialisasi SOM dengan grid 5x5 neuron
        som = MiniSom(
            x=self.grid_size, y=self.grid_size,  # Dimensi grid (5x5)
            input_len=X_scaled.shape[1],          # Jumlah fitur input
            sigma=1.0,                             # Radius neighbourhood (semi-supervised learning)
            learning_rate=0.5,                    # Kecepatan pembelajaran
            random_seed=42                         # Seed untuk reproducibility
        )
        # Inisialisasi bobot awal secara random dari data
        som.random_weights_init(X_scaled)
        # Latih SOM dengan iterasi acak dari data
        som.train_random(X_scaled, num_iteration=num_iteration)
        return som  # Kembalikan model SOM yang sudah dilatih

    # ============================
    # SOM CLUSTERING - WAKTU
    # ============================

    def run_som_time(self):
        """Jalankan SOM clustering berdasarkan pola waktu."""
        # Butuh minimal 4 data historis untuk melakukan clustering bermakna
        if len(st.session_state.history) < 4:
            return None, None, None, None

        # Konversi histori ke DataFrame untuk manipulasi data mudah
        df = pd.DataFrame(st.session_state.history)

        # Buat mapping nama task ke indeks numerik
        # Berguna untuk mengubah data kategorikal menjadi numerik
        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        # Mapping waktu ke indeks numerik
        # Mengkategorikan waktu ke dalam 4 periode hari
        time_map = {
            "Pagi (05-12)": 0,    # Indeks untuk periode pagi
            "Siang (12-15)": 1,   # Indeks untuk periode siang
            "Sore (15-18)": 2,    # Indeks untuk periode sore
            "Malam (18-21)": 3    # Indeks untuk periode malam
        }

        # Konversi nama task ke indeks numerik menggunakan mapping
        df["task_num"] = df["task"].map(task_map)
        # Konversi nama waktu ke indeks numerik menggunakan mapping
        df["time_num"] = df["time"].map(time_map)

        # Pilih fitur-fitur yang relevan untuk clustering waktu
        # Termasuk: task, waktu, durasi, energi, kesulitan, dan deadline
        X = df[["task_num", "time_num", "duration", "energy", "difficulty", "deadline"]]
        # Normalisasi semua fitur ke rentang 0-1
        X_scaled = self.scaler_time.fit_transform(X)

        # Latih SOM dengan data yang sudah dinormalisasi
        som = self.train_som(X_scaled, num_iteration=200)

        # Temukan Best Matching Unit (BMU) untuk setiap data point
        # BMU adalah neuron SOM yang paling mirip dengan data input
        bmu_positions = [som.winner(x) for x in X_scaled]
        # Hitung ID cluster (0-24) berdasarkan posisi BMU
        df["cluster"] = [pos[0] * self.grid_size + pos[1] for pos in bmu_positions]
        # Simpan koordinat X BMU untuk visualisasi
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        # Simpan koordinat Y BMU untuk visualisasi
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        # Simpan hasil clustering ke session state
        st.session_state.clusters_time = df
        # Kembalikan DataFrame, model SOM, data ter-scaling, dan error kuantisasi
        return df, som, X_scaled, som.quantization_error(X_scaled)

    # ============================
    # SOM CLUSTERING - PRIORITAS
    # ============================

    def _calc_priority_score(self, cluster_data):
        """Hitung priority score untuk satu cluster."""
        # Jika cluster kosong, tidak bisa menghitung score
        if len(cluster_data) == 0:
            return None
        # Hitung weighted average dari berbagai faktor:
        # - deadline: 30% (semakin dekat deadline, semakin tinggi prioritas)
        # - importance: 30% (task penting lebih diutamakan)
        # - effort: 20% (effort rendah lebih baik)
        # - energy: 10% (butuh energi rendah lebih baik)
        # - difficulty: 10% (semakin mudah semakin baik)
        return (
            cluster_data["deadline"].mean() / 10 * 0.3 +   # Rata-rata deadline (skala 0-10)
            cluster_data["importance"].mean() / 10 * 0.3 +  # Rata-rata importance
            cluster_data["effort"].mean() / 10 * 0.2 +     # Rata-rata effort
            cluster_data["energy"].mean() / 10 * 0.1 +    # Rata-rata energy
            cluster_data["difficulty"].mean() / 10 * 0.1   # Rata-rata difficulty
        )

    def _assign_label(self, score):
        """Assign label berdasarkan threshold ABSOLUT."""
        # Jika tidak ada data, label adalah "No Data"
        if score is None:
            return "No Data"
        # Score > 0.6 = High priority (prioritas tinggi)
        if score > 0.6:
            return "High"
        # Score > 0.4 = Medium priority (prioritas sedang)
        elif score > 0.4:
            return "Medium"
        # Score <= 0.4 = Low priority (prioritas rendah)
        return "Low"

    def run_som_priority(self):
        """Jalankan SOM clustering berdasarkan pola prioritas."""
        # Minimal perlu 4 data untuk clustering bermakna
        if len(st.session_state.history) < 4:
            return None, None, None, None

        # Konversi histori ke DataFrame
        df = pd.DataFrame(st.session_state.history)

        # Mapping nama task ke indeks numerik
        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        # Konversi task ke indeks numerik
        df["task_num"] = df["task"].map(task_map)

        # Pilih fitur-fitur yang relevan untuk prioritas:
        # task, deadline, importance, effort, energy, difficulty
        X = df[["task_num", "deadline", "importance", "effort", "energy", "difficulty"]]
        # Normalisasi fitur ke rentang 0-1
        X_scaled = self.scaler_priority.fit_transform(X)

        # Latih SOM dengan data ter-scaling
        som = self.train_som(X_scaled, num_iteration=200)

        # Temukan BMU untuk setiap data point
        bmu_positions = [som.winner(x) for x in X_scaled]
        # Hitung ID cluster dari posisi BMU
        df["priority_cluster"] = [pos[0] * self.grid_size + pos[1] for pos in bmu_positions]
        # Simpan koordinat BMU
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        # Bangun debug info untuk SEMUA 25 cluster (5x5 grid)
        cluster_debug_info = {}

        # Iterasi melalui semua posisi grid SOM
        for col in range(self.grid_size):        # Kolom 0-4
            for row in range(self.grid_size):    # Baris 0-4
                # Hitung ID cluster berdasarkan posisi
                cluster_id = col * self.grid_size + row
                # Ambil semua data yang masuk ke cluster ini
                cluster_data = df[df["priority_cluster"] == cluster_id]

                # Jika cluster memiliki data, hitung statistiknya
                if len(cluster_data) > 0:
                    # Hitung priority score untuk cluster ini
                    priority_score = self._calc_priority_score(cluster_data)
                    # Tentukan label (High/Medium/Low) berdasarkan score
                    label = self._assign_label(priority_score)
                    # Task yang paling sering muncul di cluster ini
                    dominant_task = cluster_data["task"].mode()[0]
                    # Rata-rata deadline di cluster ini
                    avg_deadline = cluster_data["deadline"].mean()
                    # Rata-rata importance di cluster ini
                    avg_importance = cluster_data["importance"].mean()
                    # Rata-rata effort di cluster ini
                    avg_effort = cluster_data["effort"].mean()
                # Jika cluster kosong
                else:
                    priority_score = None
                    label = "No Data"
                    dominant_task = None
                    avg_deadline = avg_importance = avg_effort = 0

                # Simpan semua info cluster ke dictionary
                cluster_debug_info[cluster_id] = {
                    "x": col,                       # Posisi X di grid SOM
                    "y": row,                       # Posisi Y di grid SOM
                    "samples": len(cluster_data),  # Jumlah sample di cluster
                    "priority_score": priority_score,  # Score prioritas (0-1)
                    "label": label,                 # Label (High/Medium/Low/No Data)
                    "dominant_task": dominant_task,    # Task dominan di cluster
                    "avg_deadline": avg_deadline,      # Rata-rata deadline
                    "avg_importance": avg_importance,   # Rata-rata importance
                    "avg_effort": avg_effort             # Rata-rata effort
                }

        # Tambahkan kolom label dan score ke DataFrame berdasarkan cluster
        df["priority_label"] = df["priority_cluster"].map(
            lambda c: cluster_debug_info[c]["label"]
        )
        df["priority_score"] = df["priority_cluster"].map(
            lambda c: cluster_debug_info[c]["priority_score"]
        )

        # Simpan hasil ke session state untuk digunakan UI
        st.session_state.clusters_priority = df
        st.session_state.cluster_debug_info = cluster_debug_info

        return df, som, X_scaled, cluster_debug_info

    def get_cluster_debug_df(self):
        """Get debug dataframe untuk UI."""
        # Periksa apakah info cluster ada di session state
        if not hasattr(st.session_state, 'cluster_debug_info'):
            return None
        # Pastikan formatnya adalah dictionary
        if not isinstance(st.session_state.cluster_debug_info, dict):
            return None

        # List untuk menyimpan baris-baris DataFrame
        rows = []
        # Iterasi melalui semua cluster yang tersimpan
        for cid, info in sorted(st.session_state.cluster_debug_info.items()):
            # Lewati jika format info tidak valid
            if not isinstance(info, dict):
                continue
            # Ambil priority score
            score = info.get("priority_score")
            # Format score menjadi string dengan 2 desimal
            rows.append({
                "Cluster": cid,                                  # ID cluster
                "X": info.get("x", 0),                           # Posisi X
                "Y": info.get("y", 0),                           # Posisi Y
                "Samples": info.get("samples", 0),              # Jumlah sample
                "Score": f"{score:.2f}" if score else "N/A",    # Score diformat
                "Label": info.get("label", "N/A"),               # Label prioritas
                "Dominant Task": info.get("dominant_task") or "N/A"  # Task dominan
            })
        return pd.DataFrame(rows)  # Kembalikan DataFrame debugging

    # ============================
    # SOM CLUSTERING - UNIFIED
    # ============================

    def run_som_unified(self):
        """Jalankan SOM clustering umum untuk semua pola."""
        # Minimal perlu 4 data historis
        if len(st.session_state.history) < 4:
            return None, None, None, None

        # Konversi histori ke DataFrame
        df = pd.DataFrame(st.session_state.history)

        # Mapping nama task ke indeks
        task_map = {task: idx for idx, task in enumerate(st.session_state.tasks)}
        # Mapping waktu ke indeks
        time_map = {
            "Pagi (05-12)": 0,
            "Siang (12-15)": 1,
            "Sore (15-18)": 2,
            "Malam (18-21)": 3
        }

        # Konversi task dan waktu ke indeks numerik
        df["task_num"] = df["task"].map(task_map)
        df["time_num"] = df["time"].map(time_map)

        # Pilih SEMUA fitur untuk clustering unified:
        # task, waktu, durasi, energi, kesulitan, deadline, importance, effort
        X = df[["task_num", "time_num", "duration", "energy", "difficulty",
                "deadline", "importance", "effort"]]
        # Normalisasi menggunakan scaler cluster
        X_scaled = self.scaler_cluster.fit_transform(X)

        # Latih SOM
        som = self.train_som(X_scaled, num_iteration=200)

        # Temukan BMU untuk setiap data point
        bmu_positions = [som.winner(x) for x in X_scaled]
        # Hitung ID cluster
        df["cluster"] = [pos[0] * self.grid_size + pos[1] for pos in bmu_positions]
        df["bmu_x"] = [pos[0] for pos in bmu_positions]
        df["bmu_y"] = [pos[1] for pos in bmu_positions]

        # Hitung priority label untuk setiap cluster
        for cluster_id in df["cluster"].unique():
            # Ambil semua data di cluster ini
            subset = df[df["cluster"] == cluster_id]
            # Hitung combined score dengan bobot yang sama
            combined_score = (
                subset["deadline"].mean() / 10 * 0.3 +
                subset["importance"].mean() / 10 * 0.3 +
                subset["effort"].mean() / 10 * 0.2 +
                subset["energy"].mean() / 10 * 0.1 +
                subset["difficulty"].mean() / 10 * 0.1
            )
            # Tentukan label berdasarkan threshold
            if combined_score > 0.6:
                priority_label = "High"
            elif combined_score > 0.4:
                priority_label = "Medium"
            else:
                priority_label = "Low"
            # Assign label ke semua data di cluster ini
            df.loc[df["cluster"] == cluster_id, "priority_label"] = priority_label

        # Simpan hasil ke session state
        st.session_state.clusters = df
        return df, som, X_scaled, som.quantization_error(X_scaled)

    # ============================
    # HEBBIAN LEARNING - WAKTU
    # ============================

    def learn_time(self, task_idx, time_idx, duration_minutes, energy_level, difficulty_level, deadline_score):
        """Update bobot asosiasi task-waktu menggunakan Hebbian learning."""
        # Normalisasi semua input ke rentang 0-1
        # x adalah vektor input yang merepresentasikan kondisi task
        x = np.array([
            energy_level / 10,              # Energi yang dibutuhkan (0-10 -> 0-1)
            difficulty_level / 10,          # Tingkat kesulitan (0-10 -> 0-1)
            deadline_score / 10,            # Skor deadline (urgency) (0-10 -> 0-1)
            min(duration_minutes / 120, 1.0)  # Durasi, maksimal 120 menit (2 jam)
        ])

        # y adalah vektor output (one-hot encoding untuk waktu)
        # Hanya posisi time_idx yang bernilai 1, sisanya 0
        y = np.zeros(len(self.times))
        y[time_idx] = 1.0

        # Learning rate: kecepatan pembelajaran (10%)
        lr = 0.1
        # Decay: bobot perlahan menurun agar pola lama tidak mendominasi
        decay = 0.005
        # Terapkan decay ke semua bobot sebelum pembelajaran
        st.session_state.time_weights *= (1 - decay)

        # Update bobot menggunakan rule Hebbian:
        # bobot_baru = bobot_lama + learning_rate * input * output
        # Artinya: jika input aktif dan output aktif, bobot di加强
        for i, xi in enumerate(x):
            st.session_state.time_weights[task_idx] += lr * xi * y

        # Clip bobot agar tidak keluar rentang [0, 10]
        st.session_state.time_weights = np.clip(st.session_state.time_weights, 0, 10)

    def learn_priority(self, task_idx, urgency_level, importance_level, effort_level):
        """Update bobot asosiasi task-prioritas menggunakan Hebbian learning."""
        # Normalisasi input ke rentang 0-1
        x = np.array([
            urgency_level / 10,      # Urgensi (seberapa mendesak)
            importance_level / 10,    # Kepentingan (seberapa penting)
            effort_level / 10,        # Effort (effort tinggi = nilai rendah)
        ])

        # Hitung priority score gabungan
        # Effort dibalik karena effort tinggi = prioritas rendah
        priority_score = (urgency_level * 0.4 + importance_level * 0.4 + (10 - effort_level) * 0.2)
        # Normalisasi ke rentang 0-1
        priority_score_norm = priority_score / 10

        # Tentukan indeks prioritas berdasarkan score:
        # High: score > 0.7 (indeks 2)
        # Medium: score > 0.4 (indeks 1)
        # Low: score <= 0.4 (indeks 0)
        if priority_score_norm > 0.7:
            priority_idx = 2
        elif priority_score_norm > 0.4:
            priority_idx = 1
        else:
            priority_idx = 0

        # Buat vektor output one-hot encoding untuk prioritas
        y = np.zeros(len(self.priority_levels))
        y[priority_idx] = 1.0

        # Learning rate dan decay
        lr = 0.1
        decay = 0.005
        # Terapkan decay
        st.session_state.priority_weights *= (1 - decay)

        # Update bobot menggunakan Hebbian rule
        for i, xi in enumerate(x):
            st.session_state.priority_weights[task_idx] += lr * xi * y

        # Clip bobot ke rentang [0, 10]
        st.session_state.priority_weights = np.clip(st.session_state.priority_weights, 0, 10)

    def learn(self, task_idx, time_idx, duration_minutes, energy_level,
              difficulty_level, deadline_score, importance_level, effort_level):
        """Jalankan kedua Hebbian learning sekaligus dan simpan histori."""
        # Jalankan pembelajaran untuk asosiasi waktu
        self.learn_time(task_idx, time_idx, duration_minutes, energy_level, difficulty_level, deadline_score)
        # Jalankan pembelajaran untuk asosiasi prioritas
        self.learn_priority(task_idx, deadline_score, importance_level, effort_level)

        # Tambahkan record ke histori untuk analisis later
        st.session_state.history.append({
            "task": st.session_state.tasks[task_idx],     # Nama task
            "time": self.times[time_idx],                  # Waktu yang digunakan
            "duration": duration_minutes,                 # Durasi dalam menit
            "energy": energy_level,                       # Tingkat energi (1-10)
            "difficulty": difficulty_level,                 # Tingkat kesulitan (1-10)
            "deadline": deadline_score,                   # Skor deadline/urgency (1-10)
            "importance": importance_level,               # Tingkat kepentingan (1-10)
            "effort": effort_level,                        # Effort yang dibutuhkan (1-10)
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Waktu penyimpanan
        })

        # Simpan state ke file untuk persistensi data
        save_state()

    # ============================
    # RECOMMENDATION
    # ============================

    def recommend_time(self, task_idx):
        """Rekomendasikan waktu terbaik berdasarkan Hebbian weights."""
        # Ambil bobot untuk task tertentu
        weights = st.session_state.time_weights[task_idx]
        # Jika semua bobot = 0, berarti belum ada pembelajaran
        if np.all(weights == 0):
            return None, 0
        # Temukan indeks dengan bobot tertinggi
        best_idx = np.argmax(weights)
        # Kembalikan nama waktu dan confidence score
        return self.times[best_idx], weights[best_idx]

    def recommend_priority(self, task_idx):
        """Rekomendasikan prioritas berdasarkan Hebbian weights."""
        # Ambil bobot untuk task tertentu
        weights = st.session_state.priority_weights[task_idx]
        # Jika semua bobot = 0, belum ada pembelajaran
        if np.all(weights == 0):
            return None, 0
        # Temukan indeks dengan bobot tertinggi
        best_idx = np.argmax(weights)
        # Kembalikan nama prioritas dan confidence score
        return self.priority_levels[best_idx], weights[best_idx]

    def get_priority_ranking(self):
        """Dapatkan ranking prioritas untuk semua task."""
        # List untuk menyimpan ranking
        rankings = []

        # Iterasi melalui semua task
        for idx, task in enumerate(st.session_state.tasks):
            # Dapatkan rekomendasi prioritas dari Hebbian weights
            priority, score = self.recommend_priority(idx)

            # Jika belum ada pembelajaran, fallback ke histori
            if priority is None:
                # Cari semua record histori untuk task ini
                task_history = [h for h in st.session_state.history if h["task"] == task]
                # Jika ada histori, hitung berdasarkan data historis
                if task_history:
                    # Hitung rata-rata setiap fitur dari histori
                    avg_deadline = np.mean([h["deadline"] for h in task_history])
                    avg_importance = np.mean([h["importance"] for h in task_history])
                    avg_effort = np.mean([h["effort"] for h in task_history])

                    # Hitung priority score gabungan
                    priority_score = avg_deadline * 0.4 + avg_importance * 0.4 + (10 - avg_effort) * 0.2

                    # Tentukan prioritas berdasarkan score
                    if priority_score > 7:
                        priority = "High"
                    elif priority_score > 4:
                        priority = "Medium"
                    else:
                        priority = "Low"
                    score = priority_score / 10
                # Jika tidak ada histori sama sekali, beri prioritas default Medium
                else:
                    priority = "Medium"
                    score = 0.5

            # Tambahkan ke list ranking
            rankings.append({"task": task, "priority": priority, "score": score})

        # Urutkan berdasarkan prioritas (High > Medium > Low), lalu score tertinggi
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        rankings.sort(key=lambda x: (priority_order[x["priority"]], -x["score"]))

        return rankings


# ============================
# VISUALIZATION
# ============================

def plot_som_grid(som, X_scaled, labels, clusters_df, color_by="task", title="SOM Grid"):
    """Plot SOM Grid 2D dengan U-Matrix dan nomor cluster di setiap kotak."""
    # Buat figure dan axes dengan ukuran 12x10 inci
    fig, ax = plt.subplots(figsize=(12, 10))

    # Hitung U-Matrix: jarak antar neuron Tetangga
    # Warna gelap = neuron mirip, warna terang = neuron berbeda
    umatrix = som.distance_map()
    # Plot U-Matrix sebagai heatmap dengan warna biru (inverted)
    im = ax.pcolor(umatrix, cmap='Blues_r')

    # Ambil ukuran grid SOM
    grid_size = umatrix.shape[0]
    # Set untuk menyimpan cluster yang memiliki data
    occupied_clusters = set()
    # Iterasi melalui semua data point
    for x in X_scaled:
        # Temukan BMU (neuron terbaik) untuk data ini
        bmu = som.winner(x)
        # Hitung ID cluster
        cluster_num = bmu[0] * grid_size + bmu[1]
        # Tambahkan ke set cluster yang terisi
        occupied_clusters.add(cluster_num)

    # Gambar semua cluster di grid
    for col in range(grid_size):        # Kolom 0 sampai grid_size-1
        for row in range(grid_size):     # Baris 0 sampai grid_size-1
            # Hitung ID cluster
            cluster_num = col * grid_size + row
            # Periksa apakah cluster ini memiliki data
            is_occupied = cluster_num in occupied_clusters
            # Warna teks: putih jika kosong, hitam jika terisi
            text_color = 'white' if not is_occupied else 'black'
            # Font weight: normal jika kosong, bold jika terisi
            fontweight = 'normal' if not is_occupied else 'bold'

            # Gambar nomor cluster di tengah kotak
            ax.text(col + 0.5, row + 0.5, str(cluster_num),
                   ha='center', va='center', fontsize=10, fontweight=fontweight, color=text_color)

            # Jika cluster kosong, gambar border merah putus-putus
            if not is_occupied:
                rect = plt.Rectangle((col, row), 1, 1, fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax.add_patch(rect)

    # Buat color map berdasarkan jenis pewarnaan
    if color_by == "task":
        # Jika berdasarkan task: setiap task punya warna berbeda
        colors = plt.cm.Set1(np.linspace(0, 1, len(labels)))
        color_map = {label: colors[i] for i, label in enumerate(labels)}
    else:
        # Jika berdasarkan prioritas: fixed color scheme
        color_map = {"High": "red", "Medium": "orange", "Low": "green"}

    # Gambar marker untuk setiap data point
    for i, x in enumerate(X_scaled):
        # Temukan BMU untuk data ini
        bmu = som.winner(x)
        # Tentukan warna berdasarkan jenis pewarnaan
        if color_by == "task":
            task_name = clusters_df.iloc[i]["task"]
            color = color_map.get(task_name, 'gray')
        else:
            color = color_map.get(clusters_df.iloc[i].get("priority_label", "Low"), "gray")

        # Gambar lingkaran di posisi BMU dengan warna yang sesuai
        ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o', markersize=15,
               markerfacecolor=color, markeredgecolor='black', markeredgewidth=2)

    # Buat legenda
    if color_by == "task":
        # Legenda untuk setiap task
        for idx, label in enumerate(labels):
            ax.plot([], [], 'o', markersize=10, markerfacecolor=colors[idx], markeredgecolor='black', label=label)
    else:
        # Legenda untuk setiap level prioritas
        for priority, color in color_map.items():
            ax.plot([], [], 'o', markersize=10, markerfacecolor=color, markeredgecolor='black', label=priority + " Priority")

    # Legenda untuk cluster kosong
    ax.plot([], [], '--', color='red', linewidth=2, label='Kosong (tidak ada data)')
    # Posisi legenda di luar plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    # Judul plot
    ax.set_title(title)
    # Label sumbu X dan Y
    ax.set_xlabel("SOM X (Column)")
    ax.set_ylabel("SOM Y (Row)")
    # Batasi area plot sesuai ukuran grid
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    # Buat aspek rasio sama (square cells)
    ax.set_aspect('equal')
    # Tambahkan colorbar untuk U-Matrix
    plt.colorbar(im, ax=ax, label="Jarak (U-Matrix)")

    return fig


def plot_som_priority_heatmap(som, X_scaled, clusters_df, cluster_debug_info, title="SOM Grid - Pola Prioritas"):
    """Plot SOM Grid dengan Priority Score Heatmap yang jelas."""
    # Buat figure dan axes dengan ukuran 14x11 inci
    fig, ax = plt.subplots(figsize=(14, 11))

    # Ukuran grid SOM
    grid_size = 5
    # Matriks untuk menyimpan priority score (default: NaN = tidak ada data)
    priority_matrix = np.full((grid_size, grid_size), np.nan)

    # Konversi list ke dict jika diperlukan
    if isinstance(cluster_debug_info, list):
        info_dict = {}
        for item in cluster_debug_info:
            if isinstance(item, dict) and 'x' in item and 'y' in item:
                cid = item['x'] * grid_size + item['y']
                info_dict[cid] = item
        cluster_debug_info = info_dict

    # Pastikan cluster_debug_info adalah dictionary
    if not isinstance(cluster_debug_info, dict):
        cluster_debug_info = {}

    # Isi priority matrix dengan score dari debug info
    for cid, info in cluster_debug_info.items():
        if isinstance(info, dict) and info.get("priority_score") is not None:
            # Simpan score di posisi yang benar (y, x)
            priority_matrix[info["y"], info["x"]] = info["priority_score"]

    # Plot heatmap dengan color scheme Red-Yellow-Green (inverted)
    # Merah = score tinggi (High priority), Hijau = score rendah (Low priority)
    im = ax.pcolor(priority_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, edgecolors='black', linewidths=1)

    # Konfigurasi colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # Titik tick di colorbar
    cbar.set_ticklabels(['0.0', '0.2', '0.4 (Low)', '0.6 (Med)', '0.8', '1.0 (High)'])  # Label tick
    cbar.set_label("Priority Score", fontsize=11)  # Judul colorbar

    # Gambar info detail untuk setiap cluster
    for cid, info in cluster_debug_info.items():
        if not isinstance(info, dict):
            continue

        # Ambil koordinat cluster
        x, y = info["x"], info["y"]

        # Jika cluster memiliki data
        if info.get("priority_score") is not None:
            # Ambil data dari info
            score = info["priority_score"]
            label = info["label"]
            samples = info["samples"]

            # Tentukan warna background dan teks berdasarkan label
            if label == "High":
                bg = (1, 0.85, 0.85)  # Merah muda
                tc = 'darkred'         # Teks merah tua
            elif label == "Medium":
                bg = (1, 0.95, 0.8)   # Orange muda
                tc = 'darkorange'      # Teks orange tua
            else:
                bg = (0.88, 1, 0.88)  # Hijau muda
                tc = 'darkgreen'       # Teks hijau tua

            # Gambar background rectangle dengan warna sesuai priority
            rect = plt.Rectangle((x + 0.02, y + 0.02), 0.96, 0.96, facecolor=bg, edgecolor='none', alpha=0.8)
            ax.add_patch(rect)

            # Gambar teks di dalam cluster:
            # - ID cluster (atas)
            ax.text(x + 0.5, y + 0.72, f"#{cid}", ha='center', va='center', fontsize=12, fontweight='bold', color=tc)
            # - Label priority (tengah atas)
            ax.text(x + 0.5, y + 0.50, f"[{label}]", ha='center', va='center', fontsize=10, fontweight='bold', color=tc)
            # - Priority score (tengah)
            ax.text(x + 0.5, y + 0.25, f"{score:.2f}", ha='center', va='center', fontsize=16, fontweight='bold', color=tc)
            # - Jumlah sample (bawah)
            ax.text(x + 0.5, y + 0.08, f"n={samples}", ha='center', va='center', fontsize=9, color=tc)
        # Jika cluster kosong (tidak ada data)
        else:
            # Gambar rectangle abu-abu dengan border merah putus-putus
            rect = plt.Rectangle((x, y), 1, 1, facecolor='#E8E8E8', edgecolor='red', linewidth=2, linestyle='--')
            ax.add_patch(rect)
            # Gambar teks "No Data"
            ax.text(x + 0.5, y + 0.65, f"#{cid}", ha='center', va='center', fontsize=11, fontweight='bold', color='gray')
            ax.text(x + 0.5, y + 0.35, "No Data", ha='center', va='center', fontsize=9, color='red')

    # Gambar marker BMU untuk setiap data point
    for i, x_vec in enumerate(X_scaled):
        # Temukan BMU untuk data ini
        bmu = som.winner(x_vec)
        # Ambil priority label dari data
        plabel = clusters_df.iloc[i].get("priority_label", "Low")
        # Tentukan warna marker berdasarkan label
        mc = {'High': 'darkred', 'Medium': 'darkorange', 'Low': 'darkgreen'}.get(plabel, 'gray')

        # Gambar dua lingkaran konsentris untuk efek outline
        ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o', markersize=22, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2)
        ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o', markersize=16, markerfacecolor=mc, markeredgecolor='white', markeredgewidth=2)

    # Buat legenda kustom
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=12, label='High Priority'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=12, label='Medium Priority'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=12, label='Low Priority'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='black', markersize=12, label='BMU Marker'),
        Line2D([0], [0], linestyle='--', color='red', linewidth=2, label='No Data')
    ]
    # Tampilkan legenda di luar plot
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

    # Judul dan label sumbu
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("SOM Column (X)", fontsize=11)
    ax.set_ylabel("SOM Row (Y)", fontsize=11)
    # Batasi area plot
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    # Aspect ratio sama
    ax.set_aspect('equal')

    # Atur layout agar tidak ada yang terpotong
    plt.tight_layout()
    return fig
