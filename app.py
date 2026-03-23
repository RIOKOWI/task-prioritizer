import streamlit as st  # Library untuk bikin Dashboard/Web UI secara instan
import numpy as np      # Library untuk perhitungan matriks (otak matematika AI)
import pandas as pd     # Library untuk manipulasi tabel agar enak dilihat di UI


class HebbianPrioritizer:
    def __init__(self, tasks, time_slots):
        self.tasks = tasks             # List jenis tugas (Coding, Exercise, dll)
        self.time_slots = time_slots   # List waktu (Pagi, Siang, dll)
        
        # Mengecek apakah memori 'otak' sudah ada di session_state (biar gak hilang saat refresh)
        if 'hebbian_weights' not in st.session_state:
            # Membuat matriks 0 berukuran [Jumlah Tugas x Jumlah Waktu]
            # Inisialisasi awal: semua hubungan antar saraf nilainya nol
            st.session_state.hebbian_weights = np.zeros((len(tasks), len(time_slots)))
        
    def update(self, task_idx, time_idx, efficiency_score, lr=0.1):
        """
        Rumus Hebbian: Delta W = Learning Rate * Efisiensi
        Makin cepat lo kerja, makin kuat hubungan sarafnya.
        """
        # MEKANISME DECAY : Semua bobot dikurangi 1% setiap ada update.
        # Ini supaya AI bisa 'lupa' kebiasaan lama jika lo mulai berubah jadwal (Adaptif).
        st.session_state.hebbian_weights *= 0.99
        
        # Menghitung nilai penambahan bobot (Learning Rate dikali skor performa lo)
        update_val = lr * efficiency_score
        
        # Update nilai pada koordinat tugas dan waktu yang spesifik
        st.session_state.hebbian_weights[task_idx, time_idx] += update_val
        
        # Membatasi nilai bobot antara 0 sampai 10 agar tidak meledak (Exploding Gradient)
        st.session_state.hebbian_weights = np.clip(st.session_state.hebbian_weights, 0, 10)

    def get_recommendation(self, task_idx):
        # Mengambil satu baris bobot untuk satu tugas tertentu
        weights = st.session_state.hebbian_weights[task_idx]
        
        # Jika semua nilai masih nol, artinya AI belum belajar apa-apa
        if np.all(weights == 0):
            return None, 0
            
        # Mencari index dengan nilai bobot tertinggi (Waktu paling efektif)
        best_time_idx = np.argmax(weights)
        
        # Mengembalikan nama waktu (String) dan skor kekuatannya
        return self.time_slots[best_time_idx], weights[best_time_idx]


# STREAMLIT UI
# Mengatur judul tab browser dan layout agar memenuhi layar
st.set_page_config(page_title="Hebbian Task Optimizer", layout="wide")

# Dataset statis untuk kategori
TASKS = ["Coding", "Exercise", "Washing", "Research"]
TIMES = ["Pagi (05-12)", "Siang (12-15)", "Sore (15-18)", "Malam (18-21)"]

# Inisialisasi objek model berdasarkan kategori di atas
model = HebbianPrioritizer(TASKS, TIMES)

st.title("Hebbian Task Optimizer")
st.write("Sistem memperkuat hubungan antara **Jenis Tugas** dan **Waktu Penyelesaian Terbaik** Anda.")

# SIDEBAR (Input Data)
st.sidebar.header("Log Aktivitas")
selected_task = st.sidebar.selectbox("Tugas yang baru selesai:", TASKS)
selected_time = st.sidebar.selectbox("Dikerjakan pada waktu:", TIMES)

# Slider untuk memberikan feedback ke AI
performance = st.sidebar.select_slider(
    "Seberapa cepat/fokus Anda?",
    options=[-1.0, -0.5, 0.0, 0.5, 1.0], # Negatif berarti melemahkan hubungan, Positif menguatkan
    value=0.5,
    help="1.0 = Sangat Efisien, -1.0 = Sangat Lambat/Distraksi"
)

# Tombol untuk mengeksekusi fungsi update pada model
if st.sidebar.button("Update Brain"):
    t_idx = TASKS.index(selected_task) # Mencari posisi index tugas (0, 1, 2...)
    m_idx = TIMES.index(selected_time) # Mencari posisi index waktu (0, 1, 2...)
    model.update(t_idx, m_idx, performance) # Mengirim data ke otak AI
    st.sidebar.success("Koneksi Saraf Diperkuat!")

# DASHBOARD
col1, col2 = st.columns([1, 1]) # Membagi layar menjadi 2 kolom sama besar

with col1:
    st.subheader("Matriks Kekuatan Saraf (Weights)")
    # Mengubah matriks Numpy ke DataFrame Pandas agar bisa diberi label baris/kolom
    df_weights = pd.DataFrame(
        st.session_state.hebbian_weights, 
        index=TASKS, 
        columns=TIMES
    )
    # Menampilkan tabel dengan gradasi warna biru (Heatmap)
    st.dataframe(df_weights.style.background_gradient(cmap='Blues', axis=None))
    
    # Tombol untuk menghapus semua data di session_state dan mulai dari nol
    if st.button("Reset Memori AI"):
        st.session_state.hebbian_weights = np.zeros((len(TASKS), len(TIMES)))
        st.rerun()

with col2:
    st.subheader("Rekomendasi Prioritas")
    # Dropdown untuk memilih tugas yang ingin dicek prediksinya
    target_task = st.selectbox("Pilih tugas untuk lihat waktu terbaik:", TASKS)
    best_time, score = model.get_recommendation(TASKS.index(target_task))
    
    # Jika sudah ada data, tampilkan waktu terbaik dan bar progress kekuatannya
    if best_time:
        st.info(f"Untuk **{target_task}**, waktu terbaik Anda adalah **{best_time}**.")
        # Skor dibagi 5 untuk visualisasi (karena max clip 10, 5 sudah dianggap cukup kuat)
        st.progress(min(score / 5.0, 1.0)) 
    else:
        # Jika belum ada input log sama sekali
        st.warning("Data belum cukup. Lakukan log aktivitas terlebih dahulu.")