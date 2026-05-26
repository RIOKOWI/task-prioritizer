# Kohonen SOM + Hebbian Task Prioritizer

Sistem Task Prioritizer yang menggunakan kombinasi Kohonen Self-Organizing Map (SOM) dan Hebbian Learning untuk mempelajari pola produktivitas pengguna.

---

## Deskripsi Aplikasi

Aplikasi ini membantu pengguna memprioritaskan dan menjadwalkan tugas berdasarkan:

- Preferensi waktu kerja (pagi, siang, sore, malam)
- Tingkat energi saat mengerjakan tugas
- Tingkat kesulitan tugas
- Urgensi deadline
- Durasi fokus yang dihabiskan

Sistem belajar dari setiap aktivitas yang diinputkan dan memberikan rekomendasi waktu optimal untuk setiap tugas.

---

## Arsitektur Project

Struktur file dalam project ini:

```
task-prioritizer/
|-- app_state.json
|-- app.py                    # File utama aplikasi Streamlit
|-- requirements.txt           # Daftar dependencies
|-- README.md                 # Dokumentasi project
```

### Penjelasan Struktur

- `app.py` - File utama yang berisi seluruh logika aplikasi, termasuk class HybridBrain, fungsi UI Streamlit, dan state management
- `requirements.txt` - Mendefinisikan library Python yang dibutuhkan

---

## Teknologi yang Digunakan

### Bahasa Pemrograman
- Python 3.8+

### Framework
- Streamlit - Untuk membangun web UI interaktif

### Library

| Library | Fungsi |
|---------|--------|
| numpy | Manipulasi array dan operasi matematika |
| pandas | Manipulasi dan analisis data |
| scikit-learn | Preprocessing data (MinMaxScaler) |
| minisom | Implementasi Kohonen Self-Organizing Map |
| matplotlib | Visualisasi grafik SOM Grid |

---

## Setup Project

### 1. Clone Repository

```bash
git clone https://github.com/RIOKOWI/task-prioritizer.git
cd task-prioritizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies yang diinstall:
- streamlit
- pandas
- numpy
- scikit-learn
- minisom
- matplotlib

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada URL default `http://localhost:8501`.

---

## Alur Sistem (System Flow)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Tambah Task ──> Input nama task ──> Validasi            │
│                          │                                  │
│                          v                                  │
│  2. Input Aktivitas ──> Pilih task, waktu, energi,          │
│                        difficulty, deadline, durasi         │
│                          │                                  │
│                          v                                  │
│  3. Simpan Aktivitas ──> Trigger Hebbian Learning           │
│                          │                                  │
│                          v                                  │
│  ┌───────────────────────────────────────────────────┐      │
│  │              HYBRID BRAIN CLASS                   │      │
│  │                                                   │      │
│  │  ┌──────────────────┐  ┌──────────────────────┐   │      │
│  │  │ Hebbian Learning │  │  SOM Clustering      │   │      │
│  │  │                  │  │                      │   │      │
│  │  │ - Input: features│  │ - Input: history     │   │      │
│  │  │ - Output: weight │  │ - Output: clusters   │   │      │
│  │  │   matrix update  │  │   + SOM grid         │   │      │
│  │  │                  │  │                      │   │      │
│  │  │ Formula:         │  │  MiniSOM 5x5 grid    │   │      │
│  │  │ w += lr * x * y  │  │  dengan U-Matrix     │   │      │
│  │  └──────────────────┘  └──────────────────────┘   │      │
│  └───────────────────────────────────────────────────┘      │
│                          │                                  │
│                          v                                  │
│  4. State Persistence ──> Simpan ke app_state.json          │
│                          │                                  │
│                          v                                  │
│  5. Dashboard Display ──> Tampilkan:                        │
│     - Weight Matrix                                         │
│     - Rekomendasi Waktu                                     │
│     - SOM Grid Visualization                                │
│     - Cluster Interpretation                                │
│     - Histori Aktivitas                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Penjelasan Alur

#### Input User
1. User menambahkan task baru melalui sidebar
2. User menginput aktivitas dengan memilih:
   - Task yang dikerjakan
   - Waktu (Pagi/Siang/Sore/Malam)
   - Durasi fokus (10-240 menit)
   - Level energi (1-10)
   - Tingkat kesulitan (1-10)
   - Tingkat deadline (1-10)

#### Processing (HybridBrain)
1. **Hebbian Learning Module**
   - Menerima input features: `[energy, difficulty, deadline, duration_normalized]`
   - Membuat one-hot vector untuk time slot yang dipilih
   - Update weight matrix menggunakan formula `w += lr * x * y`
   - Menerapkan weight decay untuk mencegah unbounded growth
   - Menyimpan histori aktivitas

2. **SOM Clustering Module**
   - Mengambil data histori (minimal 4 aktivitas)
   - Encode features: `[task_num, time_num, duration, energy, difficulty, deadline]`
   - Normalisasi dengan MinMaxScaler
   - Training MiniSOM dengan grid 5x5
   - Generate cluster labels dari BMU positions
   - Visualisasi dengan U-Matrix dan BMU markers

#### Output (Dashboard)
- Heatmap weight matrix (asosiasi task-waktu)
- Rekomendasi waktu terbaik untuk task yang dipilih
- SOM Grid visualization (peta produktivitas 5x5)
- Interpretasi cluster dengan priority label (High/Medium/Low)
- Tabel histori aktivitas

#### Persistence
- Setiap perubahan state disimpan ke file `app_state.json`
- Saat aplikasi direstart, state dimuat kembali dari file JSON
- Data tidak hilang saat page di-refresh

---

## Komponen Utama dalam Kode

### Class HybridBrain

```python
class HybridBrain:
    - __init__(times)           # Inisialisasi dengan time slots
    - ensure_weight_shape()     # Sesuaikan dimensi weight matrix
    - train_som(X)              # Training SOM dengan MiniSOM
    - run_som()                 # Clustering histori dengan SOM
    - learn(...)                # Hebbian learning update
    - recommend(task_idx)       # Rekomendasi waktu
```

### Fungsi Utility

```python
save_state()    # Simpan state ke JSON
load_state()    # Muat state dari JSON
plot_som_grid() # Visualisasi SOM Grid 2D
```

### State Management

```python
st.session_state.tasks     # Daftar task
st.session_state.weights   # Weight matrix Hebbian
st.session_state.history   # Histori aktivitas
st.session_state.clusters  # Hasil clustering SOM
st.session_state.brain     # Instance HybridBrain
```

---

## Time Slots

Aplikasi menggunakan 4 time slots:

| Index | Label | Jam |
|-------|-------|-----|
| 0 | Pagi (05-12) | 05:00 - 12:00 |
| 1 | Siang (12-15) | 12:00 - 15:00 |
| 2 | Sore (15-18) | 15:00 - 18:00 |
| 3 | Malam (18-21) | 18:00 - 21:00 |

---

## File Konfigurasi

### .gitignore
File yang diabaikan git:
- `__pycache__/`
- `*.pyc`
- `.streamlit/`

---

## Troubleshooting

### Error: minisom not found
```bash
pip install minisom
```

### Error: NameError HybridBrain
Pastikan class HybridBrain didefinisikan sebelum session state initialization. Lihat struktur kode di `app.py`.

### Data tidak tersimpan
Cek apakah file `app_state.json` ada di direktori project. Jika tidak ada, berarti ada masalah dengan permission write.

---
