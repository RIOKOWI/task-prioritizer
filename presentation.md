# Guide Presentasi: Hybrid SOM + Hebbian Task Prioritizer

Dokumen ini berisi panduan demo dan cheat sheet teknis untuk presentasi aplikasi.

---

## 📋 DEMO CHECKLIST

### Persiapan Sebelum Demo (5 menit)

#### 1. Setup Environment
```bash
# Clone atau buka direktori project
cd task-prioritizer

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

#### 2. Reset State (untuk demo bersih)
Klik tombol **"Reset Semua Memori"** di bagian bawah halaman untuk memulai dari nol.

#### 3. Persiapan Screen
- Buka browser di mode **fullscreen** atau maximize window
- Siapkan 2-3 browser tabs jika mau show code + app sekaligus

---

### Demo Flow (15-20 menit)

#### 🎯 Phase 1: Pengenalan & Setup (3 menit)

**Step 1: Tambah Task**
1. Buka sidebar di kiri
2. Ketik: `"Skripsi Bab 3"`
3. Klik tombol **"Tambah Task"**
4. Tambahkan task lain: `"Olahraga"`, `"Belajar Python"`, `"Meeting Client"`

**Script penjelasan:**
> "User bisa tambahkan task sesuai kebutuhan. Setiap task akan dipelajari pola produktivitasnya."

**Step 2: Verifikasi Task**
- Task muncul di dropdown "Jenis Task"
- Task juga muncul di Manajemen Task

---

#### 🚀 Phase 2: Input Aktivitas & Hebbian Learning (5 menit)

**Step 3: Input Aktivitas Pertama**
1. Pilih task: `Skripsi Bab 3`
2. Pilih waktu: `Malam (18-21)`
3. Durasi: `90 menit`
4. Energi: `8` (tinggi)
5. Kesulitan: `7`
6. Importance: `9`
7. Effort: `6`
8. Deadline/Urgency: `8`

Klik **"Simpan Aktivitas"**

**Script penjelasan:**
> "Disini user input aktivitas yang sudah dilakukan. Sistem mencatat konteks lengkap: waktu, energi, kesulitan, dan prioritas."

**Step 4: Input 3-4 Aktivitas Lain**
Buat variasi untuk show pattern:

| Task | Waktu | Energi | Kesulitan | Deadline | Effort |
|------|-------|--------|-----------|----------|--------|
| Olahraga | Pagi (05-12) | 9 | 3 | 5 | 4 |
| Belajar Python | Siang (12-15) | 6 | 6 | 7 | 5 |
| Meeting Client | Sore (15-18) | 7 | 5 | 9 | 3 |

**Step 5: Tunjukkan Weight Matrix**
1. Buka tab **"Rekomendasi Waktu"**
2. Tunjukkan **Hebbian Weight Matrix** di sebelah kiri

**Script penjelasan:**
> "Hebbian Weight Matrix menyimpan asosiasi antara task dan waktu. Warna hijau tua = asosiasi kuat. Terlihat 'Skripsi' punya bobot tinggi di waktu Malam."

---

#### 📊 Phase 3: Demo Rekomendasi (4 menit)

**Step 6: Rekomendasi Waktu**
1. Di tab "Rekomendasi Waktu", pilih task: `Skripsi Bab 3`

**Yang ditunjukkan:**
- ✅ Waktu terbaik yang direkomendasikan
- ✅ Confidence score (dari 0-10)
- ✅ Progress bar

**Script penjelasan:**
> "Berdasarkan pembelajaran Hebbian, sistem merekomendasikan waktu optimal. Score menunjukkan seberapa yakin sistem dengan rekomendasi ini."

**Step 7: Demo Tab Prioritas**
1. Buka tab **"Prioritas"**
2. Tunjukkan ranking task berdasarkan priority score

**Script penjelasan:**
> "Task di-ranking berdasarkan urgensi, importance, dan effort. Meeting Client muncul paling atas karena deadline tinggi dan effort rendah."

---

#### 🗺️ Phase 4: SOM Visualization (5 menit)

**Step 8: Buka SOM Grid**
1. Buka tab **"SOM Visualization"**
2. Tunjukkan kedua grid (Waktu & Prioritas)

**Yang ditunjukkan:**
- Grid 5x5 (25 neuron/cluster)
- U-Matrix (warna = jarak antar neuron)
- Marker untuk setiap aktivitas
- Cluster kosong (border merah putus-putus)

**Script penjelasan:**
> "SOM (Self-Organizing Map) mengelompokkan pola produktivitas ke dalam grid 2D. Setiap sel adalah satu cluster. Neuron tetangga = pola serupa."

**Step 9: Buka Debug Panel**
1. Scroll ke bawah di tab SOM
2. Tunjukkan **Debug Table** dengan semua 25 cluster
3. Tunjukkan **Interpretasi Cluster**

**Yang ditunjukkan:**
- Cluster ID, koordinat (X, Y)
- Samples (jumlah data di cluster)
- Priority Score & Label
- Dominant Task

**Script penjelasan:**
> "Debug panel menunjukkan detail setiap cluster. Terlihat cluster #15 adalah High Priority dengan task dominan Meeting Client."

---

#### 📜 Phase 5: History & Persistence (3 menit)

**Step 10: Tunjukkan History**
1. Buka tab **"History"**
2. Tunjukkan semua aktivitas yang tersimpan

**Step 11: Test Persistence**
1. **Tutup browser tab**
2. **Buka ulang** `streamlit run app.py`
3. **Verify** data masih ada

**Script penjelasan:**
> "Data tersimpan otomatis ke JSON. Tidak hilang walau aplikasi di-restart."

---

### 🎬 Closing Demo

**Step 12: Reset & Show Clean State**
Klik **"Reset Semua Memori"**

**Script penutup:**
> "Tombol reset menghapus semua memori AI. Bisa dimulai dari nol kapan saja."

---

### ⚡ Quick Demo Mode (5 menit)

Jika waktu terbatas, skip ke inti:

1. **Tambah 3 task** cepat
2. **Input 4 aktivitas** dengan variasi
3. **Show Weight Matrix** → "Ini memori AI"
4. **Show Rekomendasi** → "Sistem kasih saran"
5. **Show SOM Grid** → "Ini peta produktivitas"
6. **Restart app** → "Data persist"

---

## 🧮 CHEAT SHEET TEKNIS

### 1. Hebbian Learning Formula

```
w_baru = w_lama + lr × x × y
```

**Variabel:**
| Simbol | Arti | Nilai |
|--------|------|-------|
| `w` | Bobot asosiasi | Matriks N×M |
| `lr` | Learning rate | 0.1 (10%) |
| `x` | Input vector | [energy, difficulty, deadline, duration_norm] |
| `y` | Output vector | One-hot encoding (hanya 1 posisi = 1) |

**Contoh Perhitungan:**
```
Task: "Skripsi"
Waktu: "Malam" (index 2)

x = [energy/10, difficulty/10, deadline/10, duration/120]
  = [0.8, 0.7, 0.8, 0.75]

y = [0, 0, 1, 0]  # One-hot untuk "Malam"

w[skripsi][malam] += 0.1 × [0.8, 0.7, 0.8, 0.75] × [0, 0, 1, 0]
w[skripsi][malam] += 0.1 × 0.8 = +0.08
```

**Decay (Pencegah Unbounded Growth):**
```
w = w × (1 - decay)
w = w × 0.995  # Decay 0.5% per update
```

---

### 2. Priority Score Formula

```
priority_score = deadline×0.3 + importance×0.3 + effort×0.2 + energy×0.1 + difficulty×0.1
```

**Semua input dinormalisasi ke skala 0-10:**

| Fitur | Bobot | Arti |
|-------|-------|------|
| Deadline | 0.3 | Seberapa urgent (semakin urgent = skor tinggi) |
| Importance | 0.3 | Seberapa penting |
| Effort | 0.2 | Effort tinggi = skor rendah (dibalik) |
| Energy | 0.1 | Energi tinggi = sedikit lebih baik |
| Difficulty | 0.1 | Kesulitan rendah = sedikit lebih baik |

**Contoh:**
```
deadline=8, importance=9, effort=6, energy=7, difficulty=5

priority_score = 8×0.3 + 9×0.3 + 6×0.2 + 7×0.1 + 5×0.1
               = 2.4 + 2.7 + 1.2 + 0.7 + 0.5
               = 7.5 / 10
               = 0.75
```

**Label Assignment:**
| Score Range | Label |
|-------------|-------|
| > 0.6 | 🔴 High |
| > 0.4 | 🟡 Medium |
| ≤ 0.4 | 🟢 Low |

---

### 3. SOM (Self-Organizing Map) Parameters

```
MiniSom Parameters:
├── x = 5              # Grid width (5 neuron)
├── y = 5              # Grid height (5 neuron)
├── input_len = 8      # Jumlah fitur input
├── sigma = 1.0        # Radius neighbourhood
├── learning_rate = 0.5
└── random_seed = 42   # Reproducibility
```

**Total neuron:** 5 × 5 = **25 cluster**

**Fitur Input SOM Priority:**
```
X = [task_num, deadline, importance, effort, energy, difficulty]
```

**Fitur Input SOM Time:**
```
X = [task_num, time_num, duration, energy, difficulty, deadline]
```

---

### 4. BMU (Best Matching Unit) Finding

```
BMU = argmin ||x - w_i|| untuk semua neuron i
```

Artinya: Cari neuron yang bobotnya **paling mirip** dengan data input.

**Cluster ID Calculation:**
```
cluster_id = bmu_x × grid_size + bmu_y
           = 2 × 5 + 3
           = 13
```

---

### 5. U-Matrix (Unified Distance Matrix)

```python
umatrix = som.distance_map()
```

- **Warna gelap** = neuron tetangga mirip satu sama lain
- **Warna terang** = neuron tetangga berbeda (boundary)
- Membantu identifikasi cluster natural dalam data

---

### 6. Data Normalization (MinMaxScaler)

```python
X_scaled = (X - X_min) / (X_max - X_min)
```

Semua fitur diskala ke rentang **[0, 1]** agar tidak ada fitur yang mendominasi karena skala berbeda.

**Contoh:**
```
energy: 7/10 → 0.7
deadline: 8/10 → 0.8
duration: 90/240 → 0.375
```

---

### 7. State Persistence Schema

```json
{
  "tasks": ["Skripsi", "Olahraga", "Meeting Client"],
  "time_weights": [
    [0.0, 0.0, 0.08, 0.0],
    [0.12, 0.0, 0.0, 0.0],
    [0.0, 0.06, 0.0, 0.04]
  ],
  "priority_weights": [
    [0.0, 0.24, 0.45],
    [0.15, 0.12, 0.0],
    [0.0, 0.18, 0.36]
  ],
  "history": [
    {
      "task": "Skripsi Bab 3",
      "time": "Malam (18-21)",
      "duration": 90,
      "energy": 8,
      "difficulty": 7,
      "deadline": 8,
      "importance": 9,
      "effort": 6,
      "timestamp": "2026-06-21 14:30:00"
    }
  ]
}
```

---

### 8. Responsive Q&A

**Q: Kenapa pakai 5×5 grid?**
> 5×5 adalah sweet spot. 3×3 terlalu kecil untuk differentiate patterns, 7×7 ke atas butuh data lebih banyak agar cluster terisi.

**Q: Kenapa minimal 4 data untuk SOM?**
> SOM butuh cukup data untuk "belajar" pola. Kurang dari 4 data tidak cukup untuk menemukan struktur meaningful dalam data.

**Q: Decay 0.005 itu besar atau kecil?**
> Relatif kecil. Dengan 0.5% decay per update, bobot butuh ~200 updates untuk turun setengahnya. Ini mencegah pola lama mendominasi selamanya tapi tetap menghormati historis.

**Q: Kapan pakai Hebbian vs SOM?**
> **Hebbian** = Rekomendasi spesifik (task X paling cocok waktu Y)
> **SOM** = Visualisasi pola global (ada X cluster berbeda)

**Q: Bagaimana jika data tidak cukup?**
> Sistem fallback ke rata-rata historis. Jika tidak ada historis sama sekali, default priority = Medium.

---

### 9. Terminologi Penting

| Istilah | Penjelasan Singkat |
|---------|-------------------|
| **Hebbian Learning** | "Neuron yang fire together, wire together" - belajar asosiasi |
| **SOM (Self-Organizing Map)** | Jaringan saraf yang memetakan data high-dimensional ke grid 2D |
| **BMU (Best Matching Unit)** | Neuron SOM yang paling cocok dengan data input |
| **U-Matrix** | Visualisasi jarak antar neuron SOM |
| **Weight Matrix** | Matriks yang menyimpan pembelajaran (memory AI) |
| **Decay** | Penurunan bobot gradual untuk mencegah old patterns dominate |
| **One-hot Encoding** | Representasi kategorik: hanya 1 posisi = 1, sisanya = 0 |

---

### 10. Workflow Diagram (ASCII)

```
USER INPUT
    │
    ▼
┌─────────────────────────────────────┐
│           SIDEBAR FORM              │
│  Task, Waktu, Energi, Difficulty,   │
│  Deadline, Importance, Effort       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         HYBRIDBRAIN.LEARN()        │
├───────────────┬─────────────────────┤
│   HEBBIAN     │        SOM          │
│   ─────────   │    ───────────      │
│   Update      │    Train new         │
│   Weight      │    cluster           │
│   Matrix      │    (if ≥4 data)      │
└───────────────┴─────────────────────┘
    │
    ├──▶ RECOMMEND_TIME() ──▶ "Malam"
    │
    ├──▶ RECOMMEND_PRIORITY() ──▶ "High"
    │
    ├──▶ GET_PRIORITY_RANKING() ──▶ [Task1, Task2, ...]
    │
    └──▶ VISUALIZATION ──▶ SOM Grid + Heatmap
    │
    ▼
┌─────────────────────────────────────┐
│         SAVE TO JSON                │
│         app_state.json              │
└─────────────────────────────────────┘
```

---

## ✅ PRE-DEPLOYMENT CHECKLIST

- [ ] `pip install -r requirements.txt` berhasil
- [ ] `streamlit run app.py` bisa dijalankan
- [ ] Browser terbuka di `localhost:8501`
- [ ] Bisa tambah task
- [ ] Bisa simpan aktivitas
- [ ] Weight matrix muncul setelah 1 input
- [ ] SOM grid muncul setelah 4 input
- [ ] Data persist setelah restart

---

*Generated for presentation: Hybrid SOM + Hebbian Task Prioritizer*
