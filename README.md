# Hebbian Task Prioritizer System

Sistem Task Prioritizer berbasis **Hebbian Learning** menggunakan Python dan Streamlit.

Aplikasi ini belajar dari pola aktivitas pengguna berdasarkan:

- Jenis tugas
- Waktu pengerjaan
- Durasi fokus

Semakin sering suatu tugas dilakukan secara fokus pada waktu tertentu, maka hubungan antar neuron (weight) akan semakin kuat.

---

# Konsep Dasar

Project ini menggunakan prinsip:

> "Neuron that fires together wires together"

Artinya:
Jika suatu tugas sering dilakukan secara fokus pada waktu tertentu, maka sistem akan menganggap waktu tersebut cocok untuk tugas tersebut.

Contoh:

- Coding sering dilakukan malam hari dengan fokus tinggi
- Maka bobot hubungan `Coding -> Malam` akan meningkat

---

# Teknologi yang Digunakan

- Python
- Streamlit
- NumPy
- Pandas

---

# Fitur

## 1. Log Aktivitas
Pengguna dapat mencatat:

- Jenis tugas
- Waktu pengerjaan
- Durasi fokus

---

## 2. Hebbian Learning
Sistem akan memperkuat koneksi berdasarkan:

```python
weight += learning_rate * activation
```
---

## Installation

```bash
git clone https://github.com/RIOKOWI/task-prioritizer.git

cd task-prioritizer

pip install -r requirements.txt

streamlit run app.py
```
