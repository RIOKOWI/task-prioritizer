# Konstanta aplikasi Task Prioritizer

# Daftar periode waktu dalam sehari untuk merepresentasikan
# kapan user biasanya produktif melakukan tugas
TIMES = [
    "Pagi (05-12)",      # Waktu produktivitas pagi hari
    "Siang (12-15)",     # Waktu produktivitas siang hari
    "Sore (15-18)",      # Waktu produktivitas sore hari
    "Malam (18-21)"      # Waktu produktivitas malam hari
]

# Tingkat prioritas untuk clustering
PRIORITY_LEVELS = [
    "Low",       # Prioritas rendah
    "Medium",    # Prioritas sedang
    "High"       # Prioritas tinggi
]

# Nama file untuk menyimpan state aplikasi secara persistence
STATE_FILE = "app_state.json"
