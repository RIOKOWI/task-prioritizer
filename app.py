import streamlit as st  
import numpy as np      
import pandas as pd

class HebbianPrioritizer:
    def __init__(self, tasks, time_slots):
        self.tasks = tasks            
        self.time_slots = time_slots

    if 'hebbian_weights' not in st.session_state:
            st.session_state.hebbian_weights = np.zeros((len(tasks), len(time_slots)))
        
    def update(self, task_idx, time_idx, efficiency_score, lr=0.1):
        """
        Rumus Hebbian: Delta W = Learning Rate * Efisiensi
        Makin cepat lo kerja, makin kuat hubungan sarafnya.
        """
        st.session_state.hebbian_weights *= 0.99
        
        update_val = lr * efficiency_score
        
        st.session_state.hebbian_weights[task_idx, time_idx] += update_val
        
        st.session_state.hebbian_weights = np.clip(st.session_state.hebbian_weights, 0, 10)

    def get_recommendation(self, task_idx):
        weights = st.session_state.hebbian_weights[task_idx]
        if np.all(weights == 0):
            return None, 0
        best_time_idx = np.argmax(weights)
        return self.time_slots[best_time_idx], weights[best_time_idx]




st.set_page_config(page_title="Hebbian Task Optimizer", layout="wide")

TASKS = ["Coding", "Exercise", "Washing", "Research"]
TIMES = ["Pagi (05-12)", "Siang (12-15)", "Sore (15-18)", "Malam (18-21)"]

model = HebbianPrioritizer(TASKS, TIMES)

st.title("Hebbian Task Optimizer")
st.write("Sistem memperkuat hubungan antara **Jenis Tugas** dan **Waktu Penyelesaian Terbaik** Anda.")

st.sidebar.header("Log Aktivitas")
selected_task = st.sidebar.selectbox("Tugas yang baru selesai:", TASKS)
selected_time = st.sidebar.selectbox("Dikerjakan pada waktu:", TIMES)



performance = st.sidebar.select_slider(
    "Seberapa cepat/fokus Anda?",
    options=[-1.0, -0.5, 0.0, 0.5, 1.0], 
    value=0.5,
    help="1.0 = Sangat Efisien, -1.0 = Sangat Lambat/Distraksi"
)


if st.sidebar.button("Update Brain"):
    t_idx = TASKS.index(selected_task) 
    m_idx = TIMES.index(selected_time) 
    model.update(t_idx, m_idx, performance) 
    st.sidebar.success("Koneksi Saraf Diperkuat!")


col1, col2 = st.columns([1, 1]) 

with col1:
    st.subheader("Matriks Kekuatan Saraf (Weights)")
    
    df_weights = pd.DataFrame(
        st.session_state.hebbian_weights, 
        index=TASKS, 
        columns=TIMES
    )
    
    st.dataframe(df_weights.style.background_gradient(cmap='Blues', axis=None))
    
    
    if st.button("Reset Memori AI"):
        st.session_state.hebbian_weights = np.zeros((len(TASKS), len(TIMES)))
        st.rerun()

with col2:
    st.subheader("Rekomendasi Prioritas")
    
    target_task = st.selectbox("Pilih tugas untuk lihat waktu terbaik:", TASKS)
    best_time, score = model.get_recommendation(TASKS.index(target_task))
    
    
    if best_time:
        st.info(f"Untuk **{target_task}**, waktu terbaik Anda adalah **{best_time}**.")
        
        st.progress(min(score / 5.0, 1.0)) 
    else:
        
        st.warning("Data belum cukup. Lakukan log aktivitas terlebih dahulu.")
