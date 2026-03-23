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
