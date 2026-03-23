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
