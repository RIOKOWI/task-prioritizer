"""
Hybrid SOM + Hebbian Task Prioritizer
Streamlit application untuk mengelola dan memprioritaskan task
dengan menggunakan Kohonen SOM dan Hebbian Learning.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from config.config import TIMES, PRIORITY_LEVELS, STATE_FILE
from services.brain import HybridBrain, load_state
from widgets.components import (
    render_sidebar,
    render_priority_tab,
    render_time_tab,
    render_som_tab,
    render_history_tab,
    render_task_management
)


# ============================
# KONFIGURASI STREAMLIT
# ============================

st.set_page_config(
    page_title="Hybrid SOM + Hebbian Task Prioritizer",
    layout="wide"
)


# ============================
# LOAD STATE AWAL
# ============================

saved_state = load_state()


# ============================
# INISIALISASI SESSION STATE
# ============================

if "tasks" not in st.session_state:
    st.session_state.tasks = saved_state["tasks"] if saved_state else []

if "time_weights" not in st.session_state:
    time_weights_data = saved_state.get("time_weights", []) if saved_state else []
    st.session_state.time_weights = np.array(time_weights_data) if time_weights_data else np.zeros((0, 4))

if "priority_weights" not in st.session_state:
    priority_weights_data = saved_state.get("priority_weights", []) if saved_state else []
    st.session_state.priority_weights = np.array(priority_weights_data) if priority_weights_data else np.zeros((0, 3))

if "history" not in st.session_state:
    st.session_state.history = saved_state["history"] if saved_state else []

if "clusters" not in st.session_state:
    st.session_state.clusters = pd.DataFrame()

if "clusters_time" not in st.session_state:
    st.session_state.clusters_time = pd.DataFrame()

if "clusters_priority" not in st.session_state:
    st.session_state.clusters_priority = pd.DataFrame()

if "brain" not in st.session_state:
    st.session_state.brain = HybridBrain(TIMES, PRIORITY_LEVELS)


# ============================
# AMBIL INSTANCE BRAIN
# ============================

brain = st.session_state.brain


# ============================
# TAMPILAN UTAMA
# ============================

st.title("Hybrid Kohonen SOM + Hebbian Task Prioritizer")

st.write("""
Sistem menggunakan:

- **Hebbian Learning** -> mempelajari kebiasaan user untuk asosiasi task-waktu dan task-prioritas

- **Kohonen SOM** -> mengelompokkan pola produktivitas dan prioritas dalam grid 2D (5x5)
""")


# ============================
# RENDER SIDEBAR
# ============================

render_sidebar(brain)


# ============================
# CEK JIKA BELUM ADA TASK
# ============================

if len(st.session_state.tasks) == 0:
    st.warning("Tambahkan minimal 1 task.")
    st.stop()


# ============================
# RENDER TABS
# ============================

tab1, tab2, tab3, tab4 = st.tabs([
    "Prioritas",
    "Rekomendasi Waktu",
    "SOM Visualization",
    "History"
])

with tab1:
    render_priority_tab(brain)

with tab2:
    render_time_tab(brain)

with tab3:
    render_som_tab(brain)

with tab4:
    render_history_tab(brain)


# ============================
# TASK MANAGEMENT
# ============================

render_task_management(brain)


# ============================
# RESET SYSTEM
# ============================

st.markdown("---")

if st.button("Reset Semua Memori", type="primary"):
    st.session_state.tasks = []
    st.session_state.time_weights = np.zeros((0, 4))
    st.session_state.priority_weights = np.zeros((0, 3))
    st.session_state.history = []
    st.session_state.clusters = pd.DataFrame()
    st.session_state.clusters_time = pd.DataFrame()
    st.session_state.clusters_priority = pd.DataFrame()

    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    st.rerun()
