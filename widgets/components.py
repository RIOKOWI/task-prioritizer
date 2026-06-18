"""
UI Components untuk Task Prioritizer.
Berisi semua fungsi rendering untuk sidebar, tabs, dan task management.
"""

import streamlit as st
import numpy as np
import pandas as pd

from config.config import TIMES, PRIORITY_LEVELS, STATE_FILE
from services.brain import save_state, plot_som_grid


# ============================
# SIDEBAR COMPONENTS
# ============================

def render_sidebar(brain):
    """Render sidebar dengan input task dan aktivitas."""
    # --- Tambah Task Baru ---
    st.sidebar.header("Tambah Task Baru")

    new_task = st.sidebar.text_input("Nama Task")

    if st.sidebar.button("Tambah Task"):
        cleaned_task = new_task.strip().title()

        if not cleaned_task or len(cleaned_task) < 2:
            st.sidebar.error("Nama task minimal 2 karakter.")
        elif cleaned_task in st.session_state.tasks:
            st.sidebar.warning("Task sudah ada.")
        else:
            st.session_state.tasks.append(cleaned_task)
            brain.ensure_weight_shapes()
            save_state()
            st.sidebar.success(f"Task '{cleaned_task}' ditambahkan.")

    # --- Input Aktivitas ---
    st.sidebar.header("Input Aktivitas")

    selected_task = st.sidebar.selectbox(
        "Jenis Task",
        st.session_state.tasks
    )

    selected_time = st.sidebar.selectbox(
        "Waktu",
        TIMES
    )

    focus_duration = st.sidebar.slider(
        "Durasi Fokus (menit)",
        10, 240, 60, 10
    )

    energy_level = st.sidebar.slider(
        "Level Energi",
        1, 10, 5
    )

    difficulty_level = st.sidebar.slider(
        "Tingkat Kesulitan",
        1, 10, 5
    )

    # --- Priority-related inputs ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Prioritas Task")

    importance_level = st.sidebar.slider(
        "Tingkat Importance (1-10)",
        1, 10, 5,
        help="Seberapa penting task ini? (10 = sangat penting)"
    )

    effort_level = st.sidebar.slider(
        "Tingkat Effort (1-10)",
        1, 10, 5,
        help="Seberapa sulit effort yang dibutuhkan? (10 = effort tinggi)"
    )

    deadline_score = st.sidebar.slider(
        "Tingkat Urgency/Deadline (1-10)",
        1, 10, 5,
        help="Seberapa urgent task ini? (10 = sangat urgent)"
    )

    # --- Simpan Aktivitas ---
    if st.sidebar.button("Simpan Aktivitas"):
        task_idx = st.session_state.tasks.index(selected_task)
        time_idx = TIMES.index(selected_time)

        brain.learn(
            task_idx,
            time_idx,
            focus_duration,
            energy_level,
            difficulty_level,
            deadline_score,
            importance_level,
            effort_level
        )

        st.sidebar.success("AI memory diperbarui.")


# ============================
# TAB 1: PRIORITY RANKING
# ============================

def render_priority_tab(brain):
    """Render tab prioritas dengan ranking task."""
    st.subheader("Task Priority Ranking")

    rankings = brain.get_priority_ranking()

    if rankings:
        # Tampilkan ranking dengan visual
        priority_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}

        for rank, item in enumerate(rankings, 1):
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.markdown(f"### #{rank}")

            with col2:
                icon = priority_icons.get(item["priority"], "⚪")
                st.markdown(f"{icon} **{item['task']}**")
                st.progress(min(item["score"], 1.0), text=f"Score: {item['score']:.2f}")

            with col3:
                st.markdown(f"**{item['priority']}**")

            st.divider()

        # Tampilkan dataframe ranking
        st.markdown("### Detail Ranking")
        df_ranking = pd.DataFrame(rankings)
        st.dataframe(df_ranking, use_container_width=True)
    else:
        st.info("Belum ada data untuk ranking prioritas.")


# ============================
# TAB 2: TIME RECOMMENDATION
# ============================

def render_time_tab(brain):
    """Render tab rekomendasi waktu dan weight matrix."""
    col1, col2 = st.columns(2)

    # Weight Matrix - Waktu
    with col1:
        st.subheader("Hebbian Weight Matrix (Waktu)")

        if len(st.session_state.time_weights) > 0:
            df_time_weights = pd.DataFrame(
                st.session_state.time_weights,
                index=st.session_state.tasks,
                columns=TIMES
            )
            st.dataframe(
                df_time_weights.style.background_gradient(
                    cmap="Greens",
                    axis=None
                )
            )
        else:
            st.info("Belum ada data.")

    # Recommendation - Waktu
    with col2:
        st.subheader("Rekomendasi Waktu")

        target_task = st.selectbox(
            "Pilih Task",
            st.session_state.tasks,
            key="time_task_select"
        )

        best_time, time_score = brain.recommend_time(
            st.session_state.tasks.index(target_task)
        )

        # Priority recommendation
        priority, priority_score = brain.recommend_priority(
            st.session_state.tasks.index(target_task)
        )

        if best_time:
            st.success(f"Waktu terbaik: {best_time}")
            st.write(f"Skor asosiasi waktu: {time_score:.2f}")
            st.progress(min(time_score / 10, 1.0))
            st.divider()

        if priority:
            priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            emoji = priority_emoji.get(priority, "⚪")
            st.markdown(f"**Prioritas: {emoji} {priority}**")
            st.write(f"Skor prioritas: {priority_score:.2f}")
            st.progress(min(priority_score / 10, 1.0))
        else:
            st.warning("Belum ada pembelajaran.")


# ============================
# TAB 3: SOM VISUALIZATION
# ============================

def render_som_tab(brain):
    """Render tab visualisasi SOM."""
    col1, col2 = st.columns(2)

    # SOM - Waktu
    with col1:
        st.subheader("SOM Grid - Pola Waktu")

        if len(st.session_state.history) >= 4:
            df_time, som_time, X_scaled_time = brain.run_som_time()

            if df_time is not None:
                fig = plot_som_grid(
                    som_time, X_scaled_time,
                    st.session_state.tasks, df_time,
                    color_by="task",
                    title="SOM Grid - Pola Produktivitas per Waktu"
                )
                st.pyplot(fig)
        else:
            st.info("Minimal 4 aktivitas diperlukan.")

    # SOM - Prioritas
    with col2:
        st.subheader("SOM Grid - Pola Prioritas")

        if len(st.session_state.history) >= 4:
            df_priority, som_priority, X_scaled_priority, cluster_map = brain.run_som_priority()

            if df_priority is not None:
                fig = plot_som_grid(
                    som_priority, X_scaled_priority,
                    st.session_state.tasks, df_priority,
                    color_by="priority",
                    title="SOM Grid - Cluster Prioritas"
                )
                st.pyplot(fig)
        else:
            st.info("Minimal 4 aktivitas diperlukan.")

    # Interpretasi Cluster
    st.subheader("Interpretasi Cluster Prioritas")

    if len(st.session_state.clusters_priority) > 0:
        cluster_df = st.session_state.clusters_priority

        for cluster_id in sorted(cluster_df["priority_cluster"].unique()):
            subset = cluster_df[cluster_df["priority_cluster"] == cluster_id]

            dominant_task = subset["task"].mode()[0]
            priority_label = subset["priority_label"].mode()[0]
            avg_deadline = subset["deadline"].mean()
            avg_importance = subset["importance"].mean()
            avg_effort = subset["effort"].mean()

            priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            emoji = priority_emoji.get(priority_label, "⚪")

            st.markdown(f"""
            ### Cluster {cluster_id}: {emoji} {priority_label}

            | Metrik | Nilai |
            |--------|-------|
            | Task Dominan | {dominant_task} |
            | Avg Urgency | {avg_deadline:.1f} |
            | Avg Importance | {avg_importance:.1f} |
            | Avg Effort | {avg_effort:.1f} |
            """)
    else:
        st.warning("Belum cukup data untuk clustering.")


# ============================
# TAB 4: HISTORY
# ============================

def render_history_tab(brain):
    """Render tab histori aktivitas."""
    st.subheader("Histori Aktivitas")

    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        # Tampilkan statistik
        st.markdown("### Statistik")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Aktivitas", len(st.session_state.history))

        with col2:
            st.metric("Total Task", len(st.session_state.tasks))

        with col3:
            avg_energy = np.mean([h["energy"] for h in st.session_state.history])
            st.metric("Avg Energi", f"{avg_energy:.1f}")

        with col4:
            avg_deadline = np.mean([h["deadline"] for h in st.session_state.history])
            st.metric("Avg Urgency", f"{avg_deadline:.1f}")
    else:
        st.info("Belum ada histori.")


# ============================
# TASK MANAGEMENT
# ============================

def render_task_management(brain):
    """Render section manajemen task (hapus task, priority matrix)."""
    st.subheader("Manajemen Task")

    col1, col2 = st.columns(2)

    with col1:
        task_to_delete = st.selectbox(
            "Hapus Task",
            st.session_state.tasks,
            key="delete_task_select"
        )

        if st.button("Hapus Task", key="delete_task_btn"):
            idx = st.session_state.tasks.index(task_to_delete)
            st.session_state.tasks.pop(idx)

            if len(st.session_state.tasks) > 0:
                st.session_state.time_weights = np.delete(
                    st.session_state.time_weights, idx, axis=0
                )
                st.session_state.priority_weights = np.delete(
                    st.session_state.priority_weights, idx, axis=0
                )
            else:
                st.session_state.time_weights = np.zeros((0, 4))
                st.session_state.priority_weights = np.zeros((0, 3))

            save_state()
            st.success(f"Task '{task_to_delete}' dihapus.")
            st.rerun()

    with col2:
        st.markdown("### Priority Weight Matrix")

        if len(st.session_state.priority_weights) > 0:
            df_priority_weights = pd.DataFrame(
                st.session_state.priority_weights,
                index=st.session_state.tasks,
                columns=PRIORITY_LEVELS
            )
            st.dataframe(
                df_priority_weights.style.background_gradient(
                    cmap="Oranges",
                    axis=None
                ),
                use_container_width=True
            )
