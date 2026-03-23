# Hebbian Task Optimizer

A Streamlit-based application that learns user productivity patterns by strengthening the relationship between task types and optimal time slots using a simplified Hebbian Learning approach.

---

## Overview

This project simulates a basic learning system that tracks how efficiently tasks are completed at different times of the day.

The system reinforces patterns where:
- a specific task is performed
- at a specific time
- with good performance

Over time, it recommends the most effective time slot for each task.

---

## How It Works

The model uses a simplified version of Hebbian Learning:

> "Neurons that fire together, wire together"

In this implementation:
- Task = input
- Time slot = paired input
- Performance score = reinforcement signal

Each time the user logs an activity:
1. The system applies **weight decay** (to allow adaptation)
2. Updates the weight for the selected task–time pair
3. Strengthens or weakens the relationship based on performance

The system then:
- selects the highest weight per task
- recommends it as the optimal time

---

## Features

- Task activity logging (task + time + performance)
- Hebbian-style weight updates
- Adaptive learning with decay
- Weight matrix visualization (heatmap)
- Time recommendation per task
- Reset model functionality

---

## Tech Stack

- Python
- Streamlit
- NumPy
- Pandas

---

## Installation

```bash
git clone https://github.com/RIOKOWI/task-prioritizer.git

cd task-prioritizer

pip install -r requirements.txt

streamlit run app.py
```
