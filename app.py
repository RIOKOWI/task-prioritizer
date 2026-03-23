import streamlit as st  
import numpy as np      
import pandas as pd

class HebbianPrioritizer:
    def __init__(self, tasks, time_slots):
        self.tasks = tasks            
        self.time_slots = time_slots