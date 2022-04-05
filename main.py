# CECS 456 Project
# Group 9
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

# reading file
filename = "heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(filename, sep=',')

# converting features into numpy arrays
age = df['age'].to_numpy()
anaemia = df['anaemia'].to_numpy() # boolean (1 = anaemic, 0 = non-anaemic)
creat_phos = df['creatinine_phosphokinase'].to_numpy()
diabetes = df['diabetes'].to_numpy() # boolean (1 = diabetic, 0 = non-diabetic)
eject_fract = df['ejection_fraction'].to_numpy()
bp = df['high_blood_pressure'].to_numpy() # boolean (1 = high, 0 = low)
platelets = df['platelets'].to_numpy()
serum_creat = df['serum_creatinine'].to_numpy()
serum_sodium = df['serum_sodium'].to_numpy()
sex = df['sex'].to_numpy() # boolean (1 = male, 0 = female)
smoking = df['smoking'].to_numpy() # boolean (1 = smoker, 0 = non-smoker)
time = df['time'].to_numpy() # days
death = df['DEATH_EVENT'].to_numpy() # boolean (1 = death, 0 = no death)

