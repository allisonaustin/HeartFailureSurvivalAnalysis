# CECS 456 Project
# Group 9
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

# reading file
filename = "heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(filename, sep=',')


# creating KaplanMeierFitter objects for male and female patients
kmf_m = KaplanMeierFitter()
kmf_f = KaplanMeierFitter()

# creating KaplanMeierFitter objects for ejection fraction levels
low_ef = KaplanMeierFitter()
moderate_ef = KaplanMeierFitter()
high_ef = KaplanMeierFitter()

male = df.query("sex == 1")
female = df.query("sex == 0")

low = df.query("ejection_fraction <= 30")
moderate = df.query("ejection_fraction > 30 & ejection_fraction <= 45")
high = df.query("ejection_fraction > 45")

kmf_m.fit(durations = male["time"],event_observed = male["DEATH_EVENT"], label="Male")
kmf_f.fit(durations = female["time"],event_observed = female["DEATH_EVENT"], label="Female")

low_ef.fit(durations = low["time"],event_observed = low["DEATH_EVENT"], label="EF <= 30")
moderate_ef.fit(durations = moderate["time"],event_observed = moderate["DEATH_EVENT"], label="30 < EF <= 45")
high_ef.fit(durations = high["time"],event_observed = high["DEATH_EVENT"], label="EF > 45")

# export to excel sheet
# to run the excel sheet exports, pip install openpyxl before running
print("Male event table")
kmf_m_df = kmf_m.event_table
print(kmf_m_df)
kmf_m_df.to_excel("kmf_male.xlsx")
print("Female event table")
kmf_f_df = kmf_f.event_table
print(kmf_f_df)
kmf_f_df.to_excel("kmf_female.xlsx")

print("Predicting survival probabilities after 250 days\nMale:", kmf_m.predict(250))
print("Female:", kmf_f.predict(250))

# export to excel sheet
print("Complete list of survival possibilities")
kmf_m_survival = kmf_m.survival_function_
print(kmf_m_survival)
kmf_m_survival.to_excel("kmf_male_survival.xlsx")
kmf_f_survival = kmf_f.survival_function_
print(kmf_f_survival)
kmf_f_survival.to_excel("kmf_female_survival.xlsx")

kmf_m.plot()
kmf_f.plot()

plt.xlabel("Days passed")
plt.ylabel("Survival")
plt.title("KMF")

plt.show()


# predicting EF after 250
print("Predicting survival probabilities after 250 days\nLow EF:", low_ef.predict(250))
print("Moderate", moderate_ef.predict(250))
print("High EF", high_ef.predict(250))



low_ef.plot()
moderate_ef.plot()
high_ef.plot()

plt.xlabel("Days passed")
plt.ylabel("Survival")
plt.title("Ejection Fraction")



plt.show()



# cox proportional hazard method (cox regression)