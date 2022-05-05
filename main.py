# CECS 456 Project
# Group 9
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
# to install, run pip install lifelines
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

# creating KaplanMeierFitter objects for low BP and high BP
kmf_hbp = KaplanMeierFitter()
kmf_lbp = KaplanMeierFitter()

male = df.query("sex == 1")
female = df.query("sex == 0")

low = df.query("ejection_fraction <= 30")
moderate = df.query("ejection_fraction > 30 & ejection_fraction <= 45")
high = df.query("ejection_fraction > 45")

high_BP = df.query("high_blood_pressure == 1")
low_BP = df.query("high_blood_pressure == 0")

kmf_m.fit(durations = male["time"],event_observed = male["DEATH_EVENT"], label="Male")
kmf_f.fit(durations = female["time"],event_observed = female["DEATH_EVENT"], label="Female")

low_ef.fit(durations = low["time"],event_observed = low["DEATH_EVENT"], label="EF <= 30")
moderate_ef.fit(durations = moderate["time"],event_observed = moderate["DEATH_EVENT"], label="30 < EF <= 45")
high_ef.fit(durations = high["time"],event_observed = high["DEATH_EVENT"], label="EF > 45")

kmf_hbp.fit(durations=high_BP["time"],event_observed=high_BP["DEATH_EVENT"], label="High Blood Pressure")
kmf_lbp.fit(durations=low_BP["time"],event_observed=low_BP["DEATH_EVENT"], label="Low Blood Pressure")

# export to excel sheet
# to run the excel sheet exports, pip install openpyxl before running
kmf_m_df = kmf_m.event_table
# deleting existing xlsx file
kmf_m_df.to_excel("kmf_male.xlsx")
kmf_f_df = kmf_f.event_table
kmf_f_df.to_excel("kmf_female.xlsx")

print("Predicting survival probabilities after 250 days\nMale:", kmf_m.predict(250))
print("Female:", kmf_f.predict(250))

# export to excel sheet
kmf_m_survival = kmf_m.survival_function_
kmf_m_survival.to_excel("kmf_male_survival.xlsx")
kmf_f_survival = kmf_f.survival_function_
kmf_f_survival.to_excel("kmf_female_survival.xlsx")

# plotting kaplan meier product estimator over time
kmf_m.plot()
kmf_f.plot()

plt.xlabel("Days passed")
plt.ylabel("Survival")
plt.ylim((0,1))
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
plt.ylim((0,1))
plt.title("Ejection Fraction")



plt.show()

# predicting BP after 250
print("Predicting survival probabilities after 250 days\nLow BP:", kmf_lbp.predict(250))
print("High BP", kmf_hbp.predict(250))



kmf_lbp.plot()
kmf_hbp.plot()

plt.xlabel("Days passed")
plt.ylabel("Survival")
plt.ylim((0,1))
plt.title("Blood Pressure")



plt.show()

# cox proportional hazard
cph = CoxPHFitter()
cph.fit(df, "time", event_col="DEATH_EVENT")
cph.print_summary()
