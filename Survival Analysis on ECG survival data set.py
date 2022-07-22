#Survaival analysis on ECG survival data
#loading the packages
import pandas as pd

#loading the data set into python
ecg_data = pd.read_excel("E:/Data Science 18012022/Survival Analytics/ECG_Surv.xlsx")
ecg_data.head()
ecg_data.describe()
ecg_data.columns

ecg_data["survival_time_hr"].describe()

#EDA
#Selecting required column
ecg_data = ecg_data[["survival_time_hr","alive","group"]]
#defining followup of the customer
T = ecg_data.survival_time_hr

#importing the package required for the survival analysis
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

#Fitting the model on survival time and alive of the data set
kmf.fit(T, event_observed=ecg_data.alive)

# Time-line estimations plot 
kmf.plot()

#Fitting the model for different groups
ecg_data.group.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[ecg_data.group==1], ecg_data.alive[ecg_data.group==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[ecg_data.group==2], ecg_data.alive[ecg_data.group==2], label='2')
ax2 = kmf.plot()
kmf.plot(ax=ax2)
