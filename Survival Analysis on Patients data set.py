#Survaival analysis on patients data
#loading the packages
import pandas as pd

#loading the data set into python
patient_data = pd.read_csv("E:/Data Science 18012022/Survival Analytics/Patient.csv")
patient_data.head()
patient_data.describe()

patient_data["Followup"].describe()

#defining followup of the customer
T = patient_data.Followup

#importing the package required for the survival analysis
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

#Fitting the model on followup and event type of the data set
kmf.fit(T, event_observed=patient_data.Eventtype)

# Time-line estimations plot 
kmf.plot()


#Fitting the model for different PatientID's
patient_data.PatientID.value_counts()

# Applying KaplanMeierFitter model on different patients
kmf.fit(T[patient_data.PatientID], patient_data.Eventtype[patient_data.PatientID])
ax = kmf.plot()


