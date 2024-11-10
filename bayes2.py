import pandas as pd
import numpy as np 
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD 
from pgmpy.inference import VariableElimination 
from pgmpy.estimators import MaximumLikelihoodEstimator 

file_path = "Invistico_Airline.csv"
data = pd.read_csv(file_path)

model = BayesianNetwork(
    [
        ("Customer Type", "satisfaction"),
        ("Type of Travel", "satisfaction"),
        ("Class", "satisfaction"),
        ("Seat comfort", "satisfaction"),
        ("Age", "Customer Type"),
        ("Gender", "Customer Type"),
        ("Flight Distance", "satisfaction"),
        ("Inflight Entertainment", "satisfaction"),
        ("On-board Service", "satisfaction"),
        ("Food and Drink", "satisfaction"),
        ("Departure Delay in Minutes", "satisfaction"),
        ("Arrival Delay in Minutes", "satisfaction"),
        ("Baggage Handling", "satisfaction"),
        ("Inflight Wifi Service", "satisfaction"),
        ("Online Support", "satisfaction"),
        ("Checkin Service", "satisfaction"),
        ("Cleanliness", "satisfaction"),
    ]
)

# Estimar as CPDs
estimator = MaximumLikelihoodEstimator(model, data)
cpd_age = estimator.estimate_cpd("Age")
cpd_gender = estimator.estimate_cpd("Gender")
cpd_customer_type = estimator.estimate_cpd("Customer Type")
cpd_flight_distance = estimator.estimate_cpd("Flight Distance")
cpd_departure_delay = estimator.estimate_cpd("Departure Delay in Minutes")
cpd_arrival_delay = estimator.estimate_cpd("Arrival Delay in Minutes")

print(cpd_arrival_delay)




