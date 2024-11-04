import pandas as pd
import numpy as np
from pgmpy.factors.discrete import TabularCPD

file_path = "Invistico_Airline.csv"
data = pd.read_csv(file_path)

cpds = {}

flight_distance_dist = (
    data["Flight Distance"].value_counts(normalize=True, bins=3).sort_index()
)
cpds["cpd_flight_distance"] = TabularCPD(
    variable="Flight Distance",
    variable_card=3,
    values=[[flight_distance_dist[i]] for i in flight_distance_dist.index],
)

age_dist = data["Age"].value_counts(normalize=True, bins=3).sort_index()
cpds["cpd_age"] = TabularCPD(
    variable="Age", variable_card=3, values=[[age_dist[i]] for i in age_dist.index]
)

gender_dist = data["Gender"].value_counts(normalize=True)
cpds["cpd_gender"] = TabularCPD(
    variable="Gender",
    variable_card=2,
    values=[[gender_dist[i]] for i in gender_dist.index],
)

customer_type_dist = data["Customer Type"].value_counts(normalize=True)
cpds["cpd_customer_type"] = TabularCPD(
    variable="Customer Type",
    variable_card=2,
    values=[[customer_type_dist[i]] for i in customer_type_dist.index],
)

type_of_travel_dist = data["Type of Travel"].value_counts(normalize=True)
cpds["cpd_type_of_travel"] = TabularCPD(
    variable="Type of Travel",
    variable_card=2,
    values=[[type_of_travel_dist[i]] for i in type_of_travel_dist.index],
)

class_dist = data["Class"].value_counts(normalize=True)
cpds["cpd_class"] = TabularCPD(
    variable="Class",
    variable_card=3,
    values=[[class_dist[i]] for i in class_dist.index],
)

departure_delay_dist = (
    data["Departure Delay in Minutes"].value_counts(normalize=True, bins=3).sort_index()
)
cpds["cpd_departure_delay"] = TabularCPD(
    variable="Departure Delay in Minutes",
    variable_card=3,
    values=[[departure_delay_dist[i]] for i in departure_delay_dist.index],
)

arrival_delay_dist = (
    data["Arrival Delay in Minutes"].value_counts(normalize=True, bins=3).sort_index()
)
cpds["cpd_arrival_delay"] = TabularCPD(
    variable="Arrival Delay in Minutes",
    variable_card=3,
    values=[[arrival_delay_dist[i]] for i in arrival_delay_dist.index],
)

satisfaction_cond = (
    data.groupby(["Customer Type", "Type of Travel", "Class"])["satisfaction"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)
satisfaction_values = satisfaction_cond.values.T.tolist()
cpds["cpd_satisfaction"] = TabularCPD(
    variable="satisfaction",
    variable_card=2,
    values=satisfaction_values,
    evidence=["Customer Type", "Type of Travel", "Class"],
    evidence_card=[2, 2, 3],
)

service_vars = [
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Food and drink",
    "Inflight wifi service",
    "Cleanliness",
    "Baggage handling",
]
for var in service_vars:
    dist = data[var].value_counts(normalize=True)
    cpds[f'cpd_{var.lower().replace(" ", "_")}'] = TabularCPD(
        variable=var, variable_card=5, values=[[dist.get(i, 0)] for i in range(5)]
    )

for cpd_name, cpd in cpds.items():
    print(f"### {cpd_name} ###")
    print(cpd)
    print("\n")
