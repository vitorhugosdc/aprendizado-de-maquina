import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

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

cpd_customer_type = TabularCPD(
    variable="Customer Type",
    variable_card=2,
    values=[[0.7, 0.4, 0.6, 0.3, 0.5, 0.4], [0.3, 0.6, 0.4, 0.7, 0.5, 0.6]],
    evidence=["Age", "Gender"],
    evidence_card=[3, 2],
)

values_satisfaction = []
num_combinations = 35429400

for _ in range(num_combinations):
    p = np.random.uniform(0.3, 0.7)
    values_satisfaction.append([p, 1 - p])

values_satisfaction = np.array(values_satisfaction).T.tolist()

cpd_satisfaction = TabularCPD(
    variable="satisfaction",
    variable_card=2,
    values=values_satisfaction,
    evidence=[
        "Customer Type",
        "Type of Travel",
        "Class",
        "Seat comfort",
        "Flight Distance",
        "Inflight Entertainment",
        "On-board Service",
        "Food and Drink",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        "Baggage Handling",
        "Inflight Wifi Service",
        "Online Support",
        "Checkin Service",
        "Cleanliness",
    ],
    evidence_card=[2, 2, 3, 5, 3, 5, 3, 3, 3, 3, 3, 6, 3, 3, 3],
)

# A idade é variável então foi dividido entre 3 categorias:
# 0: Jovem, 1: Adulto, 2: idoso
cpd_age = TabularCPD(variable="Age", variable_card=3, values=[[0.36], [0.54], [0.09]])
# 0: Male, 1: Female
cpd_gender = TabularCPD(variable="Gender", variable_card=2, values=[[0.51], [0.49]])
# 0: Business, 1: Personal
cpd_type_of_travel = TabularCPD(
    variable="Type of Travel", variable_card=2, values=[[0.69], [0.31]]
)
# 0: Business, 1: Economy, 2: Other
cpd_class = TabularCPD(
    variable="Class", variable_card=3, values=[[0.47], [0.44], [0.09]]
)
# significados (definidos por mim)
# 0: Bad, 1: Average, 2: Good, 3: Very good, 4: Excellent
cpd_seat_comfort = TabularCPD(
    variable="Seat comfort",
    variable_card=5,
    values=[[0.08], [0.18], [0.25], [0.25], [0.24]],
)

# A distância também era variável então foi dividida em 3 categorias:
# 0: Curta, 1: Média, 2: Longa
cpd_flight_distance = TabularCPD(
    variable="Flight Distance",
    variable_card=3,
    values=[[0.68], [0.30], [0.02]],
)
# 0: Bad, 1: Average, 2: Good, 3: Very good, 4: Excellent
cpd_inflight_entertainment = TabularCPD(
    variable="Inflight Entertainment",
    variable_card=5,
    values=[[0.10], [0.16], [0.20], [0.20], [0.34]],
)
# 0: Bad, 1: Average, 2: Good
cpd_on_board_service = TabularCPD(
    variable="On-board Service",
    variable_card=3,
    values=[[0.23], [0.51], [0.26]],
)
# 0: Bad, 1: Average, 2: Good
cpd_food_and_drink = TabularCPD(
    variable="Food and Drink",
    variable_card=3,
    values=[[0.2], [0.41], [0.39]],
)
# 0: Bad, 1: Average, 2: Good
cpd_departure_delay = TabularCPD(
    variable="Departure Delay in Minutes",
    variable_card=3,
    values=[[0.99], [0.005], [0.005]],
)
# 0: Bad, 1: Average, 2: Good
cpd_arrival_delay = TabularCPD(
    variable="Arrival Delay in Minutes",
    variable_card=3,
    values=[[0.99], [0.005], [0.005]],
)
# 0: Bad, 1: Average, 2: Good
cpd_baggage_handling = TabularCPD(
    variable="Baggage Handling",
    variable_card=3,
    values=[[0.06], [0.28], [0.66]],
)
# 0: Bad, 1: Average, 2: Good, 3: Very good, 4: Excellent
cpd_inflight_wifi = TabularCPD(
    variable="Inflight Wifi Service",
    variable_card=6,
    values=[[0.001], [0.129], [0.20], [0.21], [0.24], [0.22]],
)
# 0: Bad, 1: Average, 2: Good
cpd_online_support = TabularCPD(
    variable="Online Support",
    variable_card=3,
    values=[[0.3], [0.4], [0.3]],
)
# 0: Bad, 1: Average, 2: Good
cpd_checkin_service = TabularCPD(
    variable="Checkin Service",
    variable_card=3,
    values=[[0.2], [0.5], [0.3]],
)
# 0: Bad, 1: Average, 2: Good
cpd_cleanliness = TabularCPD(
    variable="Cleanliness",
    variable_card=3,
    values=[[0.25], [0.5], [0.25]],
)

model.add_cpds(
    cpd_customer_type,
    cpd_satisfaction,
    cpd_age,
    cpd_gender,
    cpd_type_of_travel,
    cpd_class,
    cpd_seat_comfort,
    cpd_flight_distance,
    cpd_inflight_entertainment,
    cpd_on_board_service,
    cpd_food_and_drink,
    cpd_departure_delay,
    cpd_arrival_delay,
    cpd_baggage_handling,
    cpd_inflight_wifi,
    cpd_online_support,
    cpd_checkin_service,
    cpd_cleanliness,
)

assert model.check_model()

inference = VariableElimination(model)

query_result = inference.query(
    variables=["satisfaction"],
    evidence={
        "Customer Type": 1,
        "Class": 2,
        "Seat comfort": 1,
        "Flight Distance": 0,
        "Inflight Entertainment": 2,
        "On-board Service": 1,
        "Food and Drink": 0,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
        "Baggage Handling": 0,
        "Inflight Wifi Service": 0,
        "Online Support": 0,
        "Checkin Service": 1,
        "Cleanliness": 0,
    },
)
print(query_result)
