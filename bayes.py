import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination

data = pd.read_csv("./Invistico_Airline.csv")

data["satisfaction"] = data["satisfaction"].map(
    {"satisfied": 1, "neutral or dissatisfied": 0}
)
data["Gender"] = data["Gender"].map({"Female": 1, "Male": 0})
data["Customer Type"] = data["Customer Type"].map(
    {"Loyal Customer": 1, "disloyal Customer": 0}
)
data["Type of Travel"] = data["Type of Travel"].map(
    {"Personal Travel": 1, "Business travel": 0}
)
data["Class"] = data["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})

data = data[
    [
        "satisfaction",
        "Gender",
        "Customer Type",
        "Age",
        "Type of Travel",
        "Class",
        "Flight Distance",
        "Seat comfort",
        "Departure/Arrival time convenient",
        "Food and drink",
        "Gate location",
        "Inflight wifi service",
        "Inflight entertainment",
        "Online support",
        "Ease of Online booking",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness",
        "Online boarding",
        "Departure Delay in Minutes",
    ]
]

hc = HillClimbSearch(data)
best_model_structure = hc.estimate(scoring_method=BicScore(data))

model = BayesianNetwork(best_model_structure.edges())

model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")

assert model.check_model()

for cpd in model.get_cpds():
    print(f"CPD de {cpd.variable}:")
    print(cpd)
    print("\n")

infer = VariableElimination(model)
resultado = infer.query(
    variables=["satisfaction"],
    evidence={
        "Departure Delay in Minutes": 2,
        "Seat comfort": 1,
        "Food and drink": 2,
        "On-board service": 2,
        "Leg room service": 1,
        "Checkin service": 3,
        "Online boarding": 4,
        "Ease of Online booking": 3,
        "Inflight entertainment": 2,
    },
)

print("Distribuição da variável 'satisfaction' com evidências múltiplas:")
print(resultado)
print("\nValores da distribuição de probabilidade:")
print(resultado.values)
print("\nVariáveis com a distribuição:")
print(resultado.variables)
