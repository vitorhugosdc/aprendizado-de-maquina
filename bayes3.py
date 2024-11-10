import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Carregar os dados
file_path = "Invistico_Airline.csv"
data = pd.read_csv(file_path)
data_subset = data.sample(n=1000, random_state=42)  # Treine com 1000 amostras, por exemplo

# Definir a estrutura do modelo
model = BayesianNetwork(
    [
        ("Customer Type", "satisfaction"),
        ("Class", "satisfaction"),
        ("Seat comfort", "satisfaction"),
        ("Age", "Customer Type"),
        ("Flight Distance", "Arrival Delay in Minutes"),
        ("Inflight entertainment", "satisfaction"),
        ("On-board service", "satisfaction"),
        ("Food and drink", "satisfaction"),
        ("Departure Delay in Minutes", "satisfaction"),
        ("Arrival Delay in Minutes", "satisfaction"),
    ]
)

# Estimando as CPDs usando MLE
model.fit(data_subset, estimator=MaximumLikelihoodEstimator)

# Verificando o modelo
assert model.check_model()


inference = VariableElimination(model)

query_result = inference.query(
    variables=["satisfaction"],
    evidence={
        "Customer Type": 1,
        "Class": 2,
        "Seat comfort": 1,
        "Flight Distance": 0,
        "Inflight entertainment": 2,
        "On-board service": 1,
        "Food and drink": 0,
        "Departure Delay in Minutes": 0,
        "Arrival Delay in Minutes": 0,
        "Baggage handling": 0,
        "Inflight wifi service": 0,
        "Online support": 0,
        "Checkin service": 1,
        "Cleanliness": 0,
    },
)
print(query_result)
