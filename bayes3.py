
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Carregar os dados
file_path = "Invistico_Airline.csv"
data = pd.read_csv(file_path)

# Definir a estrutura do modelo
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

# CPDs das variáveis independentes (sem dependências)

cpd_age = TabularCPD(variable="Age", variable_card=3, values=[[0.36], [0.54], [0.09]])  # Idade: 0: Jovem, 1: Adulto, 2: Idoso
cpd_gender = TabularCPD(variable="Gender", variable_card=2, values=[[0.51], [0.49]])  # 0: Male, 1: Female
cpd_type_of_travel = TabularCPD(variable="Type of Travel", variable_card=2, values=[[0.69], [0.31]])  # 0: Business, 1: Personal
cpd_class = TabularCPD(variable="Class", variable_card=3, values=[[0.47], [0.44], [0.09]])  # 0: Business, 1: Economy, 2: Other
cpd_seat_comfort = TabularCPD(variable="Seat comfort", variable_card=5, values=[[0.08], [0.18], [0.25], [0.25], [0.24]])  # 0: Bad, 4: Excellent
cpd_flight_distance = TabularCPD(variable="Flight Distance", variable_card=3, values=[[0.68], [0.30], [0.02]])  # 0: Curta, 1: Média, 2: Longa
cpd_inflight_entertainment = TabularCPD(variable="Inflight Entertainment", variable_card=5, values=[[0.10], [0.16], [0.20], [0.20], [0.34]])  # 0: Bad, 4: Excellent
cpd_on_board_service = TabularCPD(variable="On-board Service", variable_card=3, values=[[0.23], [0.51], [0.26]])  # 0: Bad, 2: Good
cpd_food_and_drink = TabularCPD(variable="Food and Drink", variable_card=3, values=[[0.2], [0.41], [0.39]])  # 0: Bad, 2: Good
cpd_departure_delay = TabularCPD(variable="Departure Delay in Minutes", variable_card=3, values=[[0.99], [0.005], [0.005]])  # 0: Bad, 2: Good
cpd_arrival_delay = TabularCPD(variable="Arrival Delay in Minutes", variable_card=3, values=[[0.99], [0.005], [0.005]])  # 0: Bad, 2: Good
cpd_baggage_handling = TabularCPD(variable="Baggage Handling", variable_card=3, values=[[0.06], [0.28], [0.66]])  # 0: Bad, 2: Good
cpd_inflight_wifi = TabularCPD(variable="Inflight Wifi Service", variable_card=6, values=[[0.001], [0.129], [0.20], [0.21], [0.24], [0.22]])  # 0: Bad, 5: Excellent
cpd_online_support = TabularCPD(variable="Online Support", variable_card=3, values=[[0.3], [0.4], [0.3]])  # 0: Bad, 2: Good
cpd_checkin_service = TabularCPD(variable="Checkin Service", variable_card=3, values=[[0.2], [0.5], [0.3]])  # 0: Bad, 2: Good
cpd_cleanliness = TabularCPD(variable="Cleanliness", variable_card=3, values=[[0.25], [0.5], [0.25]])  # 0: Bad, 2: Good

# Adicionando as CPDs ao modelo
model.add_cpds(
    cpd_age, cpd_gender, cpd_type_of_travel, cpd_class, cpd_seat_comfort, cpd_flight_distance,
    cpd_inflight_entertainment, cpd_on_board_service, cpd_food_and_drink, cpd_departure_delay,
    cpd_arrival_delay, cpd_baggage_handling, cpd_inflight_wifi, cpd_online_support, cpd_checkin_service,
    cpd_cleanliness
)

# Estimando todas as CPDs a partir dos dados
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Verificando o modelo
assert model.check_model()

# Realizando inferência sobre "satisfaction" com base nas evidências fornecidas
inference = VariableElimination(model)

query_result = inference.query(
    variables=["satisfaction"],
    evidence={
        "Customer Type": 1,  # Tipo de cliente (1: Returning, 0: New)
        "Class": 2,  # Classe (2: Other)
        "Seat comfort": 1,  # Conforto do assento (1: Average)
        "Flight Distance": 0,  # Distância do voo (0: Short)
        "Inflight Entertainment": 2,  # Entretenimento a bordo (2: Good)
        "On-board Service": 1,  # Serviço a bordo (1: Average)
        "Food and Drink": 0,  # Comida e bebida (0: Bad)
        "Departure Delay in Minutes": 0,  # Atraso na partida (0: Good)
        "Arrival Delay in Minutes": 0,  # Atraso na chegada (0: Good)
        "Baggage Handling": 0,  # Manuseio de bagagem (0: Bad)
        "Inflight Wifi Service": 0,  # Serviço de Wifi a bordo (0: Bad)
        "Online Support": 0,  # Suporte online (0: Bad)
        "Checkin Service": 1,  # Serviço de check-in (1: Average)
        "Cleanliness": 0,  # Limpeza (0: Bad)
    },
)

# Exibindo o resultado da inferência
print(query_result)
