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


def validar_evidencias(evidencias):
    # Limites máximos definidos nas TabularCPDs
    limites_variaveis = {
        "Customer Type": 1,
        "Type of Travel": 1,
        "Class": 2,
        "Seat comfort": 4,
        "Flight Distance": 2,
        "Inflight Entertainment": 4,
        "On-board Service": 2,
        "Food and Drink": 2,
        "Departure Delay in Minutes": 2,
        "Arrival Delay in Minutes": 2,
        "Baggage Handling": 2,
        "Inflight Wifi Service": 5,
        "Online Support": 2,
        "Checkin Service": 2,
        "Cleanliness": 2,
        "Age": 2,
        "Gender": 1,
    }

    # Filtra evidências fora dos limites
    evidencias_validas = {}
    for var, valor in evidencias.items():
        if var in limites_variaveis and 0 <= valor <= limites_variaveis[var]:
            evidencias_validas[var] = valor
        else:
            print(
                f"Aviso: Valor inválido '{valor}' para a variável '{var}'. Ignorando essa evidência."
            )
    return evidencias_validas


def realizar_testes_satisfacao():
    testes = [
        # Cenários simples
        {
            "descricao": "Classe econômica com serviço de bordo ruim",
            "evidencia": {
                "Class": 1,  # Economy
                "On-board Service": 2,  # Serviço ruim
            },
        },
        {
            "descricao": "Cliente frequente em voo curto com conforto do assento médio",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Flight Distance": 0,  # Curta distância
                "Seat comfort": 2,  # Conforto médio
            },
        },
        {
            "descricao": "Cliente de classe Business sem atraso na chegada",
            "evidencia": {
                "Class": 0,  # Business
                "Arrival Delay in Minutes": 0,  # Sem atraso
            },
        },
        {
            "descricao": "Cliente de classe econômica com conforto do assento excelente",
            "evidencia": {
                "Class": 1,  # Economy
                "Seat comfort": 4,  # Excelente conforto
            },
        },
        {
            "descricao": "Jovem com Wi-Fi ruim",
            "evidencia": {
                "Age": 0,  # Jovem
                "Inflight Wifi Service": 0,  # Wi-Fi ruim
            },
        },
        # Cenários moderadamente complexos
        {
            "descricao": "Cliente idoso com atraso médio na chegada e limpeza média",
            "evidencia": {
                "Age": 2,  # Idoso
                "Arrival Delay in Minutes": 1,  # Atraso médio
                "Cleanliness": 1,  # Limpeza média
            },
        },
        {
            "descricao": "Classe Business com suporte online bom e sem atraso na decolagem",
            "evidencia": {
                "Class": 0,  # Business
                "Online Support": 2,  # Bom suporte online
                "Departure Delay in Minutes": 0,  # Sem atraso
            },
        },
        {
            "descricao": "Cliente frequente de classe econômica com serviço de bordo médio",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Class": 1,  # Economy
                "On-board Service": 1,  # Serviço médio
            },
        },
        {
            "descricao": "Viagem pessoal com conforto de assento ruim e excelente entretenimento",
            "evidencia": {
                "Type of Travel": 1,  # Viagem pessoal
                "Seat comfort": 0,  # Conforto ruim
                "Inflight Entertainment": 4,  # Excelente entretenimento
            },
        },
        {
            "descricao": "Cliente jovem em classe Business com serviço de bordo ruim",
            "evidencia": {
                "Age": 0,  # Jovem
                "Class": 0,  # Business
                "On-board Service": 0,  # Serviço ruim
            },
        },
        # Cenários mais complexos
        {
            "descricao": "Cliente em viagem pessoal, longa distância, bom conforto de assento, entretenimento médio",
            "evidencia": {
                "Customer Type": 0,  # Personal
                "Type of Travel": 1,  # Viagem pessoal
                "Flight Distance": 2,  # Longa distância
                "Seat comfort": 3,  # Bom conforto
                "Inflight Entertainment": 2,  # Entretenimento médio
            },
        },
        {
            "descricao": "Frequent flyer na classe econômica com atraso de chegada, limpeza ruim, e Wi-Fi ruim",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Class": 1,  # Economy
                "Arrival Delay in Minutes": 2,  # Grande atraso
                "Cleanliness": 0,  # Limpeza ruim
                "Inflight Wifi Service": 0,  # Wi-Fi ruim
            },
        },
        {
            "descricao": "Cliente jovem, classe Business, serviço de bordo excelente, check-in médio",
            "evidencia": {
                "Age": 0,  # Jovem
                "Class": 0,  # Business
                "On-board Service": 2,  # Excelente serviço
                "Checkin Service": 1,  # Check-in médio
            },
        },
        {
            "descricao": "Cliente idoso, voo longo, comida excelente, limpeza boa, sem atraso",
            "evidencia": {
                "Age": 2,  # Idoso
                "Flight Distance": 2,  # Longa distância
                "Food and Drink": 2,  # Excelente comida
                "Cleanliness": 2,  # Limpeza boa
                "Arrival Delay in Minutes": 0,  # Sem atraso
            },
        },
        {
            "descricao": "Cliente de viagem pessoal com conforto ruim e check-in ruim",
            "evidencia": {
                "Type of Travel": 1,  # Viagem pessoal
                "Seat comfort": 0,  # Conforto ruim
                "Checkin Service": 0,  # Check-in ruim
            },
        },
        # Cenários mais detalhados
        {
            "descricao": "Frequent flyer de classe econômica com bom conforto, atraso de chegada e Wi-Fi excelente",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Class": 1,  # Economy
                "Seat comfort": 3,  # Bom conforto
                "Arrival Delay in Minutes": 1,  # Atraso médio
                "Inflight Wifi Service": 4,  # Excelente Wi-Fi
            },
        },
        {
            "descricao": "Cliente em voo curto, limpeza excelente, sem entretenimento e sem atraso",
            "evidencia": {
                "Flight Distance": 0,  # Curta distância
                "Cleanliness": 2,  # Limpeza excelente
                "Inflight Entertainment": 0,  # Sem entretenimento
                "Departure Delay in Minutes": 0,  # Sem atraso
            },
        },
        {
            "descricao": "Cliente de classe Business, bom conforto e comida média",
            "evidencia": {
                "Class": 0,  # Business
                "Seat comfort": 3,  # Bom conforto
                "Food and Drink": 1,  # Comida média
            },
        },
        {
            "descricao": "Cliente de viagem pessoal com excelente suporte online, Wi-Fi bom e check-in excelente",
            "evidencia": {
                "Type of Travel": 1,  # Viagem pessoal
                "Online Support": 2,  # Suporte excelente
                "Inflight Wifi Service": 3,  # Wi-Fi bom
                "Checkin Service": 2,  # Check-in excelente
            },
        },
        {
            "descricao": "Cliente frequente em classe econômica com limpeza ruim e atraso de chegada",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Class": 1,  # Economy
                "Cleanliness": 0,  # Limpeza ruim
                "Arrival Delay in Minutes": 1,  # Atraso médio
            },
        },
        {
            "descricao": "Cliente em viagem pessoal, longa distância, bom conforto de assento, entretenimento médio, e serviço de bordo excelente",
            "evidencia": {
                "Customer Type": 0,  # Personal
                "Type of Travel": 1,  # Viagem pessoal
                "Flight Distance": 2,  # Longa distância
                "Seat comfort": 3,  # Bom conforto
                "Inflight Entertainment": 2,  # Entretenimento médio
                "On-board Service": 2,  # Serviço de bordo excelente
            },
        },
        {
            "descricao": "Frequent flyer na classe econômica com grande atraso de chegada, limpeza ruim, Wi-Fi ruim e comida boa",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Class": 1,  # Economy
                "Arrival Delay in Minutes": 2,  # Grande atraso
                "Cleanliness": 0,  # Limpeza ruim
                "Inflight Wifi Service": 0,  # Wi-Fi ruim
                "Food and Drink": 2,  # Comida boa
            },
        },
        {
            "descricao": "Cliente jovem, classe Business, serviço de bordo excelente, check-in médio, entretenimento bom e conforto de assento muito bom",
            "evidencia": {
                "Age": 0,  # Jovem
                "Class": 0,  # Business
                "On-board Service": 2,  # Excelente serviço
                "Checkin Service": 1,  # Check-in médio
                "Inflight Entertainment": 3,  # Entretenimento bom
                "Seat comfort": 3,  # Conforto muito bom
            },
        },
        {
            "descricao": "Cliente idoso, voo longo, comida excelente, limpeza boa, sem atraso e suporte online bom",
            "evidencia": {
                "Age": 2,  # Idoso
                "Flight Distance": 2,  # Longa distância
                "Food and Drink": 2,  # Excelente comida
                "Cleanliness": 2,  # Limpeza boa
                "Arrival Delay in Minutes": 0,  # Sem atraso
                "Online Support": 2,  # Bom suporte online
            },
        },
        {
            "descricao": "Cliente em classe econômica, voo curto, sem entretenimento, sem atraso, conforto ruim, mas limpeza excelente",
            "evidencia": {
                "Class": 1,  # Economy
                "Flight Distance": 0,  # Curta distância
                "Inflight Entertainment": 0,  # Sem entretenimento
                "Departure Delay in Minutes": 0,  # Sem atraso
                "Seat comfort": 0,  # Conforto ruim
                "Cleanliness": 4,  # Limpeza excelente
            },
        },
        {
            "descricao": "Frequent flyer em classe econômica com bom conforto, Wi-Fi excelente, grande atraso de chegada e check-in bom",
            "evidencia": {
                "Customer Type": 1,  # Frequent flyer
                "Class": 1,  # Economy
                "Seat comfort": 3,  # Bom conforto
                "Inflight Wifi Service": 4,  # Excelente Wi-Fi
                "Arrival Delay in Minutes": 2,  # Grande atraso
                "Checkin Service": 2,  # Check-in bom
            },
        },
        {
            "descricao": "Cliente em viagem pessoal, voo curto, serviço de bordo médio, atraso de partida médio e conforto do assento médio",
            "evidencia": {
                "Type of Travel": 1,  # Viagem pessoal
                "Flight Distance": 0,  # Voo curto
                "On-board Service": 1,  # Serviço médio
                "Departure Delay in Minutes": 1,  # Atraso médio
                "Seat comfort": 2,  # Conforto médio
            },
        },
        {
            "descricao": "Cliente idoso em classe Business, serviço de bordo muito bom, Wi-Fi médio, sem atraso de chegada e excelente entretenimento",
            "evidencia": {
                "Age": 2,  # Idoso
                "Class": 0,  # Business
                "On-board Service": 2,  # Serviço de bordo muito bom
                "Inflight Wifi Service": 2,  # Wi-Fi médio
                "Arrival Delay in Minutes": 0,  # Sem atraso de chegada
                "Inflight Entertainment": 4,  # Excelente entretenimento
            },
        },
        {
            "descricao": "Frequent flyer na classe econômica com conforto médio, distância curta, e todos os serviços de bordo ruins",
            "evidencia": {
                "Customer Type": 1,
                "Class": 1,
                "Seat comfort": 2,
                "Flight Distance": 0,
                "Inflight Entertainment": 1,
                "On-board Service": 0,
                "Food and Drink": 0,
                "Departure Delay in Minutes": 1,
                "Arrival Delay in Minutes": 1,
                "Baggage Handling": 0,
                "Inflight Wifi Service": 0,
                "Online Support": 0,
                "Checkin Service": 0,
                "Cleanliness": 0,
            },
        },
        {
            "descricao": "Cliente jovem em classe Business com excelente entretenimento, bom conforto, suporte online médio e Wi-Fi excelente",
            "evidencia": {
                "Age": 0,
                "Class": 0,
                "Seat comfort": 3,
                "Flight Distance": 1,
                "Inflight Entertainment": 4,
                "On-board Service": 2,
                "Food and Drink": 1,
                "Departure Delay in Minutes": 0,
                "Arrival Delay in Minutes": 0,
                "Baggage Handling": 2,
                "Inflight Wifi Service": 4,
                "Online Support": 1,
                "Checkin Service": 2,
                "Cleanliness": 2,
            },
        },
        {
            "descricao": "Cliente idoso em classe econômica com conforto ruim, voo longo, atraso na chegada, e todos os outros serviços no nível médio",
            "evidencia": {
                "Age": 2,
                "Customer Type": 0,
                "Class": 1,
                "Seat comfort": 1,
                "Flight Distance": 2,
                "Inflight Entertainment": 2,
                "On-board Service": 1,
                "Food and Drink": 1,
                "Departure Delay in Minutes": 1,
                "Arrival Delay in Minutes": 2,
                "Baggage Handling": 1,
                "Inflight Wifi Service": 1,
                "Online Support": 1,
                "Checkin Service": 1,
                "Cleanliness": 1,
            },
        },
        {
            "descricao": "Frequent flyer jovem na classe econômica, voo curto com Wi-Fi e check-in ruins, entretenimento médio e conforto excelente",
            "evidencia": {
                "Age": 0,
                "Customer Type": 1,
                "Class": 1,
                "Seat comfort": 4,
                "Flight Distance": 0,
                "Inflight Entertainment": 2,
                "On-board Service": 2,
                "Food and Drink": 1,
                "Departure Delay in Minutes": 0,
                "Arrival Delay in Minutes": 0,
                "Baggage Handling": 1,
                "Inflight Wifi Service": 0,
                "Online Support": 1,
                "Checkin Service": 0,
                "Cleanliness": 2,
            },
        },
        {
            "descricao": "Cliente de viagem pessoal em voo longo na classe Business com suporte online excelente, Wi-Fi bom, serviço de bordo ruim e sem atrasos",
            "evidencia": {
                "Type of Travel": 1,
                "Customer Type": 0,
                "Class": 0,
                "Seat comfort": 3,
                "Flight Distance": 2,
                "Inflight Entertainment": 3,
                "On-board Service": 0,
                "Food and Drink": 2,
                "Departure Delay in Minutes": 0,
                "Arrival Delay in Minutes": 0,
                "Baggage Handling": 2,
                "Inflight Wifi Service": 3,
                "Online Support": 2,
                "Checkin Service": 2,
                "Cleanliness": 2,
            },
        },
        {
            "descricao": "Cliente idoso em classe econômica, voo curto com atraso médio de partida, conforto de assento ruim e serviço de bordo médio",
            "evidencia": {
                "Age": 2,
                "Customer Type": 0,
                "Class": 1,
                "Seat comfort": 0,
                "Flight Distance": 0,
                "Inflight Entertainment": 1,
                "On-board Service": 1,
                "Food and Drink": 1,
                "Departure Delay in Minutes": 1,
                "Arrival Delay in Minutes": 0,
                "Baggage Handling": 1,
                "Inflight Wifi Service": 1,
                "Online Support": 1,
                "Checkin Service": 1,
                "Cleanliness": 1,
            },
        },
        {
            "descricao": "Frequent flyer em viagem de longa distância na classe Business, com excelente conforto, Wi-Fi médio, e todos os outros serviços de bordo no nível bom",
            "evidencia": {
                "Customer Type": 1,
                "Type of Travel": 1,
                "Class": 0,
                "Seat comfort": 4,
                "Flight Distance": 2,
                "Inflight Entertainment": 3,
                "On-board Service": 2,
                "Food and Drink": 2,
                "Departure Delay in Minutes": 0,
                "Arrival Delay in Minutes": 0,
                "Baggage Handling": 2,
                "Inflight Wifi Service": 2,
                "Online Support": 2,
                "Checkin Service": 2,
                "Cleanliness": 2,
            },
        },
        {
            "descricao": "Cliente jovem em classe Business, voo médio com conforto excelente, suporte online bom, e check-in ruim, mas limpeza excelente",
            "evidencia": {
                "Age": 0,
                "Class": 0,
                "Seat comfort": 4,
                "Flight Distance": 1,
                "Inflight Entertainment": 2,
                "On-board Service": 2,
                "Food and Drink": 1,
                "Departure Delay in Minutes": 0,
                "Arrival Delay in Minutes": 1,
                "Baggage Handling": 1,
                "Inflight Wifi Service": 2,
                "Online Support": 1,
                "Checkin Service": 0,
                "Cleanliness": 2,
            },
        },
    ]

    for teste in testes:
        evidencias_validas = validar_evidencias(teste["evidencia"])

        if not evidencias_validas:
            print(f"Teste ignorado: {teste['descricao']} (sem evidências válidas)")
            continue

        try:
            query_result = inference.query(
                variables=["satisfaction"], evidence=evidencias_validas
            )
            print(f"Teste: {teste['descricao']}")
            print("Evidências:", evidencias_validas)
            print("Resultado da probabilidade de satisfação:\n", query_result)
            print("-" * 50)
        except Exception as e:
            print(f"Erro ao realizar o teste '{teste['descricao']}': {e}")


realizar_testes_satisfacao()
