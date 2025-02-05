import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("visualizacoes", exist_ok=True)

from pgmpy.estimators import (
    PC,
    BayesianEstimator,
    BDeuScore,
    BicScore,
    HillClimbSearch,
    K2Score,
    MmhcEstimator,
)
from pgmpy.models import BayesianNetwork
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold

use_cols = [
    "satisfaction",
    "Type of Travel",
    "Class",
    "Seat comfort",
    "Age",
    "Flight Distance",
    "Inflight entertainment",
    "On-board service",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
    "Baggage handling",
]

def map_3states(val):
    # 0 ou 1 => 0 (Bad)
    # 2 ou 3 => 1 (Average)
    # 4 ou 5 => 2 (Good)
    if val <= 1:
        return 0
    elif val <= 3:
        return 1
    else:
        return 2

def map_3_baggage(val):
    
    if val <= 2:
        return 0
    elif val == 3:
        return 1
    else:
        return 2

data = pd.read_csv("Invistico_Airline.csv", usecols=use_cols)
data = data.sample(n=50000, random_state=42)

data["Age"] = pd.cut(data["Age"], bins=[0,25,60,999999], labels=["0","1","2"])

data["Flight Distance"] = pd.cut(data["Flight Distance"], bins=[0,999,1999,9999999], labels=["0","1","2"])

data["Departure Delay in Minutes"] = pd.cut(
    data["Departure Delay in Minutes"], bins=[-1,5,30,9999999], labels=["0","1","2"]
)
data["Arrival Delay in Minutes"] = pd.cut(
    data["Arrival Delay in Minutes"], bins=[-1,5,30,9999999], labels=["0","1","2"]
)

cols_3states = [
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
]
for c in cols_3states:
    if c in data.columns:
        data[c] = data[c].apply(map_3states)

if "Baggage handling" in data.columns:
    data["Baggage handling"] = data["Baggage handling"].apply(map_3_baggage)

data.dropna(inplace=True)
print(f"Shape após dropna: {data.shape}")

for col in data.columns:
    data[col] = data[col].astype("category")

print("\nCATEGORIES GLOBAIS (APÓS DISCRETIZAÇÃO)\n")
for col in data.columns:
    print(col, "=>", data[col].cat.categories.tolist())
print()

for col in data.columns:
    data[col] = data[col].cat.codes

print("Shape final do dataset:", data.shape)

def learn_structure(data, method="hc", scoring="k2"):
    if scoring == "k2":
        scoring_method = K2Score(data)
    elif scoring == "bic":
        scoring_method = BicScore(data)
    elif scoring == "bdeu":
        scoring_method = BDeuScore(data)
    else:
        scoring_method = K2Score(data)

    if method == "hc":
        est = HillClimbSearch(data)
        best_struct = est.estimate(scoring_method=scoring_method)
    elif method == "mmhc":
        mmhc = MmhcEstimator(data)
        best_struct = mmhc.estimate(scoring_method=scoring_method)
    elif method == "pc":
        pc_estimator = PC(data)
        best_struct = pc_estimator.estimate()
    else:
        est = HillClimbSearch(data)
        best_struct = est.estimate(scoring_method=scoring_method)

    return best_struct

def build_bn(best_struct, data, prior_type="BDeu"):
    if isinstance(best_struct, BayesianNetwork):
        bn_model = best_struct
    else:
        bn_model = BayesianNetwork()
        bn_model.add_nodes_from(best_struct.nodes())
        bn_model.add_edges_from(best_struct.edges())
    bn_model.fit(data, estimator=BayesianEstimator, prior_type=prior_type)
    return bn_model

def predict_bn(model, test_data, target="satisfaction"):
    test_data_no_target = test_data.drop(columns=[target])
    y_pred = model.predict(test_data_no_target)[target].values

    y_prob_df = model.predict_probability(test_data_no_target)
    prob_col = f"{target}_1"
    if prob_col not in y_prob_df.columns:
        prob_col = y_prob_df.columns[-1]
    y_prob = y_prob_df[prob_col].values
    return y_pred, y_prob

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

param_grid = [
    {"method": "hc",   "scoring": "k2",  "prior_type": "BDeu"},
    {"method": "hc",   "scoring": "bic", "prior_type": "BDeu"},
    {"method": "mmhc", "scoring": "k2",  "prior_type": "BDeu"},
]

results = []
all_fold_metrics = {}

for combo in param_grid:
    method_ = combo["method"]
    scoring_ = combo["scoring"]
    prior_   = combo["prior_type"]

    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_aucs = []
    fold_ytrue = []
    fold_yprob = []

    cms = []

    for fold_index, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Config={method_}-{scoring_}, fold={fold_index}, teste={len(test_idx)}")
        train_data = data.iloc[train_idx].copy()
        test_data  = data.iloc[test_idx].copy()

        best_struct = learn_structure(train_data, method=method_, scoring=scoring_)
        bn_model = build_bn(best_struct, train_data, prior_type=prior_)

        y_true = test_data["satisfaction"].values
        y_pred, y_prob = predict_bn(bn_model, test_data, "satisfaction")

        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.0

        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_aucs.append(auc)

        fold_ytrue.append(y_true)
        fold_yprob.append(y_prob)

        cm = confusion_matrix(y_true, y_pred)
        cms.append(cm)

    avg_prec = np.mean(fold_precisions)
    avg_rec  = np.mean(fold_recalls)
    avg_f1   = np.mean(fold_f1s)
    avg_auc  = np.mean(fold_aucs)

    std_prec = np.std(fold_precisions)
    std_rec  = np.std(fold_recalls)
    std_f1   = np.std(fold_f1s)
    std_auc  = np.std(fold_aucs)

    results.append({
        "method": method_,
        "scoring": scoring_,
        "prior": prior_,
        "precision": (avg_prec, std_prec),
        "recall": (avg_rec, std_rec),
        "f1": (avg_f1, std_f1),
        "auc": (avg_auc, std_auc),
        "cms": cms,
        "fold_precisions": fold_precisions,
        "fold_recalls": fold_recalls,
        "fold_f1s": fold_f1s,
        "fold_aucs": fold_aucs,
        "fold_ytrue": fold_ytrue,
        "fold_yprob": fold_yprob,
    })

    config_label = f"{method_}-{scoring_}-{prior_}"
    all_fold_metrics[config_label] = {
        "precisions": fold_precisions,
        "recalls": fold_recalls,
        "f1s": fold_f1s,
        "aucs": fold_aucs,
    }

print("\n========== RESULTADOS DA GRID SEARCH (categorias unificadas antes do KFold) ===========")
best_f1 = -1
best_cfg = None

for res in results:
    m, s, p = res["method"], res["scoring"], res["prior"]
    (pr_mean, pr_std) = res["precision"]
    (rc_mean, rc_std) = res["recall"]
    (f1_mean, f1_std) = res["f1"]
    (auc_mean, auc_std) = res["auc"]

    print(f"\nMétodo: {m} | Scoring: {s} | Prior: {p}")
    print(f" Precision: {pr_mean:.3f} +/- {pr_std:.3f}")
    print(f" Recall:    {rc_mean:.3f} +/- {rc_std:.3f}")
    print(f" F1:        {f1_mean:.3f} +/- {f1_std:.3f}")
    print(f" AUC:       {auc_mean:.3f} +/- {auc_std:.3f}")

    if f1_mean > best_f1:
        best_f1 = f1_mean
        best_cfg = res

for cfg_label, fold_data in all_fold_metrics.items():
    folds = range(1, len(fold_data["precisions"]) + 1)

    plt.figure(figsize=(8,6))
    plt.plot(folds, fold_data["precisions"], marker='o', label='Precision')
    plt.plot(folds, fold_data["recalls"],    marker='s', label='Recall')
    plt.plot(folds, fold_data["f1s"],        marker='^', label='F1')
    plt.plot(folds, fold_data["aucs"],       marker='x', label='AUC')
    plt.title(f"Validação Cruzada (scores por fold) - {cfg_label}")
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)

    plt.savefig(f"visualizacoes/val_cruzada_{cfg_label}.png")
    plt.close()

if best_cfg is not None:
    print("\nMelhor configuração (F1 médio):", best_cfg)

    cm_to_plot = best_cfg["cms"][0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_to_plot)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão (um dos folds) - Melhor Config")
    plt.savefig("visualizacoes/confusao_melhor_config.png")
    plt.close()

cfg_labels = []
f1_means = []
auc_means = []

for res in results:
    label = f"{res['method']}-{res['scoring']}"
    f1_means.append(res['f1'][0])
    auc_means.append(res['auc'][0])
    cfg_labels.append(label)

x_pos = np.arange(len(cfg_labels))

plt.figure(figsize=(8,6))
plt.bar(x_pos, f1_means, align='center', alpha=0.7, label='F1')
plt.xticks(x_pos, cfg_labels, rotation=45)
plt.ylabel('F1 (médio)')
plt.title('Comparação de F1 por Configuração')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("visualizacoes/comparacao_f1_por_config.png")
plt.close()

plt.figure(figsize=(8,6))
plt.bar(x_pos, auc_means, align='center', alpha=0.7, color='orange', label='AUC')
plt.xticks(x_pos, cfg_labels, rotation=45)
plt.ylabel('AUC (médio)')
plt.title('Comparação de AUC por Configuração')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("visualizacoes/comparacao_auc_por_config.png")
plt.close()

if best_cfg is not None:
    fold_ytrue = best_cfg["fold_ytrue"]
    fold_yprob = best_cfg["fold_yprob"]
    Kfolds = len(fold_ytrue)

    plt.figure(figsize=(8,6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []

    for i in range(Kfolds):
        y_true_i = fold_ytrue[i]
        y_prob_i = fold_yprob[i]
        fpr, tpr, thresholds = roc_curve(y_true_i, y_prob_i, pos_label=1)
        plt.plot(fpr, tpr, alpha=0.3, label=f"Fold {i} (AUC={roc_auc_score(y_true_i,y_prob_i):.3f})")

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = roc_auc_score(np.concatenate(fold_ytrue), np.concatenate(fold_yprob))
    plt.plot(mean_fpr, mean_tpr, 'k--', label=f"Mean ROC (AUC={mean_auc:.3f})", linewidth=2)

    plt.title("Curvas ROC (melhor configuração)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig("visualizacoes/roc_melhor_config.png")
    plt.close()

print("\nTodas as imagens foram salvas na pasta 'visualizacoes'!.\n")
