import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("visualizacoes", exist_ok=True)

import numpy as np
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
from sklearn.tree import DecisionTreeClassifier

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

def map_3_for_baggage(val):
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

cols_3states = ["Seat comfort","Inflight entertainment","On-board service"]
for c in cols_3states:
    if c in data.columns:
        data[c] = data[c].apply(map_3states)

if "Baggage handling" in data.columns:
    data["Baggage handling"] = data["Baggage handling"].apply(map_3_for_baggage)

data.dropna(inplace=True)
print(f"Shape após dropna: {data.shape}")

for col in data.columns:
    data[col] = data[col].astype("category")

for col in data.columns:
    data[col] = data[col].cat.codes

print("Shape final (após codes):", data.shape)

param_grid = [
    {"criterion": ["gini","entropy"], "max_depth": [None,5,10,20]},
]

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

X = data.drop(columns=["satisfaction"])
y = data["satisfaction"].values

results = []
all_fold_metrics = {}

for config in param_grid:
    for criterion in config["criterion"]:
        for max_d in config["max_depth"]:
            label_cfg = f"Tree(crit={criterion},maxD={max_d})"
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            fold_aucs = []
            fold_ytrue = []
            fold_yprob = []
            cms = []

            fold_idx = 0
            for train_idx, test_idx in kf.split(X):
                train_data = X.iloc[train_idx]
                train_labels= y[train_idx]
                test_data  = X.iloc[test_idx]
                test_labels= y[test_idx]

                dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_d, random_state=42)
                dt.fit(train_data, train_labels)

                y_pred = dt.predict(test_data)
                if hasattr(dt, "predict_proba"):
                    y_prob = dt.predict_proba(test_data)[:,1]
                else:
                    y_prob = np.zeros_like(y_pred, dtype=float)

                prec = precision_score(test_labels, y_pred, average="binary", zero_division=0)
                rec  = recall_score(test_labels, y_pred, average="binary", zero_division=0)
                f1   = f1_score(test_labels, y_pred, average="binary", zero_division=0)
                try:
                    auc = roc_auc_score(test_labels, y_prob)
                except Exception:
                    auc = 0.0

                fold_precisions.append(prec)
                fold_recalls.append(rec)
                fold_f1s.append(f1)
                fold_aucs.append(auc)
                fold_ytrue.append(test_labels)
                fold_yprob.append(y_prob)

                cm = confusion_matrix(test_labels, y_pred)
                cms.append(cm)

                fold_idx += 1

            avg_prec = np.mean(fold_precisions)
            avg_rec  = np.mean(fold_recalls)
            avg_f1   = np.mean(fold_f1s)
            avg_auc  = np.mean(fold_aucs)

            std_prec = np.std(fold_precisions)
            std_rec  = np.std(fold_recalls)
            std_f1   = np.std(fold_f1s)
            std_auc  = np.std(fold_aucs)

            results.append({
                "config_label": label_cfg,
                "criterion": criterion,
                "max_depth": max_d,
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

            all_fold_metrics[label_cfg] = {
                "precisions": fold_precisions,
                "recalls": fold_recalls,
                "f1s": fold_f1s,
                "aucs": fold_aucs,
            }

print("\n========== RESULTADOS DO MODELO 3 (Árvore de Decisão) ===========")
best_f1 = -1
best_cfg = None

for res in results:
    cfg_label = res["config_label"]
    (pr_mean, pr_std) = res["precision"]
    (rc_mean, rc_std) = res["recall"]
    (f1_mean, f1_std) = res["f1"]
    (auc_mean, auc_std) = res["auc"]

    print(f"\nConfig: {cfg_label}")
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

    plt.savefig(f"visualizacoes/val_cruzada_{cfg_label}_modelo3.png")
    plt.close()

if best_cfg is not None:
    print("\nMelhor config do Modelo 3 (F1 médio):", best_cfg["config_label"])
    cm_to_plot = best_cfg["cms"][0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_to_plot)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão (fold 0) - {best_cfg['config_label']}")
    plt.savefig("visualizacoes/confusao_melhor_config_modelo3.png")
    plt.close()

unique_cfg_labels = list(set([r["config_label"] for r in results]))
f1_means = []
auc_means = []

for lbl in unique_cfg_labels:
    entry = [r for r in results if r["config_label"] == lbl][0]
    f1_means.append(entry["f1"][0])
    auc_means.append(entry["auc"][0])

x_pos = np.arange(len(unique_cfg_labels))

plt.figure(figsize=(8,6))
plt.bar(x_pos, f1_means, align='center', alpha=0.7, label='F1')
plt.xticks(x_pos, unique_cfg_labels, rotation=45)
plt.ylabel('F1 (médio)')
plt.title('Comparação de F1 (Modelo 3 - Decision Tree)')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("visualizacoes/comparacao_f1_modelo3.png")
plt.close()

plt.figure(figsize=(8,6))
plt.bar(x_pos, auc_means, align='center', alpha=0.7, color='orange', label='AUC')
plt.xticks(x_pos, unique_cfg_labels, rotation=45)
plt.ylabel('AUC (médio)')
plt.title('Comparação de AUC (Modelo 3 - Decision Tree)')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("visualizacoes/comparacao_auc_modelo3.png")
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
        plt.plot(fpr, tpr, alpha=0.3,
                 label=f"Fold {i} (AUC={roc_auc_score(y_true_i,y_prob_i):.3f})")

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = roc_auc_score(np.concatenate(fold_ytrue),
                             np.concatenate(fold_yprob))
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label=f"Mean ROC (AUC={mean_auc:.3f})", linewidth=2)

    plt.title(f"Curvas ROC (Modelo 3) - {best_cfg['config_label']}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.savefig("visualizacoes/roc_melhor_config_modelo3.png")
    plt.close()

print("\nImagens do Modelo 3 (Decision Tree) salvas em 'visualizacoes'!\nFIM do script.")
