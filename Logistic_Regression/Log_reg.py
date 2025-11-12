import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================
data = pd.read_csv("diabetes_dataset.csv")

# Encoding
data["education_level_encoded"] = OrdinalEncoder().fit_transform(data[["education_level"]])
data["smoking_status_encoded"] = OrdinalEncoder().fit_transform(data[["smoking_status"]])

onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = onehot.fit_transform(data[["gender", "ethnicity", "employment_status"]])
encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(["gender", "ethnicity", "employment_status"]))
data = pd.concat([data.drop(["gender", "ethnicity", "employment_status"], axis=1), encoded_df], axis=1)

# Filtering
data = data[~data["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
data = data.dropna(subset=["hba1c"])
data["hba1c_mmolmol"] = 10.93 * data["hba1c"] - 23.5
data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)

# Features
X_home = data[["age", "bmi", "waist_to_hip_ratio", "diet_score",
               "physical_activity_minutes_per_week", "sleep_hours_per_day",
               "smoking_status_encoded", "alcohol_consumption_per_week",
               "family_history_diabetes"]]
X_clinical = data[["glucose_fasting", "insulin_level", "heart_rate"]]
y = data["hba1c_class"]

# ============================================================
# 2. MODEL EVALUATION FUNCTION
# ============================================================
def evaluate_model(X, y, label, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = LogisticRegression(**params, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_score = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    return {"Model": label, "Accuracy": acc, "Precision": prec, "Recall": rec, "ROC-AUC": roc_auc, "fpr": fpr, "tpr": tpr}

# ============================================================
# 3. RUN MODELS
# ============================================================
home_params = {'C': 1, 'class_weight': None, 'max_iter': 500, 'penalty': 'l2', 'solver': 'saga'}
clinical_params = {'C': 100, 'class_weight': 'balanced', 'max_iter': 500, 'penalty': 'l2', 'solver': 'lbfgs'}

res_home = evaluate_model(X_home, y, "Home Data", home_params)
res_clinical = evaluate_model(X_clinical, y, "Clinical Data", clinical_params)

# ============================================================
# 4. RESULTS TABLE
# ============================================================
results = pd.DataFrame([res_home, res_clinical])[["Model", "Accuracy", "Precision", "Recall", "ROC-AUC"]]
results[["Accuracy", "Precision", "Recall", "ROC-AUC"]] = results[["Accuracy", "Precision", "Recall", "ROC-AUC"]].applymap(lambda x: f"{x:.3f}")
print("\n" + "="*70)
print("FINAL EVALUATION – LOGISTIC REGRESSION")
print("="*70)
print(results.to_string(index=False))
print("="*70 + "\n")

# ============================================================
# 5. ROC CURVE PLOT
# ============================================================
plt.figure(figsize=(6, 4))
plt.plot(res_home["fpr"], res_home["tpr"], label=f'Home Data (AUC = {res_home["ROC-AUC"]:.3f})', linewidth=2)
plt.plot(res_clinical["fpr"], res_clinical["tpr"], label=f'Clinical Data (AUC = {res_clinical["ROC-AUC"]:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=300)
plt.show()

print("Saved: roc_comparison.png")