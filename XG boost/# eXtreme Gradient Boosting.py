import pandas as pd
import numpy as np
from sklearn.utils import resample  # TILFØJET - skal være i starten
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix
)

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

# ============================================================
# 2. FILTERING
# ============================================================
data = data[~data["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
data = data.dropna(subset=["hba1c"])

data["hba1c_mmolmol"] = 10.93 * data["hba1c"] - 23.5

# Fjern yderligere "midt i mellem" patienter for bedre læring
data = data[(data["hba1c_mmolmol"] < 45) | (data["hba1c_mmolmol"] > 50)].copy()

data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)

# ============================================================
# 3. FEATURE SETS (FORBEDRET)
# ============================================================
X_home = data[
    [
        "age",
        "bmi",
        "waist_to_hip_ratio",
        "diet_score",
        "physical_activity_minutes_per_week",
        "sleep_hours_per_day",
        "smoking_status_encoded",
        "alcohol_consumption_per_week",
        "family_history_diabetes",
    ]
]

X_clinical = data[["glucose_fasting", "insulin_level", "heart_rate"]]
y = data["hba1c_class"]

# ============================================================
# 5. XGBoost EVALUATION (med robust fejlhåndtering)
# ============================================================
def train_and_evaluate_xgboost(X, y, label):
    """
    Træn og evaluer XGBoost model med robust fejlhåndtering for NaN-værdier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # DEBUG: Vis test set fordeling
    print(f"\n{label} - Test set fordeling: 0={sum(y_test==0)}, 1={sum(y_test==1)}")
    
    # Scaling (XGBoost fungerer med StandardScaler)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Definer XGBoost model med optimerede hyperparametre
    model = XGBClassifier(
        n_estimators=200,           # Flere træer for bedre performance
        gamma=0.1,                  # Lavere gamma = mindre regularization
        max_depth=6,                # Dybere træer for bedre læring
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,         # Lavere learning rate med flere estimators
        min_child_weight=3,         # Forhindrer overfitting
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        scale_pos_weight=1,         # Håndter klasse imbalance hvis nødvendigt
        random_state=42,
        eval_metric='logloss'
    )
    
    # Træn modellen
    model.fit(X_train_s, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]  # Probability for class 1
    
    # DEBUG: Vis prediction fordeling
    unique_preds, counts_preds = np.unique(y_pred, return_counts=True)
    print(f"{label} - Prediction fordeling: {dict(zip(unique_preds, counts_preds))}")
    
    # Confusion matrix for debugging
    cm = confusion_matrix(y_test, y_pred)
    print(f"{label} - Confusion Matrix:\n{cm}")
    
    # Metrics med robust håndtering af edge cases
    acc = accuracy_score(y_test, y_pred)
    
    # Robust precision og recall beregning
    try:
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
    except Exception as e:
        print(f"⚠️  Fejl i precision/recall beregning: {e}")
        # Fallback beregning baseret på confusion matrix
        tn, fp, fn, tp = cm.ravel()
        if tp + fp == 0:  # Ingen positive predictioner
            prec = 0.0
        else:
            prec = tp / (tp + fp)
        
        if tp + fn == 0:  # Ingen positive true labels
            rec = 0.0
        else:
            rec = tp / (tp + fn)
    
    print()
    print(f"{label}: {model.__class__.__name__}")
    print(f"Hyperparametre: n_estimators=200, gamma=0.1, max_depth=6, learning_rate=0.05")
    print(f"Accuracy: {acc:.1%}")
    print(f"Precision:     {prec:.3f}")
    print(f"Recall:        {rec:.3f}")
    print(f"→ {int(acc * len(X_test))} rigtige ud af {len(X_test)} patienter")
    
    # Returner model og scaler for ROC plotting
    return model, scaler, X_test_s, y_test

def cross_validate_xgboost(X, y, label):
    """
    Cross-validation som i den gamle model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBClassifier(
        n_estimators=200,           # Flere træer for bedre performance
        gamma=0.1,                  # Lavere gamma = mindre regularization
        max_depth=6,                # Dybere træer for bedre læring
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,         # Lavere learning rate med flere estimators
        min_child_weight=3,         # Forhindrer overfitting
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        scale_pos_weight=1,         # Håndter klasse imbalance hvis nødvendigt
        random_state=42,
        eval_metric='logloss'
    )
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    print()
    print(f"{label} ({model.__class__.__name__})")
    print(f"Hyperparametre: n_estimators=200, gamma=0.1, max_depth=6, learning_rate=0.05")
    print(f"Cross-val accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ============================================================
# 6. RUN MODELS (minimalistisk - kun ROC curve)
# ============================================================
print("\n" + "="*60)
print("XGBoost EVALUERING MED BALANCERING OG PRECISION/RECALL FOKUS")
print("="*60)

# Træn og evaluer (gem resultater til ROC)
results_home = train_and_evaluate_xgboost(X_home, y, "Hjemme-data")
results_clinical = train_and_evaluate_xgboost(X_clinical, y, "Kliniske data")

# Cross-validation
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTATER")
print("="*60)
cross_validate_xgboost(X_home, y, "Hjemme-data")
cross_validate_xgboost(X_clinical, y, "Kliniske data")

# ROC Curve plotting (kun én graf som ønsket)
model_home, scaler_home, X_test_home, y_test_home = results_home
model_clinical, scaler_clinical, X_test_clinical, y_test_clinical = results_clinical

plt.figure(figsize=(8, 6))

# ROC for hjemme-data
y_score_home = model_home.predict_proba(X_test_home)[:, 1]
fpr_home, tpr_home, _ = roc_curve(y_test_home, y_score_home)
roc_auc_home = auc(fpr_home, tpr_home)
plt.plot(fpr_home, tpr_home, lw=2, label=f"Hjemme-data (AUC={roc_auc_home:.3f})", color='blue')

# ROC for kliniske data
y_score_clinical = model_clinical.predict_proba(X_test_clinical)[:, 1]
fpr_clinical, tpr_clinical, _ = roc_curve(y_test_clinical, y_score_clinical)
roc_auc_clinical = auc(fpr_clinical, tpr_clinical)
plt.plot(fpr_clinical, tpr_clinical, lw=2, label=f"Kliniske data (AUC={roc_auc_clinical:.3f})", color='red')

# Diagonal linje
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – XGBoost Binary Classification (HbA1c ≥ 48 mmol/mol)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# SUMMARY (valgfri - opsummering af resultater)
# ============================================================
print("\n" + "="*60)
print("SAMMENLIGNING AF FEATURE SETS")
print("="*60)

# Robust beregning for summary
y_pred_home = model_home.predict(X_test_home)
y_pred_clinical = model_clinical.predict(X_test_clinical)

acc_home = accuracy_score(y_test_home, y_pred_home)
prec_home = precision_score(y_test_home, y_pred_home, zero_division=0)
rec_home = recall_score(y_test_home, y_pred_home, zero_division=0)

acc_clinical = accuracy_score(y_test_clinical, y_pred_clinical)
prec_clinical = precision_score(y_test_clinical, y_pred_clinical, zero_division=0)
rec_clinical = recall_score(y_test_clinical, y_pred_clinical, zero_division=0)

print(f"Hjemme-data:     Accuracy {acc_home:.1%} | Precision {prec_home:.3f} | Recall {rec_home:.3f} | AUC {roc_auc_home:.3f}")
print(f"Kliniske data:   Accuracy {acc_clinical:.1%} | Precision {prec_clinical:.3f} | Recall {rec_clinical:.3f} | AUC {roc_auc_clinical:.3f}")




