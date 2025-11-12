import matplotlib
matplotlib.use("TkAgg")  # sikrer at plots kan vises i et normalt desktop-miljø

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression # kan udskiftes

# 1. Dataindlæsning og encoding

diabetes_data = pd.read_csv("diabetes_dataset.csv")

# Ordinal encoding
diabetes_data["education_level_encoded"] = OrdinalEncoder().fit_transform(
    diabetes_data[["education_level"]]
)
diabetes_data["smoking_status_encoded"] = OrdinalEncoder().fit_transform(
    diabetes_data[["smoking_status"]]
)

# One-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ethnicity_one_hot = onehot_encoder.fit_transform(
    diabetes_data[["gender", "ethnicity", "employment_status"]]
)
ethnicity_one_hot_df = pd.DataFrame(
    ethnicity_one_hot,
    columns=onehot_encoder.get_feature_names_out(["gender", "ethnicity", "employment_status"]),
)
data_encoded = pd.concat(
    [diabetes_data.drop(["gender", "ethnicity", "employment_status"], axis=1), ethnicity_one_hot_df],
    axis=1,
)

# 2. Filtrering og HbA1c-konvertering

filtered_data = data_encoded[~data_encoded["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data = filtered_data.dropna(subset=["hba1c"])
filtered_data["hba1c_mmolmol"] = 10.93 * filtered_data["hba1c"] - 23.5

# 3. Binary klassifikation: Type 2 Diabetes (>=48) vs No Diabetes (<48)
filtered_data["hba1c_class"] = (filtered_data["hba1c_mmolmol"] >= 48).astype(int)
# Class 0: No Diabetes (< 48 mmol/mol)
# Class 1: Type 2 Diabetes (>= 48 mmol/mol)

# -------------------------------------------------------------
# 4. Feature sets
# -------------------------------------------------------------
X_home = filtered_data[
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
X_clinical = filtered_data[["glucose_fasting", "insulin_level", "heart_rate"]]
y = filtered_data["hba1c_class"]

# -------------------------------------------------------------
# 5. GridSearch for optimal hyperparameters
# -------------------------------------------------------------

# Dictionary to store results for comparison
results_comparison = []

def evaluate_model(X, y, label):
    print(f"\n{'='*70}")
    print(f"GRID SEARCH - {label}")
    print(f"{'='*70}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Skalering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- GridSearchCV for optimal parameters ---
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [500, 1000],
        'class_weight': ['balanced', None]
    }
    
    logistic_model = LogisticRegression(random_state=42)
    
    grid_search = GridSearchCV(
        logistic_model,
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=1,  # Changed from -1 to avoid Python 3.13 multiprocessing bug
        verbose=1
    )
    
    print(f"Fitting GridSearchCV on {label}...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Use best model
    model = grid_search.best_estimator_

    # Forudsigelser
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # Klassiske metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC-AUC for binary classification
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    # Udskriv resultater
    print(f"\n{'='*70}")
    print(f"EVALUATION - {label}")
    print(f"{'='*70}\n")
    print(f"Best Model Parameters: {model.get_params()}\n")
    print(f"Accuracy:             {acc:.4f}")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Store results
    results_comparison.append({
        'Model': model.__class__.__name__,
        'Data': 'Hjemme' if 'Hjemme' in label else 'Kliniske',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'ROC-AUC': roc_auc
    })


# -------------------------------------------------------------
# 7. Kør modeller
# -------------------------------------------------------------
evaluate_model(X_home, y, "Hjemme-data")
evaluate_model(X_clinical, y, "Kliniske data")


# -------------------------------------------------------------
# Comparison Table
# -------------------------------------------------------------
print("\n" + "="*70)
print("COMPARISON - LOGISTIC REGRESSION")
print("="*70)

comparison_df = pd.DataFrame(results_comparison)
print("\n", comparison_df.to_string(index=False))
print("\n")

# -------------------------------------------------------------
# 8. VISUEL FORKLARING – figurer til rapporten
# -------------------------------------------------------------

# --- Figur 1: HbA1c klassifikationsgrænser (Binary) ---
plt.figure(figsize=(8, 2))
plt.axvspan(0, 48, color="#7cd992", alpha=0.9, label="No Diabetes (<48 mmol/mol)")
plt.axvspan(48, 60, color="#f46a6a", alpha=0.9, label="Type 2 Diabetes (≥48 mmol/mol)")

plt.xlim(0, 60)
plt.xlabel("HbA1c (mmol/mol)")
plt.yticks([])
plt.title("Binary HbA1c Classification Threshold")
plt.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.4))
plt.tight_layout()
plt.show()
