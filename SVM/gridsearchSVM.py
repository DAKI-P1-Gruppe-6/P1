import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.svm import SVC

# -------------------------------------------------------------
# 1. Dataindlæsning og encoding
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 2. Filtrering og HbA1c-konvertering
# -------------------------------------------------------------
filtered_data = data_encoded[~data_encoded["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data = filtered_data.dropna(subset=["hba1c"])
filtered_data["hba1c_mmolmol"] = 10.93 * filtered_data["hba1c"] - 23.5

# 3. Binær klassifikation
filtered_data["hba1c_class"] = (filtered_data["hba1c_mmolmol"] >= 48).astype(int)

# -------------------------------------------------------------
# 4. Feature sæt
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
# 5. GridSearch med 30% Subset 
# -------------------------------------------------------------

results_comparison = []

def evaluate_model(X, y, label):
    print(f"\n{'='*70}")
    print(f"GRID SEARCH SVM (30% Subset) - {label}")
    print(f"{'='*70}\n")
    
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 2. Skalering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. 30% Subset til GridSearch
    X_train_subset, _, y_train_subset, _ = train_test_split(
        X_train, y_train, 
        train_size=0.30,
        stratify=y_train, 
        random_state=42
    )
    
    print(f"Total training data: {len(X_train)} rækker")
    print(f"GridSearch subset (30%): {len(X_train_subset)} rækker")
    
    # Beregn antal kombinationer
    n_kernels = len(param_grid['kernel'])
    n_C = len(param_grid['C'])
    n_gamma = len(param_grid['gamma'])
    total_combinations = n_kernels * n_C * n_gamma
    total_fits = total_combinations * 2  # cv=2
    
    print(f"Parameter kombinationer: {total_combinations}")
    print(f"Total fits (med 2-fold CV): {total_fits}\n")

    # --- SVM Model Setup ---
    svm_base = SVC(
        max_iter=20000,  # Øget for bedre konvergens
        class_weight='balanced', 
        random_state=42, 
        probability=True
    )

    # --- Parameter Grid (optimeret for hastighed) ---
    param_grid = {
        'kernel': ['rbf', 'linear'],  # Fjernet 'poly' - er meget langsom
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1]  # 'scale' er ofte god default
    }
    
    # GridSearch på Subset med 2-fold CV (hurtigere)
    grid_search = GridSearchCV(
        svm_base,
        param_grid, 
        cv=2,  # Reduceret fra 5 til 2 for hastighed
        scoring='accuracy',
        n_jobs=1, 
        verbose=1
    )
    
    print(f"Starter GridSearch på subset...")
    grid_search.fit(X_train_subset, y_train_subset)
    
    print(f"\nBedste parametre fundet på subset: {grid_search.best_params_}")
    
    # 4. En ny model med de bedste parametre på X_train 
    best_params = grid_search.best_params_
    
    final_model = SVC(
        **best_params, 
        max_iter=20000,  # Øget for bedre konvergens
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    print("Gen-træner bedste model på fuldt træningssæt...")
    final_model.fit(X_train, y_train)

    # Evaluering på X_test
    y_pred = final_model.predict(X_test)
    y_score = final_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    # Udskriv resultater
    print(f"\n{'='*70}")
    print(f"EVALUERING - {label}")
    print(f"{'='*70}\n")
    print(f"Model Parametre: {final_model.get_params()}\n")
    print(f"Accuracy:             {acc:.4f}")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Gem til tabel
    results_comparison.append({
        'Model': 'SVM',
        'Data': 'Hjemme' if 'Hjemme' in label else 'Kliniske',
        'Best Kernel': best_params['kernel'],
        'Best C': best_params['C'],
        'Best Gamma': best_params.get('gamma', '-'),
        'Accuracy': acc,
        'ROC-AUC': roc_auc
    })

# -------------------------------------------------------------
# 6. Kørsel af modeller
# -------------------------------------------------------------
evaluate_model(X_home, y, "Hjemme-data")
evaluate_model(X_clinical, y, "Kliniske data")

# -------------------------------------------------------------
# Tabel til sammenligning
# -------------------------------------------------------------
print("\n" + "="*70)
print("SAMMENLIGNING - SVM RESULTATER")
print("="*70)
comparison_df = pd.DataFrame(results_comparison)
print("\n", comparison_df.to_string(index=False))
