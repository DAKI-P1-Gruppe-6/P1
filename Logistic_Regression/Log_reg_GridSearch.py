import matplotlib
matplotlib.use("TkAgg")  # sikrer at plots kan vises i et normalt desktop-miljø

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
import os
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
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression # kan udskiftes

# ============================================================
# IMPORT DATA FROM Data procesing.py
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_path = os.path.join(script_dir, "..", "Data procesing.py")
spec = importlib.util.spec_from_file_location("data_processing", data_processing_path)
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)

X_home = data_processing.X_home
X_clinical = data_processing.X_clinical
y = data_processing.y

# -------------------------------------------------------------
# 5. GridSearch for optimal hyperparameters
# -------------------------------------------------------------

# Dictionary to store results for comparison
results_comparison = []

def evaluate_model(X, y, label, n_iter=100):
    print(f"\n{'='*70}")
    print(f"RANDOMIZED SEARCH - {label}")
    print(f"{'='*70}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Skalering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- RandomizedSearchCV for optimal parameters ---
    param_distributions = {
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'max_iter': [100, 200, 500, 1000, 2000, 5000],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0, 0.15, 0.25, 0.5, 0.75, 0.85, 1.0]  # For elasticnet
    }
    
    logistic_model = LogisticRegression(random_state=42)
    
    random_search = RandomizedSearchCV(
        logistic_model,
        param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    print(f"Fitting RandomizedSearchCV on {label}...")
    print(f"Testing {n_iter} random combinations with 3-fold CV...")
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ RandomizedSearchCV completed!")
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {random_search.best_score_:.4f}")
    
    # Use best model
    model = random_search.best_estimator_

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
    print(f"TEST SET EVALUATION - {label}")
    print(f"{'='*70}\n")
    print(f"Best Model Parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print()
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
N_ITERATIONS = 100  # Adjust this value
evaluate_model(X_home, y, "Hjemme-data", n_iter=N_ITERATIONS)


# -------------------------------------------------------------
# Results Summary
# -------------------------------------------------------------
print("\n" + "="*70)
print("LOGISTIC REGRESSION RANDOMIZED SEARCH RESULTS")
print("="*70)

if results_comparison:
    comparison_df = pd.DataFrame(results_comparison)
    print("\n", comparison_df.to_string(index=False))
    print("\n")
    print("✅ Model optimized and ready!")
    print("⚡ RandomizedSearch completed efficiently!")
else:
    print("No results to display.")
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
