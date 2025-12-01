import pandas as pd
import numpy as np
import importlib.util
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier

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

# ============================================================
# GRID SEARCH FUNCTION
# ============================================================
def evaluate_model(X, y, label):
    print(f"\n{'='*70}")
    print(f"GRID SEARCH - {label}")
    print(f"{'='*70}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- GridSearchCV for optimal parameters ---
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [20, 30, 40, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [5, 10, 15],
        'max_leaf_nodes': [20, 30, 40, None],
        'min_impurity_decrease': [0.0, 1e-5, 1e-4],
        'max_features': ['sqrt', None]
    }
    
    dt_model = DecisionTreeClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        dt_model,
        param_grid,
        cv=2,  # Minimal cross-validation for faster execution
        scoring='roc_auc',
        n_jobs=1,  # Changed from -1 to avoid Python 3.13 multiprocessing bug
        verbose=1
    )
    
    print(f"Fitting GridSearchCV on {label}...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best training ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Use best model
    model = grid_search.best_estimator_

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Combined score: (Recall + AUC) / 2
    combined_score = (rec + roc_auc) / 2

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION - {label}")
    print(f"{'='*70}\n")
    print(f"Best Model Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nAccuracy:             {acc:.4f}")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"(Recall+AUC)/2:       {combined_score:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    return {
        'Model': 'Decision Tree',
        'Data': label,
        'Best_Params': grid_search.best_params_,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'ROC-AUC': roc_auc,
        '(Recall+AUC)/2': combined_score
    }

# ============================================================
# RUN GRID SEARCH
# ============================================================
results = []

# Test on home data
result_home = evaluate_model(X_home, y, "Hjemme-data")
results.append(result_home)

# Test on clinical data
result_clinical = evaluate_model(X_clinical, y, "Kliniske data")
results.append(result_clinical)

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n" + "="*70)
print("COMPARISON - DECISION TREE GRID SEARCH RESULTS")
print("="*70)

comparison_df = pd.DataFrame(results)
print("\n", comparison_df[['Data', 'Accuracy', 'Recall', 'ROC-AUC', '(Recall+AUC)/2']].to_string(index=False))
print("\n")

# Print best parameters for each dataset
print("\n" + "="*70)
print("BEST PARAMETERS SUMMARY")
print("="*70)
for result in results:
    print(f"\n{result['Data']}:")
    for param, value in result['Best_Params'].items():
        print(f"  {param}: {value}")

