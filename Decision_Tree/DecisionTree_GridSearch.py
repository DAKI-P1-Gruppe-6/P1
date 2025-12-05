import pandas as pd
import numpy as np
import importlib.util
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
# RANDOMIZED SEARCH FUNCTION
# ============================================================
def evaluate_model(X, y, label, n_iter=100):
    print(f"\n{'='*70}")
    print(f"RANDOMIZED SEARCH - {label}")
    print(f"{'='*70}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- RandomizedSearchCV for optimal parameters ---
    param_distributions = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, None],
        'min_samples_split': [2, 5, 10, 15, 20, 30, 50],
        'min_samples_leaf': [1, 2, 4, 7, 10, 15, 20],
        'max_features': [None, 'sqrt', 'log2', 0.5, 0.7, 0.9],
        'max_leaf_nodes': [None, 10, 20, 30, 40, 50, 75, 100],
        'min_impurity_decrease': [0.0, 0.00001, 0.0001, 0.001, 0.01],
        'class_weight': [None, 'balanced']
    }
    
    dt_model = DecisionTreeClassifier(random_state=42)
    
    random_search = RandomizedSearchCV(
        dt_model,
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
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ RandomizedSearchCV completed!")
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
    
    # Use best model
    model = random_search.best_estimator_

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
    print(f"TEST SET EVALUATION - {label}")
    print(f"{'='*70}\n")
    print(f"Best Model Parameters:")
    for param, value in random_search.best_params_.items():
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
        'Best_Params': random_search.best_params_,
        'CV_Score': random_search.best_score_,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'ROC-AUC': roc_auc,
        '(Recall+AUC)/2': combined_score
    }

# ============================================================
# RUN RANDOMIZED SEARCH
# ============================================================
print("\n" + "🔍 Randomized Search on Home Data")
N_ITERATIONS = 100  # Adjust this value

# Test on home data only
result_home = evaluate_model(X_home, y, "Hjemme-data", n_iter=N_ITERATIONS)

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "="*70)
print("DECISION TREE RANDOMIZED SEARCH RESULTS")
print("="*70)

print(f"\nData: {result_home['Data']}")
print(f"CV Score:         {result_home['CV_Score']:.4f}")
print(f"Accuracy:         {result_home['Accuracy']:.4f}")
print(f"Recall:           {result_home['Recall']:.4f}")
print(f"ROC-AUC:          {result_home['ROC-AUC']:.4f}")
print(f"(Recall+AUC)/2:   {result_home['(Recall+AUC)/2']:.4f}")

# Print best parameters
print("\n" + "="*70)
print("BEST PARAMETERS FOUND")
print("="*70)
for param, value in result_home['Best_Params'].items():
    print(f"  {param}: {value}")

print("\n" + "="*70)
print("✅ Model optimized and ready!")
print("⚡ RandomizedSearch completed efficiently!")
print("="*70)

