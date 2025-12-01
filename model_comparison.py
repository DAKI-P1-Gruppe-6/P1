"""
Model Comparison Script
Kører alle modellerne (XGB, KNN, DT) og sammenligner deres resultater.
"""
import pandas as pd
import numpy as np
import importlib.util
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

# Models
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# IMPORT DATA FROM Data procesing.py
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_path = os.path.join(script_dir, "Data procesing.py")
spec = importlib.util.spec_from_file_location("data_processing", data_processing_path)
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)

X_home = data_processing.X_home
X_clinical = data_processing.X_clinical
y = data_processing.y

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, X, y, label, scaler_type='standard'):
    """
    Evaluér en model og returner metrics.
    
    Parameters:
    - model: sklearn model instance
    - X: features
    - y: target
    - label: model name
    - scaler_type: 'standard' or 'minmax'
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scaling
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Combined score: (Recall + AUC) / 2
    combined_score = (rec + roc_auc) / 2
    
    return {
        'Model': label,
        'Accuracy': acc,
        'Recall': rec,
        'AUC': roc_auc,
        '(Recall+AUC)/2': combined_score
    }

# ============================================================
# DEFINE MODELS WITH FIXED PARAMETERS
# ============================================================

# KNN - find best k
def find_best_knn(X, y):
    best_k = 95
    return best_k

# ============================================================
# RUN ALL MODELS
# ============================================================
print("\n" + "="*80)
print("MODEL COMPARISON - ALL MODELS")
print("="*80)

results = []

# Test on X_home (home data)
print("\n" + "="*80)
print("TESTING ON HOME DATA (X_home)")
print("="*80)

# XGBoost
print("\n1. XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=50,
    gamma=0.5,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
result = evaluate_model(xgb_model, X_home, y, "XGBoost (Home)")
results.append(result)

# KNN
print("\n2. K-Nearest Neighbors...")
best_k = find_best_knn(X_home, y)
print(f"Best k: {best_k}")
knn_model = KNeighborsClassifier(n_neighbors=best_k)
result = evaluate_model(knn_model, X_home, y, "KNN (Home)", scaler_type='minmax')
results.append(result)

# Decision Tree
print("\n3. Decision Tree...")
dt_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=36,
    max_leaf_nodes=34,
    min_samples_leaf=13,
    min_impurity_decrease=1.1876178794228976e-05,
    min_samples_split=14,
    random_state=42
)
result = evaluate_model(dt_model, X_home, y, "Decision Tree (Home)")
results.append(result)

# ============================================================
# CREATE RESULTS TABLE
# ============================================================
print("\n" + "="*80)
print("RESULTS SUMMARY - HOME DATA")
print("="*80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('(Recall+AUC)/2', ascending=False)

print("\n" + df_results.to_string(index=False))

print("\n" + "="*80)
print("BEST MODEL (by (Recall+AUC)/2):", df_results.iloc[0]['Model'])
print("="*80)
print(f"Accuracy: {df_results.iloc[0]['Accuracy']:.4f}")
print(f"Recall:   {df_results.iloc[0]['Recall']:.4f}")
print(f"AUC:      {df_results.iloc[0]['AUC']:.4f}")
print(f"(Recall+AUC)/2: {df_results.iloc[0]['(Recall+AUC)/2']:.4f}")

