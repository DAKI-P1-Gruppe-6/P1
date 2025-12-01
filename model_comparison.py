"""
Model Comparison Script
Kører alle modellerne (XGBoost, Logistic Regression, Decision Tree, KNN, Random Forest, SVM) 
og sammenligner deres resultater.
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

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
def evaluate_model(model, X, y, label, scaler_type='standard', use_proba=True):
    """
    Evaluér en model og returner metrics.
    
    Parameters:
    - model: sklearn model instance
    - X: features
    - y: target
    - label: model name
    - scaler_type: 'standard' or 'minmax'
    - use_proba: whether the model supports predict_proba (False for LinearSVC)
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
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC (only if model supports predict_proba)
    if use_proba:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
    else:
        # For LinearSVC, use decision_function
        y_score = model.decision_function(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
    
    # Combined score: (Recall + AUC) / 2
    combined_score = (rec + roc_auc) / 2
    
    return {
        'Model': label,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'AUC': roc_auc,
        '(Recall+AUC)/2': combined_score
    }

# ============================================================
# RUN ALL MODELS
# ============================================================
print("\n" + "="*80)
print("MODEL COMPARISON - ALL MODELS")
print("="*80)

results_home = []
results_clinical = []

# ============================================================
# TEST ON HOME DATA (X_home)
# ============================================================
print("\n" + "="*80)
print("TESTING ON HOME DATA (X_home)")
print("="*80)

# 1. XGBoost
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
result = evaluate_model(xgb_model, X_home, y, "XGBoost")
results_home.append(result)

# 2. Logistic Regression
print("\n2. Logistic Regression...")
lr_model = LogisticRegression(
    C=10,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
result = evaluate_model(lr_model, X_home, y, "Logistic Regression")
results_home.append(result)

# 3. Decision Tree
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
result = evaluate_model(dt_model, X_home, y, "Decision Tree")
results_home.append(result)

# 4. K-Nearest Neighbors
print("\n4. K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=95)
result = evaluate_model(knn_model, X_home, y, "KNN", scaler_type='minmax')
results_home.append(result)

# 5. Random Forest
print("\n5. Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
result = evaluate_model(rf_model, X_home, y, "Random Forest")
results_home.append(result)

# 6. Linear SVC
print("\n6. Linear SVC...")
lsvc_model = LinearSVC(
    random_state=42,
    max_iter=2000
)
result = evaluate_model(lsvc_model, X_home, y, "Linear SVC", use_proba=False)
results_home.append(result)

# 7. SVC (RBF Kernel)
print("\n7. SVC (RBF Kernel)...")
svc_model = SVC(
    kernel="rbf",
    random_state=42,
    max_iter=20000,
    probability=True  # Enable probability estimates for ROC-AUC
)
result = evaluate_model(svc_model, X_home, y, "SVC (RBF)")
results_home.append(result)

# ============================================================
# TEST ON CLINICAL DATA (X_clinical)
# ============================================================
print("\n" + "="*80)
print("TESTING ON CLINICAL DATA (X_clinical)")
print("="*80)

# 1. XGBoost
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
result = evaluate_model(xgb_model, X_clinical, y, "XGBoost")
results_clinical.append(result)

# 2. Logistic Regression
print("\n2. Logistic Regression...")
lr_model = LogisticRegression(
    C=10,
    penalty='l2',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
result = evaluate_model(lr_model, X_clinical, y, "Logistic Regression")
results_clinical.append(result)

# 3. Decision Tree
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
result = evaluate_model(dt_model, X_clinical, y, "Decision Tree")
results_clinical.append(result)

# 4. K-Nearest Neighbors
print("\n4. K-Nearest Neighbors...")
knn_model = KNeighborsClassifier(n_neighbors=95)
result = evaluate_model(knn_model, X_clinical, y, "KNN", scaler_type='minmax')
results_clinical.append(result)

# 5. Random Forest
print("\n5. Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
result = evaluate_model(rf_model, X_clinical, y, "Random Forest")
results_clinical.append(result)

# 6. Linear SVC
print("\n6. Linear SVC...")
lsvc_model = LinearSVC(
    random_state=42,
    max_iter=2000
)
result = evaluate_model(lsvc_model, X_clinical, y, "Linear SVC", use_proba=False)
results_clinical.append(result)

# 7. SVC (RBF Kernel)
# print("\n7. SVC (RBF Kernel)...")
# svc_model = SVC(
#     kernel="rbf",
#     random_state=42,
#     max_iter=20000,
#     probability=True  # Enable probability estimates for ROC-AUC
# )
# result = evaluate_model(svc_model, X_clinical, y, "SVC (RBF)")
# results_clinical.append(result)

# ============================================================
# CREATE RESULTS TABLES
# ============================================================
print("\n" + "="*80)
print("RESULTS SUMMARY - HOME DATA")
print("="*80)

df_home = pd.DataFrame(results_home)
df_home = df_home.sort_values('(Recall+AUC)/2', ascending=False)

print("\n" + df_home.to_string(index=False))

print("\n" + "="*80)
print("BEST MODEL FOR HOME DATA (by (Recall+AUC)/2):", df_home.iloc[0]['Model'])
print("="*80)
print(f"Accuracy:       {df_home.iloc[0]['Accuracy']:.4f}")
print(f"Precision:      {df_home.iloc[0]['Precision']:.4f}")
print(f"Recall:         {df_home.iloc[0]['Recall']:.4f}")
print(f"AUC:            {df_home.iloc[0]['AUC']:.4f}")
print(f"(Recall+AUC)/2: {df_home.iloc[0]['(Recall+AUC)/2']:.4f}")

print("\n" + "="*80)
print("RESULTS SUMMARY - CLINICAL DATA")
print("="*80)

df_clinical = pd.DataFrame(results_clinical)
df_clinical = df_clinical.sort_values('(Recall+AUC)/2', ascending=False)

print("\n" + df_clinical.to_string(index=False))

print("\n" + "="*80)
print("BEST MODEL FOR CLINICAL DATA (by (Recall+AUC)/2):", df_clinical.iloc[0]['Model'])
print("="*80)
print(f"Accuracy:       {df_clinical.iloc[0]['Accuracy']:.4f}")
print(f"Precision:      {df_clinical.iloc[0]['Precision']:.4f}")
print(f"Recall:         {df_clinical.iloc[0]['Recall']:.4f}")
print(f"AUC:            {df_clinical.iloc[0]['AUC']:.4f}")
print(f"(Recall+AUC)/2: {df_clinical.iloc[0]['(Recall+AUC)/2']:.4f}")

# ============================================================
# OVERALL COMPARISON
# ============================================================
print("\n" + "="*80)
print("OVERALL COMPARISON - ALL MODELS & DATASETS")
print("="*80)

# Add dataset column
df_home['Dataset'] = 'Home'
df_clinical['Dataset'] = 'Clinical'

# Combine
df_all = pd.concat([df_home, df_clinical], ignore_index=True)
df_all = df_all.sort_values('(Recall+AUC)/2', ascending=False)

print("\n" + df_all[['Model', 'Dataset', 'Accuracy', 'Precision', 'Recall', 'AUC', '(Recall+AUC)/2']].to_string(index=False))

print("\n" + "="*80)
print("OVERALL BEST MODEL:", df_all.iloc[0]['Model'], f"({df_all.iloc[0]['Dataset']} data)")
print("="*80)
print(f"Accuracy:       {df_all.iloc[0]['Accuracy']:.4f}")
print(f"Precision:      {df_all.iloc[0]['Precision']:.4f}")
print(f"Recall:         {df_all.iloc[0]['Recall']:.4f}")
print(f"AUC:            {df_all.iloc[0]['AUC']:.4f}")
print(f"(Recall+AUC)/2: {df_all.iloc[0]['(Recall+AUC)/2']:.4f}")
