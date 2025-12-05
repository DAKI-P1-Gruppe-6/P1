import pandas as pd
import numpy as np
import importlib.util
import os
import joblib
from datetime import datetime
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
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# IMPORT DATA FROM Data procesing.py
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_path = os.path.join(script_dir, "..", "Data procesing.py")
spec = importlib.util.spec_from_file_location("data_processing", data_processing_path)
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)

X_home = data_processing.X_home
y = data_processing.y

# ============================================================
# RANDOMIZED SEARCH FUNCTION
# ============================================================
def evaluate_model_with_randomsearch(X, y, label, n_iter=100):
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

    # --- RandomizedSearchCV: Test RANDOM sample of combinations ---
    # Expanded parameter space for Random Forest
    param_distributions = {
        # Number of trees
        'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],
        
        # Tree depth and structure
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, None],
        'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
        'min_samples_leaf': [1, 2, 3, 4, 5, 7, 10],
        
        # Feature sampling
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7],
        
        # Sampling
        'bootstrap': [True, False],
        
        # Class balance
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        
        # Split criterion
        'criterion': ['gini', 'entropy', 'log_loss'],
        
        # Leaf nodes
        'max_leaf_nodes': [None, 10, 20, 30, 50, 75, 100],
        
        # Other parameters
        'min_impurity_decrease': [0.0, 0.00001, 0.0001, 0.001],
        'max_samples': [None, 0.5, 0.7, 0.8, 0.9]
    }
    
    # Calculate total possible combinations
    total_combinations = 1
    for param_values in param_distributions.values():
        total_combinations *= len(param_values)
    
    rf_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    )
    
    # RandomizedSearchCV: Test RANDOM sample with 3-fold CV
    random_search = RandomizedSearchCV(
        rf_model,
        param_distributions,
        n_iter=n_iter,  # Number of random combinations to test
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,  # Show progress
        random_state=42,
        return_train_score=True
    )
    
    print(f"Fitting RandomizedSearchCV on {label}...")
    print(f"Total possible combinations: {total_combinations:,}")
    print(f"Testing {n_iter} RANDOM combinations with 3-fold CV...")
    print(f"Total fits: {n_iter * 3}")
    print(f"Optimizing for: Accuracy\n")
    print("⚡ Much faster than GridSearch!\n")
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\n✅ RandomizedSearchCV completed!")
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV Accuracy: {random_search.best_score_:.4f}")
    
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
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:             {acc:.4f}")
    print(f"  Precision:            {prec:.4f}")
    print(f"  Recall:               {rec:.4f}")
    print(f"  ROC-AUC:              {roc_auc:.4f}")
    print(f"  (Recall+AUC)/2:       {combined_score:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    # ============================================================
    # SAVE MODEL (PERSISTENCE)
    # ============================================================
    # Create models directory if it doesn't exist
    models_dir = os.path.join(script_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = label.replace(" ", "_").replace("-", "_")
    model_filename = f"random_forest_{safe_label}_{timestamp}.pkl"
    scaler_filename = f"scaler_rf_{safe_label}_{timestamp}.pkl"
    model_path = os.path.join(models_dir, model_filename)
    scaler_path = os.path.join(models_dir, scaler_filename)
    
    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'label': label,
        'timestamp': timestamp,
        'search_type': 'RandomizedSearch',
        'n_iter': n_iter,
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_,
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'roc_auc': roc_auc,
            'combined_score': combined_score
        },
        'confusion_matrix': cm.tolist(),
        'model_file': model_filename,
        'scaler_file': scaler_filename
    }
    
    metadata_filename = f"metadata_rf_{safe_label}_{timestamp}.pkl"
    metadata_path = os.path.join(models_dir, metadata_filename)
    joblib.dump(metadata, metadata_path)
    
    print(f"\n📁 Model saved to: {model_path}")
    print(f"📁 Scaler saved to: {scaler_path}")
    print(f"📁 Metadata saved to: {metadata_path}")
    
    return {
        'Model': 'Random Forest',
        'Data': label,
        'Best_Params': random_search.best_params_,
        'CV_Score': random_search.best_score_,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'ROC-AUC': roc_auc,
        '(Recall+AUC)/2': combined_score,
        'Model_Path': model_path,
        'Scaler_Path': scaler_path
    }


# ============================================================
# RUN RANDOMIZED SEARCH
# ============================================================
print("\n" + "🔍 Randomized Search on Home Data")
print("Finding great hyperparameters efficiently...")
N_ITERATIONS = 100  # Adjust this value to test more/fewer combinations
result_home = evaluate_model_with_randomsearch(X_home, y, "Home-data", n_iter=N_ITERATIONS)

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "="*70)
print("RANDOM FOREST RANDOMIZED SEARCH RESULTS - HOME DATA")
print("="*70)

print(f"\nData: {result_home['Data']}")
print(f"Search Type: RandomizedSearchCV")
print(f"Combinations Tested: {N_ITERATIONS}")
print(f"\nCross-Validation Score: {result_home['CV_Score']:.4f}")
print(f"\nTest Set Performance:")
print(f"  Accuracy:             {result_home['Accuracy']:.4f}")
print(f"  Precision:            {result_home['Precision']:.4f}")
print(f"  Recall:               {result_home['Recall']:.4f}")
print(f"  ROC-AUC:              {result_home['ROC-AUC']:.4f}")
print(f"  (Recall+AUC)/2:       {result_home['(Recall+AUC)/2']:.4f}")

# Print best parameters
print("\n" + "="*70)
print("BEST PARAMETERS FOUND")
print("="*70)
for param, value in result_home['Best_Params'].items():
    print(f"  {param}: {value}")

print("\n" + "="*70)
print("SAVED MODEL FILES")
print("="*70)
print(f"  Model: {result_home['Model_Path']}")
print(f"  Scaler: {result_home['Scaler_Path']}")

print("\n" + "="*70)
print("✅ Model trained, optimized, and saved!")
print("⚡ RandomizedSearch found great parameters efficiently!")
print("="*70)

