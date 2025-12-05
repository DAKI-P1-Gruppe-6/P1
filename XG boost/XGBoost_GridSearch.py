# hejsa  igen
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
from xgboost import XGBClassifier

# ============================================================
# IMPORT DATA FROM Data procesing.py
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_path = os.path.join(script_dir, "..", "Data procesing.py")
spec = importlib.util.spec_from_file_location("data_processing", data_processing_path)
#hejsa
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)

X_home = data_processing.X_home
y = data_processing.y

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def load_saved_model(model_path, scaler_path):
    """
    Load a saved XGBoost model and its scaler.
    
    Example usage:
        model, scaler = load_saved_model('saved_models/xgboost_Home_20231201_143022.pkl',
                                         'saved_models/scaler_Home_20231201_143022.pkl')
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model loaded from: {model_path}")
    print(f"Scaler loaded from: {scaler_path}")
    return model, scaler

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
    # Diverse parameter space - reduced values, more parameter types
    param_distributions = {
        # Tree Structure Parameters
        'n_estimators': [50, 100, 200, 400],
        'max_depth': [4, 6, 8, 10],
        'max_leaves': [0, 31, 63],
        'min_child_weight': [1, 5, 10],
        
        # Learning Parameters
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        
        # Regularization Parameters
        'gamma': [0, 0.1, 0.5, 1.0],  # min_split_loss
        'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
        'reg_lambda': [1.0, 5.0, 10.0],  # L2 regularization
        
        # Sampling Parameters
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'colsample_bylevel': [0.8, 1.0],
        'colsample_bynode': [0.8, 1.0],
        
        # Tree Method and Growth Policy
        'tree_method': ['auto', 'hist'],
        'grow_policy': ['depthwise', 'lossguide'],
        'max_delta_step': [0, 1, 5],
        
        # Histogram and Binning
        'max_bin': [128, 256, 512],  # Number of bins for histogram
        
        # Sampling Method
        'sampling_method': ['uniform', 'gradient_based'],  # Sampling strategy
        
        # Parallel Trees (Random Forest style)
        'num_parallel_tree': [1, 2, 3],  # Multiple trees per iteration
        
        # Class Balance
        'scale_pos_weight': [1, 2, 4],
        
        # Booster Type
        'booster': ['gbtree', 'dart'],  # dart = dropout trees
        
        # DART specific (only used if booster='dart')
        'sample_type': ['uniform', 'weighted'],
        'normalize_type': ['tree', 'forest'],
        'rate_drop': [0.0, 0.1, 0.3],  # Dropout rate
        'skip_drop': [0.0, 0.3, 0.5]  # Probability of skipping dropout
    }
    
    # Calculate total possible combinations
    total_combinations = 1
    for param_values in param_distributions.values():
        total_combinations *= len(param_values)
    
    xgb_model = XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    # RandomizedSearchCV: Test RANDOM sample with 3-fold CV
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions,
        n_iter=n_iter,  # Number of random combinations to test
        cv=3,  # 3-fold cross-validation
        scoring='accuracy',
        n_jobs=1,  # Changed from -1 to avoid multiprocessing issues
        verbose=2,  # Show progress
        random_state=42,
        return_train_score=True
    )
    
    print(f"Fitting RandomizedSearchCV on {label}...")
    print(f"Total possible combinations: {total_combinations:,}")
    print(f"Testing {n_iter} RANDOM combinations with 3-fold CV...")
    print(f"Total fits: {n_iter * 3}")
    print(f"Optimizing for: Accuracy\n")
    print("Much faster than GridSearch!\n")
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\nRandomizedSearchCV completed!")
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
    print(f"\nAccuracy:             {acc:.4f}")
    print(f"Precision:            {prec:.4f}")
    print(f"Recall:               {rec:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"(Recall+AUC)/2:       {combined_score:.4f}")
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
    model_filename = f"xgboost_{safe_label}_{timestamp}.pkl"
    scaler_filename = f"scaler_{safe_label}_{timestamp}.pkl"
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
    
    metadata_filename = f"metadata_{safe_label}_{timestamp}.pkl"
    metadata_path = os.path.join(models_dir, metadata_filename)
    joblib.dump(metadata, metadata_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return {
        'Model': 'XGBoost',
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
print("\nRandomized Search on Home Data")
print("Finding great hyperparameters efficiently...")
N_ITERATIONS = 100  # Adjust this value
result_home = evaluate_model_with_randomsearch(X_home, y, "Hjemme-data", n_iter=N_ITERATIONS)

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "="*70)
print("XGBOOST RANDOMIZED SEARCH RESULTS - HOME DATA")
print("="*70)

print(f"\nData: {result_home['Data']}")
print(f"CV Score:             {result_home['CV_Score']:.4f}")
print(f"Accuracy:             {result_home['Accuracy']:.4f}")
print(f"Precision:            {result_home['Precision']:.4f}")
print(f"Recall:               {result_home['Recall']:.4f}")
print(f"ROC-AUC:              {result_home['ROC-AUC']:.4f}")
print(f"(Recall+AUC)/2:       {result_home['(Recall+AUC)/2']:.4f}")

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
print("Model optimized and saved!")
print("RandomizedSearch completed efficiently!")
print("="*70)

