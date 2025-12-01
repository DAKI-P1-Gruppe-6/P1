from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import pandas as pd
import numpy as np
import importlib.util
import os

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

print("\nClass counts:")
print(y.value_counts())

# ============================================================
# 4. TRAIN & EVALUATE FUNCTION WITH CROSS-VALIDATION
# ============================================================
def evaluate_model_cv(X, y, label, cv=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=100,        # færre træer = hurtigere
        max_depth=6,           # lavere dybde = mindre CPU-forbrug
        min_samples_leaf=4,     # forhindrer overfitting
        max_features="sqrt",    # hurtigere splits
        class_weight="balanced",
        n_jobs=-1,              # brug alle kerner (hurtigere)
        random_state=42
    )

    # 1️⃣ Krydsvalidering
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"\n{label} 5-fold CV accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # 2️⃣ Split og ROC på test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"{label} test set metrics:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, ROC-AUC: {roc_auc:.3f}")

# ============================================================
# 5. RUN MODELS
# ============================================================
evaluate_model_cv(X_home, y, "Home Data")
evaluate_model_cv(X_clinical, y, "Clinical Data")
