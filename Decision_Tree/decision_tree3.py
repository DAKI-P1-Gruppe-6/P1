import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.tree import  DecisionTreeClassifier

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
# 5. EVALUATION FUNCTION (BINARY)
# ============================================================
def evaluate_model(X, y, label):

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model
    model = DecisionTreeClassifier(criterion="entropy",max_depth=36,max_leaf_nodes=34,min_samples_leaf=13,min_impurity_decrease=1.1876178794228976e-05,min_samples_split=14)
    
    # Cross-validation
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    print(f"\n{label} Cross-val accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Training
    model= model.fit(X_train_s, y_train)

    # Predictions
    y_pred = model.predict(X_test_s)
    y_score = model.predict_proba(X_test_s)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Print results
    print(f"\n{'='*40}")
    print(f"{label} RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy:      {acc:.3f}")
    print(f"Precision:     {prec:.3f}")
    print(f"Recall:        {rec:.3f}")
    print(f"ROC-AUC:       {roc_auc:.3f}")

    # Plot ROC
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")


# ============================================================
# 6. RUN MODELS
# ============================================================
plt.figure(figsize=(6, 4))

evaluate_model(X_home, y, "Home Data")
evaluate_model(X_clinical, y, "Clinical Data")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Binary Decision Tree classification")
plt.legend()
plt.tight_layout()
plt.show()

