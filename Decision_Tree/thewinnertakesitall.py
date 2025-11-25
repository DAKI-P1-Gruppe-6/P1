import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
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
# 1. LOAD & PREPARE DATA
# ============================================================
data = pd.read_csv("diabetes_dataset.csv")

# Encoding
data["education_level_encoded"] = OrdinalEncoder().fit_transform(data[["education_level"]])
data["smoking_status_encoded"] = OrdinalEncoder().fit_transform(data[["smoking_status"]])

onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = onehot.fit_transform(data[["gender", "ethnicity", "employment_status"]])
encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(["gender", "ethnicity", "employment_status"]))
data = pd.concat([data.drop(["gender", "ethnicity", "employment_status"], axis=1), encoded_df], axis=1)

# ============================================================
# 2. FILTERING
# ============================================================
data = data[~data["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
data = data.dropna(subset=["hba1c"])

# hba1c (NGSP %) → mmol/mol (IFCC)
data["hba1c_mmolmol"] = 10.93 * data["hba1c"] - 23.5
data = data[~((data["hba1c_mmolmol"] < 48) & (data["diabetes_stage"] == "Type 2"))]

# ============================================================
# 3. BINARY CLASSIFICATION
# ============================================================
data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)

# ============================================================
# 4. FEATURE SETS
# ============================================================
X_home = data[
    [
        "age",
        "bmi",
        "diet_score",
        "physical_activity_minutes_per_week",
        "sleep_hours_per_day",
        "smoking_status_encoded",
        "alcohol_consumption_per_week",
        "family_history_diabetes",
    ]
]

X_clinical = data[["glucose_fasting", "insulin_level", "heart_rate"]]

y = data["diagnosed_diabetes"]


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
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=36,
        max_leaf_nodes=34,
        min_samples_leaf=13,
        min_impurity_decrease=1.1876178794228976e-05,
        min_samples_split=14
    )

    # Cross-validation
    cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
    print(f"\n{label} Cross-val accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Training
    model.fit(X_train_s, y_train)

    # --- Feature Importance (Decision Tree) ---
    importances = model.feature_importances_
    feature_names = X.columns

    # Sort by importance
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(7, 4))
    plt.barh(feature_names[idx], importances[idx], color="steelblue")
    plt.title(f"Feature Importances – {label}")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()  # Highest at top
    plt.tight_layout()
    plt.show()


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

    return {
        "model": model,
        "scaler": scaler,
        "X_test_s": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_score": y_score,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
    }


# ============================================================
# 6. RUN MODELS (NO ROC PLOTTING HERE)
# ============================================================
results_home = evaluate_model(X_home, y, "Home Data")
results_clinical = evaluate_model(X_clinical, y, "Clinical Data")


# ============================================================
# 7. PLOT FOR HOME DATA ONLY
# ============================================================

# --- ROC Curve ---
plt.figure(figsize=(6, 4))
plt.plot(results_home["fpr"], results_home["tpr"], label=f"Home Data (AUC={results_home['roc_auc']:.3f})", lw=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Home Data Only")
plt.legend()
plt.tight_layout()
plt.show()

# --- Confusion Matrix ---
cm = confusion_matrix(results_home["y_test"], results_home["y_pred"])

# Extract TN, FP, FN, TP from sklearn format
tn, fp, fn, tp = cm.ravel()

# Reorder to requested format: TP, FP, TN, FN
cm_custom = np.array([
    [tp, fp],
    [tn, fn]
])

labels = np.array([
    ["True Positive", "False Positive"],
    ["True Negative", "False Negative"]
])

plt.figure(figsize=(6, 5))
plt.imshow(cm_custom, cmap="Blues")
plt.title("Confusion Matrix – Home Data (Labeled)")
plt.colorbar()

# Axis labels
plt.xticks([0, 1], ["TP", "FP"])
plt.yticks([0, 1], ["TN", "FN"])

# Write both label + number inside squares
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            f"{labels[i, j]}\n{cm_custom[i, j]}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold"
        )

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.tight_layout()
plt.show()


# Add text inside squares
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

plt.tight_layout()
plt.show()

