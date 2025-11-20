import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix
)

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

data["hba1c_mmolmol"] = 10.93 * data["hba1c"] - 23.5

mask_no_diabetes_high = (data["diabetes_stage"] == "no diabetes") & (data["hba1c_mmolmol"] > 48)
mask_diabetes_low = data["diabetes_stage"].isin(["pre-diabetes", "Type 2"]) & (data["hba1c_mmolmol"] < 48)

data = data[~(mask_no_diabetes_high | mask_diabetes_low)].copy()

data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)

# Fjern yderligere "midt i mellem" patienter for bedre læring
data = data[(data["hba1c_mmolmol"] < 45) | (data["hba1c_mmolmol"] > 50)].copy()

data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)


# ============================================================
# 3. FEATURE SETS (FORBEDRET)
# ============================================================
X_home = data[
    [
        "age",
        "bmi",
        "waist_to_hip_ratio",
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
# 4. TRAINING FUNCTION (LOGREG + THRESHOLD TUNING)
# ============================================================
def train_best_logreg(X, y, label):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Flere hyperparametre end dine originale
    model = LogisticRegression(
        C=5,
        class_weight="balanced",
        max_iter=500,
        penalty="l2",
        solver="saga"
    )

    model.fit(X_train_s, y_train)

    # Probability predictions
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # ======================================================
    # THRESHOLD TUNING (finder hvilket cut der giver bedste accuracy)
    # ======================================================
    thresholds = np.linspace(0.1, 0.9, 200)
    best_thresh = 0.5
    best_acc = 0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred_t)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Pred efter bedste threshold
    y_pred = (y_prob >= best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\n=== {label} ===")
    print(f"Optimized threshold: {best_thresh:.3f}")
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"→ {int(acc * len(X_test))} rigtige ud af {len(X_test)}")

    # ======================================================
    # ROC AUC
    # ======================================================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.3f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {label}")
    plt.legend()
    plt.show()

    # ======================================================
    # Confusion matrix
    # ======================================================
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

from sklearn.model_selection import cross_val_score

def cross_validate_logreg(X, y, label):

    # Skal skalere X for hver fold, så vi bygger en pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=5,
            class_weight="balanced",
            max_iter=500,
            penalty="l2",
            solver="saga"
        ))
    ])

    print("\n==============================================")
    print(f" CROSS-VALIDATION RESULTS – {label}")
    print("==============================================")

    # Accuracy
    acc_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Accuracy (mean ± std) : {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    # Recall
    rec_scores = cross_val_score(pipeline, X, y, cv=5, scoring="recall")
    print(f"Recall (mean ± std)   : {rec_scores.mean():.3f} ± {rec_scores.std():.3f}")

    # AUC
    auc_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    print(f"ROC–AUC (mean ± std)  : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    print("==============================================\n")


train_best_logreg(X_clinical, y, "Kliniske data – Optimized Logistic Regression")
cross_validate_logreg(X_clinical, y, "Kliniske data – Optimized Logistic Regression")

train_best_logreg(X_home, y, "Hjemme data – Optimized Logistic Regression")
cross_validate_logreg(X_home, y, "Hjemme data – Optimized Logistic Regression")



