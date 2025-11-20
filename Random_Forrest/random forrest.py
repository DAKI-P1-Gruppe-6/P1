from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# HbA1c (NGSP %) → mmol/mol (IFCC)
data["hba1c_mmolmol"] = (data["hba1c"] - 2.15) * 10.929

# Fjern uoverensstemmende labels
mask_no_diabetes_high = (data["diabetes_stage"] == "no diabetes") & (data["hba1c_mmolmol"] >= 48)
mask_diabetes_low = data["diabetes_stage"].isin(["pre-diabetes", "Type 2"]) & (data["hba1c_mmolmol"] < 48)
data = data[~(mask_no_diabetes_high | mask_diabetes_low)].copy()

# Binær label
data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)
print("\nClass counts:")
print(data["hba1c_class"].value_counts())

# ============================================================
# 3. FEATURE SETS
# ============================================================
features_home = [
    "age", "bmi", "waist_to_hip_ratio", "diet_score",
    "physical_activity_minutes_per_week", "sleep_hours_per_day",
    "smoking_status_encoded", "alcohol_consumption_per_week",
    "family_history_diabetes"
]

features_clinical = ["glucose_fasting", "insulin_level", "heart_rate"]

X_home = data[features_home]
X_clinical = data[features_clinical]
y = data["hba1c_class"]

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

<<<<<<< Updated upstream
    # 2️⃣ Split og ROC på test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
=======
    # Model
    model = RandomForestClassifier(n_estimators=10, min_samples_leaf=21, random_state=42)
    model.fit(X_train_s, y_train)
>>>>>>> Stashed changes

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"{label} test set metrics:")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, ROC-AUC: {roc_auc:.3f}")

    # Plot ROC
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

# ============================================================
# 5. RUN MODELS
# ============================================================
plt.figure(figsize=(6, 4))
evaluate_model_cv(X_home, y, "Home Data")
evaluate_model_cv(X_clinical, y, "Clinical Data")
plt.plot([0,1],[0,1],"k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Binary Classification")
plt.legend()
plt.tight_layout()
plt.show()
