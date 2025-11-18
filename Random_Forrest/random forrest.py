from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
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
# 4. TRAIN & EVALUATE FUNCTION
# ============================================================
def train_and_evaluate(X, y, label):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Standardisering
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=21, random_state=42)
    model.fit(X_train_s, y_train)

    # Prediction
    y_pred = model.predict(X_test_s)
    y_score = model.predict_proba(X_test_s)[:, 1]  # sandsynlighed for klasse 1

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"{label} accuracy: {acc:.3f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-kurve for {label}")
    plt.legend()
    plt.show()

# ============================================================
# 5. RUN MODELS
# ============================================================
train_and_evaluate(X_home, y, "Hjemme-data")
train_and_evaluate(X_clinical, y, "Kliniske data")
