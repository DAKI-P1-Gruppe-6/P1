import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

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

# ============================================================
# 3. 
# ============================================================

#Remove all no-diabetes patients if mmol/mol > 42
mask_no_diabetes_wrong = (
    (data["diabetes_stage"] == "No Diabetes") &
    (data["hba1c_mmolmol"] >= 42)
)

#Remove all Type 2 diabetes patients if mmol/mol < 48
mask_type2_wrong = (
    (data["diabetes_stage"] == "Type 2") &
    (data["hba1c_mmolmol"] < 48)
)

#Remove all pre-diabetes patients if mmol/mol >=42 or mmol/mol <48
mask_pre_diabetes_wrong = (
    (data["diabetes_stage"] == "Pre-Diabetes") &
    (
        (data["hba1c_mmolmol"] < 42) |
        (data["hba1c_mmolmol"] >= 48)
    )
)

no_diabetes = data[data["diabetes_stage"] == "No Diabetes"]
pre_diabetes = data[data["diabetes_stage"] == "Pre-Diabetes"]
type_2 = data[data["diabetes_stage"] == "Type 2"]


# ============================================================
# 4. BALANCERING AF KLASSER (UNDERSAMPLING)
# ============================================================

label_map = {"No Diabetes": 0, "Pre-Diabetes": 1, "Type 2": 2}
data["diabetes_class_3"] = data["diabetes_stage"].map(label_map)

print("\nKlassestørrelser FØR balancering (labels):")
print(data["diabetes_stage"].value_counts())

# Find mindste klassestørrelse
class_counts = data["diabetes_class_3"].value_counts()
min_count = class_counts.min()

balanced_parts = []
for c in sorted(class_counts.index):
    subset = data[data["diabetes_class_3"] == c]
    subset_down = resample(
        subset,
        replace=False,            # undersampling
        n_samples=min_count,
        random_state=42
    )
    balanced_parts.append(subset_down)

data = pd.concat(balanced_parts).sample(frac=1, random_state=42).copy()

print("\nKlassestørrelser EFTER balancering (labels):")
print(data["diabetes_stage"].value_counts())

# ============================================================
# 5. BINARY CLASSIFICATION
# ============================================================
data["hba1c_class"] = np.where(
    data["diabetes_stage"].isin(["Pre-Diabetes", "No Diabetes"]), 1, 0
)

print("\nFordeling af binær klasse (0 = No/Pre, 1 = Type2):")
print(data["hba1c_class"].value_counts())

# ============================================================
# UNDERSAMPLING AF MAJORITY-CLASS
# ============================================================

counts = data["hba1c_class"].value_counts()
minority_class = counts.idxmin()
majority_class = counts.idxmax()

data_minority = data[data["hba1c_class"] == minority_class]
data_majority = data[data["hba1c_class"] == majority_class]

data_majority_down = resample(
    data_majority,
    replace=False,
    n_samples=len(data_minority),
    random_state=42
)

data = pd.concat([data_minority, data_majority_down]).sample(frac=1, random_state=42).copy()

print("\nFordeling af binær klasse EFTER undersampling (0 = No/Pre, 1 = Type2):")
print(data["hba1c_class"].value_counts())

print("\nFordeling af diabetes_stage EFTER undersampling:")
print(data["diabetes_stage"].value_counts())

# ============================================================
# HBA1C
# ============================================================
data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)

# ============================================================
# Data visualisation
# ============================================================
# Farver til klasser (0 = <48, 1 = ≥48)
color_map = {0: "tab:blue", 1: "tab:red"}
colors = data["hba1c_class"].map(color_map)

# 1) Histogram over HbA1c (mmol/mol) pr. klasse
plt.figure(figsize=(8, 6))
for cls, label in [(0, "Klasse 0 (<48 mmol/mol)"), (1, "Klasse 1 (≥48 mmol/mol)")]:
    subset = data[data["hba1c_class"] == cls]
    plt.hist(
        subset["hba1c_mmolmol"],
        bins=30,
        alpha=0.5,
        label=label
    )

plt.xlabel("HbA1c (mmol/mol)")
plt.ylabel("Antal patienter")
plt.title("Fordeling af HbA1c (mmol/mol) efter binær klasse")
plt.legend()
plt.tight_layout()
plt.show()

# 2) BMI vs HbA1c (mmol/mol)
plt.figure(figsize=(8, 6))
plt.scatter(
    data["bmi"],
    data["hba1c_mmolmol"],
    c=colors,
    alpha=0.5
)
plt.xlabel("BMI")
plt.ylabel("HbA1c (mmol/mol)")
plt.title("BMI vs HbA1c (mmol/mol) farvet efter klasse")
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label="Klasse 0 (<42)"),
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:red", label="Klasse 1 (≥42)")
]
plt.legend(handles=handles)
plt.tight_layout()
plt.show()

# 3) Alder vs HbA1c (mmol/mol)
plt.figure(figsize=(8, 6))
plt.scatter(
    data["age"],
    data["hba1c_mmolmol"],
    c=colors,
    alpha=0.5
)
plt.xlabel("Alder")
plt.ylabel("HbA1c (mmol/mol)")
plt.title("Alder vs HbA1c (mmol/mol) farvet efter klasse")
plt.legend(handles=handles)
plt.tight_layout()
plt.show()

# 4) Barplot: diabetes_stage efter undersampling
plt.figure(figsize=(6, 4))
data["diabetes_stage"].value_counts().plot(kind="bar")
plt.xlabel("diabetes_stage")
plt.ylabel("Antal patienter")
plt.title("Fordeling af diabetes_stage efter undersampling")
plt.tight_layout()
plt.show()
# ============================================================
# 7. FEATURE SETS
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
y = data["hba1c_class"]

def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Definerer k-range
    k_values = range(10, 200, 2)

    best_k = None
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_roc_auc = 0.0  # ny variabel til ROC-AUC

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_s, y_train)

        # Predictions (klasser)
        y_pred = model.predict(X_test_s)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # ROC-AUC: brug sandsynlighed for klasse 1
        y_proba = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # behold det k med bedst accuracy (som før)
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
            best_precision = prec
            best_recall = rec
            best_roc_auc = roc_auc   # gem tilhørende ROC-AUC

    print()
    print(f"{label}: {model.__class__.__name__}")
    print(f"Bedste k: {best_k}")
    print(f"Accuracy:       {best_accuracy:.1%}")
    print(f"Precision:      {best_precision:.3f}")
    print(f"Recall:         {best_recall:.3f}")
    print(f"ROC–AUC:        {best_roc_auc:.3f}")
    print(f"→ {int(best_accuracy * len(X_test))} rigtige ud af {len(X_test)} patienter")

train_and_evaluate(X_home, y, "Hjemme-data")
#cross_validate_model(X_home, y, "Hjemme-data")

train_and_evaluate(X_clinical, y, "Kliniske data")
#cross_validate_model(X_clinical, y, "Kliniske data")