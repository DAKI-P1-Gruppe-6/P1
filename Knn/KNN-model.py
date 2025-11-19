import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#Remove all no-diabetes patients if mmol/mol > 48
mask_no_diabetes_high = (data["diabetes_stage"] == "No Diabetes") & (data["hba1c_mmolmol"] > 48)

#Remove all pre-diabetes and diabetes patients if mmol/mol < 48
mask_diabetes_low = data["diabetes_stage"].isin(["Pre-Diabetes", "Type 2"]) & (data["hba1c_mmolmol"] < 48)

data = data[~(mask_no_diabetes_high | mask_diabetes_low)].copy()

data["diabetes_group"] = np.where(
    data["diabetes_stage"].isin(["Type 2", "Pre-Diabetes"]),
    "Type 2 / Pre-Diabetes",
    "No Diabetes"
)

print(data["diabetes_group"].value_counts())

# ============================================================
# 4. BINARY CLASSIFICATION
# ============================================================
data["hba1c_class"] = (data["hba1c_mmolmol"] >= 48).astype(int)

# ============================================================
# 5. BALANCERING AF KLASSER (UNDERSAMPLING)
# ============================================================
from sklearn.utils import resample

data_majority = data[data["hba1c_class"] == 1]
data_minority = data[data["hba1c_class"] == 0]  

data_majority_down = resample(
    data_majority,
    replace=False,
    n_samples=len(data_minority),
    random_state=42
)

data = pd.concat([data_majority_down, data_minority])

print("\nEfter balancering (hba1c_class):")
print(data["hba1c_class"].value_counts())

# ============================================================
# 6. FEATURE SETS
# ============================================================

print("\nEfter balancering (hba1c_class):")
print(data["hba1c_class"].value_counts())

# ============================================================
# 5b. VISUALISERING AF DATAPUNKTER
# ============================================================

# Farver pr. klasse (0 = No Diabetes, 1 = Diabetes)
color_map = {0: "tab:blue", 1: "tab:red"}
colors = data["hba1c_class"].map(color_map)

# 1) BMI vs HbA1c (mmol/mol)
plt.figure(figsize=(8, 6))
plt.scatter(data["bmi"], data["hba1c_mmolmol"], c=colors, alpha=0.5)
plt.xlabel("BMI")
plt.ylabel("HbA1c (mmol/mol)")
plt.title("BMI vs HbA1c (mmol/mol) farvet efter klasse")
plt.legend(handles=[
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label="Klasse 0 (hba1c < 48)"),
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:red", label="Klasse 1 (hba1c ≥ 48)")
])
plt.tight_layout()
plt.show()

# 2) Alder vs HbA1c (mmol/mol)
plt.figure(figsize=(8, 6))
plt.scatter(data["age"], data["hba1c_mmolmol"], c=colors, alpha=0.5)
plt.xlabel("Alder")
plt.ylabel("HbA1c (mmol/mol)")
plt.title("Alder vs HbA1c (mmol/mol) farvet efter klasse")
plt.legend(handles=[
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label="Klasse 0 (hba1c < 48)"),
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:red", label="Klasse 1 (hba1c ≥ 48)")
])
plt.tight_layout()
plt.show()

# 3) Kliniske data: faste glukose vs insulin
plt.figure(figsize=(8, 6))
plt.scatter(data["glucose_fasting"], data["insulin_level"], c=colors, alpha=0.5)
plt.xlabel("Faste glukose")
plt.ylabel("Insulin-niveau")
plt.title("Glucose_fasting vs Insulin_level farvet efter klasse")
plt.legend(handles=[
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label="Klasse 0 (hba1c < 48)"),
    plt.Line2D([0], [0], marker="o", linestyle="", color="tab:red", label="Klasse 1 (hba1c ≥ 48)")
])
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

    #Definerer k-range
    k_values = range(20, 100, 2)

    best_k = None
    best_accuracy = 0
    best_precision = 0
    best_recall = 0

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_s, y_train)

        # Predictions
        y_pred = model.predict(X_test_s)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k
            best_precision = prec
            best_recall = rec

    print(f"{label}: {model.__class__.__name__}")
    print(f"Bedste k: {best_k}")
    print(f"Accuracy: {best_accuracy:.1%}")
    print(f"Precision:     {best_precision:.3f}")
    print(f"Recall:        {best_recall:.3f}")
    print(f"→ {int(best_accuracy * len(X_test))} rigtige ud af {len(X_test)} patienter")

def cross_validate_model(X, y, label):

    #Definerer k-range
    k_values = range(20, 100, 2)
    scaler = MinMaxScaler()
    best_k = None
    mean_score = 0
    best_cv = 0
    

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)

        if mean_score > best_mean:
            best_mean = mean_score
            best_cv = cv_scores
            best_k = k

    print(f"{label} ({model.__class__.__name__})")
    print(f"Bedste k: {best_k}")
    print(f"\n{label} Cross-val accuracy: {best_cv.mean()*100:.2f}% ± {best_cv.std()*100:.2f}%")

train_and_evaluate(X_home, y, "Hjemme-data")
#cross_validate_model(X_home, y, "Hjemme-data")

train_and_evaluate(X_clinical, y, "Kliniske data")
#cross_validate_model(X_clinical, y, "Kliniske data")