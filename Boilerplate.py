import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression # kan udskiftes

# 1. Dataindlæsning og encoding

diabetes_data = pd.read_csv("diabetes_dataset.csv")

# Ordinal encoding
diabetes_data["education_level_encoded"] = OrdinalEncoder().fit_transform(
    diabetes_data[["education_level"]]
)
diabetes_data["smoking_status_encoded"] = OrdinalEncoder().fit_transform(
    diabetes_data[["smoking_status"]]
)

# One-hot encoding
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ethnicity_one_hot = onehot_encoder.fit_transform(
    diabetes_data[["gender", "ethnicity", "employment_status"]]
)
ethnicity_one_hot_df = pd.DataFrame(
    ethnicity_one_hot,
    columns=onehot_encoder.get_feature_names_out(["gender", "ethnicity", "employment_status"]),
)
data_encoded = pd.concat(
    [diabetes_data.drop(["gender", "ethnicity", "employment_status"], axis=1), ethnicity_one_hot_df],
    axis=1,
)

# 2. Filtrering og HbA1c-konvertering

filtered_data = data_encoded[~data_encoded["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data = filtered_data.dropna(subset=["hba1c"])
filtered_data["hba1c_mmolmol"] = 10.93 * filtered_data["hba1c"] - 23.5

# 3. Klassifikation i 3 niveauer (IFCC standard)
conditions = [
    (filtered_data["hba1c_mmolmol"] < 42),
    (filtered_data["hba1c_mmolmol"] >= 42) & (filtered_data["hba1c_mmolmol"] < 48),
    (filtered_data["hba1c_mmolmol"] >= 48),
]
values = [0, 1, 2]
filtered_data["hba1c_class"] = np.select(conditions, values)

# -------------------------------------------------------------
# 4. Feature sets
# -------------------------------------------------------------
X_home = filtered_data[
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
X_clinical = filtered_data[["glucose_fasting", "insulin_level", "heart_rate"]]
y = filtered_data["hba1c_class"]

# -------------------------------------------------------------
# 5. Evaluering af model (boilerplate + cross-validation)
# -------------------------------------------------------------
def evaluate_model(X, y, label):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Skalering
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Model (kan ændres her) ---
    model = LogisticRegression(class_weight='balanced')
    #--------------------------------------------------
    # --- Cross-validation ---
    cv_scaler = MinMaxScaler()
    X_scaled = cv_scaler.fit_transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"\n{label} ({model.__class__.__name__}) Cross-val accuracy: "
          f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # --- Modeltræning ---
    model.fit(X_train, y_train)

    # Forudsigelser
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # Klassiske metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Klinisk “rigtig nok” accuracy (±1 klasse)
    cm = confusion_matrix(y_test, y_pred)
    total = np.sum(cm)
    true_close = (
        cm[0,0] + cm[1,1] + cm[2,2] +
        cm[0,1] + cm[1,0] +
        cm[1,2] + cm[2,1]
    )
    clinically_ok = true_close / total

    # ROC-AUC pr. klasse
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc_scores = []
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

    # Udskriv resultater
    print(f"\n{'='*30}\n{label}\n{'='*30}")
    print(f"Accuracy:             {acc:.3f}")
    print(f"Precision (weighted): {prec:.3f}")
    print(f"Recall (weighted):    {rec:.3f}")
    print(f'Klinisk "rigtig nok": {clinically_ok:.3f}')
    for i, val in enumerate(auc_scores):
        print(f"Klasse {i} AUC: {val:.3f}")


# -------------------------------------------------------------
# 6. Kør modeller
# -------------------------------------------------------------
evaluate_model(X_home, y, "Hjemme-data")
evaluate_model(X_clinical, y, "Kliniske data")
