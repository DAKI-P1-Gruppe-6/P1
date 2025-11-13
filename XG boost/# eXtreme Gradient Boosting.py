import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt  # Rettet import af matplotlib
# -------------------------------------------------------------
# 1. Dataindlæsning og encoding
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 2. Filtrering og HbA1c-konvertering
# -------------------------------------------------------------
filtered_data = data_encoded[~data_encoded["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data = filtered_data.dropna(subset=["hba1c"])
filtered_data["hba1c_mmolmol"] = 10.93 * filtered_data["hba1c"] - 23.5

# -------------------------------------------------------------
# 3. Klassifikation i 3 niveauer (IFCC standard)
# -------------------------------------------------------------
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
# 5. Evaluering af model
# -------------------------------------------------------------
def evaluate_model(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Gradient Boosting model optimeret til lav CPU-belastning
    model = GradientBoostingClassifier(
        n_estimators=200,        # færre træer = hurtigere
        max_depth=10,            # lavere dybde = mindre CPU-forbrug
        min_samples_leaf=8,      # forhindrer overfitting
        max_features="sqrt",     # hurtigere splits
        random_state=42
    )
    model.fit(X_train, y_train)

    # Forudsigelser
    y_pred = model.predict(X_test)

    # Klassiske metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    from sklearn.preprocessing import label_binarize

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)

    # Klinisk ”rigtig nok” accuracy (tillader små fejl)
    cm = confusion_matrix(y_test, y_pred)
    total = np.sum(cm)
    true_close = (
        cm[0,0] + cm[1,1] + cm[2,2] +
        cm[0,1] + cm[1,0] +
        cm[1,2] + cm[2,1]
    )
    clinically_ok = true_close / total

    print(f"\n{'='*30}\n{label}\n{'='*30}")
    print(f"Accuracy:             {acc:.3f}")
    print(f"Precision (weighted): {prec:.3f}")
    print(f"Recall (weighted):    {rec:.3f}")
    print(f'Clinical "close enough": {clinically_ok:.3f}')

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        print(f"Class {i} AUC: {roc_auc:.3f}")

# -------------------------------------------------------------
# 6. Kør modeller
# -------------------------------------------------------------
evaluate_model(X_home, y, "Home data")
evaluate_model(X_clinical, y, "Clinical data")
