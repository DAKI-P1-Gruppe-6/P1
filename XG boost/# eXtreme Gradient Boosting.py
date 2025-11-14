import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

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
# 5. Evaluering af model med GridSearchCV
# -------------------------------------------------------------

def evaluate_model_with_grid_search(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*30}\n{label} - Grid Search Results\n{'='*30}")
    print(f"Accuracy:             {acc:.3f}")
    print(f"Precision (weighted): {prec:.3f}")
    print(f"Recall (weighted):    {rec:.3f}")

# -------------------------------------------------------------
# 6. Kør modeller med Grid Search
# -------------------------------------------------------------
print("dav")
evaluate_model_with_grid_search(X_home, y, "Home data")
evaluate_model_with_grid_search(X_clinical, y, "Clinical data")
