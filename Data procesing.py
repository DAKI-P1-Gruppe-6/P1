import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
# Konstruer path til diabetes_dataset.csv relativt til denne fil
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "diabetes_dataset.csv")
data = pd.read_csv(data_path)

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

# Fjern yderligere "midt i mellem" patienter for bedre læring
data = data[~((data["hba1c_mmolmol"] < 48) & (data["diabetes_stage"] == "Type 2"))]
data = data[~((data["hba1c_mmolmol"] > 48) & 
                    ((data["diabetes_stage"] == "Pre-Diabetes") | 
                     (data["diabetes_stage"] == "No Diabetes")))]

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
y = data["hba1c_class"]
