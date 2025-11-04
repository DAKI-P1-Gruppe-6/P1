import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

diabetes_data = pd.read_csv("diabetes_dataset.csv")

ordinal_encoder = OrdinalEncoder()
diabetes_data["education_level_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["education_level"]])
diabetes_data["smoking_status_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["smoking_status"]])

onehot_encoder = OneHotEncoder(sparse_output=False)
ethnicity_one_hot = onehot_encoder.fit_transform(diabetes_data[["gender","ethnicity","employment_status"]])
ethnicity_one_hot_df = pd.DataFrame(
    ethnicity_one_hot,
    columns=onehot_encoder.get_feature_names_out(["gender","ethnicity","employment_status"])
)
data_encoded = pd.concat(
    [diabetes_data.drop(["gender","ethnicity","employment_status"], axis=1),
     ethnicity_one_hot_df],
    axis=1
)

filtered_data = data_encoded[~data_encoded["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data["hba1c"] = (filtered_data["hba1c"] - 2.15) * 10.929
filtered_data = filtered_data.dropna(subset=["hba1c"])

conditions = [
    (filtered_data["hba1c"] < 42),
    (filtered_data["hba1c"] >= 42) & (filtered_data["hba1c"] < 48),
    (filtered_data["hba1c"] >= 48)
]
values = [0, 1, 2]
filtered_data["hba1c_class"] = np.select(conditions, values)

X_home = filtered_data[["age","diet_score","bmi","smoking_status_encoded","waist_to_hip_ratio","sleep_hours_per_day"]]
X_clinical = filtered_data[["heart_rate","glucose_fasting","insulin_level"]]
y = filtered_data["hba1c_class"]

def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Define and fit model
    model = 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{label}: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"→ {int(accuracy * len(X_test))} rigtige ud af {len(X_test)} patienter")

def cross_validate_model(X, y, label):
    #Define and fit model
    model = 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"{label} ({model.__class__.__name__}) cross-val accuracy: {scores.mean()*100:.3f} ± {scores.std()*100:.3f}")


#Run models
train_and_evaluate(X_home, y, "Hjemme-data")
cross_validate_model(X_home, y, "Hjemme-data")

train_and_evaluate(X_clinical, y, "Kliniske data")
cross_validate_model(X_clinical, y, "Kliniske data")
