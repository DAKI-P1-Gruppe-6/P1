import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

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

X_home = filtered_data[[
    "age","bmi","waist_to_hip_ratio","diet_score",
    "physical_activity_minutes_per_week","sleep_hours_per_day",
    "smoking_status_encoded","alcohol_consumption_per_week",
    "family_history_diabetes"
]]

X_clinical = filtered_data[[
    "glucose_fasting","insulin_level","heart_rate"
]]

X_combined = pd.concat([X_home, X_clinical], axis=1)
y = filtered_data["hba1c_class"]

def evaluate_set(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    threshold = 0.3
    y_pred = []
    for p in probs:
        if p[2] >= threshold:
            y_pred.append(2)
        else:
            y_pred.append(np.argmax(p[:2]))
    print(f"\n{label}")
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(label)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    importances = model.feature_importances_
    for f, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
        print(f"{f:35s} {imp:.3f}")
    if label == "Hjemme-data":
        total = np.sum(cm)
        true_close = (
            cm[0,0] + cm[1,1] + cm[2,2] +
            cm[0,1] + cm[1,0] +
            cm[1,2] + cm[2,1]
        )
        critical_errors = cm[0,2] + cm[2,0]
        print(f"Klassisk accuracy: {np.trace(cm)/total:.3f}")
        print(f'Klinisk "rigtig nok" accuracy: {true_close/total:.3f}')
        print(f"Kritiske fejl (0↔2): {critical_errors/total*100:.2f}%")

evaluate_set(X_home, y, "Hjemme-data")
evaluate_set(X_clinical, y, "Kliniske data")
evaluate_set(X_combined, y, "Kombineret data")
