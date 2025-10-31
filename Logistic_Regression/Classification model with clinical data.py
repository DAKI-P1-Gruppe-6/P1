import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score

################ BOILER PLATE FOR DATA ############################################################

diabetes_data = pd.read_csv("diabetes_dataset.csv")

#Encoding of object to binary
ordinal_encoder = OrdinalEncoder()
diabetes_data["education_level_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["education_level"]])
diabetes_data["smoking_status_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["smoking_status"]])

#Encoding object to groups
onehot_encoder = OneHotEncoder(sparse_output=False)
ethnicity_one_hot = onehot_encoder.fit_transform(diabetes_data[["gender","ethnicity","employment_status"]])
ethnicity_one_hot_df = pd.DataFrame(ethnicity_one_hot,columns=onehot_encoder.get_feature_names_out(["gender","ethnicity","employment_status"]))
data_encoded = pd.concat([diabetes_data.drop(["gender","ethnicity","employment_status"],axis=1),ethnicity_one_hot_df],axis=1)

#Filtering data
filtered_data = data_encoded[~data_encoded["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data["hba1c"] = (filtered_data["hba1c"] - 2.15) * 10.929
filtered_data = filtered_data.dropna(subset=["hba1c"])

#HbA1c classification
conditions = [
    (filtered_data["hba1c"] < 42),
    (filtered_data["hba1c"] >= 42) & (filtered_data["hba1c"] < 48),
    (filtered_data["hba1c"] >= 48)
]
values = [0, 1, 2]
filtered_data["hba1c_class"] = np.select(conditions, values)

#Feature sets
X_home = filtered_data[["age","diet_score","bmi","smoking_status_encoded","waist_to_hip_ratio","sleep_hours_per_day"]]
X_clinical = filtered_data[["heart_rate","glucose_fasting","insulin_level"]]
y = filtered_data["hba1c_class"]

#Model function
def train_and_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


#####################################################################################################################



    model = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{label} accuracy: {accuracy:.3f}")

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(8,6))
    for i in range(y_score.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Klasse {i} (AUC = {roc_auc:.2f})")

    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-kurve for {label}")
    plt.legend()
    plt.show()

print(filtered_data["hba1c_class"].value_counts())

#Run models
train_and_evaluate(X_home, y, "Hjemme-data")
train_and_evaluate(X_clinical, y, "Kliniske data")

