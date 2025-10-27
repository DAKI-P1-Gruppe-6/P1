import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


diabetes_data = pd.read_csv("diabetes_dataset.csv") #Read CSV file

#Encoding of object to binary

ordinal_encoder = OrdinalEncoder()
diabetes_data["education_level_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["education_level"]])
diabetes_data["smoking_status_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["smoking_status"]])

#Encoding object to groups
onehot_encoder = OneHotEncoder(sparse_output=False)
ethnicity_one_hot = onehot_encoder.fit_transform(diabetes_data[["gender","ethnicity","employment_status"]])

ethnicity_one_hot_df = pd.DataFrame(ethnicity_one_hot,columns=onehot_encoder.get_feature_names_out(["gender","ethnicity","employment_status"]))
data_encoded = pd.concat([diabetes_data.drop(["gender","ethnicity","employment_status"],axis=1),ethnicity_one_hot_df],axis=1)
#Filter data to focus on only no diabetes, pre-diabetes and type 2
filtered_data = diabetes_data[~diabetes_data["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()


filtered_data["diabetes_multi"] = filtered_data["diabetes_stage"].replace({
    "No Diabetes": 0,
    "Pre-Diabetes": 1,
    "Type 2": 2
})



#filtered_data = filtered_data.dropna(subset=["diabetes_binary"])

#Define X and y values from features to highlight clinical data
X = filtered_data[["heart_rate", "glucose_fasting", "insulin_level", "hba1c"]]
y = filtered_data["diabetes_multi"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]


#Implement OneVSrestClassifier as logistical regression to classify multiclass
model_medicin = OneVsRestClassifier(LogisticRegression(max_iter=2000))


#y_score = model_medicin.fit(X, y_bin).predict_proba(X)
#y_pred = model_medicin.predict(X_test)


# --- Model ---
model_medicin = OneVsRestClassifier(LogisticRegression(max_iter=2000))
model_medicin.fit(X_train, y_train)

# --- Accuracy ---
y_pred = model_medicin.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (test data): {accuracy:.3f}")

# --- ROC ---
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = model_medicin.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Klasse {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-kurver for 3-klasses diabetes klassifikation (test data)")
plt.legend()
plt.show()