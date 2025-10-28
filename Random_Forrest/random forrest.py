import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


filtered_data["diabetes_binary"] = filtered_data["diabetes_stage"].replace({
    "No Diabetes": 0,
    "Pre-Diabetes": 1,
    "Type 2": 2
})


X = filtered_data [["age", "diet_score", "bmi", "smoking_status_encoded", "waist_to_hip_ratio", "sleep_hours_per_day"]]
y = filtered_data ["diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# tree_clf.fit(X, y)
# Decision_Tree_Classifier = DecisionTreeClassifier(max_features=5, max_leaf_nodes=4 )
# Decision_Tree_Classifier.fit(X_train, y_train)

# Forudsig og beregn accuracy
# y_predict = Decision_Tree_Classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_predict)
# print(f"accuracy er for decision tree er {accuracy*100:.2f}%")

print("hej")

Random_Forest_Classifier = RandomForestClassifier(n_estimators=100, random_state= 42 )
Random_Forest_Classifier.fit(X_train, y_train)
y_predict = Random_Forest_Classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"acuuracy for real {accuracy*100:.2f}%")

####################### random forrest
best_accuracy = 0
best_n = 1  # Start fx med 1 (kan ikke være 0)
print (1)
# for n in range(1, 20):  # min_samples_leaf må ikke være 0, så start på 1
for n in range(2, 100):  # min_samples_leaf må ikke være 0, så start på 1
    Random_Forest_Classifier = RandomForestClassifier(n_estimators=100, max_leaf_nodes=n, random_state= 42  )
    Random_Forest_Classifier.fit(X_train, y_train)
    y_predict = Random_Forest_Classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n = n
print(f"accuracy er for random forrest er {accuracy*100:.2f}%")
# print(f"Bedste min_samples_leaf er {best_n} med accuracy på {best_accuracy*100:.2f}%")