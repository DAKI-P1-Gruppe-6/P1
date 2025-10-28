import pandas as pd
import matplotlib as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

DD = pd.read_csv("diabetes_dataset.csv")

#                               LISTER OG VARIABLER
home_feat = ["age",
             "gender",
             "bmi",
             "waist_to_hip_ratio",
             "ethnicity",
             "smoking_status",
             "alcohol_consumption_per_week",
             "physical_activity_minutes_per_week",
             "sleep_hours_per_day",
             "family_history_diabetes",
              "heart_rate" ]

num_feat = ["age",
    "bmi",
    "waist_to_hip_ratio",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "sleep_hours_per_day",
    "heart_rate"]
cat_feat = ["gender",
    "ethnicity",
    "smoking_status",
    "family_history_diabetes"]

X = DD[home_feat]
y = DD["diagnosed_diabetes"]

#                                 Pipeline
num_pipe = Pipeline([('standard', StandardScaler()),])

cat_pipe = Pipeline([("one_hot", OneHotEncoder(handle_unknown='ignore'))])

preprocesser = ColumnTransformer([('num',num_pipe,num_feat),('cat',cat_pipe,cat_feat)])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train_pre = preprocesser.fit_transform(X_train)
X_test_pre = preprocesser.transform(X_test)

#                                Linear SVC

lin_svc = LinearSVC(random_state=42,max_iter=2000)

lin_svc.fit(X_train_pre,y_train)

lin_y_pred = lin_svc.predict(X_test_pre)

#                                   Linear Scores
lin_svc_recall_score = recall_score(y_test, lin_y_pred)
lin_svc_accuracy_score = accuracy_score(y_test, lin_y_pred)
lin_svc_precision_score = precision_score(y_test, lin_y_pred)

print(f"Accuracy for LinearSVC:{lin_svc_accuracy_score}")
print(f"Precision for LinearSVC:{lin_svc_precision_score}")
print(f"Recall for LinearSVC:{lin_svc_recall_score}")

#                               SVC (Kernel)

rbf_svc = SVC(kernel="rbf", random_state=42, max_iter=20000)

rbf_svc.fit(X_train_pre,y_train)

rbf_pred = rbf_svc.predict(X_test_pre)

#                                   SVC (Kernel) Scores

rbf_svc_recall_score = recall_score(y_test, rbf_pred)
rbf_svc_accuracy_score = accuracy_score(y_test, rbf_pred)
rbf_svc_precision_score = precision_score(y_test, rbf_pred)

print(f"Accuracy for RBF SVC:{rbf_svc_accuracy_score}")
print(f"Precision for RBF SVC:{rbf_svc_precision_score}")
print(f"Recall for RBF SVC:{rbf_svc_recall_score}")