import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score

# --- Load dataset ---
diabetes_data = pd.read_csv("diabetes_dataset.csv")

# --- Encode categorical columns ---
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
    [diabetes_data.drop(["gender","ethnicity","employment_status"], axis=1), ethnicity_one_hot_df],
    axis=1
)

# --- Filter only No Diabetes, Pre-Diabetes, Type 2 ---
filtered_data = diabetes_data[~diabetes_data["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
filtered_data["diabetes_multi"] = filtered_data["diabetes_stage"].replace({
    "No Diabetes": 0,
    "Pre-Diabetes": 1,
    "Type 2": 2
})

# --- Select features and target ---
X = filtered_data[[
    "age",
    "bmi",
    "waist_to_hip_ratio",
    "alcohol_consumption_per_week",
    "physical_activity_minutes_per_week",
    "sleep_hours_per_day",
    "family_history_diabetes",
    "heart_rate"
]]
y = filtered_data["diabetes_multi"]

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Hyperparameter tuning ---
base_clf = DecisionTreeClassifier(random_state=42)
ovr_clf = OneVsRestClassifier(base_clf)

param_grid = {
    "estimator__max_depth": [5, 10, 15, 20, None],
    "estimator__min_samples_split": [2, 5, 10, 20],
    "estimator__max_leaf_nodes": [None, 5, 10, 20],
    "estimator__criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(ovr_clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# --- Evaluate on test set ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy with best hyperparameters: {accuracy:.3f}")

# --- ROC curves ---
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = best_model.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(8,6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves for 3-class diabetes classification (test data)")
plt.legend()
plt.show()
