
import pandas as pd
import matplotlib.pyplot as mpl

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score,roc_curve,accuracy_score,recall_score,auc
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint



dataset = pd.read_csv("diabetes_dataset.csv")

dataset[["education_level_encoded"]]= OrdinalEncoder().fit_transform(dataset[["education_level"]])
dataset[["smoking_status_encoded"]]= OrdinalEncoder().fit_transform(dataset[["smoking_status"]])

onehot = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
data_onehot = onehot.fit_transform(dataset[["gender","ethnicity","employment_status"]])
data_onehot_pd = pd.DataFrame(data_onehot,columns=onehot.get_feature_names_out(["gender","ethnicity","employment_status"]))
dataset = pd.concat([dataset.drop(["gender","ethnicity","employment_status"],axis=1),data_onehot_pd],axis=1)

data = dataset[~dataset["diabetes_stage"].isin(["Type 1", "Gestational"])].copy()
dataset = dataset.dropna(subset=["hba1c"])

dataset["hba1c_mmolmol"] = 10.93 * dataset["hba1c"] - 23.5
# Drop rows where hba1c_class <48 AND diabetes_stage == Type 2
dataset = dataset[~((dataset["hba1c_mmolmol"] < 48) & (dataset["diabetes_stage"] == "Type 2"))]
dataset = dataset[~((dataset["hba1c_mmolmol"] > 48) & 
                    ((dataset["diabetes_stage"] == "Pre-Diabetes") | 
                     (dataset["diabetes_stage"] == "No Diabetes")))]

dataset["hba1c_class"] = (dataset["hba1c_mmolmol"]>=48).astype(int)
X_home = dataset[
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

X_clinical = dataset[["glucose_fasting", "insulin_level", "heart_rate"]]

y= dataset["hba1c_class"]
from scipy.stats import randint, uniform
import seaborn as sns


# Add the target to homemade features to see correlations with the label
home_data_with_target = X_home.copy()
home_data_with_target["hba1c_class"] = y

# Compute correlation matrix
corr_home = home_data_with_target.corr()

# Plot the heatmap
mpl.figure(figsize=(12, 10))
sns.heatmap(corr_home, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
mpl.title("Correlation Matrix – Home Data Features + Target")
mpl.show()

param_dist = {
    "max_depth": randint(3, 40),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "max_leaf_nodes": randint(5, 50),
    "criterion": ["gini", "entropy"],
    "min_impurity_decrease": uniform(0.0, 0.05)  # random float between 0.0 and 0.05
}


def evaluation_random_search(X, y, label, n_iter=500, cv=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    dtree = DecisionTreeClassifier(random_state=42)

    # Randomized Search
    random_search = RandomizedSearchCV(
        estimator=dtree,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    print(f"\n{label} - Best Parameters: {random_search.best_params_}")

    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"\n{'='*40}")
    print(f"{label} RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy:      {acc:.3f}")
    print(f"Precision:     {prec:.3f}")
    print(f"Recall:        {rec:.3f}")
    print(f"ROC-AUC:       {roc_auc:.3f}")

    mpl.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")


# ============================================================
# 6. RUN MODELS
# ============================================================
mpl.figure(figsize=(6, 4))
evaluation_random_search(X_home, y, "Home Data")
evaluation_random_search(X_clinical, y, "Clinical Data")

mpl.plot([0, 1], [0, 1], "k--")
mpl.xlabel("False Positive Rate")
mpl.ylabel("True Positive Rate")
mpl.title("ROC Curve – Decision Tree with Random Search")
mpl.legend()
mpl.tight_layout()
mpl.show()

