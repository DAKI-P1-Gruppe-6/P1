
import pandas as pd
import matplotlib.pyplot as mpl
import importlib.util
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score,roc_curve,accuracy_score,recall_score,auc
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# ============================================================
# IMPORT DATA FROM Data procesing.py
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_processing_path = os.path.join(script_dir, "..", "Data procesing.py")
spec = importlib.util.spec_from_file_location("data_processing", data_processing_path)
data_processing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_processing)

X_home = data_processing.X_home
X_clinical = data_processing.X_clinical
y = data_processing.y
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

