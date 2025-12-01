import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib.util
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

# ============================================================
# 4. TRAINING FUNCTION (LOGREG + THRESHOLD TUNING)
# ============================================================
def train_best_logreg(X, y, label):

    # ============================================================
    # Train-test split + scaling
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ============================================================
    # Logistic Regression Model
    # ============================================================
    model = LogisticRegression(
        C=5,
        class_weight="balanced",
        max_iter=500,
        penalty="l2",
        solver="saga"
    )

    model.fit(X_train_s, y_train)

    # Probability predictions
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # ============================================================
    # THRESHOLD TUNING
    # ============================================================
    thresholds = np.linspace(0.1, 0.9, 200)
    best_thresh = 0.5
    best_acc = 0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred_t)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Final prediction
    y_pred = (y_prob >= best_thresh).astype(int)

    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # ============================================================
    # PRINT RESULTS CLEARLY
    # ============================================================
    print("\n==============================================")
    print(f" RESULTS – {label}")
    print("==============================================")
    print(f"Optimized Threshold : {best_thresh:.3f}")
    print(f"Accuracy            : {acc:.3f} ({acc*100:.1f}%)")
    print(f"Precision           : {prec:.3f}")
    print(f"Recall (TPR)        : {rec:.3f}")
    print(f"ROC–AUC             : {roc_auc:.3f}")
    print(f"Correct predictions : {int(acc * len(X_test))} / {len(X_test)}")
    print("==============================================\n")

    # ============================================================
    # PLOT ROC CURVE
    # ============================================================
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {label}")
    plt.legend()
    plt.show()

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Flere hyperparametre end dine originale
    model = LogisticRegression(
        C=5,
        class_weight="balanced",
        max_iter=500,
        penalty="l2",
        solver="saga"
    )

    model.fit(X_train_s, y_train)

    # Probability predictions
    y_prob = model.predict_proba(X_test_s)[:, 1]

    # ======================================================
    # THRESHOLD TUNING (finder hvilket cut der giver bedste accuracy)
    # ======================================================
    thresholds = np.linspace(0.1, 0.9, 200)
    best_thresh = 0.5
    best_acc = 0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        acc = accuracy_score(y_test, y_pred_t)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Pred efter bedste threshold
    y_pred = (y_prob >= best_thresh).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"\n=== {label} ===")
    print(f"Optimized threshold: {best_thresh:.3f}")
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"→ {int(acc * len(X_test))} rigtige ud af {len(X_test)}")

    # ======================================================
    # ROC AUC
    # ======================================================
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.3f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {label}")
    plt.legend()
    plt.show()

    # ======================================================
    # Confusion matrix
    # ======================================================
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix – {label}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

from sklearn.model_selection import cross_val_score

def cross_validate_logreg(X, y, label):

    # Skal skalere X for hver fold, så vi bygger en pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=5,
            class_weight="balanced",
            max_iter=500,
            penalty="l2",
            solver="saga"
        ))
    ])

    print("\n==============================================")
    print(f" CROSS-VALIDATION RESULTS – {label}")
    print("==============================================")

    # Accuracy
    acc_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Accuracy (mean ± std) : {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

    # Recall
    rec_scores = cross_val_score(pipeline, X, y, cv=5, scoring="recall")
    print(f"Recall (mean ± std)   : {rec_scores.mean():.3f} ± {rec_scores.std():.3f}")

    # AUC
    auc_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")
    print(f"ROC–AUC (mean ± std)  : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    print("==============================================\n")

def build_screening_tool(X, y, label):
    """
    Træner model + scaler + threshold OG opbygger screening-funktionen.
    Returnerer (model, scaler, best_threshold, percentiles, X_test, risk_scores_test, y_test)
    som screening-værktøjet skal bruge.
    """

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model
    model = LogisticRegression(
        C=5,
        class_weight="balanced",
        max_iter=500,
        penalty="l2",
        solver="saga"
    )
    model.fit(X_train_s, y_train)

    # Test scores
    risk_scores = model.predict_proba(X_test_s)[:, 1]

    # Threshold tuning
    thresholds = np.linspace(0.1, 0.9, 200)
    best_thresh = 0.5
    best_acc = 0

    for t in thresholds:
        preds = (risk_scores >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    # Percentiler til risikogrupper
    p25 = np.percentile(risk_scores, 25)
    p50 = np.percentile(risk_scores, 50)
    p75 = np.percentile(risk_scores, 75)

    percentiles = (p25, p50, p75)

    print(f"\nScreening-model for {label} klar.")
    print(f"→ Optimal threshold = {best_thresh:.3f}")
    print(f"→ ROC AUC = {auc(*roc_curve(y_test, risk_scores)[:2]):.3f}")
    print("----------------------------------------------------")

    return model, scaler, best_thresh, percentiles, X_test, risk_scores, y_test




# ======================================================================
# 12. USER SCREENING FUNCTION – END USER TOOL
# ======================================================================

def run_screening(model, scaler, best_threshold, percentiles,
                  X_test, risk_scores_test, y_test, label):
    """
    Selve screening-værktøjet:
    - indsamler brugerinput
    - beregner risikoscore
    - klassificerer via threshold
    - finder risikogruppe
    - lokal modelnøjagtighed
    """

    print("\n============================================")
    print(f" DIABETES SCREENING – {label}")
    print("============================================")

    # ---- USER INPUT ----
    age = float(input("Alder (år): "))
    bmi = float(input("BMI: "))

    waist = float(input("Taljeomkreds i cm: "))
    hip = float(input("Hofteomkreds i cm: "))
    whr = waist / hip

    diet = float(input("Kostscore (0–10): "))
    activity = float(input("Fysisk aktivitet (min/uge): "))
    sleep = float(input("Søvn (timer/nat): "))
    smoke = float(input("Ryger? (0/1): "))
    alcohol = float(input("Alkohol (genstande/uge): "))
    family = float(input("Familiemedlem med diabetes? (0/1): "))

    # DataFrame i præcis samme rækkefølge som X_home
    user_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "waist_to_hip_ratio": whr,
        "diet_score": diet,
        "physical_activity_minutes_per_week": activity,
        "sleep_hours_per_day": sleep,
        "smoking_status_encoded": smoke,
        "alcohol_consumption_per_week": alcohol,
        "family_history_diabetes": family
    }])

    # Scaling
    user_scaled = scaler.transform(user_df)

    # Risikoscore
    risk = model.predict_proba(user_scaled)[0, 1]

    print(f"\nDin risikoscore: {risk:.3f} (0 = lav risiko, 1 = høj risiko)")

    # ---- CLASSIFICATION ----
    diagnosis = "Høj risiko (model vurderer diabetes)" if risk >= best_threshold else "Ingen diabetes-risiko"
    print(f"Modelklassifikation: {diagnosis}")

    # ---- RISK GROUP (percentiles) ----
    p25, p50, p75 = percentiles
    if risk < p25:
        group = "Low"
    elif risk < p50:
        group = "Moderate"
    elif risk < p75:
        group = "High"
    else:
        group = "Very High"

    print(f"Risikogruppe: {group}")

    ''' Dette stykke bruges til validering af bruger i forhold til datasæt ~ 58% acc 

    tolerance = 0.05
    low = risk - tolerance
    high = risk + tolerance

    similar = pd.DataFrame({
        "RiskScore": risk_scores_test,
        "TrueLabel": y_test
    })
    similar = similar[(similar["RiskScore"] >= low) & (similar["RiskScore"] <= high)]

    print("\n============================================")
    print(" LOKAL MODELVALIDERING – PATIENTER SOM DIG")
    print("============================================")

    if len(similar) == 0:
        print("Ingen lignende patienter i testdata → kan ikke beregne lokal nøjagtighed.")
    else:
        N = len(similar)
        pct_diabetes = similar["TrueLabel"].mean()
        pred_correct = (similar["RiskScore"] >= best_threshold).astype(int) == similar["TrueLabel"]
        pct_correct = pred_correct.mean()

        print(f"Antal lignende patienter : {N}")
        print(f"Andel med diabetes (faktisk) : {pct_diabetes:.2%}")
        print(f"Modelens lokale nøjagtighed  : {pct_correct:.2%}")

    print("============================================\n")

'''

train_best_logreg(X_home, y, "Hjemme data – Optimized Logistic Regression")
cross_validate_logreg(X_home, y, "Hjemme data – Optimized Logistic Regression")

train_best_logreg(X_clinical, y, "Kliniske data – Optimized Logistic Regression")
cross_validate_logreg(X_clinical, y, "Kliniske data – Optimized Logistic Regression")



model_h, scaler_h, thr_h, pct_h, Xth, Rth, Yth = build_screening_tool(X_home, y, "Home Data")
run_screening(model_h, scaler_h, thr_h, pct_h, Xth, Rth, Yth, "Home Data")
