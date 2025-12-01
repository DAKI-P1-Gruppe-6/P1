import pandas as pd
import numpy as np
from sklearn.utils import resample  # TILFØJET - skal være i starten
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os


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
# 5. XGBoost EVALUATION (med robust fejlhåndtering)
# ============================================================
def train_and_evaluate_xgboost(X, y, label):
    """
    Træn og evaluer XGBoost model med robust fejlhåndtering for NaN-værdier.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # DEBUG: Vis test set fordeling
    print(f"\n{label} - Test set fordeling: 0={sum(y_test==0)}, 1={sum(y_test==1)}")
    
    # Scaling (XGBoost fungerer med StandardScaler)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Definer XGBoost model med dine specifikke hyperparametre
    model = XGBClassifier(
        n_estimators=50,
        gamma=0,
        max_depth=3,
        subsample=0.8,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Træn modellen
    model.fit(X_train_s, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]  # Probability for class 1
    
    # DEBUG: Vis prediction fordeling
    unique_preds, counts_preds = np.unique(y_pred, return_counts=True)
    print(f"{label} - Prediction fordeling: {dict(zip(unique_preds, counts_preds))}")
    
    # Confusion matrix for debugging
    cm = confusion_matrix(y_test, y_pred)
    print(f"{label} - Confusion Matrix:\n{cm}")
    
    # Metrics med robust håndtering af edge cases
    acc = accuracy_score(y_test, y_pred)
    
    # Robust precision og recall beregning
    try:
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
    except Exception as e:
        print(f"⚠️  Fejl i precision/recall beregning: {e}")
        # Fallback beregning baseret på confusion matrix
        tn, fp, fn, tp = cm.ravel()
        if tp + fp == 0:  # Ingen positive predictioner
            prec = 0.0
        else:
            prec = tp / (tp + fp)
        
        if tp + fn == 0:  # Ingen positive true labels
            rec = 0.0
        else:
            rec = tp / (tp + fn)
    
    print()
    print(f"{label}: {model.__class__.__name__}")
    print(f"Hyperparametre: n_estimators=50, gamma=0.5, max_depth=4, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1")
    print(f"Accuracy: {acc:.1%}")
    print(f"Precision:     {prec:.3f}")
    print(f"Recall:        {rec:.3f}")
    print(f"→ {int(acc * len(X_test))} rigtige ud af {len(X_test)} patienter")
    
    # Returner model og scaler for ROC plotting
    return model, scaler, X_test_s, y_test

def cross_validate_xgboost(X, y, label):
    """
    Cross-validation som i den gamle model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBClassifier(
        n_estimators=50,
        gamma=0.5,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    print()
    print(f"{label} ({model.__class__.__name__})")
    print(f"Hyperparametre: n_estimators=50, gamma=0.5, max_depth=4, subsample=0.8, colsample_bytree=0.8, learning_rate=0.1")
    print(f"Cross-val accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ============================================================
# 6. RUN MODELS (minimalistisk - kun ROC curve)
# ============================================================
print("\n" + "="*60)
print("XGBoost EVALUERING MED BALANCERING OG PRECISION/RECALL FOKUS")
print("="*60)

# Træn og evaluer (gem resultater til ROC)
results_home = train_and_evaluate_xgboost(X_home, y, "Hjemme-data")
results_clinical = train_and_evaluate_xgboost(X_clinical, y, "Kliniske data")

# Cross-validation
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTATER")
print("="*60)
cross_validate_xgboost(X_home, y, "Hjemme-data")
cross_validate_xgboost(X_clinical, y, "Kliniske data")

# ROC Curve plotting (kun én graf som ønsket)
model_home, scaler_home, X_test_home, y_test_home = results_home
model_clinical, scaler_clinical, X_test_clinical, y_test_clinical = results_clinical

plt.figure(figsize=(8, 6))

# ROC for hjemme-data
y_score_home = model_home.predict_proba(X_test_home)[:, 1]
fpr_home, tpr_home, _ = roc_curve(y_test_home, y_score_home)
roc_auc_home = auc(fpr_home, tpr_home)
plt.plot(fpr_home, tpr_home, lw=2, label=f"Hjemme-data (AUC={roc_auc_home:.3f})", color='blue')

# ROC for kliniske data
y_score_clinical = model_clinical.predict_proba(X_test_clinical)[:, 1]
fpr_clinical, tpr_clinical, _ = roc_curve(y_test_clinical, y_score_clinical)
roc_auc_clinical = auc(fpr_clinical, tpr_clinical)
plt.plot(fpr_clinical, tpr_clinical, lw=2, label=f"Kliniske data (AUC={roc_auc_clinical:.3f})", color='red')

# Diagonal linje
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – XGBoost Binary Classification (HbA1c ≥ 48 mmol/mol)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# SUMMARY (valgfri - opsummering af resultater)
# ============================================================
print("\n" + "="*60)
print("SAMMENLIGNING AF FEATURE SETS")
print("="*60)

# Robust beregning for summary
y_pred_home = model_home.predict(X_test_home)
y_pred_clinical = model_clinical.predict(X_test_clinical)

acc_home = accuracy_score(y_test_home, y_pred_home)
prec_home = precision_score(y_test_home, y_pred_home, zero_division=0)
rec_home = recall_score(y_test_home, y_pred_home, zero_division=0)

acc_clinical = accuracy_score(y_test_clinical, y_pred_clinical)
prec_clinical = precision_score(y_test_clinical, y_pred_clinical, zero_division=0)
rec_clinical = recall_score(y_test_clinical, y_pred_clinical, zero_division=0)

print(f"Hjemme-data:     Accuracy {acc_home:.1%} | Precision {prec_home:.3f} | Recall {rec_home:.3f} | AUC {roc_auc_home:.3f}")
print(f"Kliniske data:   Accuracy {acc_clinical:.1%} | Precision {prec_clinical:.3f} | Recall {rec_clinical:.3f} | AUC {roc_auc_clinical:.3f}")

# ============================================================
# XG boost end
# ============================================================




############################################################################
# PyQt
############################################################################

from PyQt5.QtWidgets import (QApplication, QWidget, QFormLayout, QSpinBox, QPushButton,QVBoxLayout, QLabel, QHBoxLayout, QGroupBox, QComboBox, QDoubleSpinBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from sklearn.preprocessing import OrdinalEncoder
import sys
import pandas as pd

class lineEditDemo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Window size
        self.resize(900, 300)

        ############################################################################
        # Right box (BMI calculator)
        ############################################################################

        # Age input (right box)
        self.e1 = QSpinBox()
        self.e1.setRange(0, 200)
        self.e1.setAlignment(Qt.AlignLeft)
        self.e1.setFont(QFont("Arial", 20))
        self.e1.setMinimumHeight(25)
        self.e1.setMinimumWidth(100)

        # Height input (right box)
        self.e2 = QSpinBox()
        self.e2.setRange(0, 300)
        self.e2.setAlignment(Qt.AlignLeft)
        self.e2.setFont(QFont("Arial", 20))
        self.e2.setSuffix(" cm")
        self.e2.setMinimumHeight(25)
        self.e2.setMinimumWidth(100)

        # Weight input (right box)
        self.e3 = QSpinBox()
        self.e3.setRange(0, 500)
        self.e3.setAlignment(Qt.AlignLeft)
        self.e3.setFont(QFont("Arial", 20))
        self.e3.setSuffix(" kg")
        self.e3.setMinimumHeight(25)
        self.e3.setMinimumWidth(100)

        # BMI result label (right box)
        self.bmi_label = QLabel("BMI: ")
        self.bmi_label.setFont(QFont("Arial", 15))
        self.bmi_label.setAlignment(Qt.AlignLeft)

        # Form layout for right box (age, height, weight)
        form_layout = QFormLayout()
        form_layout.addRow("Age", self.e1)
        form_layout.addRow("Height", self.e2)
        form_layout.addRow("Weight", self.e3)
        form_layout.setLabelAlignment(Qt.AlignLeft)

        # Submit button (calculate BMI)
        submit_btn = QPushButton("Submit")
        submit_btn.setFont(QFont("Arial", 16))
        submit_btn.setMinimumWidth(170)
        submit_btn.clicked.connect(self.on_submit)

        # Title label inside right box
        title_label = QLabel("BMI calculator")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignLeft)

        # Layout inside right box
        box_inner_layout = QVBoxLayout()
        box_inner_layout.addWidget(title_label)          # Title at the top
        box_inner_layout.addLayout(form_layout)          # Age / Height / Weight
        box_inner_layout.addWidget(submit_btn, alignment=Qt.AlignLeft)
        box_inner_layout.addWidget(self.bmi_label, alignment=Qt.AlignLeft)

        # Right group box
        right_box = QGroupBox()
        right_box.setLayout(box_inner_layout)
        right_box.setMaximumWidth(260)

        ############################################################################
        # Left box (patient data)
        ############################################################################

        # Title inside left box
        left_title = QLabel("Patient data")
        left_title.setFont(QFont("Arial", 20, QFont.Bold))
        left_title.setAlignment(Qt.AlignLeft)

        # Age input (left box)
        self.txt_age = QSpinBox()
        self.txt_age.setRange(0, 200)
        self.txt_age.setAlignment(Qt.AlignLeft)
        self.txt_age.setFont(QFont("Arial", 20))
        self.txt_age.setMinimumHeight(25)
        self.txt_age.setMinimumWidth(100)

        # BMI input (left box)
        self.txt_bmi = QDoubleSpinBox()
        self.txt_bmi.setRange(0.0, 100.0)
        self.txt_bmi.setDecimals(1)
        self.txt_bmi.setSingleStep(0.1)
        self.txt_bmi.setAlignment(Qt.AlignLeft)
        self.txt_bmi.setFont(QFont("Arial", 20))
        self.txt_bmi.setMinimumHeight(25)
        self.txt_bmi.setMinimumWidth(100)
        self.txt_bmi.setSpecialValueText("0")
        self.txt_bmi.setValue(0.0)

        # Waist-hip ratio input
        self.txt_waist_hip = QDoubleSpinBox()
        self.txt_waist_hip.setRange(0, 500)
        self.txt_waist_hip.setDecimals(2)
        self.txt_waist_hip.setSingleStep(0.1)
        self.txt_waist_hip.setAlignment(Qt.AlignLeft)
        self.txt_waist_hip.setFont(QFont("Arial", 20))
        self.txt_waist_hip.setMinimumHeight(25)
        self.txt_waist_hip.setMinimumWidth(100)
        self.txt_waist_hip.setSpecialValueText("0")
        self.txt_waist_hip.setValue(0.0)

        # Physical activity input (minutes per week)
        self.txt_physical_activity = QSpinBox()
        self.txt_physical_activity.setRange(0, 200)
        self.txt_physical_activity.setAlignment(Qt.AlignLeft)
        self.txt_physical_activity.setFont(QFont("Arial", 20))
        self.txt_physical_activity.setMinimumHeight(25)
        self.txt_physical_activity.setMinimumWidth(100)

        # Sleep input (hours per day)
        self.txt_sleep = QSpinBox()
        self.txt_sleep.setRange(0, 24)
        self.txt_sleep.setAlignment(Qt.AlignLeft)
        self.txt_sleep.setFont(QFont("Arial", 20))
        self.txt_sleep.setMinimumHeight(25)
        self.txt_sleep.setMinimumWidth(100)

        # Alcohol input (per week)
        self.txt_alcohol = QSpinBox()
        self.txt_alcohol.setRange(0, 200)
        self.txt_alcohol.setAlignment(Qt.AlignLeft)
        self.txt_alcohol.setFont(QFont("Arial", 20))
        self.txt_alcohol.setMinimumHeight(25)
        self.txt_alcohol.setMinimumWidth(100)

        # Diet score dropdown (1–10)
        self.combo_diet_score = QComboBox()
        self.combo_diet_score.setFont(QFont("Arial", 18))
        self.combo_diet_score.setMinimumHeight(25)
        self.combo_diet_score.setMinimumWidth(110)
        self.combo_diet_score.setMaximumWidth(110)
        self.combo_diet_score.addItems(["1","2","3","4","5","6","7","8","9","10"])

        # Smoking status dropdown
        self.combo_smoking_status = QComboBox()
        self.combo_smoking_status.setFont(QFont("Arial", 18))
        self.combo_smoking_status.setMinimumHeight(25)
        self.combo_smoking_status.setMinimumWidth(110)
        self.combo_smoking_status.setMaximumWidth(110)
        self.combo_smoking_status.addItems(["Never","Former","Current"])

        # Family history of diabetes dropdown
        self.combo_family_history = QComboBox()
        self.combo_family_history.setFont(QFont("Arial", 18))
        self.combo_family_history.setMinimumHeight(25)
        self.combo_family_history.setMaximumWidth(110)
        self.combo_family_history.setMinimumWidth(110)
        self.combo_family_history.addItems(["Yes","No"])

        # Form layout for left box (patient data)
        left_form_layout = QFormLayout()
        left_form_layout.addRow("Age", self.txt_age)
        left_form_layout.addRow("BMI", self.txt_bmi)
        left_form_layout.addRow("Waist-hip ratio (cm)", self.txt_waist_hip)
        left_form_layout.addRow("Physical activity (min/week)", self.txt_physical_activity)
        left_form_layout.addRow("Sleep (hours/day)", self.txt_sleep)
        left_form_layout.addRow("Alcohol (per week)", self.txt_alcohol)
        left_form_layout.addRow("Diet score (1–10)", self.combo_diet_score)
        left_form_layout.addRow("Smoking status", self.combo_smoking_status)
        left_form_layout.addRow("Family history of diabetes", self.combo_family_history)

        left_form_layout.setLabelAlignment(Qt.AlignLeft)
        left_form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        # Layout inside left box
        left_inner_layout = QVBoxLayout()
        left_inner_layout.addWidget(left_title)
        left_inner_layout.addSpacing(5)
        left_inner_layout.addLayout(left_form_layout)

        # Submit-knap nederst i venstre felt
        left_submit_btn = QPushButton("Submit")
        left_submit_btn.setFont(QFont("Arial", 16))
        left_submit_btn.setMinimumWidth(285)
        left_submit_btn.clicked.connect(self.on_left_submit)
        left_inner_layout.addWidget(left_submit_btn, alignment=Qt.AlignLeft)

        # Left group box
        left_box = QGroupBox()
        left_box.setLayout(left_inner_layout)

        ############################################################################
        # Middle box
        ############################################################################
        middle_box = QGroupBox()

        middle_label = QLabel("Results")
        middle_label.setFont(QFont("Arial", 20, QFont.Bold))
        middle_label.setAlignment(Qt.AlignLeft)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(middle_label)

        middle_box.setLayout(middle_layout)

        ############################################################################
        # Main layout: left, middle and right box
        ############################################################################

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_box)      # Left box 
        main_layout.addWidget(middle_box)    # Middle box
        main_layout.addWidget(right_box)     # Right box 

        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(1, 1)

        self.setLayout(main_layout)
        self.setWindowTitle("Diabetes Screening Tool")

        # Saves patient data
        self.patient_data = None

    def on_submit(self):
        """Called when the Submit button (right box) is pressed"""
        height_cm = self.e2.value()
        weight_kg = self.e3.value()

        # BMI calculation
        if height_cm > 0:
            height_m = height_cm / 100
            bmi = weight_kg / (height_m ** 2)
            bmi_text = f"{bmi:.1f}"
        else:
            bmi_text = "No input detected"

        # Update BMI label
        self.bmi_label.setText(f"BMI: {bmi_text}")

    def on_left_submit(self):
        """Saves all inputs from left box"""
        self.patient_data = {
            "age": self.txt_age.value(),
            "bmi": self.txt_bmi.value(),
            "waist_hip": self.txt_waist_hip.value() / 100,
            "diet_score": int(self.combo_diet_score.currentText()),
            "physical_activity": self.txt_physical_activity.value(),
            "sleep": self.txt_sleep.value(),
            "smoking_status": self.combo_smoking_status.currentText(),
            "alcohol": self.txt_alcohol.value(),
            "family_history": self.combo_family_history.currentText(),
        }

        self.smoking_encoder = OrdinalEncoder(categories=[["Never","Former","Current"]])
        self.family_encoder = OrdinalEncoder(categories=[["Yes","No"]])

        homedata = list(self.patient_data.values())

        self.patient_data = pd.DataFrame([self.patient_data])

        smoking_map = {"Never": 0, "Former": 1, "Current": 2}
        family_map = {"No": 0, "Yes": 1}

        self.patient_data["smoking_status"] = (self.patient_data["smoking_status"].map(smoking_map).astype(int))
        self.patient_data["family_history"] = (self.patient_data["family_history"].map(family_map).astype(int))

        self.homedata = self.patient_data.iloc[0].tolist()
        print(self.homedata)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = lineEditDemo()
    win.show()
    sys.exit(app.exec_())

############################################################################
# PyQt end
############################################################################