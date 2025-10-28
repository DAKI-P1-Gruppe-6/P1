import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- Setup ---
seed = random.randint(0, 100)
print(f"Seed: {seed}")

# --- Load data ---
diabetes_data = pd.read_csv("diabetes_dataset.csv")

# --- Encoding ---
ordinal_encoder = OrdinalEncoder()
diabetes_data["education_level_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["education_level"]])
diabetes_data["smoking_status_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["smoking_status"]])

onehot_encoder = OneHotEncoder(sparse_output=False)
ethnicity_one_hot = onehot_encoder.fit_transform(diabetes_data[["gender", "ethnicity", "employment_status"]])
ethnicity_one_hot_df = pd.DataFrame(
    ethnicity_one_hot,
    columns=onehot_encoder.get_feature_names_out(["gender", "ethnicity", "employment_status"])
)
data_encoded = pd.concat(
    [diabetes_data.drop(["gender", "ethnicity", "employment_status"], axis=1),
     ethnicity_one_hot_df],
    axis=1
)

# --- Filter kun No Diabetes og Type 2 ---
filtered_data = data_encoded[
    diabetes_data["diabetes_stage"].isin(["No Diabetes", "Type 2"])
].copy()

# --- Lav binær target ---
filtered_data["diabetes_binary"] = filtered_data["diabetes_stage"].replace({
    "No Diabetes": 0,
    "Type 2": 1
})

# --- Features og labels ---
#X = filtered_data[["heart_rate", "glucose_fasting", "insulin_level", "hba1c"]]
X = filtered_data[["age", "diet_score", "bmi", "smoking_status_encoded", "waist_to_hip_ratio", "sleep_hours_per_day"]]
y = filtered_data["diabetes_binary"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#Angiver k-værdi
k = 199

#Fitter modellen
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

#Laver prediction på modellen
y_pred = model.predict(X_test)

# 7. Beregner accuracy
accuracy = accuracy_score(y_test, y_pred)

# 8. Print resultatet
print(f"KNN med k = {k}")
print(f"Accuracy: {accuracy:.1%}")
print(f"→ {int(accuracy * len(X_test))} rigtige ud af {len(X_test)} patienter")