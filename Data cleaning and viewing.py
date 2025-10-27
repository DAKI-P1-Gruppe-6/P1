import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv("diabetes_dataset.csv") #Read CSV file

#Encoding of object to binary

ordinal_encoder = OrdinalEncoder()
diabetes_data["education_level_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["education_level"]])
diabetes_data["smoking_status_encoded"] = ordinal_encoder.fit_transform(diabetes_data[["smoking_status"]])

onehot_encoder = OneHotEncoder(sparse_output=False)
ethnicity_one_hot = onehot_encoder.fit_transform(diabetes_data[["gender","ethnicity","employment_status"]])

ethnicity_one_hot_df = pd.DataFrame(ethnicity_one_hot,columns=onehot_encoder.get_feature_names_out(["gender","ethnicity","employment_status"]))
data_encoded = pd.concat([diabetes_data.drop(["gender","ethnicity","employment_status"],axis=1),ethnicity_one_hot_df],axis=1)


#Descripction of dataset: name, amount, datatype e.g
diabetes_data.head()
diabetes_data.info()
diabetes_data.describe()


data_encoded.iloc[:, 0:10].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()


data_encoded.iloc[:, 10:20].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

data_encoded.iloc[:, 20:30].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

data_encoded.iloc[:, 30:42].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

data_encoded.iloc[:, 40:50].hist(bins=50, figsize=(20, 15))
plt.tight_layout()
plt.show()

