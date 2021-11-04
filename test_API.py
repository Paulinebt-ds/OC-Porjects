# Import des librairies nécéssaires
from sklearn.preprocessing import LabelEncoder
import json

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
from joblib import load
import numpy as np
import requests

# Importation de la base "application_train"
df = pd.read_csv("C:/Users/pbliv/Documents/Data Science/P7/application_train.csv")

# Suppression des colonnes vides, s'il y en a
if df.shape[1] >= 122:
    if "" in df.columns:
        df.drop("", axis=1, inplace=True)

# Transformation en nombre entier des colonnes DAYS
df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])
df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])

# Suppression de la TARGET
df.drop(['TARGET'], axis=1, inplace=True)

# print(df.head())
# print(df.shape)

# Création de l'objet "LabelEncoder()"
le = LabelEncoder()

# Si une colonne a au plus deux labels, on applique le LabelEncoder
for col in df:
    if df[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df[col].unique())) <= 2:
            # Train on the training data
            le.fit(df[col])
            # Transform training data
            df[col] = le.transform(df[col])

#print(df.shape)
# One-hot encoding (dichotomisation) pour les variables catégorielles

df = pd.get_dummies(df)
#print(df.shape)


df.drop(['CODE_GENDER_XNA', 'NAME_FAMILY_STATUS_Unknown',
       'NAME_INCOME_TYPE_Maternity leave'], axis=1, inplace=True)

# Remplacement des anomalies par des valeurs manquantes
df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

# Dataframe pour les variables polynomiales
poly_features = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# Imputation des valeurs manquantes des fonctions polynomiales par la moyenne
imputer = load("C:/Users/pbliv/Documents/Data Science/P7/imputer_poly.joblib")
poly_imputer = imputer.transform(poly_features)

# Création des polynômes avec un degré spécifique
poly_transformer = PolynomialFeatures(degree=3)
# Entraînement sur les variables polynomiales
poly_transformer.fit(poly_imputer)
# Transformation des variables
poly_features = poly_transformer.transform(poly_imputer)
print('Polynomial Features shape: ', poly_features.shape)

# get_features_names nous permet de récupérer toutes les variables polynomiales créées
# Create a dataframe of the features
poly_features = pd.DataFrame(poly_features,
                             columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                         'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Concaténation des variables polynamiales avec l'ensemble d'entraînement
poly_features['SK_ID_CURR'] = df['SK_ID_CURR']
app_train_poly = df.merge(poly_features, on='SK_ID_CURR', how='left', suffixes=('', '_y'))
app_train_poly.drop(app_train_poly.filter(regex='_y$').columns.tolist(), axis=1, inplace=True)

# Variables métier
data_domain = app_train_poly.copy()

data_domain["CREDIT_INCOME_PERCENT"] = data_domain['AMT_CREDIT'] / data_domain['AMT_INCOME_TOTAL']
data_domain["ANNUITY_INCOME_PERCENT"] = data_domain['AMT_ANNUITY'] / data_domain['AMT_INCOME_TOTAL']
data_domain["CREDIT_TERM"] = data_domain['AMT_ANNUITY'] / data_domain['AMT_CREDIT']
data_domain["DAYS_EMPLOYED_PERCENT"] = data_domain['DAYS_EMPLOYED'] / data_domain['DAYS_BIRTH']

df = data_domain.loc[2, :]
print(data_domain.shape)
data_domain.to_csv("C:/Users/pbliv/Documents/Data Science/P7/data_domain.csv")
json_df = df.to_json()
print(json_df)
url = "http://127.0.0.1:5000/predict"
r = requests.post(url, json=json_df)
print(r.text)