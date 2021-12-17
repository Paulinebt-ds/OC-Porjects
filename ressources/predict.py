import json

from flask import Flask, request
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
from joblib import load
import pickle
import numpy as np
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        print(request)
        row = request.get_json()
        print(row)
        row = json.loads(row)
        print(row)
    row = {k: [v] for k, v in row.items()}
    print(row)
    data_client = pd.DataFrame.from_dict(row)
    data_client = data_client.drop(["TARGET", "SK_ID_CURR"], axis=1)
    cols = data_client.columns

    print(data_client.head())
    # calcul prédiction défaut et probabilité de défaut
    payload = data_client
    print("Payload, head :", payload.head())

    # Modèle sauvegardé par LGBM
    # Load the Model back from file
    LGBM_model = lgb.Booster(model_file="C:/Users/pbliv/Documents/Data Science/P7/model.bin")

    # Importation du scaler
    scaler = load("C:/Users/pbliv/Documents/Data Science/P7/scaler.joblib")
    scaler.clip = False
    print(scaler)

    # Importation de l'imputer
    imputer = load("C:/Users/pbliv/Documents/Data Science/P7/imputer.joblib")
    print(imputer)
    print(LGBM_model)

    # Imputation des valeurs manquantes
    payload = imputer.transform(payload)

    # Réduction
    payload = scaler.transform(payload)

    # Prédiction
    pred = LGBM_model.predict(payload)
    print(pred)
    pred_score = ''.join(str(e) for e in pred)
    # Affichage du résultat
    ## Sous format json
    req_data = {"ligne client": 2,
                "id": data_client.iloc[0, 0],
                "prediction": pred}

    print('Nouvelle Prédiction : \n', req_data)

    return pred_score



if __name__ == '__main__':
    app.run(debug=True)

