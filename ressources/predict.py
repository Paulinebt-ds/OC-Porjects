import json

from flask import Flask, request
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from ressources.components.functions import lime_explain
from ressources.pages.layout_client import lime_explainer, predict_method
import lightgbm as lgb
import lime
from lime.lime_tabular import LimeTabularExplainer
from joblib import load
import pickle
import numpy as np
import requests
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
    print(type(payload))
    print(payload)
    print(payload[0].shape)

    # Prédiction
    pred = LGBM_model.predict(payload)
    print(pred)
    pred_score = ''.join(str(e) for e in pred)
    payload = pd.DataFrame(payload, columns=cols)
    payload["score_pred"] = pred_score
    payload = payload.to_json()
    return payload
    # Affichage du résultat
    ## Sous format json
    req_data = {"ligne client": 2,
                "id": data_client.iloc[0, 0],
                "prediction": pred}
    print(payload)
    print('Nouvelle Prédiction : \n', req_data)

@app.route('/lime', methods=['POST'])
def lime_test():
    print(request)
    row = request.get_json()
    print("Réponse de l'API de prédiction: ", row)
    row = json.loads(row)
    print(row)
    print(type(row))
    row = {k: [v] for k, v in row.items()}
    print(row)
    data = pd.DataFrame.from_dict(row)
    data.drop(columns="score_pred", axis=1, inplace=True)
    print("Data pour lime_explainer :",data)
    data = np.array(data)
    print(data)
    print(data.shape)
    print(data[0])
    print(data[0].shape)
    n_samples = 2
    exp = lime_explain(lime_explainer, data[0], predict_method, n_samples)
    print("Type de l'explain_instance : ",type(exp))
    exp_list = exp.as_list()
    print(type(exp_list))
    print("Explain_instance sous forme de liste :", exp_list)
    exp_json = json.dumps(dict(exp_list))

    return exp_json



    #return pred_score




if __name__ == '__main__':
    app.run(debug=True)

