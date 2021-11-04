from flask import Flask, jsonify, request
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load
import csv

app = Flask(__name__)


@app.route('/form', methods=['GET', 'POST'])
def get_row():
    if request.method == 'POST':
        row = request.form.get('row')
        return row

    # otherwise handle the GET request
    return '''
               <form method="POST">
                   <div><label>Row: <input type="text" name="row"></label></div>
                   <input type="submit" value="Submit">
               </form>'''

# Chemin des fichiers
path = "C:/Users/pbliv/Documents/Data Science/P7"
#Ligne du client que l'on souhaite charger
## Ouverture du fichier
f = open("./app_train.csv", 'r')
## Lecture  du fichier
r = csv.reader(f, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
liste = list(r)
i = get_row()
data_client = liste[i]
data_client = pd.DataFrame(data_client)
# Fermeture du fichier
f.close()

# Dataframe pour les variables polynomiales
poly_features = data_client[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

# Imputation des valeurs manquantes par la moyenne
imputer = load("./imputer_poly.joblib")
poly_features = imputer.fit_transform(poly_features)

# Création des polynômes avec un degré spécifique
poly_transformer = PolynomialFeatures(degree=3)

# Entraînement sur les variables polynomiales
poly_transformer.fit(poly_features)
# Transformation des variables
poly_features = poly_transformer.transform(poly_features)

# Concaténation des variables polynamiales avec l'ensemble d'entraînement
poly_features['SK_ID_CURR'] = data_client['SK_ID_CURR']
app_train_poly = data_client.merge(poly_features, on='SK_ID_CURR', how='left')

# Variables métier
data_domain = app_train_poly.copy()

data_domain["CREDIT_INCOME_PERCENT"] = data_domain['AMT_CREDIT'] / data_domain['AMT_INCOME_TOTAL']
data_domain["ANNUITY_INCOME_PERCENT"] = data_domain['AMT_ANNUITY'] / data_domain['AMT_INCOME_TOTAL']
data_domain["CREDIT_TERM"] = data_domain['AMT_ANNUITY'] / data_domain['AMT_CREDIT']
data_domain["DAYS_EMPLOYED_PERCENT"] = data_domain['DAYS_EMPLOYED'] / data_domain['DAYS_BIRTH']

@app.route('/predict', methods = ['POST'])
def predict():
    # calcul prédiction défaut et probabilité de défaut

    # Modèle sauvegardé via joblib
    my_model = load("C:/Users/pbliv/Documents/Data Science/P7/mymodel.joblib")
    payload = data_domain
    pred = my_model.predict(payload)
    req_data = {"row": i,
                "id": data_client.loc[:, 1],
                "prediction": pred}
    id_client = data_client.loc[:, 1]
    print('Nouvelle Prédiction : \n', req_data)
    return '''
                      <h1>Ligne du client choisi: {}</h1>
                      <h1>ID du client : {}</h1>
                      <h1>Prédiction du client : {}</h1>'''.format(i, id_client, pred)
    #return jsonify(req_data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

