from flask import Flask, request
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_package.functions import *
from dash_package import *

import plotly.express as px
import plotly.graph_objects as go

from joblib import dump, load

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
#Chargement du dataset
df = pd.read_csv('C:/Users/pbliv/Documents/Data Science/P7/application_train.csv')

df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])
df['DAYS_REGISTRATION'] = abs(df['DAYS_REGISTRATION'])
df['DAYS_ID_PUBLISH'] = abs(df['DAYS_ID_PUBLISH'])

print(df.head())

# Create a label encoder object
le = LabelEncoder()

# Iterate through the columns
for col in df:
    if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            # if len(list(data_client[col].unique())) <= 2:
            if col in ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
                # Train on the training data
                le.fit(df[col])
                # Transform training data
                df[col] = le.transform(df[col])

# one-hot encoding of categorical variables
df = pd.get_dummies(df)
cols = df.columns

# Replace the anomalous values with nan
df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

# Cartes représentant les informations (Score, ID, accepté ou non")
id = df.loc[i, "SK_ID_CURR"]
#appel de l'API de prediction
score = request.post("http://127.0.0.1:5000/result", data=df.loc[i, :])
score = score.text

# Jauge du score
#Figure plotly
 fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = score,
    mode = "gauge+number+delta",
    title = {'text': "Score"},
    delta = {'reference': 0.5},
    gauge = {'axis': {'range': [None, 1]},
             'steps' : [
                 {'range': [0, 0.5], 'color': "lightgray"},
                 {'range': [0.5, 1], 'color': "gray"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}}))

#La jauge est intégrée à l'application Dash
app.layout = html.Div([
    dcc.Graph(figure=fig)
])
#Individu acccepté ou non selon son score
if score <=0.5:
    accepted = "oui"
else :
    accepted = "non"

#Affichage des cartes

cards = [
    ## Carte affichant l'id de l'individu
    dbc.Card(
        [
            html.H2(f"{id:.2f}%", className="card-title"),
            html.P("ID de l'individu", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    ## Carte affichant si l'individu est accepté
    dbc.Card(
        [
            html.H2(f"{accepted}", className="card-title"),
            html.P("Client accepté", className="card-text"),
        ],
        body=True,
        color="primary",
        inverse=True,
    ),
]
#Import du modèle
my_model = load("C:/Users/pbliv/Documents/Data Science/P7/mymodel.joblib")

#Dataframe de features importances
results = pd.DataFrame({"var":data.drop(columns = ['TARGET']).columns,"features_importance": my_model.named_steps['classifier'].feature_importances_})

# Importance des variables globale
coef_fig = px.bar(
    data = results.sort_values(by='features_importance', ascending=False).head(50),
    x=my_model.feature_importances_,
    y=df.drop(columns = ['TARGET']).columns,
    orientation="h",
    labels={"x": "Importance", "y": "Variables"},
    title="Importance des variables dans la prédiction de défaut de crédit"
)

#Graph features importances (globale) intégré à l'app Dash
app.layout = html.Div([
    dcc.Graph(figure=coef_fig)
])

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)