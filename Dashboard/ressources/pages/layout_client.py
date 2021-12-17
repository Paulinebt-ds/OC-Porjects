# Importation des packages
from typing import Union, Any

import dill
from flask import Flask, request
import os
from flask_caching import Cache
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
from dash import dash_table
from io import BytesIO

from plotly.graph_objs import Figure

from header import *
import dash_alternative_viz as dav

from bokeh.embed import json_item

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import lime
import time
from joblib import dump, load
import self

# Import des fonctions de preprocess
from ressources.components.preprocess import *
from ressources.components.functions import *
# from components.functions import *
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.pipeline import Pipeline
import json

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
from joblib import load
import requests

####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### Corporate css formatting
corporate_colors = {
    'dark-blue-grey': 'rgb(62, 64, 76)',
    'medium-blue-grey': 'rgb(77, 79, 91)',
    'superdark-green': 'rgb(41, 56, 55)',
    'dark-green': 'rgb(57, 81, 85)',
    'medium-green': 'rgb(93, 113, 120)',
    'light-green': 'rgb(186, 218, 212)',
    'pink-red': 'rgb(255, 101, 131)',
    'dark-pink-red': 'rgb(247, 80, 99)',
    'white': 'rgb(251, 251, 252)',
    'light-grey': 'rgb(208, 206, 206)'
}
externalgraph_rowstyling = {
    'margin-left': '15px',
    'margin-right': '15px'
}

externalgraph_colstyling = {
    'border-radius': '10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': corporate_colors['superdark-green'],
    'background-color': corporate_colors['superdark-green'],
    'box-shadow': '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top': '10px'
}

navbarcurrentpage = {
    'text-decoration': 'underline',
    'color': 'white',
    'text-decoration-color': corporate_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
}

filterdiv_borderstyling = {
    'border-radius': '0px 0px 10px 10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': corporate_colors['light-green'],
    'background-color': corporate_colors['light-green'],
    'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'
}
recapdiv = {
    'border-radius': '10px',
    'border-style': 'solid',
    'border-width': '1px',
    'border-color': 'rgb(251, 251, 252, 0.1)',
    'margin-left': '15px',
    'margin-right': '15px',
    'margin-top': '15px',
    'margin-bottom': '15px',
    'padding-top': '5px',
    'padding-bottom': '5px',
    'background-color': 'rgb(251, 251, 252, 0.1)'
}
## APP FLASK
# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
# Importation de la base "application_train"
path = "C:/Users/pbliv/Documents/Data Science/P7/application_train.csv"
path_imputer = "C:/Users/pbliv/Documents/Data Science/P7/imputer.joblib"
path_scaler = "C:/Users/pbliv/Documents/Data Science/P7/scaler.joblib"
train_domain_path = "C:/Users/pbliv/Documents/Data Science/P7/app_train_all.csv"
model_Pkl = "C:/Users/pbliv/Documents/Data Science/P7/model.pkl"
model_prediction_pkl = "C:/Users/pbliv/Documents/Data Science/P7/model_prediction.pkl"
lime_explainer_pkl = "C:/Users/pbliv/Documents/Data Science/P7/lime_explainer.pkl"
desc_col = "C:/Users/pbliv/Documents/Data Science/P7/Desc_col.xlsx"

@cache.memoize(timeout=60)  # mise en cache de la fonction pour exécution unique
def get_data():
    df = chargement_data(path)
    return df


data_domain = get_data()
desc_col = pd.read_excel(desc_col)
# def transform_data():
# df = normalize_data(data_domain, path_scaler, path_imputer)
# return df
# data_domain = transform_data()

train_domain = pd.read_csv(train_domain_path)
# Import du modèle
# Avec LGBM Booster
LGBM_model = lgb.Booster(model_file="C:/Users/pbliv/Documents/Data Science/P7/model.bin")

# Avec dill

with open(model_Pkl, 'rb') as file:
    model = dill.load(file)

## Import du lime_explainer
with open(lime_explainer_pkl, 'rb') as file:
    lime_explainer = dill.load(file)
# Dataframe de features importances
results = pd.DataFrame({"var": train_domain.drop(["TARGET", "SK_ID_CURR"], axis=1).columns,
                        "features_importance": LGBM_model.feature_importance()})

# Dropdown des colonnes les plus importantes
var_imp = results.sort_values(by='features_importance', ascending=False).head(50)['var'].unique().tolist()

top_results = results.sort_values(by='features_importance', ascending=False).head(20)
# X matrice des var. explicatives
X = train_domain.drop(['SK_ID_CURR', 'TARGET'], axis=1)

# Y matrice de la variable cible : expliquée
y = train_domain['TARGET']
y = y.values.reshape(-1, 1)

X = normalize_data(X, path_scaler, path_imputer)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

scaler = load(path_scaler)

scaled_test_data = scaler.transform(X_val)

predict_method = model.predict_proba

## Layout de la page client
app.layout = html.Div([
    dcc.Store(id='memory-output'),
    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar('client'),
    #####################
    # Row 3 : Filters
    html.Div([  # External row

        html.Div([  # External 12-column

            html.Div([  # Internal row

                # Internal columns
                html.Div([
                ],
                    className='col-2'),  # Blank 2 columns

                # Filter pt 1
                html.Div([

                    html.Div([
                        html.H5(
                            children='Filtres:',
                            style={'text-align': 'left', 'color': corporate_colors['medium-blue-grey']}
                        ),

                        # ID du client
                        html.Div([
                            html.Label("Chosissez l'ID du client : "),
                            html.Br(),
                            dcc.Dropdown(
                                id='case-dropdown',
                                options=[
                                    {'label': id_value, 'value': id_value} for id_value in
                                    data_domain['SK_ID_CURR'].values
                                ],
                                value=data_domain['SK_ID_CURR'].values,
                            ),
                            html.Div(),
                            html.Hr(),
                            html.Div(id="number-out"),
                        ]),
                        html.Label('Nombre de variables importantes à montrer (local + global) :'),
                        dcc.Input(
                            id='num-samples-input',
                            type='number',
                            min=0,
                            max=20,
                            step=1
                        ),

                        html.Br(),
                        html.Div(
                            [html.Button(id='submit-button', n_clicks=0, children='Predict'),
                             html.Button(id='reset-button', n_clicks=0, children='Reset',
                                         style={'backgroundColor': 'white', 'color': '#515151'})],
                            style={'display': 'flex', 'justifyContent': 'center'}
                        ), ],
                        style={'margin-top': '10px',
                               'margin-bottom': '5px',
                               'text-align': 'left',
                               'paddingLeft': 5})

                ],
                    className='col-4'),  # Filter part 1
            ],
                className='row')  # Internal row
        ],
            className='col-12',
            style=filterdiv_borderstyling)  # External 12-column

    ],
        className='row sticky-top'),  # External row

    #####################
    # Row 4
    get_emptyrow(),

    #####################
    # Row 5 : Charts
    html.Div([  # External row

        html.Div([
        ],
            className='col-1'),  # Blank 1 column

        html.Div([  # External 10-column

            html.H2(children="Prédiction du client",
                    style={'color': corporate_colors['white']}),

            html.Div([  # Internal row - RECAPS

                # Intégration des cartes
                html.Div([dbc.Card(
                    [
                        html.H2(id="update_id", className="card-title"),
                        html.P("ID de l'individu", className="card-text"),
                    ],
                    body=True,
                    color="light",
                )
                ],
                    className='col-4'),  # Empty column
                html.Div([
                    dcc.Graph(id="update_gauge-fig")],
                    className='col-4'),
                ## Carte affichant si l'individu est accepté
                html.Div([
                    dbc.Card([
                        html.H2(id="update_card", className="card-title"),
                        html.H3(id="update_score", className="card-subtitle"),
                        html.P("Prédiction", className="card-text"),
                    ],
                        body=True,
                        color="light",
                    ),
                ], className='col-4')  # Empty column

            ],
                className='row',
                style=recapdiv
            ),  # Internal row - RECAPS

            html.Div([  # Internal row
                html.Div([
                    dav.Svg(id="seaborn")
                ],
                    className='col-6'),
                html.Div([
                    dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in desc_col.columns],
                    data=desc_col.to_dict('records'),
                    )],
                    className='col-6'),
                html.Br(),
                dcc.Loading(
                    id='explainer-obj',
                    type="default"
                ),
            ],
                className='row'),  # Internal row
        ],
            className='col-10',
            style=externalgraph_colstyling),  # External 10-column

        html.Div([
        ],
            className='col-1'),  # Blank 1 column

    ],
        className='row',
        style=externalgraph_rowstyling
    ),  # External row
])


@app.callback(
    Output("memory-output", "data"),
    Input("case-dropdown", "value"),
)
def filter_id(caseval):
    if type(caseval) is int:
        data = data_domain[data_domain["SK_ID_CURR"] == int(str(caseval))]
    else:
        data = data_domain
    return data.to_dict('records')


@app.callback(
    Output("number-out", "children"),
    Input("case-dropdown", "value"),
)
def number_render(caseval):
    return "ID choisi: {}".format(caseval)


@app.callback(
    Output("update_id", "children"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("case-dropdown", "value"),
)
def update_input(submit_n_clicks, reset_n_clicks, case):
    ctx = dash.callback_context
    if type(case) is list or "reset" in ctx.triggered[0]["prop_id"]:
        # Return empty iFrame
        id_client = data_domain[0, "SK_ID_CURR"]
    else:
        id_client = case
    return id_client


@app.callback(
    Output("update_score", "children"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("case-dropdown", "value"),
)
def update_score(submit_n_clicks, reset_n_clicks, case):
    ctx = dash.callback_context
    if type(case) is int or "predict" in ctx.triggered[0]["prop_id"]:
        id_client = case
        row_client = data_domain[data_domain["SK_ID_CURR"] == int(str(id_client))].index
        row_client = row_client[0]
        df = data_domain.loc[row_client, :]
        json_df = df.to_json()
        print(json_df)
        # appel de l'API de prediction
        url = "http://127.0.0.1:5000/predict"
        r = requests.post(url, json=json_df)
        print(r.text)
        score = float(r.text)

        return score


@app.callback(
    Output("update_card", "children"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("update_score", "children"),
)
def update_card(submit_n_clicks, reset_n_clicks, score):
    ctx = dash.callback_context
    if "predict" in ctx.triggered[0]["prop_id"]:
        if update_score <= 0.5:
            accepted = "accepté"
        else:
            accepted = "refusé"

        msg_card = ["Client ", f"{accepted}", html.Br(), "avec un score de : ", f"{score: .2f}"]
        return msg_card


@app.callback(Output('case-dropdown', 'value'),
              Input('reset-button', 'n_clicks'))
def clear_form(n_clicks):
    """Empty input textarea"""
    return ""


# Mise à jour de la jauge
@app.callback(
    Output("update_gauge-fig", "figure"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("update_score", "children"),
)
def update_gauge(submit_n_clicks, reset_n_clicks, score):
    ctx = dash.callback_context
    if "predict" in ctx.triggered[0]["prop_id"]:
        accepted_percent = (1 - score) * 100
        gauge_figure = go.Figure(go.Indicator(
            domain={'x': [1, 0], 'y': [0, 1]},
            value=accepted_percent,
            mode="gauge+number+delta",
            title="Pourcentage d'acceptation du client",
            delta={'reference': 50},
            # align = "center",
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': accepted_percent}}))
        return gauge_figure


@app.callback(Output("seaborn", "contents"), [Input('num-samples-input', 'value')])
def seaborn_fig(n_samples):
    empty_obj = html.Iframe(
        srcDoc='''<div>Entrer le nombre de variables pour l'importance des variables (global).</div>''',
        width='100%',
        height='100px',
        style={'border': '2px #d3d3d3 solid'},
        hidden=True,
    )
    if type(n_samples) is not int:
        # Return empty iFrame
        obj = empty_obj
    else:
        fig, ax = plt.subplots()
        sns.barplot(data=results.sort_values(by="features_importance", ascending=False).head(n_samples),
                    x="features_importance",
                    y="var")
        ax.set_title("Les " + str(n_samples) + " variables les plus importantes")
        fig.set_size_inches(5.5, 3.7)
        fig.tight_layout()

        b_io = BytesIO()
        fig.savefig(b_io, format="svg")
        obj = b_io.getvalue().decode("utf-8")
        return obj


@app.callback(Output('explainer-obj', 'children'),
              Input('submit-button', 'n_clicks'),
              Input('reset-button', 'n_clicks'),
              State('case-dropdown', 'value'),
              State('num-samples-input', 'value'))
def generate_explainer_html(submit_n_clicks, reset_n_clicks, case, n_samples):
    ctx = dash.callback_context  # Capture callback context to track button clicks
    empty_obj = html.Iframe(
        srcDoc='''<div>Enter input text to see LIME explanations.</div>''',
        width='100%',
        height='100px',
        style={'border': '2px #d3d3d3 solid'},
        hidden=True,
    )
    if type(case) is list or not n_samples or "reset" in ctx.triggered[0]["prop_id"]:
        # Return empty iFrame
        obj = empty_obj
    else:
        index = data_domain[data_domain["SK_ID_CURR"] == int(str(case))].index
        index = index[0]
        exp = lime_explain(lime_explainer, scaled_test_data[index], predict_method, num_features=int(n_samples))

        obj = html.Iframe(
            # Javascript is disabled from running in an IFrame for security reasons
            # Static HTML only!!!
            srcDoc=exp.as_html(),
            width='100%',
            height='800px',
            style={'border': '2px #d3d3d3 solid',
                   'backgroundColor': 'white'}
        )
    return obj


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
