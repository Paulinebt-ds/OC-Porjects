# Importation des packages
import dill
import plotly
from flask import Flask, request
from flask_caching import Cache
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
from dash import callback
from dash import dash_table
from io import BytesIO
from plotly.graph_objs import Figure

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

# Import des fonctions de preprocess
from plotly.offline import iplot

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
path = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/application_train.csv"
path_imputer = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/imputer.joblib"
path_scaler = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/scaler.joblib"
train_domain_path = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/app_train_all.csv"
model_Pkl = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/model.pkl"
model_prediction_pkl = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/model_prediction.pkl"
lime_explainer_pkl = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/lime_explainer.pkl"
desc_col = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/data/Desc_col.xlsx"

@cache.memoize(timeout=60)  # mise en cache de la fonction pour exécution unique
def get_data():
    df = chargement_data(path)
    return df


data_domain = get_data()
desc_col = pd.read_excel(desc_col)
columns = [{"name": i, "id": i} for i in desc_col.columns]
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
print(scaled_test_data)
print(scaled_test_data.shape)
print(scaled_test_data[1].shape)
predict_method = model.predict_proba

## Layout de la page client
layout = html.Div([
    #dcc.Store(id='memory-output'),

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
                                value=int(100002.0),
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
                    dcc.Graph(id="credit-term-figure")],
                    className='col-6'),
                html.Div([
                    dcc.Graph(id="days-birth-figure")],
                    className='col-6'),
                html.Div([
                    dcc.Graph(id="amt-annuity-figure")],
                    className='col-6'),
                html.Div([
                    dcc.Graph(id="id-publish-figure")],
                    className='col-6'),
                html.Div([
                    dcc.Graph(id="employed-percent-figure")],
                    className='col-6'),
                html.Div([
                    dash_table.DataTable(
                    id='table-desc',
                    columns=[{"name": i, "id": i} for i in desc_col.columns],
                    data=desc_col.to_dict('records'),
                    style_header={
                        'backgroundColor': 'blue',
                        'fontWeight': 'bold',
                        'color': 'white',
                        'border': '0px'
                    },
                    style_cell={'padding': '5px', 'border': '0px', 'textAlign': 'center'},
                    style_as_list_view=True,
                    page_action='native',
                    fixed_rows={'headers': True},
                    style_table={'height': '400px', 'overflowY': 'auto'},
                    sort_action='native'
                    )]),
                html.Div([dav.Svg(
                    id='explainer-obj'
                )],
                 className='col-6'),
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


@callback(
    Output("memory-output", "data"),
    Input("case-dropdown", "value"),
)
def filter_id(caseval):
    if type(caseval) is int:
        data = data_domain[data_domain["SK_ID_CURR"] == int(float(str(caseval)))]
    else:
        data = data_domain[data_domain["SK_ID_CURR"] == int(float(str(100002)))]
    return data.to_dict('records')


@callback(
    Output("number-out", "children"),
    Input("case-dropdown", "value"),
)
def number_render(caseval):
    return "ID choisi: {}".format(caseval)


@callback(
    Output("update_id", "children"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("case-dropdown", "value"),
)
def update_input(submit_n_clicks, reset_n_clicks, case):
    id_client = case
    return id_client

@callback(
    Output("memory-API", "data"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("case-dropdown", "value"))

def data_from_api(submit_n_clicks, reset_n_clicks, case):
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
        r = r.json()
        print(r)
        data_row = pd.DataFrame.from_dict(r)
        print(data_row)
        print(data_row.shape)
        data_row = data_row.to_dict('records')
        print(type(data_row))
        return data_row

@callback(
    Output("update_score", "children"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("case-dropdown", "value"),
    State('memory-API', "data")
)
def update_score(submit_n_clicks, reset_n_clicks, case, data):
    ctx = dash.callback_context
    if type(case) is int and type(data) is list or "predict" in ctx.triggered[0]["prop_id"]:
        data_row = pd.DataFrame.from_dict(data)
        print(data_row)
        score = data_row["score_pred"].values
        print(score)
        score = score[0]
        score = float(str(score))
        print(score)
        return score

@callback(
    Output("update_card", "children"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("update_score", "children"),
)
def update_card(submit_n_clicks, reset_n_clicks, score):
    print(type(score))
    if type(score) is float:
        if score <= 0.5:
            accepted = "accepté"
        else:
            accepted = "refusé"

        msg_card = ["Client ", f"{accepted}", html.Br(), "avec un score de : ", f"{score: .2f}"]
        return msg_card


@callback(Output('case-dropdown', 'value'),
           Input('reset-button', 'n_clicks'))
def clear_form(n_clicks):
    """Empty input textarea"""
    return ""

@callback(
    Output("update_gauge-fig", "figure"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("update_score", "children"),
)
def update_gauge(submit_n_clicks, reset_n_clicks, score):
    ctx = dash.callback_context
    if type(score) is float:
        accepted_percent = (1 - score) * 100
        print(accepted_percent)
        gauge_figure = go.Figure(go.Indicator(
            domain={'x': [1, 0], 'y': [0, 1]},
            value=accepted_percent,
            mode="gauge+number+delta",
            title="Pourcentage d'acceptation du client",
            delta={'reference': 50},
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': accepted_percent}}))
    else:
        gauge_figure = go.Figure(go.Indicator(
            domain={'x': [1, 0], 'y': [0, 1]},
            value=50,
            mode="gauge+number+delta",
            title="Pourcentage d'acceptation du client",
            delta={'reference': 50},
            gauge={'axis': {'range': [0, 100]},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                             {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
    return gauge_figure


@callback(Output("seaborn", "contents"), [Input('num-samples-input', 'value')])
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


@callback(
    dash.dependencies.Output("credit-term-figure", "figure"),
    [dash.dependencies.Input("case-dropdown", "value"),
     dash.dependencies.State('memory-output', "data")])
def update_credit_term(id_client, data):
    col_to_plot = "CREDIT_TERM"
    dff = data_domain
    fig = create_moy_hist(id_client, col_to_plot, dff, data)
    return go.Figure(fig)


@callback(
    dash.dependencies.Output("days-birth-figure", "figure"),
    [dash.dependencies.Input("case-dropdown", "value"),
     dash.dependencies.State('memory-output', "data")])
def update_days_birth(id_client, data):
    col_to_plot = "DAYS_BIRTH"
    dff = data_domain
    fig = create_moy_hist(id_client, col_to_plot, dff, data)
    return go.Figure(fig)

@callback(
    dash.dependencies.Output("amt-annuity-figure", "figure"),
    [dash.dependencies.Input("case-dropdown", "value"),
     dash.dependencies.State('memory-output', "data")])
def update_amt_annuity(id_client, data):
    col_to_plot = "AMT_ANNUITY"
    dff = data_domain
    fig = create_moy_hist(id_client, col_to_plot, dff, data)
    return go.Figure(fig)

@callback(
    dash.dependencies.Output("id-publish-figure", "figure"),
    [dash.dependencies.Input("case-dropdown", "value"),
     dash.dependencies.State('memory-output', "data")])
def update_id_publish(id_client, data):
    col_to_plot = "DAYS_ID_PUBLISH"
    dff = data_domain
    fig = create_moy_hist(id_client, col_to_plot, dff, data)
    return go.Figure(fig)

@callback(
    dash.dependencies.Output("employed-percent-figure", "figure"),
    [dash.dependencies.Input("case-dropdown", "value"),
     dash.dependencies.State('memory-output', "data")])
def update_employed_percent(id_client, data):
    col_to_plot = "DAYS_EMPLOYED_PERCENT"
    dff = data_domain
    fig = create_moy_hist(id_client, col_to_plot, dff, data)
    return go.Figure(fig)

@callback(Output("table-desc", "data"),
              [Input('num-samples-input', 'value')])
def update_table(n_samples):
    if type(n_samples) is not int:
        features = results.sort_values(by="features_importance", ascending=False).head(5)['var'].values
        print(features)
        print(desc_col[desc_col['Row'].isin(features)])
        filtered_df = desc_col[desc_col["Row"].isin(features)]

    else:
        features = results.sort_values(by="features_importance", ascending=False).head(n_samples)['var'].values
        filtered_df = desc_col[desc_col['Row'].isin(features)]

    return filtered_df.to_dict('records')


@callback(Output('explainer-obj', 'contents'),
              Input('submit-button', 'n_clicks'),
              Input('reset-button', 'n_clicks'),
              State('case-dropdown', 'value'),
              State('num-samples-input', 'value'),
              State('memory-API', 'data'))
def generate_explainer_html(submit_n_clicks, reset_n_clicks, case, n_samples, data):
    if type(n_samples) is int:
        data_row = pd.DataFrame.from_dict(data)
        data_row = data_row.loc[0, :]
        json_df = data_row.to_json()
        exp = requests.post('http://127.0.0.1:5000/lime', json=json_df)
        print(exp)
        print(type(exp))
        exp = exp.json()
        print(exp)
        print(exp.items())
        print(list(exp.items()))
        exp = list(exp.items())
        fig = plt.figure(figsize=(12,6))
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names, fontsize=8)
        title = 'Local explanation for class 1'
        plt.title(title)
        b_io = BytesIO()
        fig.savefig(b_io, format="svg")
        obj = b_io.getvalue().decode("utf-8")

        return obj


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
