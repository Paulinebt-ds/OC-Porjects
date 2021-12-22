# Importation des packages
import dill
from flask import Flask, request
import os
from flask_caching import Cache
import dash
from dash import dcc
import selenium
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output, State

import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.embed import json_item
import dash_alternative_viz as dav

import plotly.express as px
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import lime
import time
from joblib import dump, load
import self
from bokeh.io import output_file, show
# Import des fonctions de preprocess
from ressources.components.preprocess import *
from ressources.components.functions import *
from header import *
# from components.functions import *
import pandas as pd
import numpy as np
from bokeh.io import export_svg

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.pipeline import Pipeline
import json

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
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
td_style = {"width": "50%", "margin": "20px"}
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Importation de la base "application_train"
path = "C:/Users/pbliv/Documents/Data Science/P7/application_train.csv"

app_train = pd.read_csv(path)

features = app_train.drop(["SK_ID_CURR", "TARGET"], axis=1).columns
## Layout de la page client
app.layout = html.Div([

    #####################
    # Row 1 : Header
    get_header(),

    #####################
    # Row 2 : Nav bar
    get_navbar('bivariate'),
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
                        # 1ère colonne à choisir
                        html.Div([
                            html.Label("Chosissez la première colonne: "),
                            html.Br(),
                            dcc.Dropdown(
                                id='crossfilter-xaxis-column',
                                options=[
                                    {'label': var, 'value': var} for var in features
                                ],
                                value="ORGANIZATION_TYPE",
                            ),
                            html.Div(),
                            html.Hr(),
                            html.Div(id="col1-out"),
                        ]),

                        html.Br(),
                        # 2ème colonne à choisir
                        html.Div([
                            html.Label("Chosissez la deuxième colonne: "),
                            html.Br(),
                            dcc.Dropdown(
                                id='crossfilter-yaxis-column',
                                options=[
                                    {'label': var, 'value': var} for var in features
                                ],
                                value="EXT_SOURCE_1",
                            ),
                            html.Div(),
                            html.Hr(),
                            html.Div(id="col2-out"),
                        ]),
                        html.Div([
                            dcc.RadioItems(
                                id="checkbox-value",
                                options=[
                                    {'label': 'Ajout ligne client', 'value': 'True'},
                                    {'label': 'Graphiques de base', 'value': 'False'}
                                ],
                                value='False',
                                labelStyle={'display': 'inline-block'},

                            )],
                            className='col2-out'),
                        html.Div(
                            [html.Button(id='submit-button', n_clicks=0, children='Submit'),
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

            html.H2(children="Analyse bivariée",
                    style={'color': corporate_colors['white']}),

            html.Div([  # Internal row - RECAPS

            ], className='col-4')  # Empty column

        ],
            className='row'
        ),  # Internal row - RECAPS

        html.Div([  # Internal row
            html.Div([
                dcc.Graph(
                    id='x-time-series'
                ),
            ],
                className='col-6'),

            html.Div([
                dcc.Graph(
                    id='y-time-series'
                )
            ],
                className='col-6'),

            html.Div([
                dcc.Graph(
                    id='distrib-col1-target'
                )
            ],
                className='col-6'),

            html.Div([
                dcc.Graph(
                    id='distrib-col2-target'
                )
            ],
                className='col-6'),
            html.Div([
                dav.Svg(
                    id='crossfilter-indicator-scatter'
                )
            ],
                className='col-6'
            ),
        ],
            className='row'),  # Internal row
    ],
        className='row',
        style=externalgraph_rowstyling
    ),  # External row
])


@app.callback(
    Output("col1-out", "children"),
    Input('crossfilter-xaxis-column', "value"),
)
def col1_render(col1_val):
    return "1ère colonne choisie: {}".format(col1_val)


@app.callback(
    Output("col2-out", "children"),
    Input('crossfilter-yaxis-column', "value"),
)
def col2_render(col2_val):
    return "2ème colonne choisie: {}".format(col2_val)

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'contents'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value')])

def update_graph(xaxis_column_name, yaxis_column_name):
    if pd.api.types.is_numeric_dtype(app_train[xaxis_column_name]) and pd.api.types.is_numeric_dtype(app_train[yaxis_column_name]):
        x = [xaxis_column_name, yaxis_column_name, "TARGET"]
        data_corr = app_train[x]
        corr = data_corr.corr()
        fig, ax = plt.subplots()
        mask = np.triu(np.ones_like(corr, dtype=np.bool))
        sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
        ax.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
        fig.set_size_inches(5.5, 3.7)
        fig.tight_layout()

        b_io = BytesIO()
        fig.savefig(b_io, format="svg")
        obj = b_io.getvalue().decode("utf-8")
    else:
        obj = html.Iframe(
            srcDoc='''<div>Choisissez des variables quantitatives.</div>''',
            width='100%',
            height='100px',
            style={'border': '2px #d3d3d3 solid'},
            hidden=True,
        )
        b_io = BytesIO()
        obj = b_io.getvalue().decode("utf-8")
    return obj



def create_time_series(dff, col, color, title):
    fig = px.histogram(dff, x=col, color=color)

    fig.update_xaxes(showgrid=False)

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    #fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig

def create_bar_plot(dff, col, color, title):
    occ_by_target = pd.DataFrame(dff.groupby(["TARGET", col])[col].count())
    df = pd.DataFrame({'count': occ_by_target[col]})
    df = df.reset_index()
    fig = px.bar(df, x=col, y="count", color="TARGET", title=title)
    return fig

def create_box_plot(dff, col, color, title):

    fig = px.box(dff, y=col, x="TARGET", color=color)

    fig.update_xaxes(showgrid=False)

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       text=title)

    #fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    return fig


@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('checkbox-value', 'value')])
def update_x_timeseries(xaxis_column_name, checkbox):
    if checkbox=='False':
        dff = app_train
        title = '<b>{}</b>'.format(xaxis_column_name)
        col = xaxis_column_name
        color = None
        return go.Figure(create_time_series(dff, col, color, title))

    else:
        dff = app_train
        title = '<b>{}</b>'.format(xaxis_column_name)
        col = xaxis_column_name
        color = None
        fig = create_time_series(dff, col, color, title)
        fig.add_shape(type="line",
                      x0=2, y0=2, x1=5, y1=2,
                      line=dict(
                          color="LightSeaGreen",
                          width=4,
                          dash="dashdot",
                      )
                      )
        return go.Figure(fig)

@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('checkbox-value', 'value')])
def update_y_timeseries(yaxis_column_name, checkbox):
    if checkbox=='False':
        dff = app_train
        title = '<b>{}</b>'.format(yaxis_column_name)
        col = yaxis_column_name
        color = None
        return go.Figure(create_time_series(dff, col, color, title))

    else:
        dff = app_train
        title = '<b>{}</b>'.format(yaxis_column_name)
        col = yaxis_column_name
        color = None
        fig = create_time_series(dff, col, color, title)
        fig.add_shape(type="line",
                      x0=2, y0=2, x1=5, y1=2,
                      line=dict(
                          color="LightSeaGreen",
                          width=4,
                          dash="dashdot",
                      )
                      )
        return go.Figure(fig)


@app.callback(
    dash.dependencies.Output('distrib-col1-target', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('checkbox-value', 'value')])
def update_x_target(xaxis_column_name, checkbox):
    if pd.api.types.is_object_dtype(app_train[xaxis_column_name]) and checkbox == 'False':
        dff = app_train
        title = '<b>{}</b>'.format(xaxis_column_name)
        col = xaxis_column_name
        color = None
        return go.Figure(create_bar_plot(dff, col, color, title))

    elif pd.api.types.is_object_dtype(app_train[xaxis_column_name]) and checkbox == 'True':
        dff = app_train
        title = '<b>{}</b>'.format(xaxis_column_name)
        col = xaxis_column_name
        color = None
        fig = create_bar_plot(dff, col, color, title)
        fig.add_shape(type="line",
                      x0=2, y0=2, x1=5, y1=2,
                      line=dict(
                          color="LightSeaGreen",
                          width=4,
                          dash="dashdot",
                      )
                      )
        return go.Figure(fig)

    if pd.api.types.is_numeric_dtype(app_train[xaxis_column_name]):
        dff = app_train
        title = '<b>{}</b>'.format(xaxis_column_name)
        col = xaxis_column_name
        color = "TARGET"
        return go.Figure(create_box_plot(dff, col, color, title))


@app.callback(
    dash.dependencies.Output('distrib-col2-target', 'figure'),
    [dash.dependencies.Input('crossfilter-yaxis-column', 'value')])
def update_y_target(yaxis_column_name):
    if pd.api.types.is_object_dtype(app_train[yaxis_column_name]):
        dff = app_train
        title = '<b>{}</b>'.format(yaxis_column_name)
        col = yaxis_column_name
        color = "TARGET"
        return go.Figure(create_bar_plot(dff, col, color, title))

    if pd.api.types.is_numeric_dtype(app_train[yaxis_column_name]):
        dff = app_train
        title = '<b>{}</b>'.format(yaxis_column_name)
        col = yaxis_column_name
        color = "TARGET"
        return go.Figure(create_box_plot(dff, col, color, title))


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
