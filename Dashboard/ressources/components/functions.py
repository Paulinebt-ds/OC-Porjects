import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import lime
import time
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular
from sklearn.model_selection import train_test_split
import dill
from io import BytesIO
from lime.lime_tabular import LimeTabularExplainer
COLOR_BR_r = ['dodgerblue', 'indianred']
COLOR_BR = ['#EF553B', '#00CC96']

def get_lime_explainer(data, X):

  cat_feat_ix = [i for i, c in enumerate(data.columns) if pd.api.types.is_categorical_dtype(data[c])]
  feat_names = list(data.columns)
  class_names = [0, 1]
  lime_explainer = LimeTabularExplainer(np.array(X),
                                      feature_names=feat_names,
                                      class_names=class_names,
                                      categorical_features=cat_feat_ix,
                                      mode="classification"
                                      )
  return lime_explainer


def create_time_series(dff, col, y_col, color, title):
    fig = px.histogram(dff, x=col, y=y_col, color=color, title=title)

    fig.update_xaxes(showgrid=False)

    return fig

def create_moy_hist(id_client, col_to_plot, dff, data):
    if type(id_client) is not int:
        moy_regle = np.mean(dff[dff["TARGET"]==0][col_to_plot])
        moy_defaut = np.mean(dff[dff["TARGET"]==1][col_to_plot])
        moy_globale = np.mean(dff[col_to_plot])
        d = {'groupe': ["globale", "en règle", "défaut"], 'moyenne': [moy_globale, moy_regle, moy_defaut]}
        dff = pd.DataFrame(data=d)
        title = 'Moyenne par groupe de la variable %s'%(col_to_plot)
        col = "groupe"
        y_col = "moyenne"
        color = "groupe"
        fig = create_time_series(dff, col, y_col, color, title)
        return fig

    else:
        print(id_client)
        print(type(id_client))
        data_client = pd.DataFrame.from_dict(data)
        print(data_client)
        data = data_client[col_to_plot]
        print(data)
        data = data.values
        print(data)
        data = data[0]
        print(data)
        moy_regle = np.mean(dff[dff["TARGET"]==0][col_to_plot])
        moy_defaut = np.mean(dff[dff["TARGET"]==1][col_to_plot])
        moy_globale = np.mean(dff[col_to_plot])
        d = {'groupe': ["globale", "en règle", "défaut", "client"], 'moyenne': [moy_globale, moy_regle, moy_defaut, data]}
        dff = pd.DataFrame(data=d)
        title = 'Moyenne par groupe de la variable %s'%(col_to_plot)
        col = "groupe"
        y_col = "moyenne"
        color = "groupe"
        fig = create_time_series(dff, col, y_col, color, title)
        return fig

def lime_explain(explainer, data, predict_method, num_features):
    explanation = explainer.explain_instance(data, predict_method, num_features=num_features, num_samples=100)
    return explanation


corporate_colors = {
    'dark-blue-grey' : 'rgb(62, 64, 76)',
    'medium-blue-grey' : 'rgb(77, 79, 91)',
    'superdark-green' : 'rgb(41, 56, 55)',
    'dark-green' : 'rgb(57, 81, 85)',
    'medium-green' : 'rgb(93, 113, 120)',
    'light-green' : 'rgb(186, 218, 212)',
    'pink-red' : 'rgb(255, 101, 131)',
    'dark-pink-red' : 'rgb(247, 80, 99)',
    'white' : 'rgb(251, 251, 252)',
    'light-grey' : 'rgb(208, 206, 206)'
}
externalgraph_rowstyling = {
    'margin-left' : '15px',
    'margin-right' : '15px'
}

externalgraph_colstyling = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['superdark-green'],
    'background-color' : corporate_colors['superdark-green'],
    'box-shadow' : '0px 0px 17px 0px rgba(186, 218, 212, .5)',
    'padding-top' : '10px'
}

navbarcurrentpage = {
    'text-decoration': 'underline',
    'color': 'white',
    'text-decoration-color': corporate_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)',
    'background-color': corporate_colors['dark-green']
    }

filterdiv_borderstyling = {
    'border-radius' : '0px 0px 10px 10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : corporate_colors['light-green'],
    'background-color' : corporate_colors['light-green'],
    'box-shadow' : '2px 5px 5px 1px rgba(255, 101, 131, .5)'
    }
recapdiv = {
    'border-radius' : '10px',
    'border-style' : 'solid',
    'border-width' : '1px',
    'border-color' : 'rgb(251, 251, 252, 0.1)',
    'margin-left' : '15px',
    'margin-right' : '15px',
    'margin-top' : '15px',
    'margin-bottom' : '15px',
    'padding-top' : '5px',
    'padding-bottom' : '5px',
    'background-color' : 'rgb(251, 251, 252, 0.1)'
    }

#####################
# Nav bar
def get_navbar(p = 'Global'):

    navbar_global = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Global',
                        style = navbarcurrentpage),
                href='/apps/global'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Client',
                        style = navbarcurrentpage),
                href='/apps/Client'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Bivariée',
                        style=navbarcurrentpage),
                href='/apps/bivariate'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_client = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Global',
                        style=navbarcurrentpage),
                href='/apps/global'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Client',
                        style = navbarcurrentpage),
                href='/apps/Client'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Bivariée',
                        style=navbarcurrentpage),
                href='/apps/bivariate'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3',
                 style=navbarcurrentpage)

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    navbar_bivariate = html.Div([

        html.Div([], className = 'col-3'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Global',
                        style=navbarcurrentpage),
                href='/apps/global'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Client',
                        style=navbarcurrentpage),
                href='/apps/client'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Bivariée',
                        style = navbarcurrentpage),
                href='/apps/bivariate'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    if p == 'global':
        return navbar_global
    elif p == 'client':
        return navbar_client
    else:
        return navbar_bivariate

# Empty row

def get_emptyrow(h='45px'):
    """This returns an empty row of a defined height"""

    emptyrow = html.Div([
        html.Div([
            html.Br()
        ], className = 'col-12')
    ],
    className = 'row',
    style = {'height' : h})

    return emptyrow

def histogram(df, x='str', legend=True, client=None):
    '''client = [df_test, input_client] '''
    if x == "TARGET":
        fig = px.histogram(df,
                           x=x,
                           color="TARGET",
                           width=300,
                           height=200,
                           category_orders={"TARGET": [1, 0]},
                           color_discrete_sequence=COLOR_BR)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=50))
    else:
        fig = px.histogram(df,
                           x=x,
                           color="TARGET",
                           width=300,
                           height=200,
                           category_orders={"TARGET": [1, 0]},
                           color_discrete_sequence=COLOR_BR,
                           barmode="group",
                           histnorm='percent')
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if legend == True:
        fig.update_layout(legend=dict(yanchor="top", xanchor="right"))
    else:
        fig.update_layout(showlegend=False)
    if client:
        client_data = client[0][client[0].SK_ID_CURR == client[1]]
        vline = client_data[x].to_numpy()[0]
        print(vline)

        fig.add_vline(x=vline, line_width=3, line_dash="dash", line_color="black")
    return fig

