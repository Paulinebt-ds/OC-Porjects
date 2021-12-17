# Importation des packages
from flask import Flask, request
import os
from flask_caching import Cache
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output


import plotly.express as px
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import lime
import time
from joblib import dump, load

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
import json


from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

import lightgbm as lgb
from joblib import load
import requests
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

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
    'text-decoration' : 'underline',
    'color' : 'white',
    'text-decoration-color' : corporate_colors['pink-red'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)'
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
# Header with logo
def get_header():

    header = html.Div([

        html.Div([], className = 'col-2'), #Same as img width, allowing to have the title centrally aligned

        html.Div([
            html.H1(children='Dashboard',
                    style = {'textAlign' : 'center',
                            'color': 'white'}
            )],
            className='col-8',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.Img(
                    src = app.get_asset_url("logo_pret_a_depenser.png"),
                    height = '100',
                    width = '200')
            ],
            className = 'col-2',
            style = {
                    'align-items': 'center',
                    'padding-top' : '1%',
                    'height' : '100'})

        ],
        className = 'row',
        style = {'height' : '4%',
                'background-color' : corporate_colors['superdark-green']}
        )

    return header
#####################
# Nav bar
def get_navbar(p):

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