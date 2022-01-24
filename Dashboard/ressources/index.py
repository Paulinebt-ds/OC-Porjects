import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_labs as dl  # pip install dash-labs
from ressources.app import app, server
from components.functions import navbarcurrentpage, corporate_colors
from pages.layout_client import data_domain

# Connect to your app pages
from pages import layout_global, layout_bivariee, layout_client

df = data_domain[data_domain["SK_ID_CURR"] == int(100002)]
df = df.to_dict('records')
# layout rendu par l'application
header = html.Div([

        html.Div([
            html.H1(children='Dashboard',
                    style={'textAlign': 'center',
                           'color': 'white'})
                ],
                className='col-8',
                style={'padding-top': '1%'}
                ),

        html.Div([
            html.Img(
                    src=app.get_asset_url("logo_pret_a_depenser.PNG"),
                    height='100',
                    width='200')
            ],
            className='col-2',
            style={
                    'align-items': 'center',
                    'padding-top': '1%',
                    'height': '100'})

        ],
        className='row',
        style={'height': '4%',
               'background-color': 'rgb(41, 56, 55)'}
        )
navbar = html.Div([

        html.Div([], className='col-3'),

        html.Div([
            dcc.Link(
                html.H4(children='Global',
                        style=navbarcurrentpage),
                href='/apps/global'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Client',
                        style=navbarcurrentpage),
                href='/apps/client'
                )
        ],
        className='col-2'),

        html.Div([
            dcc.Link(
                html.H4(children='Bivariée',
                        style=navbarcurrentpage),
                href='/apps/bivariate'
                )
        ],
        className='col-2'),

        html.Div([], className='col-3')

    ],
    className='row',
    style={'background-color': corporate_colors['dark-green'],
           'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    header,
    navbar,
    dcc.Store(id="memory-output", data=df),
    dcc.Store(id="memory-API", data=df),
    html.Div(id='page-content', children=[])
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return layout_global.layout
    if pathname == '/apps/global':
        return layout_global.layout
    if pathname == '/apps/client':
        return layout_client.layout
    if pathname == '/apps/bivariate':
        return layout_bivariee.layout
    else:
        return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=True)