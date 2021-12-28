import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_labs as dl  # pip install dash-labs
from ressources.app import app, server
from components.pages_plugin import *
from pages.layout_global import layout as layout_global
from pages.layout_client import layout as layout_client
from pages.layout_client import data_domain
from pages.layout_bivariee import layout as layout_bivariee

df = data_domain[data_domain["SK_ID_CURR"] == int(100002)]
df = df.to_dict('records')

# layout rendu par l'application
app.layout = html.Div([
    dcc.Store(id='memory-output', data=df),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# callback pour mettre à jour les pages
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/global':
        return layout_global
    elif pathname == '/apps/client':
        return layout_client
    elif pathname == '/apps/bivariate':
        return layout_bivariee
    else:
        return layout_global  # This is the "home page"


if __name__ == '__main__':
    app.run_server(debug=False)