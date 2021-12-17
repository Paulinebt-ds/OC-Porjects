import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import callbacks
from pages.header import *
from pages import layout_global
from pages import layout_client
from app import app, server


# layout rendu par l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    get_navbar(),
    html.Div(id='page-content')
])


# callback pour mettre à jour les pages
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/global' or pathname == '/':
        return layout_global
    elif pathname == '/client':
        return layout_client


if __name__ == '__main__':
    app.run_server(debug=True)