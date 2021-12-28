import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_labs as dl  # pip install dash-labs
from ressources.app import app, server
from components.pages_plugin import *

from pages.layout_client import data_domain


df = data_domain[data_domain["SK_ID_CURR"] == int(100002)]
df = df.to_dict('records')
# layout rendu par l'application
navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="More Pages",
    ),
    brand="Multi Page App Plugin Demo",
    color="primary",
    dark=True,
    className="mb-2",
)

app.layout = dbc.Container(
    [navbar,
     dcc.Store(id="memory-output", data=df),
     dl.plugins.page_container],
    fluid=True,
)


if __name__ == '__main__':
    app.run_server(debug=True)S