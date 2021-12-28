import dash
import dash_bootstrap_components as dbc
import components.pages_plugin as pages_plugin
import dash_labs as dl  # pip install dash-labs

bootstrap_theme = [dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.9.0/css/all.css']
app = dash.Dash(__name__,
                plugins=[pages_plugin], external_stylesheets=bootstrap_theme, suppress_callback_exceptions=True)

#print(list(dash.page_registry.values()))
server = app.server

