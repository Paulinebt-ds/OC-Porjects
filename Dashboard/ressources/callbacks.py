#Import des librairies
import dash
from dash import html
from dash.dependencies import Input, Output, State
import holoviews as hv
from holoviews.plotting.plotly.dash import to_dash
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.embed import json_item
import dash_alternative_viz as dav
from ressources.components.functions import *
from ressources.components.preprocess import *
from pages.layout_bivariee import app_train
from pages.layout_client import data_domain, results, desc_col, lime_explainer, scaled_test_data, predict_method
from app import app
import plotly.express as px
import plotly.graph_objects as go
from lime.lime_tabular import LimeTabularExplainer
import lime
from bokeh.io import output_file, show
import pandas as pd
import numpy as np

#Callbacks pour layout_client
@app.callback(
    Output("memory-output", "data"),
    Input("case-dropdown", "value"),
)
def filter_id(caseval):
    if type(caseval) is int:
        data = data_domain[data_domain["SK_ID_CURR"] == int(float(str(caseval)))]
    else:
        data = data_domain[data_domain["SK_ID_CURR"] == int(float(str(100002)))]
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
    id_client = case
    return id_client

@app.callback(
    Output("memory-API", "data"),
    Input('submit-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input("case-dropdown", "value"))

def data_from_API(submit_n_clicks, reset_n_clicks, case):
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

@app.callback(
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

@app.callback(
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


@app.callback(Output('case-dropdown', 'value'),
           Input('reset-button', 'n_clicks'))
def clear_form(n_clicks):
    """Empty input textarea"""
    return ""

@app.callback(
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

@app.callback(Output("table-desc", "data"),
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

@app.callback(Output('explainer-obj', 'children'),
              Input('submit-button', 'n_clicks'),
              Input('reset-button', 'n_clicks'),
              State('case-dropdown', 'value'),
              State('num-samples-input', 'value'),
              State('memory-API', 'data'))
def generate_explainer_html(submit_n_clicks, reset_n_clicks, case, n_samples, data):
    ctx = dash.callback_context  # Capture callback context to track button clicks
    empty_obj = html.Iframe(
        srcDoc='''<div>Enter input text to see LIME explanations.</div>''',
        width='100%',
        height='100px',
        style={'border': '2px #d3d3d3 solid'},
        hidden=True,
    )
    if type(case) is list or type(n_samples) is not int or "reset" in ctx.triggered[0]["prop_id"]:
        # Return empty iFrame
        obj = empty_obj
    else:
        data = pd.DataFrame.from_dict(data)
        data.drop(columns="score_pred", axis=1, inplace=True)
        print(data)
        data = np.array(data)
        print(data)
        print(data.shape)
        print(data[0])
        print(data[0].shape)
        exp = lime_explain(lime_explainer, data[0], predict_method, num_features=int(n_samples))

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


# Callbacks pour layout_bivariee
@app.callback(
    Output("col2-out", "children"),
    Input('crossfilter-yaxis-column', "value"),
)
def col2_render(col2_val):
    return "2ème colonne choisie: {}".format(col2_val)


@app.callback(
    Output("col1-out", "children"),
    Input('crossfilter-xaxis-column', "value"),
)
def col1_render(col1_val):
    return "1ère colonne choisie: {}".format(col1_val)

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

    return fig

def create_bar_plot(dff, col, color, title):
    temp = dff[col].value_counts()
    temp_y0 = []
    temp_y1 = []
    count_occ = []
    for val in temp.index:
        temp_y1.append(np.sum(dff["TARGET"][dff[col] == val] == 1))
        temp_y0.append(np.sum(app_train["TARGET"][dff[col] == val] == 0))
        count_occ.append(np.sum([dff[col] == val]))
    trace1 = go.Bar(
        x=(temp_y1 / pd.Series(count_occ)) * 100,
        y=temp.index,
        orientation='h',
        name='NO'
    )
    trace2 = go.Bar(
        x=(temp_y0 / pd.Series(count_occ)) * 100,
        y=temp.index,
        orientation='h',
        name='YES'
    )

    data = [trace1, trace2]
    layout = go.Layout(
        title=title,
        # Barmode=stack permet d'avoir des graphiques empilés
        barmode='stack',
        xaxis=dict(
            title='Pourcentage',
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        ),
        yaxis=dict(
            title=col,
            titlefont=dict(
                size=16,
                color='rgb(107, 107, 107)'
            ),
            tickfont=dict(
                size=14,
                color='rgb(107, 107, 107)'
            )
        )
    )

    fig = go.Figure(data=data, layout=layout)
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
     dash.dependencies.Input('checkbox-value', 'value'),
     dash.dependencies.State('memory-output', 'data')])
def update_x_timeseries(xaxis_column_name, checkbox, data):
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
        data = data[col]
        fig.add_shape(type="line",
                      x0=data.values, y0=min(dff[col]), x1=data.values, y1=max(dff[col]),
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
        return create_bar_plot(dff, col, color, title)

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
        return create_bar_plot(dff, col, color, title)

    if pd.api.types.is_numeric_dtype(app_train[yaxis_column_name]):
        dff = app_train
        title = '<b>{}</b>'.format(yaxis_column_name)
        col = yaxis_column_name
        color = "TARGET"
        return go.Figure(create_box_plot(dff, col, color, title))
