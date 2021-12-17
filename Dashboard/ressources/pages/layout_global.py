# Importation des packages
from flask import Flask, request
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np

import requests


####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

####################### Corporate css formatting
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
def get_navbar(p = 'sales'):

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
                html.H4(children = 'Page 3',
                        style=navbarcurrentpage),
                href='/apps/page3'
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
                html.H4(children = 'Page 3',
                        style=navbarcurrentpage),
                href='/apps/page3'
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

    navbar_page3 = html.Div([

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
                html.H4(children = 'Page 3',
                        style = navbarcurrentpage),
                href='/apps/page3'
                )
        ],
        className='col-2'),

        html.Div([], className = 'col-3')

    ],
    className = 'row',
    style = {'background-color' : corporate_colors['dark-green'],
            'box-shadow': '2px 5px 5px 1px rgba(255, 101, 131, .5)'}
    )

    if p == 'sales':
        return navbar_global
    elif p == 'page2':
        return navbar_client
    else:
        return navbar_page3
#####################
#####################
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
path = "C:/Users/pbliv/Documents/Data Science/P7/application_train.csv"
app_train = pd.read_csv(path)

temp = app_train["TARGET"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
fig_target = px.pie(df, values='values', names='labels', title='Loan Repayed or not')


temp = app_train["NAME_CONTRACT_TYPE"].value_counts()
fig_set = {
    "data": [
        {
            "values": temp.values,
            "labels": temp.index,
            "domain": {"x": [0, .48]},
            # "name": "Types of Loans",
            # "hoverinfo":"label+percent+name",
            "hole": .7,
            "type": "pie"
        },

    ],
    "layout": {
        "title": "Types of loan",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "",
                "x": 0.17,
                "y": 0.5
            }

        ]
    }
}
fig_contract_type = go.Figure(fig_set)


temp = app_train["NAME_INCOME_TYPE"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
fig_income_type = px.pie(df, values='values', names='labels', title='Source de revenus pour les demandeurs',hole=0.5)




temp = app_train["NAME_INCOME_TYPE"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
for val in temp.index:
    temp_y1.append(np.sum(app_train["TARGET"][app_train["NAME_INCOME_TYPE"]==val] == 1))
    temp_y0.append(np.sum(app_train["TARGET"][app_train["NAME_INCOME_TYPE"]==val] == 0))
trace1 = go.Bar(
    x = temp.index,
    y = (temp_y1 / temp.sum()) * 100,
    name='NO'
)
trace2 = go.Bar(
    x = temp.index,
    y = (temp_y0 / temp.sum()) * 100,
    name='YES'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Income sources of Applicant's in terms of loan is repayed or not  in %",
    #barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Income source',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count in %',
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

fig_income_type_target = go.Figure(data=data, layout=layout)


temp = app_train["OCCUPATION_TYPE"].value_counts()
#print(temp.values)
temp_y0 = []
temp_y1 = []
count_occ = []
for val in temp.index:
    temp_y1.append(np.sum(app_train["TARGET"][app_train["OCCUPATION_TYPE"]==val] == 1))
    temp_y0.append(np.sum(app_train["TARGET"][app_train["OCCUPATION_TYPE"]==val] == 0))
    count_occ.append(np.sum([app_train["OCCUPATION_TYPE"]==val]))
trace1 = go.Bar(
    x = (temp_y1 / pd.Series(count_occ)) * 100,
    y = temp.index,
    orientation='h',
    name='NO'
)
trace2 = go.Bar(
    x = (temp_y0 /  pd.Series(count_occ)) * 100,
    y = temp.index,
    orientation='h',
    name='YES'
)

data = [trace1, trace2]
layout = go.Layout(
    title = "Occupation of Applicant's in terms of loan is repayed or not in %",
    #Barmode=stack permet d'avoir des graphiques empilés
    barmode='stack',
    width = 1000,
    xaxis=dict(
        title='Pourcentage',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Emploi du demandeur',
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

fig_occupation_type_target = go.Figure(data=data, layout=layout)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

## Layout de la page client
app.layout = html.Div([

    #####################
    #Row 1 : Header
    get_header(),

    #####################
    #Row 2 : Nav bar
    get_navbar('Client'),
    #####################
    #Row 3 : Filters
    html.Div([  # External row

        html.Div([  # External 12-column

            html.Div([  # Internal row

                         # Internal columns
                         html.Div([
                         ],
                             className='col-2'),  # Blank 2 columns
        ],
            className = 'row') # Internal row
        ],
            className='col-12',
            style=filterdiv_borderstyling)  # External 12-column

    ],
        className='row sticky-top'),  # External row

#####################
    #Row 4
    get_emptyrow(),

#####################
    #Row 5 : Charts
    html.Div([ # External row

        html.Div([
        ],
        className = 'col-1'), # Blank 1 column

        html.Div([ # External 10-column

            html.H2(children = "Description globale",
                    style = {'color' : corporate_colors['white']}),

            html.Div([ # Internal row - RECAPS

            ],
                className='row',
                style=recapdiv
            ),  # Internal row - RECAPS

            html.Div([ # Internal row

                # Distribution de la target
                html.Div([dcc.Graph(
                    figure=fig_target
                )
                ],
                    className='col-6'),  # Empty column
                #Distribution de 'income_type'
                html.Div([
                    dcc.Graph(figure=fig_income_type)],
                    className='col-6'),
                #Distribution de contract_type
                html.Div([
                    dcc.Graph(figure=fig_contract_type)]
                    , className='col-6'),  # Empty column
                #Distribution de income type selon la target
                html.Div([
                    dcc.Graph(figure=fig_income_type_target)]
                    , className='col-6'),  # Empty column
                #Distribution de occupation type selon la target
                html.Div([
                    dcc.Graph(figure=fig_occupation_type_target)]
                    , className='col-6'),  # Empty column
            ],
            className = 'row'), # Internal row
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
#@app.callback(
    #Output("output", "children"),
    #Input("row_client", "value"),
#)


#@app.callback(
    #Output("table-container", "data"),
    #Input("filter_dropdown", "value")
#)

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)

if rad == '🔎 Further explore data':
    with eda:
        st.header("**Overview of exploratory data analysis.** \n ----")
        st.subheader("Plotting distributions of target and some features.")

        col1, col2, col3 = st.columns(3)  # 3 cols with histogram = home-made func
        col1.plotly_chart(histogram(df_train, x='TARGET'), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='CODE_GENDER'), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='EXT_SOURCE_1'), use_container_width=True)

        st.subheader("Let's plot some extra numerical features of your choice.")
        # letting user choose num & cat feats from dropdown
        col1, col2, col3 = st.columns(3)
        num_col = df_train.select_dtypes(include=np.number).columns.sort_values()
        input1 = col1.selectbox('1st plot', num_col)
        input2 = col2.selectbox('2nd plot', num_col[1:])
        input3 = col3.selectbox('3rd plot', num_col[2:])

        st.subheader("Now, you may pick some categorical features to plot.")
        col4, col5, col6 = st.columns(3)
        cat_col = df_train.select_dtypes(exclude=np.number).columns.sort_values()
        input4 = col4.selectbox('1st plot', cat_col[1:])
        input5 = col5.selectbox('2nd plot', cat_col[2:])
        input6 = col6.selectbox('3rd plot', cat_col[3:])

        button = st.button('Plot it! ')
        if button:
            col1.plotly_chart(histogram(df_train, x=input1, legend=False), use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=input2, legend=False), use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=input3, legend=False), use_container_width=True)
            col4.plotly_chart(histogram(df_train, x=input4, legend=False), use_container_width=True)
            col5.plotly_chart(histogram(df_train, x=input5, legend=False), use_container_width=True)
            col6.plotly_chart(histogram(df_train, x=input6, legend=False), use_container_width=True)