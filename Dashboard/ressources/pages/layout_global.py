# Importation des packages
from flask import Flask, request
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash.dependencies import Input, Output
from ressources.components.pages_plugin import *
import plotly.express as px
import plotly.graph_objects as go
import base64
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
image_filename = 'C:/Users/pbliv/PycharmProjects/flaskProject/ressources/assets/logo_pret_a_depenser.PNG'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
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
                    src = "C:/Users/pbliv/PycharmProjects/flaskProject/ressources/assets/logo_pret_a_depenser.PNG/png;base64,{}".format(encoded_image),
                    height = '100',
                    width = '200',
                    className='img')

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

## Effect of age on the repayment
# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data["DAYS_BIRTH"] = abs(age_data["DAYS_BIRTH"])
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365
# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins=np.linspace(20, 70, num=11))
# Group by the bin and calculate averages
age_groups = age_data.groupby('YEARS_BINNED').mean()

age_groups["YEARS_BINNED"] = age_groups.index
age_groups["YEARS_BINNED"] = age_groups["YEARS_BINNED"].astype(str)
age_groups["TARGET"] = 100*age_groups["TARGET"]

# Graph the age bins and the average of the target as a bar plot
fig_years_binned = go.Figure(px.bar(age_groups, x="YEARS_BINNED", y="TARGET", labels={"YEARS_BINNED": "Bins of age","TARGET": "Failure to Repay (%)"},
                                    title='Failure to Repay by Age Group'))

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
layout_income = go.Layout(
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

fig_income_type_target = go.Figure(data=data, layout=layout_income)


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
layout_occupation = go.Layout(
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

fig_occupation_type_target = go.Figure(data=data, layout=layout_occupation)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
dash.register_page(__name__, path="/apps/global")
## Layout de la page globale
layout = html.Div([

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
                html.Div([dcc.Markdown('''
                             # Implémentez un modèle de scoring
                             Prêt à dépenser souhaite mettre en oeuvre un outil de "scoring crédit" 
                             pour calculer la probabilité qu'un client rembourse son crédit, puis classifier la demande en accordée/refusée.
                             Le dashboard sera mis à disposition pour que les chargés de relation client puissent à la fois expliquer 
                             les décisions d'octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement

                             Les données sont disponibles à cette [adresse](https://www.kaggle.com/c/home-credit-default-risk/data)
                             ''')
            ],
                className='col-4',
                style = {'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'color': 'white'})

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
                #Distribution des tranches d'âge selon la target
                html.Div([
                    dcc.Graph(figure=fig_years_binned)]
                    , className='col-6')  # Empty column
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