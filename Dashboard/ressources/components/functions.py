import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import lime
import time
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

  cat_feat_ix = [i for i, c in enumerate(data.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns) if pd.api.types.is_categorical_dtype(data.drop(['SK_ID_CURR', 'TARGET'], axis=1)[c])]
  feat_names = list(data.drop(['SK_ID_CURR', 'TARGET'], axis=1).columns)
  class_names = list(data["TARGET"].unique())
  lime_explainer = LimeTabularExplainer(np.array(X),
                                      feature_names=feat_names,
                                      class_names=class_names,
                                      categorical_features=cat_feat_ix,
                                      mode="classification"
                                      )
  return lime_explainer

def lime_explain(explainer, data, predict_method, num_features):
  explanation = explainer.explain_instance(data, predict_method, num_features=num_features)
  return explanation

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

