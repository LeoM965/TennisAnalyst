import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_year(match_str):
    match = re.search(r'\b(20\d{2})\b', match_str)
    return int(match.group(1)) if match else None


def learn_serve_weights(df):
    pca = PCA(n_components=1)
    scaler = StandardScaler()

    serve_cols = ['Overall_Unret', 'Overall_W3', 'Overall_RiP']
    available_serve_cols = [col for col in serve_cols if col in df.columns]

    if len(available_serve_cols) > 0:
        serve_data = df[available_serve_cols].dropna()
        if len(serve_data) > 0:
            scaled_data = scaler.fit_transform(serve_data)
            pca.fit(scaled_data)
            serve_weights = abs(pca.components_[0])
            serve_weights = serve_weights / serve_weights.sum()
        else:
            serve_weights = [0.4, 0.3, 0.3]
    else:
        serve_weights = [0.4, 0.3, 0.3]

    if 'First_Unret' in df.columns and 'Second_Unret' in df.columns:
        first_second_data = df[['First_Unret', 'Second_Unret']].dropna()

        if len(first_second_data) > 0:
            first_importance = first_second_data['First_Unret'].std()
            second_importance = first_second_data['Second_Unret'].std()
            total_importance = first_importance + second_importance + 0.001
            type_weights = [first_importance / total_importance, second_importance / total_importance]
        else:
            type_weights = [0.7, 0.3]
    else:
        type_weights = [0.7, 0.3]

    return serve_weights, type_weights