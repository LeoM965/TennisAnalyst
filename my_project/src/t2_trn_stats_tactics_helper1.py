import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_year(match_str):
    match = re.search(r'\b(20\d{2})\b', match_str)
    return int(match.group(1)) if match else None


def learn_tactics_weights(df):
    pca = PCA(n_components=1)
    scaler = StandardScaler()

    tactics_cols = ['Net_Freq', 'Net_W_Pct', 'FH_Wnr_Pct', 'BH_Wnr_Pct']
    available_cols = [col for col in tactics_cols if col in df.columns]

    if len(available_cols) > 0:
        tactics_data = df[available_cols].dropna()
        if len(tactics_data) > 0:
            scaled_data = scaler.fit_transform(tactics_data)
            pca.fit(scaled_data)
            tactics_weights = abs(pca.components_[0])
            tactics_weights = tactics_weights / tactics_weights.sum()
        else:
            tactics_weights = [0.3, 0.3, 0.2, 0.2]
    else:
        tactics_weights = [0.3, 0.3, 0.2, 0.2]

    return tactics_weights