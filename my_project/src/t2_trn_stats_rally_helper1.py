import pandas as pd
import numpy as np
import os
import shutil
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_year(match_str):
    match = re.search(r'\b(20\d{2})\b', match_str)
    return int(match.group(1)) if match else None


def learn_weights_from_data(df):
    pca = PCA(n_components=1)
    scaler = StandardScaler()

    rally_cols = ['1-3 W%', '4-6 W%', '7-9 W%', '10+ W%']
    rally_data = df[rally_cols].dropna()

    if len(rally_data) > 0:
        scaled_data = scaler.fit_transform(rally_data)
        pca.fit(scaled_data)
        rally_weights = abs(pca.components_[0])
        rally_weights = rally_weights / rally_weights.sum()
    else:
        rally_weights = [0.25, 0.25, 0.25, 0.25]

    power_cols = ['FHP', 'BHP']
    power_data = df[power_cols].dropna()

    if len(power_data) > 0:
        power_corr = power_data.corr().iloc[0, 1]
        if abs(power_corr) > 0.5:
            tactical_weights = [0.4, 0.4, 0.2]
        else:
            tactical_weights = [0.3, 0.3, 0.4]
    else:
        tactical_weights = [0.3, 0.3, 0.4]

    return rally_weights, tactical_weights


def learn_match_control_weights(df):
    if 'Result' not in df.columns:
        return 1.0, 1.0

    win_data = df[df['Result'].str.contains('W', na=False)]
    loss_data = df[df['Result'].str.contains('L', na=False)]

    if len(win_data) > 0 and len(loss_data) > 0:
        win_avg_short = win_data['1-3 W%'].mean()
        loss_avg_short = loss_data['1-3 W%'].mean()
        win_avg_long = win_data['10+ W%'].mean()
        loss_avg_long = loss_data['10+ W%'].mean()

        short_importance = abs(win_avg_short - loss_avg_short) if not pd.isna(win_avg_short) and not pd.isna(
            loss_avg_short) else 0.4
        long_importance = abs(win_avg_long - loss_avg_long) if not pd.isna(win_avg_long) and not pd.isna(
            loss_avg_long) else 0.6

        total_importance = short_importance + long_importance + 0.001
        short_weight = short_importance / total_importance
        long_weight = long_importance / total_importance

        return short_weight, long_weight
    else:
        return 0.4, 0.6