import pandas as pd
import numpy as np

from t2_trn_stats_return_helper1 import learn_return_weights


def calculate_return_indicators(df):
    indicators = df.copy()

    cols = df.columns.tolist()
    column_mapping = {}

    if len(cols) >= 18:
        column_mapping = {
            cols[3]: 'Overall_RiP',  # RiP%
            cols[4]: 'Overall_RiP_W',  # RiP W%
            cols[5]: 'Overall_RetWnr',  # RetWnr%
            cols[6]: 'FH_BH',  # FH/BH
            cols[7]: 'Overall_RDI',  # RDI
            cols[8]: 'Overall_Slice',  # Slice%
            cols[9]: 'First_RiP',  # 1st: RiP%
            cols[10]: 'First_RiP_W',  # RiP W% (first serve)
            cols[11]: 'First_RetWnr',  # RetWnr% (first serve)
            cols[12]: 'First_RDI',  # RDI (first serve)
            cols[13]: 'First_Slice',  # Slice% (first serve)
            cols[14]: 'Second_RiP',  # 2nd: RiP%
            cols[15]: 'Second_RiP_W',  # RiP W% (second serve)
            cols[16]: 'Second_RetWnr',  # RetWnr% (second serve)
            cols[17]: 'Second_RDI',  # RDI (second serve)
            cols[18]: 'Second_Slice'  # Slice% (second serve)
        }

    indicators = indicators.rename(columns=column_mapping)

    pct_cols = ['Overall_RiP', 'Overall_RiP_W', 'Overall_RetWnr', 'Overall_Slice',
                'First_RiP', 'First_RiP_W', 'First_RetWnr', 'First_Slice',
                'Second_RiP', 'Second_RiP_W', 'Second_RetWnr', 'Second_Slice']

    for col in pct_cols:
        if col in indicators.columns:
            indicators[col] = indicators[col].astype(str).replace(['-', '', 'nan', 'NaN'], np.nan)
            indicators[col] = pd.to_numeric(indicators[col].astype(str).str.rstrip('%'), errors='coerce') / 100

    rdi_cols = ['Overall_RDI', 'First_RDI', 'Second_RDI']
    for col in rdi_cols:
        if col in indicators.columns:
            indicators[col] = indicators[col].astype(str).replace(['-', '', 'nan', 'NaN'], np.nan)
            indicators[col] = pd.to_numeric(indicators[col], errors='coerce')

    if 'FH_BH' in indicators.columns:
        fh_bh_data = indicators['FH_BH'].astype(str).str.split('/', expand=True)
        if len(fh_bh_data.columns) >= 2:
            fh_count = pd.to_numeric(fh_bh_data[0], errors='coerce').fillna(0)
            bh_count = pd.to_numeric(fh_bh_data[1], errors='coerce').fillna(0)
            total_shots = fh_count + bh_count
            indicators['Return_Forehand_Ratio'] = np.where(total_shots > 0, fh_count / total_shots, 0.5)
        else:
            indicators['Return_Forehand_Ratio'] = 0.5
    else:
        indicators['Return_Forehand_Ratio'] = 0.5

    return_weights, serve_weights = learn_return_weights(indicators)

    indicators['Return_In_Play_Rate'] = indicators['Overall_RiP'].fillna(0)
    indicators['Return_Win_Efficiency'] = indicators['Overall_RiP_W'].fillna(0)
    indicators['Return_Aggression'] = indicators['Overall_RetWnr'].fillna(0)
    indicators['Return_Depth_Index'] = indicators['Overall_RDI'].fillna(0)
    indicators['Return_Defense_Rate'] = indicators['Overall_Slice'].fillna(0)

    first_serve_score = (
            indicators['First_RiP'].fillna(0) * 0.4 +
            indicators['First_RiP_W'].fillna(0) * 0.4 +
            indicators['First_RetWnr'].fillna(0) * 0.2
    )

    second_serve_score = (
            indicators['Second_RiP'].fillna(0) * 0.4 +
            indicators['Second_RiP_W'].fillna(0) * 0.4 +
            indicators['Second_RetWnr'].fillna(0) * 0.2
    )

    indicators['First_Serve_Return_Quality'] = first_serve_score
    indicators['Second_Serve_Return_Quality'] = second_serve_score

    indicators['Serve_Return_Adaptability'] = (
            indicators['First_Serve_Return_Quality'] * serve_weights[0] +
            indicators['Second_Serve_Return_Quality'] * serve_weights[1]
    )

    first_rdi = indicators['First_RDI'].fillna(indicators['Overall_RDI'].fillna(0))
    second_rdi = indicators['Second_RDI'].fillna(indicators['Overall_RDI'].fillna(0))
    depth_variance = abs(first_rdi - second_rdi)
    indicators['Return_Consistency'] = np.exp(-depth_variance / 0.5)

    attack_defense_balance = indicators['Return_Aggression'] / (indicators['Return_Defense_Rate'] + 0.01)
    indicators['Return_Tactical_Balance'] = np.tanh(attack_defense_balance)

    indicators['Return_Positioning_Intelligence'] = (
            (indicators['Return_Depth_Index'] / 3.0) * (1 + indicators['Return_In_Play_Rate'])
    )

    first_vs_second_diff = abs(indicators['First_Serve_Return_Quality'] - indicators['Second_Serve_Return_Quality'])
    indicators['Service_Type_Adaptability'] = np.exp(-first_vs_second_diff * 2)

    first_slice = indicators['First_Slice'].fillna(0)
    second_slice = indicators['Second_Slice'].fillna(0)
    indicators['Defensive_Versatility'] = (first_slice + second_slice) / 2

    if 'Result' in indicators.columns:
        is_win = indicators['Result'].str.contains('W', na=False)
        win_multiplier = np.where(is_win, 1.1, 0.9)
        indicators['Return_Match_Impact'] = indicators['Serve_Return_Adaptability'] * win_multiplier
    else:
        indicators['Return_Match_Impact'] = indicators['Serve_Return_Adaptability']

    indicators['Overall_Return_Game'] = (
            indicators['Return_In_Play_Rate'] * 0.2 +
            indicators['Return_Win_Efficiency'] * 0.2 +
            indicators['Return_Aggression'] * 0.15 +
            indicators['Serve_Return_Adaptability'] * 0.25 +
            indicators['Return_Tactical_Balance'] * 0.2
    )

    return indicators