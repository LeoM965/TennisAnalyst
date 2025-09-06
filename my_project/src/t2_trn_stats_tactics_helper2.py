import pandas as pd
import numpy as np

from t2_trn_stats_tactics_helper1 import learn_tactics_weights


def calculate_tactics_indicators(df):
    indicators = df.copy()

    cols = df.columns.tolist()
    column_mapping = {}

    if len(cols) >= 15:
        column_mapping = {
            cols[3]: 'SnV_Freq',
            cols[4]: 'SnV_W_Pct',
            cols[5]: 'Net_Freq',
            cols[6]: 'Net_W_Pct',
            cols[7]: 'FH_Wnr_Pct',
            cols[8]: 'FH_DTL_Wnr_Pct',
            cols[9]: 'FH_IO_Wnr_Pct',
            cols[10]: 'BH_Wnr_Pct',
            cols[11]: 'BH_DTL_Wnr_Pct',
            cols[12]: 'Drop_Freq',
            cols[13]: 'Drop_Wnr_Pct',
            cols[14]: 'RallyAgg',
            cols[15]: 'ReturnAgg'
        }

    indicators = indicators.rename(columns=column_mapping)

    pct_cols = ['SnV_Freq', 'SnV_W_Pct', 'Net_Freq', 'Net_W_Pct', 'FH_Wnr_Pct',
                'FH_DTL_Wnr_Pct', 'FH_IO_Wnr_Pct', 'BH_Wnr_Pct', 'BH_DTL_Wnr_Pct',
                'Drop_Freq', 'Drop_Wnr_Pct']

    for col in pct_cols:
        if col in indicators.columns:
            indicators[col] = indicators[col].astype(str).replace(['-', '', 'nan', 'NaN'], np.nan)
            indicators[col] = pd.to_numeric(indicators[col].astype(str).str.rstrip('%'), errors='coerce') / 100

    agg_cols = ['RallyAgg', 'ReturnAgg']
    for col in agg_cols:
        if col in indicators.columns:
            indicators[col] = indicators[col].astype(str).replace(['-', '', 'nan', 'NaN'], np.nan)
            indicators[col] = pd.to_numeric(indicators[col], errors='coerce')

    tactics_weights = learn_tactics_weights(indicators)

    indicators['Net_Game_Frequency'] = indicators['Net_Freq'].fillna(0)
    indicators['Net_Game_Effectiveness'] = indicators['Net_W_Pct'].fillna(0)
    indicators['Net_Game_Impact'] = (
            indicators['Net_Game_Frequency'] * indicators['Net_Game_Effectiveness']
    )

    indicators['Serve_Volley_Frequency'] = indicators['SnV_Freq'].fillna(0)
    indicators['Serve_Volley_Effectiveness'] = indicators['SnV_W_Pct'].fillna(0)
    indicators['Serve_Volley_Impact'] = (
            indicators['Serve_Volley_Frequency'] * indicators['Serve_Volley_Effectiveness']
    )

    indicators['Forehand_Power'] = indicators['FH_Wnr_Pct'].fillna(0)
    indicators['Backhand_Power'] = indicators['BH_Wnr_Pct'].fillna(0)
    indicators['Groundstroke_Balance'] = abs(indicators['Forehand_Power'] - indicators['Backhand_Power'])
    indicators['Overall_Groundstroke_Power'] = (
                                                       indicators['Forehand_Power'] + indicators['Backhand_Power']
                                               ) / 2

    indicators['Forehand_DTL_Control'] = indicators['FH_DTL_Wnr_Pct'].fillna(0)
    indicators['Forehand_IO_Control'] = indicators['FH_IO_Wnr_Pct'].fillna(0)
    indicators['Backhand_DTL_Control'] = indicators['BH_DTL_Wnr_Pct'].fillna(0)

    fh_direction_variety = indicators['Forehand_DTL_Control'] + indicators['Forehand_IO_Control']
    bh_direction_variety = indicators['Backhand_DTL_Control']
    indicators['Directional_Versatility'] = (fh_direction_variety + bh_direction_variety) / 2

    indicators['Drop_Shot_Usage'] = indicators['Drop_Freq'].fillna(0)
    indicators['Drop_Shot_Effectiveness'] = indicators['Drop_Wnr_Pct'].fillna(0)
    indicators['Drop_Shot_Impact'] = (
            indicators['Drop_Shot_Usage'] * indicators['Drop_Shot_Effectiveness']
    )

    max_rally_agg = 200
    max_return_agg = 200

    indicators['Rally_Aggression'] = np.clip(
        indicators['RallyAgg'].fillna(0) / max_rally_agg, 0, 1
    )
    indicators['Return_Aggression'] = np.clip(
        indicators['ReturnAgg'].fillna(0) / max_return_agg, 0, 1
    )

    indicators['Overall_Aggression'] = (
            indicators['Rally_Aggression'] * 0.6 +
            indicators['Return_Aggression'] * 0.4
    )

    net_approach_rate = indicators['Net_Game_Frequency'] + indicators['Serve_Volley_Frequency']
    baseline_dominance = indicators['Overall_Groundstroke_Power']

    indicators['Court_Position_Strategy'] = np.tanh(
        net_approach_rate / (baseline_dominance + 0.01)
    )

    weapon_variety = (
            (indicators['Net_Game_Impact'] > 0).astype(int) +
            (indicators['Serve_Volley_Impact'] > 0).astype(int) +
            (indicators['Drop_Shot_Impact'] > 0).astype(int) +
            (indicators['Directional_Versatility'] > 0.1).astype(int) +
            (indicators['Overall_Aggression'] > 0.3).astype(int)
    )
    indicators['Tactical_Versatility'] = weapon_variety / 5.0

    power_score = indicators['Overall_Groundstroke_Power'] + indicators['Overall_Aggression']
    finesse_score = indicators['Drop_Shot_Impact'] + indicators['Directional_Versatility']

    indicators['Power_Finesse_Balance'] = np.tanh(
        power_score / (finesse_score + 0.01)
    )

    indicators['Tactical_Adaptability'] = (
            indicators['Tactical_Versatility'] * 0.5 +
            indicators['Power_Finesse_Balance'] * 0.3 +
            indicators['Court_Position_Strategy'] * 0.2
    )

    if 'Result' in indicators.columns:
        is_win = indicators['Result'].str.contains('W', na=False)
        win_multiplier = np.where(is_win, 1.1, 0.9)
        indicators['Tactical_Match_Impact'] = indicators['Tactical_Adaptability'] * win_multiplier
    else:
        indicators['Tactical_Match_Impact'] = indicators['Tactical_Adaptability']

    indicators['Offensive_Efficiency'] = (
            indicators['Overall_Groundstroke_Power'] * 0.4 +
            indicators['Net_Game_Impact'] * 0.3 +
            indicators['Overall_Aggression'] * 0.3
    )

    indicators['Tactical_Intelligence'] = (
            indicators['Tactical_Versatility'] * 0.3 +
            indicators['Directional_Versatility'] * 0.25 +
            indicators['Tactical_Adaptability'] * 0.25 +
            indicators['Power_Finesse_Balance'] * 0.2
    )

    indicators['Overall_Tactical_Game'] = (
            indicators['Offensive_Efficiency'] * 0.3 +
            indicators['Tactical_Intelligence'] * 0.25 +
            indicators['Net_Game_Impact'] * 0.2 +
            indicators['Overall_Aggression'] * 0.15 +
            indicators['Tactical_Versatility'] * 0.1
    )

    return indicators