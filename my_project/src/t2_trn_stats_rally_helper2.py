import numpy as np
import pandas as pd

from t2_trn_stats_rally_helper1 import learn_match_control_weights, learn_weights_from_data


def calculate_rally_indicators(df):
    indicators = df.copy()

    pct_cols = ['1-3 W%', '4-6 W%', '7-9 W%', '10+ W%', 'FH/GS', 'BH Slice%']
    for col in pct_cols:
        if col in indicators.columns:
            indicators[col] = pd.to_numeric(indicators[col].astype(str).str.rstrip('%'), errors='coerce') / 100

    rally_weights, tactical_weights = learn_weights_from_data(indicators)
    short_weight, long_weight = learn_match_control_weights(indicators)

    indicators['Rally_Length_Efficiency'] = np.exp(-indicators['RallyLen'] / 5)

    serve_return_diff = abs(indicators['RLen-Serve'] - indicators['RLen-Return'])
    serve_return_sum = indicators['RLen-Serve'] + indicators['RLen-Return']
    indicators['Serve_Return_Rally_Balance'] = 1 - (serve_return_diff / (serve_return_sum + 1))

    indicators['Short_Rally_Control'] = indicators['1-3 W%']
    indicators['Long_Rally_Endurance'] = indicators['10+ W%']

    rally_data = np.column_stack([
        indicators['1-3 W%'] * rally_weights[0],
        indicators['4-6 W%'] * rally_weights[1],
        indicators['7-9 W%'] * rally_weights[2],
        indicators['10+ W%'] * rally_weights[3]
    ])
    indicators['Rally_Progression_Score'] = np.sum(rally_data, axis=1)

    indicators['Forehand_Dominance'] = indicators['FH/GS']
    indicators['Backhand_Versatility'] = 1 - indicators['BH Slice%']

    indicators['Forehand_Power_Index'] = indicators['FHP'] / (indicators['FHP/100'] + 0.1)
    indicators['Backhand_Power_Index'] = indicators['BHP'] / (indicators['BHP/100'] + 0.1)

    fh_power = abs(indicators['FHP'])
    bh_power = abs(indicators['BHP'])
    total_power = fh_power + bh_power
    power_diff = abs(fh_power - bh_power)
    indicators['Power_Balance_Index'] = np.where(total_power > 0, 1 - power_diff / total_power, 0.5)

    rally_matrix = np.column_stack([
        indicators['1-3 W%'],
        indicators['4-6 W%'],
        indicators['7-9 W%'],
        indicators['10+ W%']
    ])
    rally_std = np.nanstd(rally_matrix, axis=1)
    indicators['Rally_Adaptability'] = np.exp(-rally_std * 4)

    if 'Result' in indicators.columns:
        is_win = indicators['Result'].str.contains('W', na=False)
        win_multiplier = np.where(is_win, 1.15, 0.85)
        base_control = (indicators['Short_Rally_Control'] * short_weight +
                        indicators['Long_Rally_Endurance'] * long_weight)
        indicators['Match_Control_Metric'] = base_control * win_multiplier
    else:
        indicators['Match_Control_Metric'] = indicators['Rally_Progression_Score']

    tactical_components = np.column_stack([
        indicators['Forehand_Dominance'] * tactical_weights[0],
        indicators['Backhand_Versatility'] * tactical_weights[1],
        indicators['Rally_Adaptability'] * tactical_weights[2]
    ])
    indicators['Tactical_Intelligence'] = np.sum(tactical_components, axis=1)

    serve_coverage = np.exp(-indicators['RLen-Serve'] / 6)
    return_coverage = np.exp(-indicators['RLen-Return'] / 6)
    indicators['Court_Coverage_Efficiency'] = (serve_coverage + return_coverage) / 2

    return indicators