import pandas as pd
import numpy as np

from t2_trn_stats_serve_helper1 import learn_serve_weights


def calculate_serve_indicators(df):
    indicators = df.copy()

    cols = df.columns.tolist()

    column_mapping = {}
    if len(cols) >= 19:
        column_mapping = {
            cols[3]: 'Overall_Unret',
            cols[4]: 'Overall_W3',
            cols[5]: 'Overall_RiP',
            cols[6]: 'First_Unret',
            cols[7]: 'First_W3',
            cols[8]: 'First_RiP',
            cols[9]: 'First_D_Wide',
            cols[10]: 'First_A_Wide',
            cols[11]: 'First_BP_Wide',
            cols[12]: 'Second_Unret',
            cols[13]: 'Second_W3',
            cols[14]: 'Second_RiP',
            cols[15]: 'Second_D_Wide',
            cols[16]: 'Second_A_Wide',
            cols[17]: 'Second_BP_Wide',
            cols[18]: '2ndAgg'
        }

    indicators = indicators.rename(columns=column_mapping)

    pct_cols = ['Overall_Unret', 'Overall_W3', 'Overall_RiP', 'First_Unret', 'First_W3', 'First_RiP',
                'First_D_Wide', 'First_A_Wide', 'First_BP_Wide', 'Second_Unret', 'Second_W3',
                'Second_RiP', 'Second_D_Wide', 'Second_A_Wide', 'Second_BP_Wide']

    for col in pct_cols:
        if col in indicators.columns:
            indicators[col] = indicators[col].astype(str).replace(['-', '', 'nan', 'NaN'], np.nan)
            indicators[col] = pd.to_numeric(indicators[col].astype(str).str.rstrip('%'), errors='coerce') / 100

    if '2ndAgg' in indicators.columns:
        indicators['2ndAgg'] = pd.to_numeric(indicators['2ndAgg'], errors='coerce')
        indicators['2ndAgg'] = np.abs(indicators['2ndAgg']) / 100

    serve_weights, type_weights = learn_serve_weights(indicators)
    indicators['Serve_Power'] = indicators['Overall_Unret'].fillna(0)
    indicators['Serve_Quick_Points'] = indicators['Overall_W3'].fillna(0)
    indicators['Serve_Rally_Control'] = indicators['Overall_RiP'].fillna(0)
    indicators['First_Serve_Power'] = indicators['First_Unret'].fillna(0)

    indicators['First_Serve_Dominance'] = (
            indicators['First_Unret'].fillna(0) * 0.4 +
            indicators['First_W3'].fillna(0) * 0.35 +
            indicators['First_RiP'].fillna(0) * 0.25
    )

    indicators['Second_Serve_Aggression'] = indicators['2ndAgg'].fillna(0)

    indicators['Second_Serve_Effectiveness'] = (
            indicators['Second_Unret'].fillna(0) * 0.3 +
            indicators['Second_W3'].fillna(0) * 0.4 +
            indicators['Second_RiP'].fillna(0) * 0.3
    )

    first_placement_cols = ['First_D_Wide', 'First_A_Wide', 'First_BP_Wide']
    second_placement_cols = ['Second_D_Wide', 'Second_A_Wide', 'Second_BP_Wide']

    first_placement_values = []
    for col in first_placement_cols:
        if col in indicators.columns:
            first_placement_values.append(indicators[col].fillna(0))

    if first_placement_values:
        indicators['First_Serve_Placement_Strategy'] = np.mean(first_placement_values, axis=0)
    else:
        indicators['First_Serve_Placement_Strategy'] = 0

    second_placement_values = []
    for col in second_placement_cols:
        if col in indicators.columns:
            second_placement_values.append(indicators[col].fillna(0))

    if second_placement_values:
        indicators['Second_Serve_Placement_Strategy'] = np.mean(second_placement_values, axis=0)
    else:
        indicators['Second_Serve_Placement_Strategy'] = 0

    indicators['Serve_Type_Adaptability'] = (
            indicators['First_Serve_Dominance'] * type_weights[0] +
            indicators['Second_Serve_Effectiveness'] * type_weights[1]
    )

    first_second_diff = abs(
        indicators['First_Serve_Dominance'] - indicators['Second_Serve_Effectiveness']
    )
    indicators['Serve_Consistency'] = np.exp(-first_second_diff * 2)

    power_score = (indicators['Serve_Power'] + indicators['First_Serve_Power']) / 2
    control_score = (indicators['Serve_Rally_Control'] + indicators['Serve_Quick_Points']) / 2
    indicators['Power_Control_Balance'] = np.tanh(power_score / (control_score + 0.01))

    bp_performance_values = []
    bp_cols = ['First_BP_Wide', 'Second_BP_Wide']
    for col in bp_cols:
        if col in indicators.columns:
            bp_performance_values.append(indicators[col].fillna(0))

    if bp_performance_values:
        indicators['Clutch_Serving'] = np.mean(bp_performance_values, axis=0)
    else:
        indicators['Clutch_Serving'] = 0.5

    placement_versatility = (
                                    indicators['First_Serve_Placement_Strategy'] +
                                    indicators['Second_Serve_Placement_Strategy']
                            ) / 2

    indicators['Serve_Tactical_Intelligence'] = (
            placement_versatility * 0.4 +
            indicators['Serve_Type_Adaptability'] * 0.4 +
            indicators['Clutch_Serving'] * 0.2
    )

    if 'Result' in indicators.columns:
        is_win = indicators['Result'].str.contains('W', na=False)
        win_multiplier = np.where(is_win, 1.1, 0.9)
        indicators['Serve_Match_Impact'] = indicators['Serve_Type_Adaptability'] * win_multiplier
    else:
        indicators['Serve_Match_Impact'] = indicators['Serve_Type_Adaptability']

    indicators['Overall_Serve_Game'] = (
            indicators['Serve_Power'] * 0.2 +
            indicators['Serve_Quick_Points'] * 0.15 +
            indicators['Serve_Rally_Control'] * 0.15 +
            indicators['Serve_Type_Adaptability'] * 0.25 +
            indicators['Serve_Tactical_Intelligence'] * 0.15 +
            indicators['Power_Control_Balance'] * 0.1
    )

    return indicators