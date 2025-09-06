import pandas as pd
import numpy as np
import re


def extract_year(match_str):
    match = re.search(r'\b(20\d{2})\b', match_str)
    return int(match.group(1)) if match else None


def calculate_tennis_indicators(df):
    indicators = df.copy()

    if 'Wnr/Pt' in indicators.columns:
        winner_pct = pd.to_numeric(indicators['Wnr/Pt'].str.rstrip('%'), errors='coerce') / 100
        indicators['Total_Points_Calc'] = indicators['Winners'] / (winner_pct + 0.001)
    else:
        indicators['Total_Points_Calc'] = (indicators['Winners'] + indicators['UFEs']) * 2.5

    if 'RallyWinners' in indicators.columns and 'RallyUFEs' in indicators.columns:
        rally_total = indicators['RallyWinners'] + indicators['RallyUFEs']
        indicators['Rally_Dominance_Index'] = np.where(rally_total > 0,
                                                       indicators['RallyWinners'] / rally_total, 0.5)
    else:
        indicators['Rally_Dominance_Index'] = 0.5

    if 'Wnr/Pt' in indicators.columns and 'UFE/Pt' in indicators.columns:
        winner_pct = pd.to_numeric(indicators['Wnr/Pt'].str.rstrip('%'), errors='coerce') / 100
        ufe_pct = pd.to_numeric(indicators['UFE/Pt'].str.rstrip('%'), errors='coerce') / 100
        indicators['Tactical_Balance_Score'] = (winner_pct * 2) / (winner_pct + ufe_pct + 0.01)
    else:
        winner_pct = indicators['Winners'] / indicators['Total_Points_Calc']
        ufe_pct = indicators['UFEs'] / indicators['Total_Points_Calc']
        indicators['Tactical_Balance_Score'] = (winner_pct * 2) / (winner_pct + ufe_pct + 0.01)

    if 'FH Wnr/Pt' in indicators.columns and 'BH Wnr/Pt' in indicators.columns:
        fh_rate = pd.to_numeric(indicators['FH Wnr/Pt'].str.rstrip('%'), errors='coerce').fillna(0) / 100
        bh_rate = pd.to_numeric(indicators['BH Wnr/Pt'].str.rstrip('%'), errors='coerce').fillna(0) / 100
        total_baseline = fh_rate + bh_rate + 0.001
        indicators['Power_Asymmetry_Index'] = abs(fh_rate - bh_rate) / total_baseline
    else:
        indicators['Power_Asymmetry_Index'] = 0.3

    if 'vs UFE/Pt' in indicators.columns and 'vs Wnr/Pt' in indicators.columns:
        opp_ufe_rate = pd.to_numeric(indicators['vs UFE/Pt'].str.rstrip('%'), errors='coerce').fillna(0) / 100
        opp_winner_rate = pd.to_numeric(indicators['vs Wnr/Pt'].str.rstrip('%'), errors='coerce').fillna(0) / 100
        indicators['Pressure_Creation_Index'] = opp_ufe_rate / (opp_winner_rate + 0.01)
    else:
        indicators['Pressure_Creation_Index'] = 1.0

    if 'Result' in indicators.columns:
        win_bonus = np.where(indicators['Result'].str.contains('W', na=False), 1.2, 0.8)
        ratio_score = indicators['Winners'] / (indicators['UFEs'] + 1)
        indicators['Match_Control_Efficiency'] = ratio_score * win_bonus
    else:
        indicators['Match_Control_Efficiency'] = indicators['Winners'] / (indicators['UFEs'] + 1)

    if 'RallyWinners' in indicators.columns:
        rally_proportion = indicators['RallyWinners'] / (indicators['Winners'] + 0.001)
        indicators['Shot_Selection_IQ'] = np.where(rally_proportion > 1, 1, rally_proportion)
    else:
        indicators['Shot_Selection_IQ'] = 0.6

    close_match_indicator = np.where(abs(indicators['Winners'] - indicators['UFEs']) <= 5, 1.5, 1.0)
    base_consistency = 1 - (indicators['UFEs'] / indicators['Total_Points_Calc'])
    indicators['Pressure_Consistency_Index'] = base_consistency * close_match_indicator

    return indicators