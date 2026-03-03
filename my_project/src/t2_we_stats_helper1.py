import re
import numpy as np
import pandas as pd
from constants import WEB_PCT_COLS

def extract_year(match_string):
    if not isinstance(match_string, str):
        return None
        
    year_capture_pattern = r'\b(20\d{2})\b'
    match_result = re.search(year_capture_pattern, match_string)
    
    if match_result:
        year_string_value = match_result.group(1)
        return int(year_string_value)
        
    return None

def calculate_tennis_indicators(dataframe):
    tennis_indicators_df = dataframe.copy()
    
    for pct_metric_col in WEB_PCT_COLS:
        if pct_metric_col in tennis_indicators_df.columns:
            raw_pct_strings = tennis_indicators_df[pct_metric_col].astype(str)
            clean_pct_numbers = raw_pct_strings.str.rstrip('%')
            
            calculated_fractions = pd.to_numeric(clean_pct_numbers, errors='coerce') / 100
            tennis_indicators_df[pct_metric_col] = calculated_fractions
            
    winners_per_point = tennis_indicators_df['Wnr/Pt'].fillna(0)
    errors_per_point = tennis_indicators_df['UFE/Pt'].fillna(0)
    errors_forced_per_point = tennis_indicators_df['vs UFE/Pt'].fillna(0)
    winners_allowed_per_point = tennis_indicators_df['vs Wnr/Pt'].fillna(0)
    
    raw_dominance_score = winners_per_point - winners_allowed_per_point
    dominance_bonus = errors_forced_per_point - errors_per_point
    
    tennis_indicators_df['Rally_Dominance_Index'] = raw_dominance_score + dominance_bonus
    
    forehand_winners = tennis_indicators_df['FH Wnr/Pt'].fillna(0)
    backhand_winners = tennis_indicators_df['BH Wnr/Pt'].fillna(0)
    
    combined_winner_power = forehand_winners + backhand_winners + 0.001
    forehand_contribution_ratio = forehand_winners / combined_winner_power
    
    tennis_indicators_df['Tactical_Balance_Score'] = 1 - abs(0.5 - forehand_contribution_ratio)
    
    power_imbalance_gap = abs(forehand_winners - backhand_winners)
    tennis_indicators_df['Power_Asymmetry_Index'] = power_imbalance_gap
    
    winners_weight = 0.6
    forced_errors_weight = 0.4
    
    pressure_formula = (winners_per_point * winners_weight) + (errors_forced_per_point * forced_errors_weight)
    tennis_indicators_df['Pressure_Creation_Index'] = pressure_formula
    
    combined_activity_metric = winners_per_point + errors_per_point + 0.001
    efficiency_calculation = (winners_per_point - errors_per_point) / combined_activity_metric
    
    tennis_indicators_df['Match_Control_Efficiency'] = efficiency_calculation
    
    rally_winners_count = tennis_indicators_df['RallyWinners'].fillna(0)
    rally_errors_count = tennis_indicators_df['RallyUFEs'].fillna(0)
    
    total_rally_events = rally_winners_count + rally_errors_count + 0.001
    selection_iq_calculation = rally_winners_count / total_rally_events
    
    tennis_indicators_df['Shot_Selection_IQ'] = selection_iq_calculation
    
    volatility_between_metrics = abs(pressure_formula - selection_iq_calculation)
    tennis_indicators_df['Pressure_Consistency_Index'] = 1 - volatility_between_metrics
    
    return tennis_indicators_df