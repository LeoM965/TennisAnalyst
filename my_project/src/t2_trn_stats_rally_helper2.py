import numpy as np
import pandas as pd
from t2_trn_stats_rally_helper1 import learn_match_control_weights, learn_weights_from_data
from constants import RALLY_PERCENTAGE_COLS

def calculate_rally_indicators(dataframe):
    rally_indicators_df = dataframe.copy()
    
    percentage_column_list = RALLY_PERCENTAGE_COLS + ['FH/GS', 'BH Slice%']
    
    for col_name in percentage_column_list:
        if col_name in rally_indicators_df.columns:
            raw_column_values = rally_indicators_df[col_name].astype(str)
            numeric_strings = raw_column_values.str.rstrip('%')
            
            calculated_fractions = pd.to_numeric(numeric_strings, errors='coerce') / 100
            rally_indicators_df[col_name] = calculated_fractions
            
    rally_importance_weights, tactical_mix_weights = learn_weights_from_data(rally_indicators_df)
    short_weight, long_weight = learn_match_control_weights(rally_indicators_df)
    
    average_rally_length = rally_indicators_df['RallyLen']
    rally_indicators_df['Rally_Length_Efficiency'] = np.exp(-average_rally_length / 5)
    
    serve_rally_length = rally_indicators_df['RLen-Serve']
    return_rally_length = rally_indicators_df['RLen-Return']
    
    length_difference = abs(serve_rally_length - return_rally_length)
    length_sum = serve_rally_length + return_rally_length
    
    rally_indicators_df['Serve_Return_Rally_Balance'] = 1 - (length_difference / (length_sum + 1))
    
    rally_indicators_df['Short_Rally_Control'] = rally_indicators_df['1-3 W%']
    rally_indicators_df['Long_Rally_Endurance'] = rally_indicators_df['10+ W%']
    
    progression_components = [
        rally_indicators_df['1-3 W%'] * rally_importance_weights[0],
        rally_indicators_df['4-6 W%'] * rally_importance_weights[1],
        rally_indicators_df['7-9 W%'] * rally_importance_weights[2],
        rally_indicators_df['10+ W%'] * rally_importance_weights[3]
    ]
    
    stacked_progression = np.column_stack(progression_components)
    rally_indicators_df['Rally_Progression_Score'] = np.sum(stacked_progression, axis=1)
    
    rally_indicators_df['Forehand_Dominance'] = rally_indicators_df['FH/GS']
    rally_indicators_df['Backhand_Versatility'] = 1 - rally_indicators_df['BH Slice%']
    
    effective_fh_power_ratio = rally_indicators_df['FHP/100'] + 0.1
    effective_bh_power_ratio = rally_indicators_df['BHP/100'] + 0.1
    
    rally_indicators_df['Forehand_Power_Index'] = rally_indicators_df['FHP'] / effective_fh_power_ratio
    rally_indicators_df['Backhand_Power_Index'] = rally_indicators_df['BHP'] / effective_bh_power_ratio
    
    fh_raw_strength = abs(rally_indicators_df['FHP'])
    bh_raw_strength = abs(rally_indicators_df['BHP'])
    
    combined_power_strength = fh_raw_strength + bh_raw_strength
    power_strength_gap = abs(fh_raw_strength - bh_raw_strength)
    
    power_balance_calculation = 1 - (power_strength_gap / combined_power_strength)
    rally_indicators_df['Power_Balance_Index'] = np.where(combined_power_strength > 0, power_balance_calculation, 0.5)
    
    performance_matrix_values = [
        rally_indicators_df['1-3 W%'], 
        rally_indicators_df['4-6 W%'], 
        rally_indicators_df['7-9 W%'], 
        rally_indicators_df['10+ W%']
    ]
    
    stacked_performance = np.column_stack(performance_matrix_values)
    performance_spread = np.nanstd(stacked_performance, axis=1)
    rally_indicators_df['Rally_Adaptability'] = np.exp(-performance_spread * 4)
    
    if 'Result' in rally_indicators_df.columns:
        match_results = rally_indicators_df['Result']
        won_mask = match_results.str.contains('W', na=False)
        result_bonus_multiplier = np.where(won_mask, 1.15, 0.85)
        
        weighted_rally_performance = (
            rally_indicators_df['Short_Rally_Control'] * short_weight + 
            rally_indicators_df['Long_Rally_Endurance'] * long_weight
        )
        rally_indicators_df['Match_Control_Metric'] = weighted_rally_performance * result_bonus_multiplier
    else:
        rally_indicators_df['Match_Control_Metric'] = rally_indicators_df['Rally_Progression_Score']
        
    tactical_logic_components = [
        rally_indicators_df['Forehand_Dominance'] * tactical_mix_weights[0],
        rally_indicators_df['Backhand_Versatility'] * tactical_mix_weights[1],
        rally_indicators_df['Rally_Adaptability'] * tactical_mix_weights[2]
    ]
    
    stacked_tactical = np.column_stack(tactical_logic_components)
    rally_indicators_df['Tactical_Intelligence'] = np.sum(stacked_tactical, axis=1)
    
    serve_length_limit = rally_indicators_df['RLen-Serve'] / 6
    return_length_limit = rally_indicators_df['RLen-Return'] / 6
    
    serve_coverage = np.exp(-serve_length_limit)
    return_coverage = np.exp(-return_length_limit)
    
    rally_indicators_df['Court_Coverage_Efficiency'] = (serve_coverage + return_coverage) / 2
    
    return rally_indicators_df
