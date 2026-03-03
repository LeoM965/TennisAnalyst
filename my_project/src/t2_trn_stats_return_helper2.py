import pandas as pd
import numpy as np
from t2_trn_stats_return_helper1 import learn_return_weights
from constants import RETURN_PCT_COLS

def calculate_return_indicators(dataframe):
    return_indicators_df = dataframe.copy()
    
    original_column_names = dataframe.columns.tolist()
    
    has_sufficient_columns = len(original_column_names) >= 19
    if has_sufficient_columns:
        return_column_mapping = {
            original_column_names[3]: 'Overall_RiP',
            original_column_names[4]: 'Overall_RiP_W',
            original_column_names[5]: 'Overall_RetWnr',
            original_column_names[6]: 'FH_BH',
            original_column_names[7]: 'Overall_RDI',
            original_column_names[8]: 'Overall_Slice',
            original_column_names[9]: 'First_RiP',
            original_column_names[10]: 'First_RiP_W',
            original_column_names[11]: 'First_RetWnr',
            original_column_names[12]: 'First_RDI',
            original_column_names[13]: 'First_Slice',
            original_column_names[14]: 'Second_RiP',
            original_column_names[15]: 'Second_RiP_W',
            original_column_names[16]: 'Second_RetWnr',
            original_column_names[17]: 'Second_RDI',
            original_column_names[18]: 'Second_Slice'
        }
        return_indicators_df = return_indicators_df.rename(columns=return_column_mapping)
        
    for pct_col_name in RETURN_PCT_COLS:
        if pct_col_name in return_indicators_df.columns:
            raw_pct_strings = return_indicators_df[pct_col_name].astype(str)
            clean_pct_values = raw_pct_strings.str.rstrip('%')
            
            calculated_ratios = pd.to_numeric(clean_pct_values, errors='coerce') / 100
            return_indicators_df[pct_col_name] = calculated_ratios
            
    rdi_metric_list = ['Overall_RDI', 'First_RDI', 'Second_RDI']
    for rdi_col in rdi_metric_list:
        if rdi_col in return_indicators_df.columns:
            placeholder_values = ['-', 'nan', 'NaN']
            clean_rdi_strings = return_indicators_df[rdi_col].astype(str).replace(placeholder_values, np.nan)
            
            numeric_rdi_values = pd.to_numeric(clean_rdi_strings, errors='coerce')
            return_indicators_df[rdi_col] = numeric_rdi_values
            
    if 'FH_BH' in return_indicators_df.columns:
        shot_split_data = return_indicators_df['FH_BH'].astype(str).str.split('/', expand=True)
        
        if len(shot_split_data.columns) >= 2:
            forehand_return_count = pd.to_numeric(shot_split_data[0], errors='coerce').fillna(0)
            backhand_return_count = pd.to_numeric(shot_split_data[1], errors='coerce').fillna(0)
            
            combined_return_shots = forehand_return_count + backhand_return_count
            forehand_ratio_calc = forehand_return_count / combined_return_shots
            
            return_indicators_df['Return_Forehand_Ratio'] = np.where(combined_return_shots > 0, forehand_ratio_calc, 0.5)
        else:
            return_indicators_df['Return_Forehand_Ratio'] = 0.5
    else:
        return_indicators_df['Return_Forehand_Ratio'] = 0.5
        
    calc_return_weights, calc_serve_type_weights = learn_return_weights(return_indicators_df)
    
    return_indicators_df['Return_In_Play_Rate'] = return_indicators_df['Overall_RiP'].fillna(0)
    return_indicators_df['Return_Win_Efficiency'] = return_indicators_df['Overall_RiP_W'].fillna(0)
    return_indicators_df['Return_Aggression'] = return_indicators_df['Overall_RetWnr'].fillna(0)
    return_indicators_df['Return_Depth_Index'] = return_indicators_df['Overall_RDI'].fillna(0)
    return_indicators_df['Return_Defense_Rate'] = return_indicators_df['Overall_Slice'].fillna(0)
    
    first_serve_components = [
        return_indicators_df['First_RiP'].fillna(0) * 0.4,
        return_indicators_df['First_RiP_W'].fillna(0) * 0.4,
        return_indicators_df['First_RetWnr'].fillna(0) * 0.2
    ]
    first_serve_return_score = sum(first_serve_components)
    
    second_serve_components = [
        return_indicators_df['Second_RiP'].fillna(0) * 0.4,
        return_indicators_df['Second_RiP_W'].fillna(0) * 0.4,
        return_indicators_df['Second_RetWnr'].fillna(0) * 0.2
    ]
    second_serve_return_score = sum(second_serve_components)
    
    return_indicators_df['First_Serve_Return_Quality'] = first_serve_return_score
    return_indicators_df['Second_Serve_Return_Quality'] = second_serve_return_score
    
    quality_adaptation_logic = (
        first_serve_return_score * calc_serve_type_weights[0] + 
        second_serve_return_score * calc_serve_type_weights[1]
    )
    return_indicators_df['Serve_Return_Adaptability'] = quality_adaptation_logic
    
    fallback_overall_depth = return_indicators_df['Overall_RDI'].fillna(0)
    first_serve_depth = return_indicators_df['First_RDI'].fillna(fallback_overall_depth)
    second_serve_depth = return_indicators_df['Second_RDI'].fillna(fallback_overall_depth)
    
    depth_consistency_gap = abs(first_serve_depth - second_serve_depth)
    return_indicators_df['Return_Consistency'] = np.exp(-depth_consistency_gap / 0.5)
    
    aggression_to_defense_ratio = return_indicators_df['Return_Aggression'] / (return_indicators_df['Return_Defense_Rate'] + 0.01)
    return_indicators_df['Return_Tactical_Balance'] = np.tanh(aggression_to_defense_ratio)
    
    depth_multiplier = return_indicators_df['Return_Depth_Index'] / 3.0
    in_play_bonus = 1 + return_indicators_df['Return_In_Play_Rate']
    return_indicators_df['Return_Positioning_Intelligence'] = depth_multiplier * in_play_bonus
    
    quality_performance_gap = abs(first_serve_return_score - second_serve_return_score)
    return_indicators_df['Service_Type_Adaptability'] = np.exp(-quality_performance_gap * 2)
    
    average_defensive_utility = (return_indicators_df['First_Slice'].fillna(0) + return_indicators_df['Second_Slice'].fillna(0)) / 2
    return_indicators_df['Defensive_Versatility'] = average_defensive_utility
    
    if 'Result' in return_indicators_df.columns:
        match_result_mask = return_indicators_df['Result'].str.contains('W', na=False)
        result_impact_multiplier = np.where(match_result_mask, 1.1, 0.9)
        return_indicators_df['Return_Match_Impact'] = return_indicators_df['Serve_Return_Adaptability'] * result_impact_multiplier
    else:
        return_indicators_df['Return_Match_Impact'] = return_indicators_df['Serve_Return_Adaptability']
        
    total_return_game_score = (
        return_indicators_df['Return_In_Play_Rate'] * 0.2 + 
        return_indicators_df['Return_Win_Efficiency'] * 0.2 + 
        return_indicators_df['Return_Aggression'] * 0.15 + 
        return_indicators_df['Serve_Return_Adaptability'] * 0.25 + 
        return_indicators_df['Return_Tactical_Balance'] * 0.2
    )
    return_indicators_df['Overall_Return_Game'] = total_return_game_score
    
    return return_indicators_df