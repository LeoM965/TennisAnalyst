import pandas as pd
import numpy as np
from t2_trn_stats_serve_helper1 import learn_serve_weights
from constants import SERVE_PCT_COLS

def calculate_serve_indicators(dataframe):
    serve_indicators_df = dataframe.copy()
    
    original_column_list = dataframe.columns.tolist()
    
    has_necessary_columns = len(original_column_list) >= 19
    if has_necessary_columns:
        serve_column_mapping = {
            original_column_list[3]: 'Overall_Unret',
            original_column_list[4]: 'Overall_W3',
            original_column_list[5]: 'Overall_RiP',
            original_column_list[6]: 'First_Unret',
            original_column_list[7]: 'First_W3',
            original_column_list[8]: 'First_RiP',
            original_column_list[9]: 'First_D_Wide',
            original_column_list[10]: 'First_A_Wide',
            original_column_list[11]: 'First_BP_Wide',
            original_column_list[12]: 'Second_Unret',
            original_column_list[13]: 'Second_W3',
            original_column_list[14]: 'Second_RiP',
            original_column_list[15]: 'Second_D_Wide',
            original_column_list[16]: 'Second_A_Wide',
            original_column_list[17]: 'Second_BP_Wide',
            original_column_list[18]: 'Second_Serve_Aggression_Raw'
        }
        serve_indicators_df = serve_indicators_df.rename(columns=serve_column_mapping)
        
    for serve_pct_col in SERVE_PCT_COLS:
        if serve_pct_col in serve_indicators_df.columns:
            raw_serve_strings = serve_indicators_df[serve_pct_col].astype(str)
            clean_serve_pct = raw_serve_strings.str.rstrip('%')
            
            numeric_serve_ratios = pd.to_numeric(clean_serve_pct, errors='coerce') / 100
            serve_indicators_df[serve_pct_col] = numeric_serve_ratios
            
    if 'Second_Serve_Aggression_Raw' in serve_indicators_df.columns:
        raw_aggression_values = pd.to_numeric(serve_indicators_df['Second_Serve_Aggression_Raw'], errors='coerce')
        absolute_aggression_magnitude = np.abs(raw_aggression_values)
        
        serve_indicators_df['Second_Serve_Aggression'] = absolute_aggression_magnitude / 100
    else:
        serve_indicators_df['Second_Serve_Aggression'] = 0
        
    calc_serve_weights, calc_type_weights = learn_serve_weights(serve_indicators_df)
    
    serve_indicators_df['Serve_Power'] = serve_indicators_df['Overall_Unret'].fillna(0)
    serve_indicators_df['Serve_Quick_Points'] = serve_indicators_df['Overall_W3'].fillna(0)
    serve_indicators_df['Serve_Rally_Control'] = serve_indicators_df['Overall_RiP'].fillna(0)
    serve_indicators_df['First_Serve_Power'] = serve_indicators_df['First_Unret'].fillna(0)
    
    first_serve_logic_components = [
        serve_indicators_df['First_Unret'].fillna(0) * 0.4,
        serve_indicators_df['First_W3'].fillna(0) * 0.35,
        serve_indicators_df['First_RiP'].fillna(0) * 0.25
    ]
    serve_indicators_df['First_Serve_Dominance'] = sum(first_serve_logic_components)
    
    second_serve_logic_components = [
        serve_indicators_df['Second_Unret'].fillna(0) * 0.3,
        serve_indicators_df['Second_W3'].fillna(0) * 0.4,
        serve_indicators_df['Second_RiP'].fillna(0) * 0.3
    ]
    serve_indicators_df['Second_Serve_Effectiveness'] = sum(second_serve_logic_components)
    
    first_placement_headers = ['First_D_Wide', 'First_A_Wide', 'First_BP_Wide']
    first_placement_metrics = [serve_indicators_df[col].fillna(0) for col in first_placement_headers if col in serve_indicators_df.columns]
    
    if first_placement_metrics:
        serve_indicators_df['First_Serve_Placement_Strategy'] = np.mean(first_placement_metrics, axis=0)
    else:
        serve_indicators_df['First_Serve_Placement_Strategy'] = 0
        
    second_placement_headers = ['Second_D_Wide', 'Second_A_Wide', 'Second_BP_Wide']
    second_placement_metrics = [serve_indicators_df[col].fillna(0) for col in second_placement_headers if col in serve_indicators_df.columns]
    
    if second_placement_metrics:
        serve_indicators_df['Second_Serve_Placement_Strategy'] = np.mean(second_placement_metrics, axis=0)
    else:
        serve_indicators_df['Second_Serve_Placement_Strategy'] = 0
        
    serve_adaptation_formula = (
        serve_indicators_df['First_Serve_Dominance'] * calc_type_weights[0] + 
        serve_indicators_df['Second_Serve_Effectiveness'] * calc_type_weights[1]
    )
    serve_indicators_df['Serve_Type_Adaptability'] = serve_adaptation_formula
    
    score_gap_between_serves = abs(serve_indicators_df['First_Serve_Dominance'] - serve_indicators_df['Second_Serve_Effectiveness'])
    serve_indicators_df['Serve_Consistency'] = np.exp(-score_gap_between_serves * 2)
    
    aggregated_power_score = (serve_indicators_df['Serve_Power'] + serve_indicators_df['First_Serve_Power']) / 2
    aggregated_control_score = (serve_indicators_df['Serve_Rally_Control'] + serve_indicators_df['Serve_Quick_Points']) / 2
    serve_indicators_df['Power_Control_Balance'] = np.tanh(aggregated_power_score / (aggregated_control_score + 0.01))
    
    critical_point_headers = ['First_BP_Wide', 'Second_BP_Wide']
    critical_point_metrics = [serve_indicators_df[col].fillna(0) for col in critical_point_headers if col in serve_indicators_df.columns]
    
    if critical_point_metrics:
        serve_indicators_df['Clutch_Serving'] = np.mean(critical_point_metrics, axis=0)
    else:
        serve_indicators_df['Clutch_Serving'] = 0.5
        
    average_placement_versatility = (
        serve_indicators_df['First_Serve_Placement_Strategy'] + 
        serve_indicators_df['Second_Serve_Placement_Strategy']
    ) / 2
    
    tactical_intelligence_score = (
        average_placement_versatility * 0.4 + 
        serve_indicators_df['Serve_Type_Adaptability'] * 0.4 + 
        serve_indicators_df['Clutch_Serving'] * 0.2
    )
    serve_indicators_df['Serve_Tactical_Intelligence'] = tactical_intelligence_score
    
    if 'Result' in serve_indicators_df.columns:
        match_result_strings = serve_indicators_df['Result']
        win_status_mask = match_result_strings.str.contains('W', na=False)
        result_driven_multiplier = np.where(win_status_mask, 1.1, 0.9)
        
        serve_indicators_df['Serve_Match_Impact'] = serve_indicators_df['Serve_Type_Adaptability'] * result_driven_multiplier
    else:
        serve_indicators_df['Serve_Match_Impact'] = serve_indicators_df['Serve_Type_Adaptability']
        
    total_serve_game_formula = (
        serve_indicators_df['Serve_Power'] * 0.2 + 
        serve_indicators_df['Serve_Quick_Points'] * 0.15 + 
        serve_indicators_df['Serve_Rally_Control'] * 0.15 + 
        serve_indicators_df['Serve_Type_Adaptability'] * 0.25 + 
        serve_indicators_df['Serve_Tactical_Intelligence'] * 0.15 + 
        serve_indicators_df['Power_Control_Balance'] * 0.1
    )
    serve_indicators_df['Overall_Serve_Game'] = total_serve_game_formula
    
    return serve_indicators_df