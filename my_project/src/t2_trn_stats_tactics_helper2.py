import pandas as pd
import numpy as np
from t2_trn_stats_tactics_helper1 import learn_tactics_weights
from constants import TACTICS_PCT_COLS

def calculate_tactics_indicators(dataframe):
    tactical_indicators_df = dataframe.copy()
    
    original_column_list = dataframe.columns.tolist()
    
    has_sufficient_columns = len(original_column_list) >= 16
    if has_sufficient_columns:
        tactical_column_mapping = {
            original_column_list[3]: 'SnV_Freq',
            original_column_list[4]: 'SnV_W_Pct',
            original_column_list[5]: 'Net_Freq',
            original_column_list[6]: 'Net_W_Pct',
            original_column_list[7]: 'FH_Wnr_Pct',
            original_column_list[8]: 'FH_DTL_Wnr_Pct',
            original_column_list[9]: 'FH_IO_Wnr_Pct',
            original_column_list[10]: 'BH_Wnr_Pct',
            original_column_list[11]: 'BH_DTL_Wnr_Pct',
            original_column_list[12]: 'Drop_Freq',
            original_column_list[13]: 'Drop_Wnr_Pct',
            original_column_list[14]: 'RallyAgg',
            original_column_list[15]: 'ReturnAgg'
        }
        tactical_indicators_df = tactical_indicators_df.rename(columns=tactical_column_mapping)
        
    for tactics_pct_col in TACTICS_PCT_COLS:
        if tactics_pct_col in tactical_indicators_df.columns:
            raw_tactics_strings = tactical_indicators_df[tactics_pct_col].astype(str)
            clean_tactics_pct = raw_tactics_strings.str.rstrip('%')
            
            numeric_tactics_ratios = pd.to_numeric(clean_tactics_pct, errors='coerce') / 100
            tactical_indicators_df[tactics_pct_col] = numeric_tactics_ratios
            
    aggression_numeric_cols = ['RallyAgg', 'ReturnAgg']
    for agg_col in aggression_numeric_cols:
        if agg_col in tactical_indicators_df.columns:
            placeholder_vals = ['-', 'nan', 'NaN']
            clean_agg_strings = tactical_indicators_df[agg_col].astype(str).replace(placeholder_vals, np.nan)
            
            numeric_agg_values = pd.to_numeric(clean_agg_strings, errors='coerce')
            tactical_indicators_df[agg_col] = numeric_agg_values
            
    tactical_learned_weights = learn_tactics_weights(tactical_indicators_df)
    
    tactical_indicators_df['Net_Game_Frequency'] = tactical_indicators_df['Net_Freq'].fillna(0)
    tactical_indicators_df['Net_Game_Effectiveness'] = tactical_indicators_df['Net_W_Pct'].fillna(0)
    
    net_impact_calc = tactical_indicators_df['Net_Game_Frequency'] * tactical_indicators_df['Net_W_Pct'].fillna(0)
    tactical_indicators_df['Net_Game_Impact'] = net_impact_calc
    
    tactical_indicators_df['Serve_Volley_Frequency'] = tactical_indicators_df['SnV_Freq'].fillna(0)
    tactical_indicators_df['Serve_Volley_Effectiveness'] = tactical_indicators_df['SnV_W_Pct'].fillna(0)
    
    snv_impact_calc = tactical_indicators_df['Serve_Volley_Frequency'] * tactical_indicators_df['SnV_W_Pct'].fillna(0)
    tactical_indicators_df['Serve_Volley_Impact'] = snv_impact_calc
    
    tactical_indicators_df['Forehand_Power'] = tactical_indicators_df['FH_Wnr_Pct'].fillna(0)
    tactical_indicators_df['Backhand_Power'] = tactical_indicators_df['BH_Wnr_Pct'].fillna(0)
    
    power_difference = abs(tactical_indicators_df['Forehand_Power'] - tactical_indicators_df['Backhand_Power'])
    tactical_indicators_df['Groundstroke_Balance'] = power_difference
    
    average_gs_power = (tactical_indicators_df['Forehand_Power'] + tactical_indicators_df['Backhand_Power']) / 2
    tactical_indicators_df['Overall_Groundstroke_Power'] = average_gs_power
    
    tactical_indicators_df['Forehand_DTL_Control'] = tactical_indicators_df['FH_DTL_Wnr_Pct'].fillna(0)
    tactical_indicators_df['Forehand_IO_Control'] = tactical_indicators_df['FH_IO_Wnr_Pct'].fillna(0)
    tactical_indicators_df['Backhand_DTL_Control'] = tactical_indicators_df['BH_DTL_Wnr_Pct'].fillna(0)
    
    total_variety_score = (
        tactical_indicators_df['Forehand_DTL_Control'] + 
        tactical_indicators_df['Forehand_IO_Control'] + 
        tactical_indicators_df['Backhand_DTL_Control']
    )
    tactical_indicators_df['Directional_Versatility'] = total_variety_score / 2
    
    tactical_indicators_df['Drop_Shot_Usage'] = tactical_indicators_df['Drop_Freq'].fillna(0)
    tactical_indicators_df['Drop_Shot_Effectiveness'] = tactical_indicators_df['Drop_Wnr_Pct'].fillna(0)
    
    drop_shot_impact_calc = tactical_indicators_df['Drop_Shot_Usage'] * tactical_indicators_df['Drop_Shot_Effectiveness']
    tactical_indicators_df['Drop_Shot_Impact'] = drop_shot_impact_calc
    
    rally_agg_normalized = tactical_indicators_df['RallyAgg'].fillna(0) / 200
    tactical_indicators_df['Rally_Aggression'] = np.clip(rally_agg_normalized, 0, 1)
    
    return_agg_normalized = tactical_indicators_df['ReturnAgg'].fillna(0) / 200
    tactical_indicators_df['Return_Aggression'] = np.clip(return_agg_normalized, 0, 1)
    
    weighted_aggression_sum = (
        tactical_indicators_df['Rally_Aggression'] * 0.6 + 
        tactical_indicators_df['Return_Aggression'] * 0.4
    )
    tactical_indicators_df['Overall_Aggression'] = weighted_aggression_sum
    
    combined_net_freq = tactical_indicators_df['Net_Game_Frequency'] + tactical_indicators_df['Serve_Volley_Frequency']
    baseline_power_ref = tactical_indicators_df['Overall_Groundstroke_Power'] + 0.01
    
    tactical_indicators_df['Court_Position_Strategy'] = np.tanh(combined_net_freq / baseline_power_ref)
    
    offensive_weapon_checklist = [
        (tactical_indicators_df['Net_Game_Impact'] > 0).astype(int),
        (tactical_indicators_df['Serve_Volley_Impact'] > 0).astype(int),
        (tactical_indicators_df['Drop_Shot_Impact'] > 0).astype(int),
        (tactical_indicators_df['Directional_Versatility'] > 0.1).astype(int),
        (tactical_indicators_df['Overall_Aggression'] > 0.3).astype(int)
    ]
    
    total_weapon_count = sum(offensive_weapon_checklist)
    tactical_indicators_df['Tactical_Versatility'] = total_weapon_count / 5.0
    
    power_intensity = tactical_indicators_df['Overall_Groundstroke_Power'] + tactical_indicators_df['Overall_Aggression']
    finesse_intensity = tactical_indicators_df['Drop_Shot_Impact'] + tactical_indicators_df['Directional_Versatility']
    
    tactical_indicators_df['Power_Finesse_Balance'] = np.tanh(power_intensity / (finesse_intensity + 0.01))
    
    adaptation_logic_mix = (
        tactical_indicators_df['Tactical_Versatility'] * 0.5 + 
        tactical_indicators_df['Power_Finesse_Balance'] * 0.3 + 
        tactical_indicators_df['Court_Position_Strategy'] * 0.2
    )
    tactical_indicators_df['Tactical_Adaptability'] = adaptation_logic_mix
    
    if 'Result' in tactical_indicators_df.columns:
        match_result_mask = tactical_indicators_df['Result'].str.contains('W', na=False)
        tactical_bonus_multiplier = np.where(match_result_mask, 1.1, 0.9)
        
        tactical_indicators_df['Tactical_Match_Impact'] = tactical_indicators_df['Tactical_Adaptability'] * tactical_bonus_multiplier
    else:
        tactical_indicators_df['Tactical_Match_Impact'] = tactical_indicators_df['Tactical_Adaptability']
        
    efficiency_mix = (
        tactical_indicators_df['Overall_Groundstroke_Power'] * 0.4 + 
        tactical_indicators_df['Net_Game_Impact'] * 0.3 + 
        tactical_indicators_df['Overall_Aggression'] * 0.3
    )
    tactical_indicators_df['Offensive_Efficiency'] = efficiency_mix
    
    intelligence_mix = (
        tactical_indicators_df['Tactical_Versatility'] * 0.3 + 
        tactical_indicators_df['Directional_Versatility'] * 0.25 + 
        tactical_indicators_df['Tactical_Adaptability'] * 0.25 + 
        tactical_indicators_df['Power_Finesse_Balance'] * 0.2
    )
    tactical_indicators_df['Tactical_Intelligence'] = intelligence_mix
    
    overall_tactical_formula = (
        tactical_indicators_df['Offensive_Efficiency'] * 0.3 + 
        tactical_indicators_df['Tactical_Intelligence'] * 0.25 + 
        tactical_indicators_df['Net_Game_Impact'] * 0.2 + 
        tactical_indicators_df['Overall_Aggression'] * 0.15 + 
        tactical_indicators_df['Tactical_Versatility'] * 0.1
    )
    tactical_indicators_df['Overall_Tactical_Game'] = overall_tactical_formula
    
    return tactical_indicators_df