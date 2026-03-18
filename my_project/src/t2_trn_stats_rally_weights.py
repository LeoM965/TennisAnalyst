import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from constants import RALLY_PERCENTAGE_COLS, POWER_COLS

def extract_year(match_string):
    if not isinstance(match_string, str):
        return None
        
    year_pattern = r'\b(20\d{2})\b'
    year_match = re.search(year_pattern, match_string)
    
    if year_match:
        extracted_year = year_match.group(1)
        return int(extracted_year)
        
    return None

def learn_weights_from_data(dataframe):
    pca_analyzer = PCA(n_components=1)
    data_scaler = StandardScaler()
    
    rally_features = dataframe[RALLY_PERCENTAGE_COLS]
    rally_features_cleaned = rally_features.dropna()
    
    if not rally_features_cleaned.empty:
        scaled_rally_data = data_scaler.fit_transform(rally_features_cleaned)
        pca_analyzer.fit(scaled_rally_data)
        
        raw_components = pca_analyzer.components_[0]
        absolute_importance = abs(raw_components)
        
        total_importance_sum = absolute_importance.sum()
        rally_weights = absolute_importance / total_importance_sum
    else:
        default_weight = 0.25
        rally_weights = np.array([default_weight] * 4)
        
    power_features = dataframe[POWER_COLS]
    power_features_cleaned = power_features.dropna()
    
    if not power_features_cleaned.empty:
        correlation_matrix = power_features_cleaned.corr()
        power_correlation_value = correlation_matrix.iloc[0, 1]
        
        is_highly_correlated = abs(power_correlation_value) > 0.5
        
        if is_highly_correlated:
            tactical_weights = [0.4, 0.4, 0.2]
        else:
            tactical_weights = [0.3, 0.3, 0.4]
    else:
        tactical_weights = [0.3, 0.3, 0.4]
        
    return rally_weights, tactical_weights

def learn_match_control_weights(dataframe):
    if 'Result' not in dataframe.columns:
        return 0.4, 0.6
        
    result_column = dataframe['Result']
    
    winning_matches = dataframe[result_column.str.contains('W', na=False)]
    losing_matches = dataframe[result_column.str.contains('L', na=False)]
    
    has_sufficient_data = not winning_matches.empty and not losing_matches.empty
    
    if has_sufficient_data:
        win_short_avg = winning_matches['1-3 W%'].mean()
        loss_short_avg = losing_matches['1-3 W%'].mean()
        
        win_long_avg = winning_matches['10+ W%'].mean()
        loss_long_avg = losing_matches['10+ W%'].mean()
        
        short_rally_impact = abs(win_short_avg - loss_short_avg)
        long_rally_impact = abs(win_long_avg - loss_long_avg)
        
        impact_sum = short_rally_impact + long_rally_impact + 0.001
        
        short_weight = short_rally_impact / impact_sum
        long_weight = long_rally_impact / impact_sum
        
        return short_weight, long_weight
        
    return 0.4, 0.6
