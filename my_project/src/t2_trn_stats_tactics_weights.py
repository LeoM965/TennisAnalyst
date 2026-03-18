import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_year(match_string):
    year_capture_pattern = r'\b(20\d{2})\b'
    match_result = re.search(year_capture_pattern, match_string)
    
    if match_result:
        year_value_string = match_result.group(1)
        return int(year_value_string)
        
    return None

def learn_tactics_weights(dataframe):
    pca_analyzer = PCA(n_components=1)
    scale_transformer = StandardScaler()
    
    tactical_metric_columns = ['Net_Freq', 'Net_W_Pct', 'FH_Wnr_Pct', 'BH_Wnr_Pct']
    
    available_df_columns = dataframe.columns
    valid_tactical_metrics = [col for col in tactical_metric_columns if col in available_df_columns]
    
    if valid_tactical_metrics:
        tactical_data_subset = dataframe[valid_tactical_metrics]
        cleaned_tactical_data = tactical_data_subset.dropna()
        
        if not cleaned_tactical_data.empty:
            standardized_tactical_data = scale_transformer.fit_transform(cleaned_tactical_data)
            pca_analyzer.fit(standardized_tactical_data)
            
            raw_tactical_importance = pca_analyzer.components_[0]
            absolute_tactical_importance = abs(raw_tactical_importance)
            
            importance_weight_sum = absolute_tactical_importance.sum()
            normalized_tactical_weights = absolute_tactical_importance / importance_weight_sum
            
            return normalized_tactical_weights
            
    default_importance_balance = [0.3, 0.3, 0.2, 0.2]
    return np.array(default_importance_balance)