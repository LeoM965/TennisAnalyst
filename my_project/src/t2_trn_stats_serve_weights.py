import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_year(match_string):
    year_regex_pattern = r'\b(20\d{2})\b'
    regex_match = re.search(year_regex_pattern, match_string)
    
    if regex_match:
        year_string = regex_match.group(1)
        return int(year_string)
        
    return None

def learn_serve_weights(dataframe):
    pca_calculator = PCA(n_components=1)
    data_normalizer = StandardScaler()
    
    serve_power_metrics = ['Overall_Unret', 'Overall_W3', 'Overall_RiP']
    
    existing_df_columns = dataframe.columns
    valid_serve_metrics = [col for col in serve_power_metrics if col in existing_df_columns]
    
    if valid_serve_metrics:
        serve_data_subset = dataframe[valid_serve_metrics]
        cleaned_serve_data = serve_data_subset.dropna()
        
        if not cleaned_serve_data.empty:
            normalized_serve_data = data_normalizer.fit_transform(cleaned_serve_data)
            pca_calculator.fit(normalized_serve_data)
            
            raw_influence_components = pca_calculator.components_[0]
            absolute_metric_influence = abs(raw_influence_components)
            
            total_influence_sum = absolute_metric_influence.sum()
            normalized_serve_weights = absolute_metric_influence / total_influence_sum
        else:
            normalized_serve_weights = np.array([0.4, 0.3, 0.3])
    else:
        normalized_serve_weights = np.array([0.4, 0.3, 0.3])
        
    has_type_metrics = 'First_Unret' in dataframe.columns and 'Second_Unret' in dataframe.columns
    
    if has_type_metrics:
        serve_type_subset = dataframe[['First_Unret', 'Second_Unret']]
        cleaned_type_data = serve_type_subset.dropna()
        
        if not cleaned_type_data.empty:
            first_serve_variability = cleaned_type_data['First_Unret'].std()
            second_serve_variability = cleaned_type_data['Second_Unret'].std()
            
            combined_variability_sum = first_serve_variability + second_serve_variability + 0.001
            
            type_importance_balance = [
                first_serve_variability / combined_variability_sum, 
                second_serve_variability / combined_variability_sum
            ]
        else:
            type_importance_balance = [0.7, 0.3]
    else:
        type_importance_balance = [0.7, 0.3]
        
    return normalized_serve_weights, type_importance_balance