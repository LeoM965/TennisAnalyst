import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_year(match_string):
    year_pattern = r'\b(20\d{2})\b'
    year_match = re.search(year_pattern, match_string)
    
    if year_match:
        extracted_year = year_match.group(1)
        return int(extracted_year)
        
    return None

def learn_return_weights(dataframe):
    pca_analyzer = PCA(n_components=1)
    data_scaler = StandardScaler()
    
    return_metric_columns = ['Overall_RiP', 'Overall_RiP_W', 'Overall_RetWnr']
    
    existing_columns = dataframe.columns
    available_metrics = [col for col in return_metric_columns if col in existing_columns]
    
    if available_metrics:
        return_data_subset = dataframe[available_metrics]
        cleaned_return_data = return_data_subset.dropna()
        
        if not cleaned_return_data.empty:
            scaled_return_data = data_scaler.fit_transform(cleaned_return_data)
            pca_analyzer.fit(scaled_return_data)
            
            raw_pca_components = pca_analyzer.components_[0]
            absolute_contribution = abs(raw_pca_components)
            
            total_contribution_sum = absolute_contribution.sum()
            normalized_return_weights = absolute_contribution / total_contribution_sum
            
            serve_type_importance = [0.6, 0.4]
            
            return normalized_return_weights, serve_type_importance
            
    default_return_weights = [0.4, 0.4, 0.2]
    default_serve_type_weights = [0.6, 0.4]
    
    return default_return_weights, default_serve_type_weights