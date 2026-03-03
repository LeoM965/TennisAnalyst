import pandas as pd
import numpy as np
import os
import shutil
from t2_trn_stats_rally_helper1 import extract_year
from t2_trn_stats_serve_helper2 import calculate_serve_indicators
from constants import WTA_MCP_SERVE, SERVE_INDICATORS

def analyze_serve_data(csv_path=WTA_MCP_SERVE):
    output_directory = 'output_serve'
    
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    
    dataframe = pd.read_csv(csv_path)
    dataframe.columns = dataframe.columns.str.strip()
    
    dataframe['Year'] = dataframe['Match'].apply(extract_year)
    dataframe = dataframe.dropna(subset=['Year'])
    
    dataframe_with_indicators = calculate_serve_indicators(dataframe)
    
    if 'Overall_Unret' in dataframe_with_indicators.columns:
        main_aggregation_key = 'Overall_Unret'
    else:
        main_aggregation_key = 'Serve_Power'
        
    aggregation_rules = {
        main_aggregation_key: ['mean', 'count']
    }
    
    for indicator in SERVE_INDICATORS:
        if indicator in dataframe_with_indicators.columns:
            aggregation_rules[indicator] = 'mean'
            
    yearly_stats = dataframe_with_indicators.groupby(['Player', 'Year']).agg(aggregation_rules).reset_index()
    
    formatted_columns = ['Player', 'Year']
    for column_name, aggregation_type in yearly_stats.columns[2:]:
        if aggregation_type == 'mean':
            formatted_columns.append(column_name)
        elif aggregation_type == 'count':
            formatted_columns.append('Matches')
            
    yearly_stats.columns = formatted_columns
    yearly_stats = yearly_stats.round(3)
    
    yearly_stats_path = os.path.join(output_directory, 'yearly_serve_stats.csv')
    yearly_stats.to_csv(yearly_stats_path, index=False)
    
    summary_columns = [col for col in yearly_stats.columns if col not in ['Player', 'Year']]
    summary_aggregation = {
        column: ('sum' if column == 'Matches' else 'mean') 
        for column in summary_columns
    }
    
    career_stats = yearly_stats.groupby('Player')[summary_columns].agg(summary_aggregation).reset_index()
    career_stats = career_stats.round(3)
    
    career_stats_path = os.path.join(output_directory, 'career_serve_stats.csv')
    career_stats.to_csv(career_stats_path, index=False)
    
    match_view_columns = ['Player', 'Match', 'Result', 'Year'] + [
        col for col in SERVE_INDICATORS if col in dataframe_with_indicators.columns
    ]
    match_stats = dataframe_with_indicators[match_view_columns].round(3)
    
    match_stats_path = os.path.join(output_directory, 'match_serve_stats.csv')
    match_stats.to_csv(match_stats_path, index=False)
    
    available_metrics = [col for col in SERVE_INDICATORS if col in career_stats.columns]
    
    if available_metrics:
        top_performers_collection = []
        
        for metric in available_metrics:
            if career_stats[metric].notna().sum() > 0:
                top_10 = career_stats.nlargest(10, metric)[['Player', metric, 'Matches']].copy()
                top_10['Metric'] = metric
                top_10['Rank'] = range(1, 11)
                top_10['Value'] = top_10[metric]
                
                selected_cols = ['Player', 'Metric', 'Rank', 'Value', 'Matches']
                top_performers_collection.append(top_10[selected_cols])
                
        if top_performers_collection:
            top_performers_df = pd.concat(top_performers_collection, ignore_index=True)
            top_performers_path = os.path.join(output_directory, 'top_serve_performers.csv')
            top_performers_df.to_csv(top_performers_path, index=False)
            
    return dataframe_with_indicators

if __name__ == "__main__":
    analyze_serve_data()