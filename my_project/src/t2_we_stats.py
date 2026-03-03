import pandas as pd
import os
import shutil
from t2_we_stats_helper1 import extract_year, calculate_tennis_indicators
from constants import WTA_WINNERS_UE

def analyze_evolution_with_indicators(csv_path=WTA_WINNERS_UE):
    output_directory = 'output_we'
    
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)
    
    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as error:
        print(f"Error loading {csv_path}: {error}")
        return
        
    dataframe.columns = dataframe.columns.str.replace('\u00a0', ' ').str.strip()
    
    for column in dataframe.select_dtypes(include=['object']).columns:
        dataframe[column] = dataframe[column].astype(str).str.replace('\u00a0', ' ').str.strip()
        
    dataframe['Winners'] = pd.to_numeric(dataframe['Winners'], errors='coerce')
    dataframe['UFEs'] = pd.to_numeric(dataframe['UFEs'], errors='coerce')
    dataframe['Year'] = dataframe['Match'].apply(extract_year)
    
    dataframe = dataframe.dropna(subset=['Winners', 'UFEs', 'Year'])
    dataframe_with_indicators = calculate_tennis_indicators(dataframe)
    
    advanced_metrics = [
        'Rally_Dominance_Index',
        'Tactical_Balance_Score',
        'Power_Asymmetry_Index',
        'Pressure_Creation_Index',
        'Match_Control_Efficiency',
        'Shot_Selection_IQ',
        'Pressure_Consistency_Index'
    ]
    
    aggregation_rules = {
        'Winners': ['mean', 'count'],
        'UFEs': 'mean'
    }
    
    for metric in advanced_metrics:
        if metric in dataframe_with_indicators.columns:
            aggregation_rules[metric] = 'mean'
            
    yearly_stats = dataframe_with_indicators.groupby(['Player', 'Year']).agg(aggregation_rules).reset_index()
    
    formatted_columns = ['Player', 'Year', 'Avg_Winners', 'Matches_Played', 'Avg_UFEs']
    for metric in advanced_metrics:
        if metric in dataframe_with_indicators.columns:
            formatted_columns.append(metric)
            
    yearly_stats.columns = formatted_columns
    yearly_stats['Winners_UFEs_Ratio'] = yearly_stats['Avg_Winners'] / (yearly_stats['Avg_UFEs'] + 0.001)
    
    yearly_stats = yearly_stats.round(3)
    yearly_stats_path = os.path.join(output_directory, 'player_indicators_by_year.csv')
    yearly_stats.to_csv(yearly_stats_path, index=False)
    
    summary_metrics = ['Avg_Winners', 'Avg_UFEs', 'Winners_UFEs_Ratio', 'Matches_Played']
    for metric in advanced_metrics:
        if metric in yearly_stats.columns:
            summary_metrics.append(metric)
            
    summary_aggregation = {
        column: ('sum' if column == 'Matches_Played' else 'mean') 
        for column in summary_metrics
    }
    
    career_summary = yearly_stats.groupby('Player')[summary_metrics].agg(summary_aggregation).reset_index()
    career_summary = career_summary.round(3)
    
    career_summary_path = os.path.join(output_directory, 'career_summary.csv')
    career_summary.to_csv(career_summary_path, index=False)
    
    match_view_cols = ['Player', 'Match', 'Winners', 'UFEs', 'Year']
    for metric in advanced_metrics:
        if metric in dataframe_with_indicators.columns:
            match_view_cols.append(metric)
            
    match_analysis = dataframe_with_indicators[match_view_cols].round(3)
    match_analysis_path = os.path.join(output_directory, 'match_analysis.csv')
    match_analysis.to_csv(match_analysis_path, index=False)
    
    top_performer_metrics = [
        metric for metric in career_summary.columns 
        if metric not in ['Player', 'Avg_Winners', 'Avg_UFEs', 'Winners_UFEs_Ratio', 'Matches_Played']
    ]
    
    if top_performer_metrics:
        top_performers_collection = []
        
        for metric in top_performer_metrics:
            if metric in career_summary.columns and not career_summary[metric].isna().all():
                top_10 = career_summary.nlargest(10, metric)[['Player', metric, 'Matches_Played']].copy()
                top_10['Metric'] = metric
                top_10['Rank'] = range(1, len(top_10) + 1)
                top_10 = top_10.rename(columns={metric: 'Value'})
                top_performers_collection.append(top_10)
                
        if top_performers_collection:
            top_performers_df = pd.concat(top_performers_collection, ignore_index=True)
            top_performers_path = os.path.join(output_directory, 'top_performers.csv')
            top_performers_df.to_csv(top_performers_path, index=False)
            
if __name__ == "__main__":
    analyze_evolution_with_indicators()