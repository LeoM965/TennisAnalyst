import os
import shutil

import pandas as pd

from t2_trn_stats_rally_helper1 import extract_year
from t2_trn_stats_rally_helper2 import calculate_rally_indicators


def analyze_rally_data(csv_path='wta_mcp_rally.csv'):
    output_dir = 'output_rally'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    df['Year'] = df['Match'].apply(extract_year)
    df = df.dropna(subset=['Year'])

    df_with_indicators = calculate_rally_indicators(df)

    indicators = [
        'Rally_Length_Efficiency',
        'Serve_Return_Rally_Balance',
        'Short_Rally_Control',
        'Long_Rally_Endurance',
        'Rally_Progression_Score',
        'Forehand_Dominance',
        'Backhand_Versatility',
        'Forehand_Power_Index',
        'Backhand_Power_Index',
        'Power_Balance_Index',
        'Rally_Adaptability',
        'Match_Control_Metric',
        'Tactical_Intelligence',
        'Court_Coverage_Efficiency'
    ]

    agg_dict = {'RallyLen': ['mean', 'count']}
    for col in indicators:
        agg_dict[col] = 'mean'

    grouped = df_with_indicators.groupby(['Player', 'Year']).agg(agg_dict).reset_index()
    grouped.columns = ['Player', 'Year', 'Avg_Rally_Length', 'Matches'] + indicators
    grouped = grouped.round(3)
    grouped.to_csv(os.path.join(output_dir, 'yearly_stats.csv'), index=False)

    summary_agg = {'Avg_Rally_Length': 'mean', 'Matches': 'sum'}
    for col in indicators:
        summary_agg[col] = 'mean'

    summary = grouped.groupby('Player')[['Avg_Rally_Length', 'Matches'] + indicators].agg(summary_agg).reset_index()
    summary = summary.round(3)
    summary.to_csv(os.path.join(output_dir, 'career_stats.csv'), index=False)

    match_cols = ['Player', 'Match', 'Result', 'RallyLen', 'Year'] + indicators
    match_stats = df_with_indicators[match_cols].round(3)
    match_stats.to_csv(os.path.join(output_dir, 'match_stats.csv'), index=False)

    top_performers = []
    for metric in indicators:
        top_10 = summary.nlargest(10, metric)[['Player', metric, 'Matches']].copy()
        top_10['Metric'] = metric
        top_10['Rank'] = range(1, 11)
        top_10['Value'] = top_10[metric]
        top_performers.append(top_10[['Player', 'Metric', 'Rank', 'Value', 'Matches']])

    all_tops = pd.concat(top_performers, ignore_index=True)
    all_tops.to_csv(os.path.join(output_dir, 'top_performers.csv'), index=False)

    print("Files created:")
    print("- yearly_stats.csv")
    print("- career_stats.csv")
    print("- match_stats.csv")
    print("- top_performers.csv")


if __name__ == "__main__":
    analyze_rally_data()