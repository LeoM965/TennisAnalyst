import pandas as pd
import numpy as np
import os
import shutil

from t2_trn_stats_rally_helper1 import extract_year
from t2_trn_stats_tactics_helper2 import calculate_tactics_indicators


def analyze_tactics_data(csv_path='wta_mcp_tactics.csv'):
    output_dir = 'output_tactics'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df['Year'] = df['Match'].apply(extract_year)
    df = df.dropna(subset=['Year'])

    df_with_indicators = calculate_tactics_indicators(df)

    indicators = [
        'Net_Game_Frequency',
        'Net_Game_Effectiveness',
        'Net_Game_Impact',
        'Serve_Volley_Frequency',
        'Serve_Volley_Effectiveness',
        'Serve_Volley_Impact',
        'Forehand_Power',
        'Backhand_Power',
        'Groundstroke_Balance',
        'Overall_Groundstroke_Power',
        'Forehand_DTL_Control',
        'Forehand_IO_Control',
        'Backhand_DTL_Control',
        'Directional_Versatility',
        'Drop_Shot_Usage',
        'Drop_Shot_Effectiveness',
        'Drop_Shot_Impact',
        'Rally_Aggression',
        'Return_Aggression',
        'Overall_Aggression',
        'Court_Position_Strategy',
        'Tactical_Versatility',
        'Power_Finesse_Balance',
        'Tactical_Adaptability',
        'Tactical_Match_Impact',
        'Offensive_Efficiency',
        'Tactical_Intelligence',
        'Overall_Tactical_Game'
    ]

    agg_dict = {}
    if 'Net_Freq' in df_with_indicators.columns:
        agg_dict['Net_Freq'] = ['mean', 'count']
    else:
        agg_dict['Net_Game_Frequency'] = ['mean', 'count']

    for col in indicators:
        if col in df_with_indicators.columns:
            agg_dict[col] = 'mean'

    grouped = df_with_indicators.groupby(['Player', 'Year']).agg(agg_dict).reset_index()

    new_cols = ['Player', 'Year']
    for col in grouped.columns[2:]:
        if isinstance(col, tuple):
            if col[1] == 'mean':
                new_cols.append(col[0])
            elif col[1] == 'count':
                new_cols.append('Matches')
        else:
            new_cols.append(col)

    grouped.columns = new_cols
    grouped = grouped.round(3)
    grouped.to_csv(os.path.join(output_dir, 'yearly_tactics_stats.csv'), index=False)

    summary_cols = [col for col in grouped.columns if col not in ['Player', 'Year']]
    summary_agg = {}
    for col in summary_cols:
        if col == 'Matches':
            summary_agg[col] = 'sum'
        else:
            summary_agg[col] = 'mean'

    summary = grouped.groupby('Player')[summary_cols].agg(summary_agg).reset_index()
    summary = summary.round(3)
    summary.to_csv(os.path.join(output_dir, 'career_tactics_stats.csv'), index=False)

    match_cols = ['Player', 'Match', 'Result', 'Year'] + [col for col in indicators if
                                                          col in df_with_indicators.columns]
    match_stats = df_with_indicators[match_cols].round(3)
    match_stats.to_csv(os.path.join(output_dir, 'match_tactics_stats.csv'), index=False)

    available_indicators = [col for col in indicators if col in summary.columns]
    if available_indicators:
        top_performers = []
        for metric in available_indicators:
            if summary[metric].notna().sum() > 0:
                top_10 = summary.nlargest(10, metric)[['Player', metric, 'Matches']].copy()
                top_10['Metric'] = metric
                top_10['Rank'] = range(1, 11)
                top_10['Value'] = top_10[metric]
                top_performers.append(top_10[['Player', 'Metric', 'Rank', 'Value', 'Matches']])

        if top_performers:
            all_tops = pd.concat(top_performers, ignore_index=True)
            all_tops.to_csv(os.path.join(output_dir, 'top_tactics_performers.csv'), index=False)

    print("Tactics analysis files created:")
    print("- yearly_tactics_stats.csv")
    print("- career_tactics_stats.csv")
    print("- match_tactics_stats.csv")
    if available_indicators:
        print("- top_tactics_performers.csv")

    return df_with_indicators


if __name__ == "__main__":
    analyze_tactics_data()