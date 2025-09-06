import pandas as pd
import os
import shutil
from t2_we_stats_helper1 import extract_year, calculate_tennis_indicators


def analyze_evolution_with_indicators(csv_path='wta_winners_unforced_errors.csv'):
    output_dir = 'output_we'

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return

    df.columns = df.columns.str.replace('\u00a0', ' ').str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.replace('\u00a0', ' ').str.strip()

    df['Winners'] = pd.to_numeric(df['Winners'], errors='coerce')
    df['UFEs'] = pd.to_numeric(df['UFEs'], errors='coerce')
    df['Year'] = df['Match'].apply(extract_year)
    df = df.dropna(subset=['Winners', 'UFEs', 'Year'])

    df_with_indicators = calculate_tennis_indicators(df)

    advanced_indicators = [
        'Rally_Dominance_Index',
        'Tactical_Balance_Score',
        'Power_Asymmetry_Index',
        'Pressure_Creation_Index',
        'Match_Control_Efficiency',
        'Shot_Selection_IQ',
        'Pressure_Consistency_Index'
    ]

    agg_dict = {
        'Winners': ['mean', 'count'],
        'UFEs': 'mean'
    }

    for col in advanced_indicators:
        if col in df_with_indicators.columns:
            agg_dict[col] = 'mean'

    grouped = df_with_indicators.groupby(['Player', 'Year']).agg(agg_dict).reset_index()

    new_columns = ['Player', 'Year', 'Avg_Winners', 'Matches_Played', 'Avg_UFEs']
    for col in advanced_indicators:
        if col in df_with_indicators.columns:
            new_columns.append(col)

    grouped.columns = new_columns
    grouped['Winners_UFEs_Ratio'] = grouped['Avg_Winners'] / (grouped['Avg_UFEs'] + 0.001)
    grouped = grouped.round(3)
    grouped.to_csv(os.path.join(output_dir, 'player_indicators_by_year.csv'), index=False)

    summary_cols = ['Avg_Winners', 'Avg_UFEs', 'Winners_UFEs_Ratio', 'Matches_Played']
    for col in advanced_indicators:
        if col in grouped.columns:
            summary_cols.append(col)

    summary_agg = {}
    for col in summary_cols:
        if col == 'Matches_Played':
            summary_agg[col] = 'sum'
        else:
            summary_agg[col] = 'mean'

    summary = grouped.groupby('Player')[summary_cols].agg(summary_agg).reset_index()
    summary = summary.round(3)
    summary.to_csv(os.path.join(output_dir, 'career_summary.csv'), index=False)

    match_cols = ['Player', 'Match', 'Winners', 'UFEs', 'Year']
    for col in advanced_indicators:
        if col in df_with_indicators.columns:
            match_cols.append(col)

    match_stats = df_with_indicators[match_cols].copy()
    match_stats = match_stats.round(3)
    match_stats.to_csv(os.path.join(output_dir, 'match_analysis.csv'), index=False)

    career_metrics = [col for col in summary.columns if
                      col not in ['Player', 'Avg_Winners', 'Avg_UFEs', 'Winners_UFEs_Ratio', 'Matches_Played']]

    if career_metrics:
        top_performers_list = []
        for metric in career_metrics:
            if metric in summary.columns and not summary[metric].isna().all():
                top_players = summary.nlargest(10, metric)[['Player', metric, 'Matches_Played']].copy()
                top_players['Metric'] = metric
                top_players['Rank'] = range(1, len(top_players) + 1)
                top_players = top_players.rename(columns={metric: 'Value'})
                top_performers_list.append(top_players)

        if top_performers_list:
            top_performers = pd.concat(top_performers_list, ignore_index=True)
            top_performers.to_csv(os.path.join(output_dir, 'top_performers.csv'), index=False)

    print(f"Analysis complete! Files saved in {output_dir}/:")
    print("- player_indicators_by_year.csv")
    print("- career_summary.csv")
    print("- match_analysis.csv")
    if career_metrics:
        print("- top_performers.csv")


if __name__ == "__main__":
    analyze_evolution_with_indicators()