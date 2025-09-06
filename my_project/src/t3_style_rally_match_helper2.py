import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import warnings
from t3_style_rally_match_helper1 import *

warnings.filterwarnings('ignore')

@st.cache_data
def load_data():
    try:
        matches_df = pd.read_csv(DATA_PATHS['matches'])
        career_df = pd.read_csv(DATA_PATHS['career'])
        return matches_df, career_df
    except FileNotFoundError:
        st.error("Required CSV files not found")
        return None, None

def get_player_style(career_df, player_name):
    if career_df is None:
        return None, None

    X = StandardScaler().fit_transform(career_df[CAREER_FEATURES])
    career_df = career_df.copy()
    career_df['Cluster'] = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X)
    career_df['Style'] = career_df['Cluster'].map(STYLE_NAMES)

    player_data = career_df[career_df['Player'] == player_name]
    return (player_data.iloc[0]['Style'], player_data.iloc[0]['Cluster']) if len(player_data) > 0 else (None, None)

def process_data(matches_df, player, year):
    data = matches_df[(matches_df['Player'] == player) & (matches_df['Year'] == year)].copy()
    if data.empty:
        return None

    data['Win'] = data['Result'].str.startswith('W').astype(int)
    data['Tournament'] = data['Match'].str.split().str[1]
    data[METRICS] = data[METRICS].fillna(data[METRICS].mean())
    return data

def get_tournament_stats(data):
    return data.groupby('Tournament').agg({
        'Win': ['sum', 'count', 'mean'],
        'Match_Control_Metric': 'mean'
    }).round(3)

def get_comparison(data):
    wins, losses = data[data['Win'] == 1], data[data['Win'] == 0]
    return pd.DataFrame({
        'Wins': wins[KEY_METRICS].mean(),
        'Losses': losses[KEY_METRICS].mean()
    }).round(3) if len(wins) > 0 and len(losses) > 0 else None

def run_ml_analysis(data):
    if len(data) < 6 or data['Win'].nunique() < 2:
        return None

    X = StandardScaler().fit_transform(data[METRICS])
    y = data['Win']

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    importance = pd.DataFrame({
        'Metric': METRICS,
        'Score': model.feature_importances_
    }).sort_values('Score', ascending=False)

    pca_coords = PCA(n_components=2).fit_transform(X)
    clusters = KMeans(n_clusters=6, random_state=42).fit_predict(X)

    cluster_stats = []
    for i in range(6):
        mask = clusters == i
        if mask.sum() > 0:
            cluster_stats.append({
                'Style': STYLE_NAMES.get(i, f'Style {i + 1}'),
                'Matches': mask.sum(),
                'Win Rate': data[mask]['Win'].mean()
            })

    return {
        'importance': importance,
        'pca': pca_coords,
        'clusters': clusters,
        'cluster_stats': pd.DataFrame(cluster_stats),
        'wins': y.values
    }

def create_plots(data, ml_data, player, year, player_style):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{player} - {year} ({player_style or "Unknown Style"})', fontsize=16, fontweight='bold')

    tourney_results = get_tournament_stats(data)
    tourney_results.columns = ['Wins', 'Matches', 'Win_Rate', 'Control']
    wins_losses = tourney_results[['Wins', 'Matches']].copy()
    wins_losses['Losses'] = wins_losses['Matches'] - wins_losses['Wins']
    wins_losses[['Wins', 'Losses']].plot(kind='bar', color=['green', 'red'], ax=axes[0, 0])
    axes[0, 0].set_title('Tournament Results')
    axes[0, 0].tick_params(axis='x', rotation=45)

    if ml_data:
        top_factors = ml_data['importance'].head(6)
        axes[0, 1].barh(range(len(top_factors)), top_factors['Score'])
        axes[0, 1].set_yticks(range(len(top_factors)))
        axes[0, 1].set_yticklabels(top_factors['Metric'].str.replace('_', ' '))
        axes[0, 1].set_title('Success Factors')

    data.groupby('Tournament')['Match_Control_Metric'].mean().plot(kind='bar', color='blue', ax=axes[0, 2])
    axes[0, 2].set_title('Control by Tournament')
    axes[0, 2].tick_params(axis='x', rotation=45)

    comparison = get_comparison(data)
    if comparison is not None:
        comparison.iloc[:4].plot(kind='bar', color=['green', 'red'], ax=axes[1, 0])
        axes[1, 0].set_title('Wins vs Losses')
        axes[1, 0].tick_params(axis='x', rotation=45)

    data.groupby('Tournament')['RallyLen'].mean().plot(kind='bar', color='orange', ax=axes[1, 1])
    axes[1, 1].set_title('Rally Length by Tournament')
    axes[1, 1].tick_params(axis='x', rotation=45)

    if ml_data:
        pca, clusters = ml_data['pca'], ml_data['clusters']
        for i in range(6):
            cluster_points = pca[clusters == i]
            if len(cluster_points) > 0:
                win_rate = data[clusters == i]['Win'].mean()
                axes[1, 2].scatter(cluster_points[:, 0], cluster_points[:, 1],
                                   c=COLORS[i], label=f'{STYLE_NAMES.get(i, f"Style {i + 1}")} ({win_rate:.2f})',
                                   alpha=0.7, s=60)
        axes[1, 2].set_title('Performance Styles Map')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    return fig