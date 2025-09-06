import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from t3_style_rally_yearly_helper1 import *


@st.cache_data
def load_and_process_data():
    try:
        df_yearly = pd.read_csv(DATA_PATHS['yearly'])
        df_career = pd.read_csv(DATA_PATHS['career'])

        scaler = StandardScaler()
        X_career = scaler.fit_transform(df_career[FEATURES])
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        kmeans.fit(X_career)

        X_yearly = scaler.transform(df_yearly[FEATURES])
        df_yearly['Cluster'] = kmeans.predict(X_yearly)
        df_yearly['Style'] = df_yearly['Cluster'].map(STYLE_NAMES)

        return df_yearly, df_career
    except FileNotFoundError:
        st.error("CSV files not found. Please ensure 'output_rally/yearly_stats.csv' and 'output_rally/career_stats.csv' exist.")
        return None, None


def add_jitter(values, jitter_amount=0.05):
    return values + np.random.normal(0, jitter_amount, len(values))


def smart_label_placement(ax, x_vals, y_vals, labels, offset_scale=0.02):
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = y_range * offset_scale

    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        nearby_points = [(x2, y2) for j, (x2, y2) in enumerate(zip(x_vals, y_vals))
                         if j != i and abs(x2 - x) < 0.3 and abs(y2 - y) < offset * 3]

        label_y = y + offset * (1 + len(nearby_points) * 0.5) if nearby_points else y + offset
        ax.annotate(label, (x, y), (x, label_y),
                    fontsize=8, ha='center', va='bottom',
                    arrowprops=dict(arrowstyle='-', alpha=0.5, lw=0.5))


def create_line_plot(data, title, metrics=None):
    metrics = metrics or KEY_METRICS
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        ax.plot(data.index, data[metric], marker='o', linewidth=3, markersize=8, color='#2E86AB')
        ax.set_title(metric.replace("_", " "), fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_heatmap(data, title, xlabel='Year', ylabel='Metrics'):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.T, annot=True, cmap='RdYlBu_r', fmt='.3f', cbar_kws={'label': 'Average Score'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    return fig


def display_player_metrics(player_info, cols):
    metrics = [
        ('Playing Style', 'Style', ''),
        ('Avg Rally Length', 'Avg_Rally_Length', '.3f'),
        ('Power Balance Index', 'Power_Balance_Index', '.3f'),
        ('Rally Adaptability', 'Rally_Adaptability', '.3f'),
        ('Tactical Intelligence', 'Tactical_Intelligence', '.3f'),
        ('Court Coverage', 'Court_Coverage_Efficiency', '.3f')
    ]

    for i, (label, key, fmt) in enumerate(metrics):
        with cols[i % 3]:
            value = player_info[key]
            display_value = value if fmt == '' else f"{value:{fmt}}"
            st.metric(label, display_value)


def create_style_distribution_plot(style_by_year):
    fig, ax = plt.subplots(figsize=(14, 8))
    style_by_year.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', alpha=0.8)
    ax.set_title('Playing Style Distribution by Year', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Players')
    ax.legend(title='Playing Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def create_player_performance_plot(year_data, selected_year_analysis):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Players Performance: {selected_year_analysis}', fontsize=16, fontweight='bold')

    for i, attr in enumerate(KEY_METRICS):
        ax = axes[i // 2, i % 2]
        all_x_positions, all_y_values, all_labels = [], [], []
        x_pos = 0

        for j, style in enumerate(year_data['Style'].unique()):
            style_data = year_data[year_data['Style'] == style]
            n_players = len(style_data)

            x_positions = np.linspace(x_pos, x_pos + 0.8, n_players)
            x_jittered = add_jitter(x_positions, 0.05)
            y_values = style_data[attr].values
            labels = [name.split()[-1] for name in style_data['Player'].values]

            ax.scatter(x_jittered, y_values, c=COLORS[j % len(COLORS)], label=style,
                       s=150, alpha=0.8, edgecolors='black', linewidth=0.5)

            all_x_positions.extend(x_jittered)
            all_y_values.extend(y_values)
            all_labels.extend(labels)
            x_pos += 1.2

        smart_label_placement(ax, all_x_positions, all_y_values, all_labels)
        style_positions = np.arange(0.4, x_pos, 1.2)
        ax.set_xticks(style_positions)
        ax.set_xticklabels(year_data['Style'].unique(), rotation=45, ha='right')
        ax.set_title(attr.replace('_', ' '), fontweight='bold')
        ax.set_ylabel(attr.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    return fig


def generate_csv_files(df_yearly):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    yearly_stats = df_yearly.groupby('Year')[HEATMAP_METRICS].mean()
    yearly_stats.round(3).to_csv(f'{OUTPUT_DIR}/yearly_statistics.csv')

    style_stats = df_yearly.groupby('Style')[HEATMAP_METRICS].mean()
    style_stats.round(3).to_csv(f'{OUTPUT_DIR}/styles_statistics.csv')

    evolution_cols = ['Player', 'Year', 'Style'] + HEATMAP_METRICS + ['Forehand_Power_Index', 'Backhand_Power_Index']
    df_yearly[evolution_cols].round(3).to_csv(f'{OUTPUT_DIR}/player_evolution.csv', index=False)

    player_summary = []
    for player in df_yearly['Player'].unique():
        player_data = df_yearly[df_yearly['Player'] == player].sort_values('Year')
        if len(player_data) >= 2:
            first, last = player_data.iloc[0], player_data.iloc[-1]
            player_summary.append({
                'Player': player,
                'First_Year': first['Year'],
                'Last_Year': last['Year'],
                'Years_Active': len(player_data),
                'First_Style': first['Style'],
                'Last_Style': last['Style'],
                'Power_Change': round(last['Power_Balance_Index'] - first['Power_Balance_Index'], 3),
                'Rally_Length_Change': round(last['Avg_Rally_Length'] - first['Avg_Rally_Length'], 3),
                'Adaptability_Change': round(last['Rally_Adaptability'] - first['Rally_Adaptability'], 3)
            })

    pd.DataFrame(player_summary).to_csv(f'{OUTPUT_DIR}/player_career_changes.csv', index=False)