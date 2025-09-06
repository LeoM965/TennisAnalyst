import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import streamlit as st
import warnings
from t3_style_rally_career_helper1 import FEATURES, STYLE_NAMES, COLORS, KEY_FEATURES_HEATMAP, DATA_PATH, OUTPUT_DIR

warnings.filterwarnings('ignore')


@st.cache_data
def load_and_cluster_data():
    try:
        df = pd.read_csv(DATA_PATH)

        scaler = StandardScaler()
        X = scaler.fit_transform(df[FEATURES])

        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        df['Style'] = df['Cluster'].map(STYLE_NAMES)

        df['Backhand_Dominance'] = 1 - df['Forehand_Dominance']
        df['FH_BH_Power_Diff'] = df['Forehand_Power_Index'] - df['Backhand_Power_Index']
        df['FH_BH_Dom_Diff'] = df['Forehand_Dominance'] - df['Backhand_Dominance']

        return df
    except FileNotFoundError:
        st.error(f"Fișierul '{DATA_PATH}' nu a fost găsit")
        return None


def add_jitter(x_vals, y_vals, jitter=0.01):
    x_range = max(x_vals) - min(x_vals) if len(set(x_vals)) > 1 else 1
    y_range = max(y_vals) - min(y_vals) if len(set(y_vals)) > 1 else 1

    x_jittered = x_vals + np.random.normal(0, x_range * jitter, len(x_vals))
    y_jittered = y_vals + np.random.normal(0, y_range * jitter, len(y_vals))

    return x_jittered, y_jittered


def create_scatter_plot(df, x_col, y_col, title, diagonal=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, style in enumerate(df['Style'].unique()):
        style_data = df[df['Style'] == style]

        x_vals = style_data[x_col].values
        y_vals = style_data[y_col].values

        if len(x_vals) > 0:
            x_jittered, y_jittered = add_jitter(x_vals, y_vals)

            ax.scatter(x_jittered, y_jittered, c=COLORS[i], label=style,
                       s=120, alpha=0.8, edgecolors='black', linewidth=0.5)

            for j, (x, y) in enumerate(zip(x_jittered, y_jittered)):
                player_name = style_data.iloc[j]['Player'].split()[-1]
                ax.annotate(player_name, (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, alpha=0.8)

    if diagonal:
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)

    ax.set_xlabel(x_col.replace('_', ' '), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_bar_plot(df, col, title, horizontal=True):
    df_sorted = df.sort_values(col)
    fig, ax = plt.subplots(figsize=(12, 8))

    style_colors = {style: COLORS[i] for i, style in enumerate(df['Style'].unique())}
    bar_colors = [style_colors[style] for style in df_sorted['Style']]

    if horizontal:
        bars = ax.barh(range(len(df_sorted)), df_sorted[col], color=bar_colors,
                       alpha=0.8, edgecolor='black', linewidth=0.5)

        for i, (bar, player) in enumerate(zip(bars, df_sorted['Player'])):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    player.split()[-1], ha='left' if width >= 0 else 'right',
                    va='center', fontsize=8)

        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels([''] * len(df_sorted))
        ax.set_xlabel(col.replace('_', ' '), fontsize=12)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.grid(axis='x', alpha=0.3)
    else:
        bars = ax.bar(range(len(df_sorted)), df_sorted[col], color=bar_colors,
                      alpha=0.8, edgecolor='black', linewidth=0.5)

        for bar, player in zip(bars, df_sorted['Player']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    player.split()[-1], ha='center', va='bottom',
                    fontsize=8, rotation=45)

        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels([''] * len(df_sorted))
        ax.set_ylabel(col.replace('_', ' '), fontsize=12)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.grid(axis='y', alpha=0.3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_style_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    style_counts = df['Style'].value_counts()
    bars = ax.bar(style_counts.index, style_counts.values,
                  color=COLORS[:len(style_counts)], alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    for bar, count in zip(bars, style_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Playing Style', fontsize=12)
    ax.set_ylabel('Number of Players', fontsize=12)
    ax.set_title('Playing Style Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def create_heatmap(df):
    heatmap_data = df.groupby('Style')[KEY_FEATURES_HEATMAP].mean()

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto')

    ax.set_xticks(range(len(KEY_FEATURES_HEATMAP)))
    ax.set_xticklabels([f.replace('_', '\n') for f in KEY_FEATURES_HEATMAP],
                       rotation=0, ha='center', fontsize=10)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=11)

    for i in range(len(heatmap_data.index)):
        for j in range(len(KEY_FEATURES_HEATMAP)):
            ax.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                    ha="center", va="center", color="black",
                    fontweight='bold', fontsize=10)

    plt.colorbar(im, ax=ax, label='Score', shrink=0.8)
    ax.set_title('Advanced Metrics by Playing Style', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def save_analysis_files(df):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    results_df = df[['Player', 'Style', 'Forehand_Power_Index', 'Backhand_Power_Index',
                     'Forehand_Dominance', 'Backhand_Dominance', 'Backhand_Versatility',
                     'FH_BH_Power_Diff', 'FH_BH_Dom_Diff'] + FEATURES].round(2)
    results_df.to_csv(f'{OUTPUT_DIR}/player_analysis.csv', index=False)

    style_stats = df.groupby('Style')[FEATURES].mean().round(2)
    style_stats.to_csv(f'{OUTPUT_DIR}/style_stats.csv')

    return len(results_df), len(style_stats)