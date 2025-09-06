import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from t3_style_rally_match_helper1 import COLORS, STREAMLIT_CONFIG
from t3_style_rally_match_helper2 import (
    load_data, get_player_style, process_data, get_tournament_stats,
    get_comparison, run_ml_analysis, create_plots
)

st.set_page_config(page_title=STREAMLIT_CONFIG['page_title'], layout=STREAMLIT_CONFIG['layout'])

st.title("Player Tournament Analysis")
st.markdown("Advanced tournament performance analysis with playing style identification")

matches_df, career_df = load_data()
if matches_df is None:
    st.stop()

col1, col2 = st.columns(2)
with col1:
    player = st.selectbox("Player:", sorted(matches_df['Player'].unique()))
with col2:
    years = sorted(matches_df[matches_df['Player'] == player]['Year'].unique())
    year = st.selectbox("Year:", years)

player_style, style_cluster = get_player_style(career_df, player)
if player_style:
    st.info(f"**{player}** plays as: **{player_style}**")
    style_color = COLORS[style_cluster] if style_cluster is not None else '#888888'
    st.markdown(
        f'<div style="background-color: {style_color}; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">{player_style}</div>',
        unsafe_allow_html=True)

if st.button("Analyze Performance", type="primary"):
    data = process_data(matches_df, player, year)

    if data is None:
        available = matches_df[matches_df['Year'] == year]['Player'].unique()
        st.error(f"No data for {player} in {year}")
        with st.expander("Available players"):
            st.write(', '.join(sorted(available)))
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    wins = int(data['Win'].sum())
    win_rate = data['Win'].mean()

    with col1:
        st.metric("Matches", len(data))
        st.metric("Wins", wins)
    with col2:
        st.metric("Win Rate", f"{win_rate:.3f}")
    with col3:
        st.metric("Avg Control", f"{data['Match_Control_Metric'].mean():.3f}")
        st.metric("Avg Rally Length", f"{data['RallyLen'].mean():.3f}")
    with col4:
        st.metric("Playing Style", player_style or "Unknown")

    ml_results = run_ml_analysis(data)
    tourney_stats = get_tournament_stats(data)
    tourney_stats.columns = ['Wins', 'Matches', 'Win Rate', 'Avg Control']

    st.subheader("Tournament Performance")
    st.dataframe(tourney_stats, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Performance Chart", "Win/Loss Analysis", "Success Factors", "Playing Patterns"])

    with tab1:
        fig = create_plots(data, ml_results, player, year, player_style)
        st.pyplot(fig)

    with tab2:
        comparison = get_comparison(data)
        if comparison is not None:
            st.subheader("Wins vs Losses Comparison")
            comparison['Difference'] = comparison['Wins'] - comparison['Losses']
            st.dataframe(comparison, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            comparison[['Wins', 'Losses']].plot(kind='bar', color=['green', 'red'], ax=ax)
            ax.set_title('Performance Metrics: Wins vs Losses')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Not enough data for win/loss comparison")

    with tab3:
        if ml_results and not ml_results['importance'].empty:
            st.subheader("Success Factors")
            importance_df = ml_results['importance'].head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(importance_df)), importance_df['Score'])
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['Metric'].str.replace('_', ' '))
            ax.set_title('Feature Importance for Success')
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(importance_df.round(4), use_container_width=True)
        else:
            st.warning("Not enough data for ML analysis (need 6+ matches with mixed results)")

    with tab4:
        if ml_results and not ml_results['cluster_stats'].empty:
            st.subheader("Playing Pattern Analysis")
            st.write("Matches grouped by similar playing patterns:")

            cluster_df = ml_results['cluster_stats']
            if not cluster_df.empty:
                st.dataframe(cluster_df.round(3), use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(cluster_df['Style'], cluster_df['Win Rate'],
                              color=[COLORS[i] for i in range(len(cluster_df))])
                ax.set_title('Win Rate by Playing Pattern')
                ax.set_ylabel('Win Rate')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No distinct playing patterns found")
        else:
            st.warning("Not enough data for pattern analysis")