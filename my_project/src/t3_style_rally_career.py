import streamlit as st
import pandas as pd
from t3_style_rally_career_helper1 import FEATURES, SCATTER_OPTIONS, STREAMLIT_CONFIG
from t3_style_rally_career_helper2 import (
    load_and_cluster_data, create_scatter_plot, create_bar_plot,
    create_style_distribution, create_heatmap, save_analysis_files
)

st.set_page_config(page_title=STREAMLIT_CONFIG['page_title'], layout=STREAMLIT_CONFIG['layout'])

st.title("Career Playing Styles Analysis")
st.markdown("Comprehensive analysis of tennis playing styles based on career statistics")

df = load_and_cluster_data()
if df is None:
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Style Overview", "Scatter Analysis", "Power Analysis", "Advanced Metrics", "Player Details"])

with tab1:
    st.header("Playing Style Distribution")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = create_style_distribution(df)
        st.pyplot(fig)

    with col2:
        st.subheader("Style Breakdown")
        style_counts = df['Style'].value_counts()
        for style, count in style_counts.items():
            percentage = (count / len(df)) * 100
            st.metric(style, f"{count} players", f"{percentage:.1f}%")

    st.subheader("Style Characteristics")
    style_summary = df.groupby('Style')[['Avg_Rally_Length', 'Power_Balance_Index',
                                         'Rally_Adaptability', 'Tactical_Intelligence']].mean().round(3)
    st.dataframe(style_summary, use_container_width=True)

with tab2:
    st.header("Scatter Plot Analysis")

    selected_plot = st.selectbox("Select Analysis:", list(SCATTER_OPTIONS.keys()))
    x_col, y_col = SCATTER_OPTIONS[selected_plot]

    diagonal = "vs" in selected_plot and ("Power" in selected_plot or "Dominance" in selected_plot)

    fig = create_scatter_plot(df, x_col, y_col, selected_plot, diagonal)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top 5 - {x_col.replace('_', ' ')}")
        top_x = df.nlargest(5, x_col)[['Player', 'Style', x_col]]
        st.dataframe(top_x, use_container_width=True)

    with col2:
        st.subheader(f"Top 5 - {y_col.replace('_', ' ')}")
        top_y = df.nlargest(5, y_col)[['Player', 'Style', y_col]]
        st.dataframe(top_y, use_container_width=True)

with tab3:
    st.header("Power and Balance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Power Balance by Player")
        fig1 = create_bar_plot(df, 'FH_BH_Power_Diff', 'Forehand vs Backhand Power Difference')
        st.pyplot(fig1)

    with col2:
        st.subheader("Dominance Balance by Player")
        fig2 = create_bar_plot(df, 'FH_BH_Dom_Diff', 'Forehand vs Backhand Dominance Difference')
        st.pyplot(fig2)

    st.subheader("Power Statistics by Style")
    power_stats = df.groupby('Style')[['Forehand_Power_Index', 'Backhand_Power_Index',
                                       'FH_BH_Power_Diff', 'Power_Balance_Index']].mean().round(3)
    st.dataframe(power_stats, use_container_width=True)

with tab4:
    st.header("Advanced Metrics Heatmap")

    fig = create_heatmap(df)
    st.pyplot(fig)

    st.subheader("Detailed Style Statistics")
    detailed_stats = df.groupby('Style')[FEATURES].mean().round(3)
    st.dataframe(detailed_stats, use_container_width=True)

    selected_style = st.selectbox("Select Style for Details:", df['Style'].unique())
    style_players = df[df['Style'] == selected_style][['Player'] + FEATURES[:8]].round(3)
    st.dataframe(style_players, use_container_width=True)

with tab5:
    st.header("Individual Player Analysis")

    selected_player = st.selectbox("Select Player:", sorted(df['Player'].unique()))
    player_data = df[df['Player'] == selected_player].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Playing Style", player_data['Style'])
        st.metric("Rally Length", f"{player_data['Avg_Rally_Length']:.3f}")
        st.metric("Power Balance", f"{player_data['Power_Balance_Index']:.3f}")

    with col2:
        st.metric("Rally Adaptability", f"{player_data['Rally_Adaptability']:.3f}")
        st.metric("Tactical Intelligence", f"{player_data['Tactical_Intelligence']:.3f}")
        st.metric("Court Coverage", f"{player_data['Court_Coverage_Efficiency']:.3f}")

    with col3:
        st.metric("Forehand Power", f"{player_data['Forehand_Power_Index']:.3f}")
        st.metric("Backhand Power", f"{player_data['Backhand_Power_Index']:.3f}")
        st.metric("Match Control", f"{player_data['Match_Control_Metric']:.3f}")

    st.subheader(f"{selected_player} - Complete Stats")
    player_stats = pd.DataFrame({
        'Metric': FEATURES,
        'Value': [player_data[feature] for feature in FEATURES]
    }).round(3)
    st.dataframe(player_stats, use_container_width=True)

    same_style_players = df[df['Style'] == player_data['Style']]['Player'].tolist()
    same_style_players.remove(selected_player)
    if same_style_players:
        st.subheader(f"Other {player_data['Style']} Players")
        st.write(", ".join(same_style_players))

st.sidebar.header("Export Data")
if st.sidebar.button("Save Analysis Files"):
    player_count, style_count = save_analysis_files(df)
    st.sidebar.success(f"Files saved!")
    st.sidebar.info(f"Player analysis: {player_count} players")
    st.sidebar.info(f"Style statistics: {style_count} styles")
    st.sidebar.info("Location: output_style_rally_career/")

st.sidebar.header("Dataset Info")
st.sidebar.metric("Total Players", len(df))
st.sidebar.metric("Playing Styles", df['Style'].nunique())
st.sidebar.metric("Features Analyzed", len(FEATURES))