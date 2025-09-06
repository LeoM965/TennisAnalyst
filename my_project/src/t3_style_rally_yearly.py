import streamlit as st
import pandas as pd
from t3_style_rally_yearly_helper1 import STREAMLIT_CONFIG, KEY_METRICS, HEATMAP_METRICS, INSIGHTS
from t3_style_rally_yearly_helper2 import (
    load_and_process_data, create_line_plot, create_heatmap, display_player_metrics,
    create_style_distribution_plot, create_player_performance_plot, generate_csv_files
)

st.set_page_config(page_title=STREAMLIT_CONFIG['page_title'], layout=STREAMLIT_CONFIG['layout'])

st.title("Tennis Analysis Dashboard")
st.markdown("Interactive analysis of tennis playing styles and performance metrics")

df_yearly, df_career = load_and_process_data()

if df_yearly is not None and df_career is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Evolution Overview", "Style Distribution", "Yearly Heatmap", "Player Analysis", "Style Comparison"])

    with tab1:
        st.header("Tennis Evolution: Key Metrics by Year")
        yearly_avg = df_yearly.groupby('Year')[KEY_METRICS].mean()
        fig = create_line_plot(yearly_avg, 'Tennis Evolution: Key Metrics by Year')
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Insights")
            for insight in INSIGHTS:
                st.write(f"• {insight}")

        with col2:
            selected_metric = st.selectbox("Select metric for detailed view:", KEY_METRICS)
            st.line_chart(yearly_avg[selected_metric])

    with tab2:
        st.header("Playing Style Distribution by Year")
        style_by_year = df_yearly.groupby(['Year', 'Style']).size().unstack(fill_value=0)

        fig = create_style_distribution_plot(style_by_year)
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Year:", sorted(df_yearly['Year'].unique()))
            year_styles = df_yearly[df_yearly['Year'] == selected_year]['Style'].value_counts()
            st.bar_chart(year_styles)

        with col2:
            st.subheader(f"Style Distribution in {selected_year}")
            for style, count in year_styles.items():
                percentage = (count / year_styles.sum()) * 100
                st.write(f"**{style}**: {count} players ({percentage:.1f}%)")

    with tab3:
        st.header("Tennis Metrics by Year: Heatmap")
        yearly_stats = df_yearly.groupby('Year')[HEATMAP_METRICS].mean()
        fig = create_heatmap(yearly_stats, 'Tennis Metrics by Year: Heatmap')
        st.pyplot(fig)
        st.subheader("Yearly Statistics Table")
        st.dataframe(yearly_stats.round(3))

    with tab4:
        st.header("Player Performance Analysis")
        available_years = sorted(df_yearly['Year'].unique())
        selected_year_analysis = st.selectbox("Select year for player analysis:", available_years, key="player_year")
        year_data = df_yearly[df_yearly['Year'] == selected_year_analysis]

        if len(year_data) >= 5:
            fig = create_player_performance_plot(year_data, selected_year_analysis)
            st.pyplot(fig)
        else:
            st.warning(f"Not enough data for {selected_year_analysis}. Need at least 5 players.")

        st.subheader("Player Details")
        selected_player = st.selectbox("Select player:", sorted(year_data['Player'].unique()))
        player_info = year_data[year_data['Player'] == selected_player].iloc[0]

        cols = st.columns(3)
        display_player_metrics(player_info, cols)

    with tab5:
        st.header("Playing Style Comparison")
        style_stats = df_yearly.groupby('Style')[HEATMAP_METRICS].mean()
        fig = create_heatmap(style_stats, 'Playing Styles: Metrics Heatmap', 'Metrics', 'Playing Style')
        st.pyplot(fig)

        st.subheader("Style Statistics")
        st.dataframe(style_stats.round(3))

        col1, col2 = st.columns(2)
        with col1:
            selected_styles = st.multiselect("Compare styles:", options=style_stats.index.tolist(),
                                             default=style_stats.index.tolist()[:2])
        with col2:
            selected_metrics = st.multiselect("Select metrics:", options=HEATMAP_METRICS,
                                              default=HEATMAP_METRICS[:3])

        if selected_styles and selected_metrics:
            comparison_data = style_stats.loc[selected_styles, selected_metrics]
            st.bar_chart(comparison_data.T)

    st.sidebar.header("Download Data")
    if st.sidebar.button("Generate CSV Files"):
        generate_csv_files(df_yearly)
        st.sidebar.success("CSV files generated in output_style_rally_yearly/")

else:
    st.error("Could not load data. Please check that the required CSV files exist.")