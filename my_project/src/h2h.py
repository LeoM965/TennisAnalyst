import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import streamlit as st
import warnings

warnings.filterwarnings('ignore')


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('output_rally/match_stats.csv')
        df['Opponent'] = df['Result'].str.extract(r'vs(\w+)')[0]
        df['Win'] = df['Result'].str.startswith('W').astype(int)
        return df.dropna(subset=['Opponent'])
    except:
        st.error("CSV not found")
        return None


def get_h2h_matches(df, p1, p2):
    m1 = df[(df['Player'] == p1) & (df['Opponent'] == p2)].copy()
    m2 = df[(df['Player'] == p2) & (df['Opponent'] == p1)].copy()

    if not m1.empty:
        m1['Target'] = m1['Win']
    if not m2.empty:
        m2['Target'] = 1 - m2['Win']
        m2['Player'], m2['Opponent'] = p1, p2

    return pd.concat([m1, m2], ignore_index=True)


def get_basic_stats(matches):
    total = len(matches)
    wins = matches['Target'].sum()
    return {
        'matches': total,
        'wins': int(wins),
        'losses': int(total - wins),
        'win_pct': wins / total,
        'recent_form': matches.tail(3)['Target'].mean() if total >= 3 else wins / total
    }


def get_ml_analysis(matches):
    if len(matches) < 5 or matches['Target'].nunique() < 2:
        return None

    features = ['RallyLen', 'Match_Control_Metric', 'Rally_Length_Efficiency', 'Power_Balance_Index']
    X = matches[features].fillna(matches[features].mean())
    y = matches['Target']

    X_scaled = StandardScaler().fit_transform(X)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_scaled, y)

    return {
        'accuracy': accuracy_score(y, rf.predict(X_scaled)),
        'feature_importance': dict(zip(features, rf.feature_importances_))
    }


def create_charts(matches, player1, player2):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{player1} vs {player2} Analysis', fontsize=14)

    yearly = matches.groupby('Year')['Target'].agg(['sum', 'count'])
    yearly.columns = ['Wins', 'Total']
    yearly['Losses'] = yearly['Total'] - yearly['Wins']
    yearly[['Wins', 'Losses']].plot(kind='bar', color=['green', 'red'], ax=axes[0, 0])
    axes[0, 0].set_title('Results by Year')
    axes[0, 0].tick_params(axis='x', rotation=45)

    matches['RallyLen'].hist(bins=10, ax=axes[0, 1], color='blue', alpha=0.7)
    axes[0, 1].axvline(matches['RallyLen'].mean(), color='red', linestyle='--')
    axes[0, 1].set_title('Rally Length Distribution')

    matches['Match_Control_Metric'].hist(bins=10, ax=axes[1, 0], color='purple', alpha=0.7)
    axes[1, 0].axvline(matches['Match_Control_Metric'].mean(), color='red', linestyle='--')
    axes[1, 0].set_title('Match Control Distribution')

    wins = matches[matches['Target'] == 1]
    losses = matches[matches['Target'] == 0]

    if len(wins) > 0 and len(losses) > 0:
        metrics = ['RallyLen', 'Match_Control_Metric', 'Power_Balance_Index']
        win_means = [wins[m].mean() for m in metrics]
        loss_means = [losses[m].mean() for m in metrics]

        x = range(len(metrics))
        width = 0.35
        axes[1, 1].bar([i - width / 2 for i in x], win_means, width, label='Wins', color='green')
        axes[1, 1].bar([i + width / 2 for i in x], loss_means, width, label='Losses', color='red')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].set_title('Wins vs Losses Comparison')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Need both wins and losses', ha='center', va='center',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Performance Comparison')

    plt.tight_layout()
    return fig


def run_analysis():
    st.title("Tennis H2H Analysis")

    df = load_data()
    if df is None:
        return

    players = sorted(df['Player'].unique())

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Player 1:", players)
    with col2:
        opponents = sorted(df[df['Player'] == p1]['Opponent'].unique())
        p2 = st.selectbox("Player 2:", opponents)

    if st.button("Analyze"):
        matches = get_h2h_matches(df, p1, p2)

        if matches.empty:
            st.error("No matches found")
            return

        stats = get_basic_stats(matches)
        ml_results = get_ml_analysis(matches)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Matches", stats['matches'])
        with col2:
            st.metric("Wins", stats['wins'])
        with col3:
            st.metric("Losses", stats['losses'])
        with col4:
            st.metric("Win Rate", f"{stats['win_pct']:.1%}")
        with col5:
            st.metric("Recent Form", f"{stats['recent_form']:.1%}")

        # Match history table
        st.subheader("Match History")
        history = matches[['Match', 'Year', 'Target', 'RallyLen', 'Match_Control_Metric']].copy()
        history['Result'] = history['Target'].map({1: 'Win', 0: 'Loss'})
        history = history.drop('Target', axis=1).sort_values('Year', ascending=False)
        st.dataframe(history, use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["Charts", "ML Analysis", "Summary"])

        with tab1:
            fig = create_charts(matches, p1, p2)
            st.pyplot(fig)

        with tab2:
            if ml_results:
                st.write(f"Model Accuracy: {ml_results['accuracy']:.3f}")
                st.write("Feature Importance:")
                for feature, importance in sorted(ml_results['feature_importance'].items(),
                                                  key=lambda x: x[1], reverse=True):
                    st.write(f"- {feature}: {importance:.3f}")
            else:
                st.info("Need more matches for ML analysis")

        with tab3:
            st.write("Performance Summary")

            if len(matches[matches['Target'] == 1]) > 0:
                wins_data = matches[matches['Target'] == 1]
                st.write("**When Winning:**")
                st.write(f"Average Rally Length: {wins_data['RallyLen'].mean():.2f}")
                st.write(f"Average Match Control: {wins_data['Match_Control_Metric'].mean():.3f}")

            if len(matches[matches['Target'] == 0]) > 0:
                losses_data = matches[matches['Target'] == 0]
                st.write("**When Losing:**")
                st.write(f"Average Rally Length: {losses_data['RallyLen'].mean():.2f}")
                st.write(f"Average Match Control: {losses_data['Match_Control_Metric'].mean():.3f}")


if __name__ == "__main__":
    run_analysis()