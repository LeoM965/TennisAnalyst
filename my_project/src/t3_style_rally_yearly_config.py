FEATURES = ['Avg_Rally_Length', 'Rally_Length_Efficiency', 'Serve_Return_Rally_Balance',
            'Short_Rally_Control', 'Long_Rally_Endurance', 'Rally_Progression_Score',
            'Forehand_Dominance', 'Backhand_Versatility', 'Forehand_Power_Index',
            'Backhand_Power_Index', 'Power_Balance_Index', 'Rally_Adaptability',
            'Match_Control_Metric', 'Tactical_Intelligence', 'Court_Coverage_Efficiency']

KEY_METRICS = ['Avg_Rally_Length', 'Power_Balance_Index', 'Rally_Adaptability', 'Tactical_Intelligence']

HEATMAP_METRICS = ['Avg_Rally_Length', 'Power_Balance_Index', 'Rally_Adaptability',
                   'Tactical_Intelligence', 'Court_Coverage_Efficiency', 'Match_Control_Metric']

STYLE_NAMES = {0: 'Power Hitter', 1: 'Baseline Grinder', 2: 'All-Court Player',
               3: 'Defender', 4: 'Counter Puncher', 5: 'Aggressive Baseliner'}

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#98D8C8', '#F7DC6F', '#BB8FCE']

STREAMLIT_CONFIG = {
    'page_title': "Tennis Analysis Dashboard",
    'layout': "wide"
}

DATA_PATHS = {
    'yearly': 'output_rally/yearly_stats.csv',
    'career': 'output_rally/career_stats.csv'
}

OUTPUT_DIR = 'output_style_rally_yearly'

INSIGHTS = [
    "Average rally length evolution over time",
    "Power balance development in tennis",
    "Rally adaptability improvements",
    "Court coverage efficiency trends"
]