FEATURES = [
    'Avg_Rally_Length', 'Rally_Length_Efficiency', 'Serve_Return_Rally_Balance',
    'Short_Rally_Control', 'Long_Rally_Endurance', 'Rally_Progression_Score',
    'Forehand_Dominance', 'Backhand_Versatility', 'Forehand_Power_Index',
    'Backhand_Power_Index', 'Power_Balance_Index', 'Rally_Adaptability',
    'Match_Control_Metric', 'Tactical_Intelligence', 'Court_Coverage_Efficiency'
]

STYLE_NAMES = {
    0: 'Power Hitter',
    1: 'Baseline Grinder',
    2: 'All-Court Player',
    3: 'Defender',
    4: 'Counter Puncher',
    5: 'Aggressive Baseliner'
}

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#98D8C8', '#F7DC6F', '#BB8FCE']

KEY_FEATURES_HEATMAP = [
    'Serve_Return_Rally_Balance',
    'Rally_Progression_Score',
    'Power_Balance_Index',
    'Match_Control_Metric',
    'Court_Coverage_Efficiency'
]

SCATTER_OPTIONS = {
    "Rally Length vs Efficiency": ('Avg_Rally_Length', 'Rally_Length_Efficiency'),
    "Power Balance vs Adaptability": ('Power_Balance_Index', 'Rally_Adaptability'),
    "Serve Balance vs Tactical Intelligence": ('Serve_Return_Rally_Balance', 'Tactical_Intelligence'),
    "Rally Progression vs Match Control": ('Rally_Progression_Score', 'Match_Control_Metric'),
    "Forehand vs Backhand Power": ('Forehand_Power_Index', 'Backhand_Power_Index'),
    "Forehand vs Backhand Dominance": ('Forehand_Dominance', 'Backhand_Dominance'),
    "Forehand Dominance vs Backhand Versatility": ('Forehand_Dominance', 'Backhand_Versatility')
}

STREAMLIT_CONFIG = {
    'page_title': "Career Playing Styles Analysis",
    'layout': "wide"
}

DATA_PATH = 'output_rally/career_stats.csv'
OUTPUT_DIR = 'output_style_rally_career'