import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Raw Data Files
WTA_MCP_RALLY = os.path.join(RAW_DATA_DIR, 'wta_mcp_rally.csv')
WTA_MCP_RETURN = os.path.join(RAW_DATA_DIR, 'wta_mcp_return.csv')
WTA_MCP_SERVE = os.path.join(RAW_DATA_DIR, 'wta_mcp_serve.csv')
WTA_MCP_TACTICS = os.path.join(RAW_DATA_DIR, 'wta_mcp_tactics.csv')
WTA_WINNERS_UE = os.path.join(RAW_DATA_DIR, 'wta_winners_unforced_errors.csv')
WTA_PLAYERS = os.path.join(RAW_DATA_DIR, 'wta_players.csv')

# Output Directories
RALLY_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_rally')
RETURN_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_return')
SERVE_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_serve')
TACTICS_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_tactics')
WE_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_we')
TRN_ANALYSIS_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_player_tournament_analysis')
ST_RALLY_YEARLY_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_style_rally_yearly')
ST_RALLY_CAREER_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_style_rally_career')

# Column Groups
RALLY_PERCENTAGE_COLS = ['1-3 W%', '4-6 W%', '7-9 W%', '10+ W%']
POWER_COLS = ['FHP', 'BHP']
DOMINANCE_COLS = ['FHD', 'BHD']

# Analysis Indicators
RALLY_INDICATORS = [
    'Rally_Length_Efficiency',
    'Serve_Return_Rally_Balance',
    'Short_Rally_Control',
    'Long_Rally_Endurance',
    'Rally_Progression_Score',
    'Forehand_Dominance',
    'Backhand_Versatility',
    'Forehand_Power_Index',
    'Backhand_Power_Index',
    'Power_Balance_Index',
    'Rally_Adaptability',
    'Match_Control_Metric',
    'Tactical_Intelligence',
    'Court_Coverage_Efficiency'
]

RETURN_INDICATORS = [
    'Return_In_Play_Rate',
    'Return_Win_Efficiency',
    'Return_Aggression',
    'Return_Forehand_Ratio',
    'Return_Depth_Index',
    'Return_Defense_Rate',
    'First_Serve_Return_Quality',
    'Second_Serve_Return_Quality',
    'Serve_Return_Adaptability',
    'Return_Consistency',
    'Return_Tactical_Balance',
    'Return_Positioning_Intelligence',
    'Service_Type_Adaptability',
    'Defensive_Versatility',
    'Return_Match_Impact',
    'Overall_Return_Game'
]

RETURN_PCT_COLS = [
    'Overall_RiP', 'Overall_RiP_W', 'Overall_RetWnr', 'Overall_Slice',
    'First_RiP', 'First_RiP_W', 'First_RetWnr', 'First_Slice',
    'Second_RiP', 'Second_RiP_W', 'Second_RetWnr', 'Second_Slice'
]

SERVE_INDICATORS = [
    'Serve_Power',
    'Serve_Quick_Points',
    'Serve_Rally_Control',
    'First_Serve_Power',
    'First_Serve_Dominance',
    'Second_Serve_Aggression',
    'Second_Serve_Effectiveness',
    'First_Serve_Placement_Strategy',
    'Second_Serve_Placement_Strategy',
    'Serve_Type_Adaptability',
    'Serve_Consistency',
    'Power_Control_Balance',
    'Clutch_Serving',
    'Serve_Tactical_Intelligence',
    'Serve_Match_Impact',
    'Overall_Serve_Game'
]

SERVE_PCT_COLS = [
    'Overall_Unret', 'Overall_W3', 'Overall_RiP', 'First_Unret', 'First_W3', 'First_RiP',
    'First_D_Wide', 'First_A_Wide', 'First_BP_Wide', 'Second_Unret', 'Second_W3',
    'Second_RiP', 'Second_D_Wide', 'Second_A_Wide', 'Second_BP_Wide'
]

TACTICS_INDICATORS = [
    'Net_Game_Frequency',
    'Net_Game_Effectiveness',
    'Net_Game_Impact',
    'Serve_Volley_Frequency',
    'Serve_Volley_Effectiveness',
    'Serve_Volley_Impact',
    'Forehand_Power',
    'Backhand_Power',
    'Groundstroke_Balance',
    'Overall_Groundstroke_Power',
    'Forehand_DTL_Control',
    'Forehand_IO_Control',
    'Backhand_DTL_Control',
    'Directional_Versatility',
    'Drop_Shot_Usage',
    'Drop_Shot_Effectiveness',
    'Drop_Shot_Impact',
    'Rally_Aggression',
    'Return_Aggression',
    'Overall_Aggression',
    'Court_Position_Strategy',
    'Tactical_Versatility',
    'Power_Finesse_Balance',
    'Tactical_Adaptability',
    'Tactical_Match_Impact',
    'Offensive_Efficiency',
    'Tactical_Intelligence',
    'Overall_Tactical_Game'
]

TACTICS_PCT_COLS = [
    'SnV_Freq', 'SnV_W_Pct', 'Net_Freq', 'Net_W_Pct', 'FH_Wnr_Pct',
    'FH_DTL_Wnr_Pct', 'FH_IO_Wnr_Pct', 'BH_Wnr_Pct', 'BH_DTL_Wnr_Pct',
    'Drop_Freq', 'Drop_Wnr_Pct'
]

WEB_INDICATORS = [
    'Rally_Dominance_Index',
    'Tactical_Balance_Score',
    'Power_Asymmetry_Index',
    'Pressure_Creation_Index',
    'Match_Control_Efficiency',
    'Shot_Selection_IQ',
    'Pressure_Consistency_Index'
]

WEB_PCT_COLS = [
    'Wnr/Pt', 'UFE/Pt', 'FH Wnr/Pt', 'BH Wnr/Pt', 'vs UFE/Pt', 'vs Wnr/Pt', 'RallyWinners', 'RallyUFEs'
]

# Scraping Config
TABLE_CONFIGS = {
   'mcp-rally': {'key_headers': ['Match', 'RLen-Serve', 'RLen-Return'], 'filename': 'wta_mcp_rally.csv'},
   'mcp-serve': {'key_headers': ['Match', 'Unret%'], 'filename': 'wta_mcp_serve.csv'},
   'mcp-tactics': {'key_headers': ['Match', 'SnV Freq', 'SnV W%'], 'filename': 'wta_mcp_tactics.csv'},
   'mcp-return': {'key_headers': ['Match', 'RiP%'], 'filename': 'wta_mcp_return.csv'}
}
