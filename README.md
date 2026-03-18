# TennisAnalyst — WTA Advanced Performance Analytics Platform

A comprehensive data analytics platform designed to scrape, process, and analyze advanced performance statistics for top WTA (Women's Tennis Association) players. The system combines automated web scraping from TennisAbstract with machine learning–driven statistical modeling to produce deep insights into rally patterns, serve effectiveness, return quality, tactical tendencies, and overall playing styles.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Data Pipeline](#data-pipeline)
5. [Tier 1 — Data Collection (Web Scraping)](#tier-1--data-collection-web-scraping)
6. [Tier 2 — Statistical Analysis & Indicator Engineering](#tier-2--statistical-analysis--indicator-engineering)
7. [Tier 3 — Playing Style Classification & Interactive Dashboards](#tier-3--playing-style-classification--interactive-dashboards)
8. [Head-to-Head (H2H) Analysis Module](#head-to-head-h2h-analysis-module)
9. [Constants & Configuration](#constants--configuration)
10. [Computed Indicators Reference](#computed-indicators-reference)
11. [Output Files](#output-files)
12. [Installation & Setup](#installation--setup)
13. [Usage Guide](#usage-guide)
14. [Dependencies](#dependencies)
15. [Data Sources](#data-sources)
16. [Tracked Players](#tracked-players)

---

## Overview

TennisAnalyst is a multi-tier Python application that provides an end-to-end workflow for WTA tennis performance analytics:

- **Automated Data Acquisition** — Selenium-based web scraping from [TennisAbstract](https://www.tennisabstract.com) collects match-level statistics for 17 top WTA players across four statistical categories (Rally, Serve, Return, Tactics) plus Winners/Unforced Errors data.
- **Advanced Indicator Engineering** — Raw scraped data is transformed into 70+ computed performance indicators using PCA-learned feature weights, exponential decay functions, correlation-based weighting, and result-adjusted multipliers. Every indicator is derived from the raw match data using statistical methods — no arbitrary hard-coded weights.
- **Machine Learning Style Classification** — KMeans clustering with StandardScaler normalization classifies players into six distinct playing styles (Power Hitter, Baseline Grinder, All-Court Player, Defender, Counter Puncher, Aggressive Baseliner) based on 15 performance features.
- **Interactive Dashboards** — Three Streamlit dashboards provide rich, interactive visualizations for career-level analysis, yearly evolution tracking, and per-tournament performance breakdowns with ML-powered success factor identification.
- **Head-to-Head Analysis** — A dedicated module extracts head-to-head match histories between any two players and uses a RandomForest classifier to identify the key performance factors that predict match outcomes.

---

## Architecture

The project follows a strict three-tier architecture, where each tier depends on the outputs of the previous tier:

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TIER 1 — DATA COLLECTION                        │
│  t1_trn_scrapping.py    →  Scrapes rally/serve/return/tactics data   │
│  t1_we_scrapping.py     →  Scrapes winners/unforced errors data      │
│  Source: TennisAbstract (Selenium + BeautifulSoup)                   │
│  Output: data/raw/*.csv                                              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│              TIER 2 — STATISTICAL ANALYSIS & INDICATORS              │
│  t2_trn_stats_rally.py      →  14 rally performance indicators       │
│  t2_trn_stats_serve.py      →  16 serve performance indicators       │
│  t2_trn_stats_return.py     →  16 return performance indicators      │
│  t2_trn_stats_tactics.py    →  28 tactical performance indicators    │
│  t2_we_stats.py             →   7 winners/errors indicators          │
│  Method: PCA-learned weights, correlation analysis, exponential      │
│          decay, tanh normalization, result-adjusted multipliers       │
│  Output: output_rally/, output_serve/, output_return/,               │
│          output_tactics/, output_we/                                  │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│        TIER 3 — STYLE CLASSIFICATION & INTERACTIVE DASHBOARDS        │
│  t3_style_rally_career.py   →  Career-level style analysis dashboard │
│  t3_style_rally_yearly.py   →  Yearly evolution dashboard            │
│  t3_style_rally_match.py    →  Tournament/match-level dashboard      │
│  h2h.py                    →  Head-to-Head analysis dashboard        │
│  Method: KMeans clustering, RandomForest, PCA dimensionality         │
│          reduction, StandardScaler normalization                     │
│  Output: output_style_rally_career/, output_style_rally_yearly/,     │
│          output_player_tournament_analysis/                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
TennisAnalyst/
├── data/
│   ├── raw/                              # Raw scraped CSV files
│   │   ├── wta_players.csv               # Player registry (IDs and names)
│   │   ├── wta_mcp_rally.csv             # Raw rally statistics per match
│   │   ├── wta_mcp_serve.csv             # Raw serve statistics per match
│   │   ├── wta_mcp_return.csv            # Raw return statistics per match
│   │   ├── wta_mcp_tactics.csv           # Raw tactics statistics per match
│   │   └── wta_winners_unforced_errors.csv  # Raw winners/UFE per match
│   └── processed/                        # (Reserved for processed outputs)
│
├── my_project/
│   └── src/                              # All Python source code
│       ├── constants.py                  # Central configuration & indicator definitions
│       │
│       ├── t1_trn_scrapping.py           # Tier 1: Scrape rally/serve/return/tactics
│       ├── t1_we_scrapping.py            # Tier 1: Scrape winners/unforced errors
│       │
│       ├── t2_trn_stats_rally.py         # Tier 2: Rally analysis pipeline
│       ├── t2_trn_stats_rally_weights.py # ├── Year extraction, PCA weight learning
│       ├── t2_trn_stats_rally_indicators.py # └── Rally indicator calculation engine
│       │
│       ├── t2_trn_stats_serve.py         # Tier 2: Serve analysis pipeline
│       ├── t2_trn_stats_serve_weights.py # ├── PCA serve weight learning
│       ├── t2_trn_stats_serve_indicators.py # └── Serve indicator calculation engine
│       │
│       ├── t2_trn_stats_return.py        # Tier 2: Return analysis pipeline
│       ├── t2_trn_stats_return_weights.py# ├── PCA return weight learning
│       ├── t2_trn_stats_return_indicators.py# └── Return indicator calculation engine
│       │
│       ├── t2_trn_stats_tactics.py       # Tier 2: Tactics analysis pipeline
│       ├── t2_trn_stats_tactics_weights.py# ├── PCA tactics weight learning
│       ├── t2_trn_stats_tactics_indicators.py# └── Tactics indicator calculation engine
│       │
│       ├── t2_we_stats.py                # Tier 2: Winners/Errors analysis pipeline
│       ├── t2_we_stats_indicators.py        # └── WE indicator calculation engine
│       │
│       ├── t3_style_rally_career.py      # Tier 3: Career style Streamlit dashboard
│       ├── t3_style_rally_career_config.py# ├── Configuration, feature lists, style names
│       ├── t3_style_rally_career_analysis.py# └── Clustering, scatter/bar/heatmap plots
│       │
│       ├── t3_style_rally_yearly.py      # Tier 3: Yearly evolution Streamlit dashboard
│       ├── t3_style_rally_yearly_config.py# ├── Configuration, heatmap metrics
│       ├── t3_style_rally_yearly_analysis.py# └── Data loading, line/heatmap/distribution plots
│       │
│       ├── t3_style_rally_match.py       # Tier 3: Tournament analysis Streamlit dashboard
│       ├── t3_style_rally_match_config.py# ├── Configuration, data paths
│       ├── t3_style_rally_match_analysis.py# └── ML analysis, tournament stats, plots
│       │
│       ├── h2h.py                        # Head-to-Head analysis Streamlit dashboard
│       │
│       ├── output_rally/                 # Rally analysis output CSVs
│       ├── output_serve/                 # Serve analysis output CSVs
│       ├── output_return/                # Return analysis output CSVs
│       ├── output_tactics/               # Tactics analysis output CSVs
│       ├── output_we/                    # Winners/Errors analysis output CSVs
│       ├── output_style_rally_career/    # Career style analysis output CSVs
│       ├── output_style_rally_yearly/    # Yearly style evolution output CSVs
│       └── output_player_tournament_analysis/ # Tournament analysis output CSVs
│
├── requirements.txt                      # Python dependencies
└── README.md                             # This documentation file
```

---

## Data Pipeline

The data flows through three sequential stages:

### Stage 1: Raw Data Acquisition
Selenium-based scrapers navigate to each player's TennisAbstract profile page and extract HTML tables containing match-by-match statistics. The scrapers use headless Chrome with retry logic (up to 3 attempts) and `WebDriverWait` for reliable table detection. Each row represents one match, and the data is saved as CSV files in `data/raw/`.

### Stage 2: Indicator Computation
Each Tier 2 module reads the corresponding raw CSV, cleans percentage columns (stripping `%` symbols and converting to fractions), then applies a series of statistical transformations:

1.  **PCA Weight Learning** — Principal Component Analysis is applied to groups of related metrics (e.g., rally win percentages at different lengths) to learn data-driven importance weights. The first principal component's loadings are normalized to produce weights that reflect each metric's contribution to the overall variance.
2.  **Correlation-Based Weighting** — For power metrics (FHP/BHP), the correlation between forehand and backhand power determines whether to weight tactical diversity more or less heavily.
3.  **Result-Adjusted Multipliers** — Winning matches receive a 1.1–1.15x bonus, losing matches a 0.85–0.9x penalty, reflecting that performance in winning conditions is inherently more valuable.
4.  **Exponential Decay** — Used for metrics like Rally Length Efficiency (`exp(-length/5)`) and Court Coverage Efficiency (`exp(-length/6)`) to produce diminishing-returns curves.
5.  **Hyperbolic Tangent Normalization** — `tanh()` is used to bound metrics like Power-Control Balance and Court Position Strategy into the (−1, 1) range.

### Stage 3: Aggregation & Output
Each Tier 2 module produces four standard output files:
- `match_*_stats.csv` — Per-match indicator values
- `yearly_*_stats.csv` — Grouped by Player × Year (means + match counts)
- `career_*_stats.csv` — Grouped by Player across all years
- `top_*_performers.csv` — Top 10 players for each indicator

---

## Tier 1 — Data Collection (Web Scraping)

### `t1_trn_scrapping.py` — Tournament Statistics Scraper

Scrapes four categories of match-level statistics from [TennisAbstract](https://www.tennisabstract.com):

| Table Type      | Key Headers Detected           | Output File             |
|-----------------|-------------------------------|-------------------------|
| `mcp-rally`     | `Match`, `RLen-Serve`, `RLen-Return` | `wta_mcp_rally.csv` |
| `mcp-serve`     | `Match`, `Unret%`              | `wta_mcp_serve.csv`  |
| `mcp-tactics`   | `Match`, `SnV Freq`, `SnV W%` | `wta_mcp_tactics.csv`|
| `mcp-return`    | `Match`, `RiP%`                | `wta_mcp_return.csv` |

**Key Implementation Details:**
- Players are loaded from `wta_players.csv` (player ID + names)
- URLs follow the pattern: `https://www.tennisabstract.com/cgi-bin/wplayer-more.cgi?p={id}/{name}&table={type}`
- Headless Chrome with custom user-agent to avoid detection
- Automatic retry logic (3 attempts) with driver re-initialization on failure
- Table validation via key header matching before data extraction
- All data rows are prepended with the player's name

### `t1_we_scrapping.py` — Winners/Unforced Errors Scraper

Scrapes the `winners-errors` table type from TennisAbstract, collecting per-match counts of winners, unforced errors, forehand/backhand winners, forced errors from opponents, and rally-specific winners/errors. The implementation mirrors `t1_trn_scrapping.py` with identical retry and validation logic, but targets the `Winners` and `UFEs` key headers.

---

## Tier 2 — Statistical Analysis & Indicator Engineering

Each Tier 2 module follows an identical architectural pattern:

```
Main Script (t2_*_stats_*.py)
├── Reads raw CSV from data/raw/
├── Extracts year from match identifiers
├── Calls indicators helper for engine
├── Aggregates: match → yearly → career → top performers
└── Saves 4 CSV files to output_*/

Weights Helper (t2_*_weights.py)
├── extract_year() — regex extraction of YYYY from match strings
└── learn_*_weights() — PCA-based weight learning

Indicators Helper (t2_*_indicators.py)
├── Percentage column cleaning (strip %, divide by 100)
├── Semantic column detection (hardcoded mapping)
├── Calls weights helper for weight learning
└── Computes all indicator formulas with type hints and docstrings
```

### Rally Analysis (`t2_trn_stats_rally.py` + helpers)

Processes serve/return rally lengths and win percentages at different rally lengths (1–3, 4–6, 7–9, 10+ shots).

**PCA Weight Learning (t2_trn_stats_rally_weights.py):**
- Learns importance weights for rally win percentage columns (`1-3 W%`, `4-6 W%`, `7-9 W%`, `10+ W%`) using PCA's first component loadings
- Learns tactical weights from forehand/backhand power correlation: if `|corr(FHP, BHP)| > 0.5`, then `[0.4, 0.4, 0.2]`; otherwise `[0.3, 0.3, 0.4]` (more weight on adaptability when power is unbalanced)
- Learns match control weights by comparing short-rally and long-rally impact on winning vs losing

**Indicator Formulas (t2_trn_stats_rally_indicators.py):**

| Indicator | Formula |
|-----------|---------|
| Rally_Length_Efficiency | `exp(-avg_rally_length / 5)` |
| Serve_Return_Rally_Balance | `1 - (|serve_len - return_len| / (serve_len + return_len + 1))` |
| Short_Rally_Control | `1-3 Win%` (direct mapping) |
| Long_Rally_Endurance | `10+ Win%` (direct mapping) |
| Rally_Progression_Score | Weighted sum of all rally win%s using PCA weights |
| Forehand_Dominance | `FH/GS` ratio |
| Backhand_Versatility | `1 - BH_Slice%` |
| Forehand_Power_Index | `FHP / (FHP/100 + 0.1)` |
| Backhand_Power_Index | `BHP / (BHP/100 + 0.1)` |
| Power_Balance_Index | `1 - (|FHP - BHP| / (FHP + BHP))` |
| Rally_Adaptability | `exp(-std(all_rally_win_pcts) * 4)` |
| Match_Control_Metric | `(short_control * w1 + long_endurance * w2) * result_multiplier` |
| Tactical_Intelligence | Weighted sum of [Forehand_Dominance, Backhand_Versatility, Rally_Adaptability] |
| Court_Coverage_Efficiency | `(exp(-serve_len/6) + exp(-return_len/6)) / 2` |

### Serve Analysis (`t2_trn_stats_serve.py` + helpers)

Processes unreturned serve percentages, win-within-3-shots rates, rally-in-play rates, and serve placement breakdowns for first/second serves.

**PCA Weight Learning:** Learns serve importance weights from `Overall_Unret`, `Overall_W3`, `Overall_RiP` using PCA. Learns first/second serve type importance from the variability (standard deviation) of each serve type's unreturned rate.

**Key Indicators:**
- `Serve_Power`, `Serve_Quick_Points`, `Serve_Rally_Control` — direct mappings
- `First_Serve_Dominance` — weighted composite: `0.4×Unret + 0.35×W3 + 0.25×RiP`
- `Second_Serve_Effectiveness` — weighted composite: `0.3×Unret + 0.4×W3 + 0.3×RiP`
- `First/Second_Serve_Placement_Strategy` — average of deuce/ad/break-point wide percentages
- `Serve_Type_Adaptability` — PCA-weighted combination of first/second serve scores
- `Serve_Consistency` — `exp(-|first_dominance - second_effectiveness| × 2)`
- `Power_Control_Balance` — `tanh(power / (control + 0.01))`
- `Clutch_Serving` — average break-point wide percentages
- `Serve_Tactical_Intelligence` — `0.4×placement + 0.4×adaptability + 0.2×clutch`
- `Serve_Match_Impact` — Adaptability × result multiplier (1.1 win / 0.9 loss)
- `Overall_Serve_Game` — composite: `0.2×power + 0.15×quick + 0.15×rally + 0.25×adapt + 0.15×intelligence + 0.1×balance`

### Return Analysis (`t2_trn_stats_return.py` + helpers)

Processes return-in-play rates, return winner percentages, return depth indices (RDI), slice rates, and forehand/backhand return splits for overall/first-serve/second-serve returns.

**Key Indicators:**
- `Return_In_Play_Rate`, `Return_Win_Efficiency`, `Return_Aggression` — direct mappings
- `Return_Forehand_Ratio` — parsed from `FH/BH` column (e.g., `"25/15"` → `25/(25+15)`)
- `Return_Depth_Index` — numeric RDI value
- `First/Second_Serve_Return_Quality` — `0.4×RiP + 0.4×RiP_W + 0.2×RetWnr`
- `Serve_Return_Adaptability` — PCA-weighted blend of first/second serve return quality
- `Return_Consistency` — `exp(-|first_depth - second_depth| / 0.5)`
- `Return_Tactical_Balance` — `tanh(aggression / (defense_rate + 0.01))`
- `Return_Positioning_Intelligence` — `(depth / 3.0) × (1 + in_play_rate)`
- `Service_Type_Adaptability` — `exp(-|first_quality - second_quality| × 2)`
- `Defensive_Versatility` — average of first/second serve slice rates
- `Return_Match_Impact` — Adaptability × result multiplier
- `Overall_Return_Game` — `0.2×in_play + 0.2×win_eff + 0.15×aggression + 0.25×adaptability + 0.2×tactical_balance`

### Tactics Analysis (`t2_trn_stats_tactics.py` + helpers)

Processes net approach frequency/win rates, serve-and-volley frequency/win rates, forehand/backhand winner and down-the-line percentages, drop shot frequency/winner rates, and rally/return aggression scores.

**PCA Weight Learning:** Learns tactical importance weights from `Net_Freq`, `Net_W_Pct`, `FH_Wnr_Pct`, `BH_Wnr_Pct` using PCA.

**Key Indicators:**
- `Net_Game_Impact` — `Net_Frequency × Net_Win_Pct`
- `Serve_Volley_Impact` — `SnV_Frequency × SnV_Win_Pct`
- `Drop_Shot_Impact` — `Drop_Frequency × Drop_Win_Pct`
- `Groundstroke_Balance` — `|FH_Power - BH_Power|`
- `Overall_Groundstroke_Power` — `(FH_Power + BH_Power) / 2`
- `Directional_Versatility` — `(FH_DTL + FH_IO + BH_DTL) / 2`
- `Rally/Return_Aggression` — normalized to [0, 1] range by dividing by 200 and clipping
- `Overall_Aggression` — `0.6×rally + 0.4×return`
- `Court_Position_Strategy` — `tanh(net_freq / baseline_power)`
- `Tactical_Versatility` — counts how many of 5 offensive weapons are active (normalized to [0, 1])
- `Power_Finesse_Balance` — `tanh(power_intensity / (finesse_intensity + 0.01))`
- `Tactical_Adaptability` — `0.5×versatility + 0.3×power_finesse + 0.2×court_position`
- `Offensive_Efficiency` — `0.4×groundstroke_power + 0.3×net_impact + 0.3×aggression`
- `Tactical_Intelligence` — `0.3×versatility + 0.25×directional + 0.25×adaptability + 0.2×power_finesse`
- `Overall_Tactical_Game` — composite of 5 weighted components

### Winners/Errors Analysis (`t2_we_stats.py` + helper)

Processes raw winners, unforced errors, forehand/backhand winners per point, opponent errors/winners per point, and rally-specific winners/errors.

**Key Indicators:**
- `Rally_Dominance_Index` — `(wnr/pt - vs_wnr/pt) + (vs_ufe/pt - ufe/pt)`
- `Tactical_Balance_Score` — `1 - |0.5 - FH_ratio|` (closer to 0.5 = more balanced)
- `Power_Asymmetry_Index` — `|FH_winners - BH_winners|`
- `Pressure_Creation_Index` — `0.6×winners + 0.4×forced_errors`
- `Match_Control_Efficiency` — `(winners - errors) / (winners + errors)`
- `Shot_Selection_IQ` — `rally_winners / (rally_winners + rally_errors)`
- `Pressure_Consistency_Index` — `1 - |pressure - shot_selection_iq|`

---

## Tier 3 — Playing Style Classification & Interactive Dashboards

### KMeans Style Classification

All three Tier 3 dashboards use KMeans clustering (k=6, StandardScaler preprocessing) on 15 career-level rally features to classify players into six distinct playing styles:

| Cluster | Style Name           | Description |
|---------|---------------------|-------------|
| 0       | Power Hitter        | Dominates through raw shot power and aggressive play |
| 1       | Baseline Grinder    | Focuses on long rallies, consistency, and wearing opponents down |
| 2       | All-Court Player    | Versatile player effective in all areas of the court |
| 3       | Defender            | Prioritizes defensive positioning and returns |
| 4       | Counter Puncher     | Absorbs pressure and converts defensive positions into winning opportunities |
| 5       | Aggressive Baseliner| Combines baseline play with aggressive shot selection |

### `t3_style_rally_career.py` — Career Style Analysis Dashboard

A five-tab Streamlit dashboard for analyzing career-level playing styles:

- **Tab 1 — Style Overview:** Bar chart distribution of playing styles, percentage breakdown, and grouped statistics table showing average metrics per style
- **Tab 2 — Scatter Analysis:** Interactive scatter plots with 7 preset axis pairs (e.g., Rally Length vs Efficiency, Forehand vs Backhand Power), color-coded by style with player name annotations and optional diagonal reference lines
- **Tab 3 — Power Analysis:** Horizontal bar plots of forehand–backhand power difference and dominance difference per player, with style-level power statistics table
- **Tab 4 — Advanced Metrics Heatmap:** Color-coded heatmap of 5 key features across all playing styles, plus filterable detailed style statistics with per-player drilldown
- **Tab 5 — Player Details:** Individual player analysis showing 9 key metrics, complete stats table, and list of similarly-styled players

**Sidebar:** Export functionality to save `player_analysis.csv` and `style_stats.csv` to `output_style_rally_career/`

### `t3_style_rally_yearly.py` — Yearly Evolution Dashboard

A five-tab Streamlit dashboard for tracking how playing styles evolve over time:

- **Tab 1 — Evolution Overview:** Line plots of 4 key metrics averaged across all players by year, with selectable single-metric deep-dive and key insights
- **Tab 2 — Style Distribution:** Stacked bar chart showing how the number of players in each style changes year-over-year, with selectable year detail showing percentage breakdown
- **Tab 3 — Yearly Heatmap:** Seaborn heatmap of 6 metrics by year, with corresponding numerical statistics table
- **Tab 4 — Player Analysis:** Per-year scatter plots showing individual player positions within their styles across 4 key metrics, with 6-metric individual player detail cards
- **Tab 5 — Style Comparison:** Heatmap of metrics by playing style, with multi-select comparison bar charts

**Sidebar:** CSV generation (yearly stats, style stats, player evolution, career changes) to `output_style_rally_yearly/`

### `t3_style_rally_match.py` — Player Tournament Analysis Dashboard

A Streamlit dashboard for match-level analysis within specific years:

- **Player/Year Selection** — Dropdown selectors for player and year, with automatic playing style identification via KMeans clustering on career data
- **Match Metrics** — Win count, win rate, average match control, rally length, and identified playing style
- **Tournament Table** — Grouped by tournament: wins, matches, win rate, average control
- **Tab 1 — Performance Chart:** 6-panel matplotlib figure showing tournament results, success factors (feature importance), control by tournament, wins vs losses comparison, rally length trends, and PCA performance styles map
- **Tab 2 — Win/Loss Analysis:** Side-by-side comparison of key metrics in winning vs losing matches
- **Tab 3 — Success Factors:** RandomForest feature importance ranking with horizontal bar chart showing which indicators most predict winning
- **Tab 4 — Playing Patterns:** KMeans cluster analysis of match-level performance showing win rate by identified pattern

---

## Head-to-Head (H2H) Analysis Module

### `h2h.py`

A Streamlit dashboard for head-to-head analysis between any two players:

- **Match Extraction** — Combines matches from both players' perspectives, inverting win/loss status for the second player to create a unified view
- **Basic Statistics** — Total matches, wins, losses, win percentage, and recent form (last 3 matches rolling average)
- **ML Analysis** — RandomForest classifier (50 estimators) trained on 4 features (`RallyLen`, `Match_Control_Metric`, `Rally_Length_Efficiency`, `Power_Balance_Index`) to predict match outcomes. Reports training accuracy and feature importance rankings.
- **Visualization Tabs:**
  - **Charts** — 4-panel figure: results by year (stacked bar), rally length distribution (histogram with mean line), match control distribution, wins vs losses metric comparison
  - **ML Analysis** — Feature importance display with sorted rankings
  - **Summary** — Average rally length and match control for winning and losing matches separately

---

## Constants & Configuration

### `constants.py`

Central configuration file containing:

- **Path Definitions** — Base directory auto-detection via `os.path.abspath(__file__)`, with derived paths for `data/raw/`, `data/processed/`, and all output directories
- **Raw Data File Paths** — Constants for all 6 CSV file paths (`WTA_MCP_RALLY`, `WTA_MCP_RETURN`, `WTA_MCP_SERVE`, `WTA_MCP_TACTICS`, `WTA_WINNERS_UE`, `WTA_PLAYERS`)
- **Indicator Lists** — Complete lists of all indicator names for each analysis type (`RALLY_INDICATORS` with 14 entries, `RETURN_INDICATORS` with 16, `SERVE_INDICATORS` with 16, `TACTICS_INDICATORS` with 28, `WEB_INDICATORS` with 7)
- **Percentage Column Groups** — Lists of columns that need `%` stripping during preprocessing
- **Scraping Configuration** — `TABLE_CONFIGS` dictionary mapping table types to their key headers and output filenames

---

## Computed Indicators Reference

### Rally Indicators (14 total)
`Rally_Length_Efficiency`, `Serve_Return_Rally_Balance`, `Short_Rally_Control`, `Long_Rally_Endurance`, `Rally_Progression_Score`, `Forehand_Dominance`, `Backhand_Versatility`, `Forehand_Power_Index`, `Backhand_Power_Index`, `Power_Balance_Index`, `Rally_Adaptability`, `Match_Control_Metric`, `Tactical_Intelligence`, `Court_Coverage_Efficiency`

### Serve Indicators (16 total)
`Serve_Power`, `Serve_Quick_Points`, `Serve_Rally_Control`, `First_Serve_Power`, `First_Serve_Dominance`, `Second_Serve_Aggression`, `Second_Serve_Effectiveness`, `First_Serve_Placement_Strategy`, `Second_Serve_Placement_Strategy`, `Serve_Type_Adaptability`, `Serve_Consistency`, `Power_Control_Balance`, `Clutch_Serving`, `Serve_Tactical_Intelligence`, `Serve_Match_Impact`, `Overall_Serve_Game`

### Return Indicators (16 total)
`Return_In_Play_Rate`, `Return_Win_Efficiency`, `Return_Aggression`, `Return_Forehand_Ratio`, `Return_Depth_Index`, `Return_Defense_Rate`, `First_Serve_Return_Quality`, `Second_Serve_Return_Quality`, `Serve_Return_Adaptability`, `Return_Consistency`, `Return_Tactical_Balance`, `Return_Positioning_Intelligence`, `Service_Type_Adaptability`, `Defensive_Versatility`, `Return_Match_Impact`, `Overall_Return_Game`

### Tactics Indicators (28 total)
`Net_Game_Frequency`, `Net_Game_Effectiveness`, `Net_Game_Impact`, `Serve_Volley_Frequency`, `Serve_Volley_Effectiveness`, `Serve_Volley_Impact`, `Forehand_Power`, `Backhand_Power`, `Groundstroke_Balance`, `Overall_Groundstroke_Power`, `Forehand_DTL_Control`, `Forehand_IO_Control`, `Backhand_DTL_Control`, `Directional_Versatility`, `Drop_Shot_Usage`, `Drop_Shot_Effectiveness`, `Drop_Shot_Impact`, `Rally_Aggression`, `Return_Aggression`, `Overall_Aggression`, `Court_Position_Strategy`, `Tactical_Versatility`, `Power_Finesse_Balance`, `Tactical_Adaptability`, `Tactical_Match_Impact`, `Offensive_Efficiency`, `Tactical_Intelligence`, `Overall_Tactical_Game`

### Winners/Errors Indicators (7 total)
`Rally_Dominance_Index`, `Tactical_Balance_Score`, `Power_Asymmetry_Index`, `Pressure_Creation_Index`, `Match_Control_Efficiency`, `Shot_Selection_IQ`, `Pressure_Consistency_Index`

---

## Output Files

Each Tier 2 analysis module generates four output files following a consistent naming pattern:

| Output Directory   | Files Generated |
|--------------------|-----------------|
| `output_rally/`    | `match_rally_stats.csv`, `yearly_rally_stats.csv`, `career_rally_stats.csv`, `top_rally_performers.csv` |
| `output_serve/`    | `match_serve_stats.csv`, `yearly_serve_stats.csv`, `career_serve_stats.csv`, `top_serve_performers.csv` |
| `output_return/`   | `match_return_stats.csv`, `yearly_return_stats.csv`, `career_return_stats.csv`, `top_return_performers.csv` |
| `output_tactics/`  | `match_tactics_stats.csv`, `yearly_tactics_stats.csv`, `career_tactics_stats.csv`, `top_tactics_performers.csv` |
| `output_we/`       | `match_analysis.csv`, `player_indicators_by_year.csv`, `career_summary.csv`, `top_performers.csv` |

Tier 3 dashboards generate additional output files:

| Output Directory                   | Files Generated |
|------------------------------------|-----------------|
| `output_style_rally_career/`       | `player_analysis.csv`, `style_stats.csv` |
| `output_style_rally_yearly/`       | `yearly_statistics.csv`, `styles_statistics.csv`, `player_evolution.csv`, `player_career_changes.csv` |
| `output_player_tournament_analysis/` | Generated via dashboard interactions |

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Google Chrome browser (for Selenium web scraping)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LeoM965/TennisAnalyst.git
   cd TennisAnalyst
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data scraping pipeline (optional — raw data is included):**
   ```bash
   cd my_project/src
   python t1_trn_scrapping.py
   python t1_we_scrapping.py
   ```

4. **Run the statistical analysis:**
   ```bash
   python t2_trn_stats_rally.py
   python t2_trn_stats_serve.py
   python t2_trn_stats_return.py
   python t2_trn_stats_tactics.py
   python t2_we_stats.py
   ```

5. **Launch the interactive dashboards:**
   ```bash
   streamlit run t3_style_rally_career.py
   streamlit run t3_style_rally_yearly.py
   streamlit run t3_style_rally_match.py
   streamlit run h2h.py
   ```

---

## Usage Guide

### Running the Full Pipeline

Execute modules in order (Tier 1 → Tier 2 → Tier 3):

```bash
cd my_project/src

# Tier 1: Scrape data (requires Chrome + internet)
python t1_trn_scrapping.py    # ~15-30 min for 17 players × 4 tables
python t1_we_scrapping.py     # ~10-15 min for 17 players

# Tier 2: Compute indicators
python t2_trn_stats_rally.py
python t2_trn_stats_serve.py
python t2_trn_stats_return.py
python t2_trn_stats_tactics.py
python t2_we_stats.py

# Tier 3: Launch dashboards
streamlit run t3_style_rally_career.py   # Career style analysis
streamlit run t3_style_rally_yearly.py   # Yearly evolution
streamlit run t3_style_rally_match.py    # Tournament analysis
streamlit run h2h.py                     # Head-to-head comparison
```

### Adding New Players

Edit `data/raw/wta_players.csv` to add new players:
```csv
player_id,first_name,last_name
216347,Iga,Swiatek
```
The `player_id` corresponds to TennisAbstract's internal player ID (found in the player page URL). After adding, re-run the Tier 1 scrapers.

---

## Dependencies

| Package            | Purpose |
|--------------------|---------|
| `pandas`           | DataFrame operations, CSV I/O, data aggregation |
| `numpy`            | Array operations, mathematical functions (exp, tanh, clip) |
| `scikit-learn`     | PCA, KMeans clustering, StandardScaler, RandomForest |
| `selenium`         | Browser automation for web scraping |
| `beautifulsoup4`   | HTML parsing for table extraction |
| `webdriver-manager`| Automatic ChromeDriver management |
| `matplotlib`       | Static plots (bar, scatter, histogram, heatmap) |
| `seaborn`          | Enhanced heatmap visualizations |
| `streamlit`        | Interactive web dashboard framework |

---

## Data Sources

All raw data is scraped from **[TennisAbstract](https://www.tennisabstract.com)**, a comprehensive tennis statistics website created by Jeff Sackmann. The data includes match-level statistics for WTA tour matches.

---

## Tracked Players

The current player registry (`data/raw/wta_players.csv`) includes 17 top WTA players:

| Player | ID |
|--------|-----|
| Iga Swiatek | 216347 |
| Aryna Sabalenka | 214544 |
| Coco Gauff | 221103 |
| Jessica Pegula | 202468 |
| Mirra Andreeva | 259799 |
| Jasmine Paolini | 211148 |
| Madison Keys | 201619 |
| Qinwen Zheng | 221012 |
| Elena Rybakina | 214981 |
| Amanda Anisimova | 216153 |
| Emma Navarro | 215613 |
| Karolina Muchova | 214096 |
| Elina Svitolina | 202494 |
| Ekaterina Alexandrova | 206420 |
| Clara Tauson | 220704 |
| Paula Badosa | 211651 |
| Tatjana Maria | 213583 |
