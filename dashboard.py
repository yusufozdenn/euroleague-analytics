"""
Euroleague Analytics - Streamlit Dashboard
===============================================
Team-first interactive dashboard. Select a team, then explore rosters,
player performance, league standings, feature analysis, model results,
and forecasts.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"
PLOTLY_TEMPLATE = "plotly_dark"

st.set_page_config(
    page_title="Euroleague Analytics",
    page_icon="\U0001F3C0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d3d5c;
    }
    .metric-card h3 {
        color: #a0a0c0;
        font-size: 0.85rem;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card h1 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: #80ffaa;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .metric-card p.down {
        color: #ff8080;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data Loading — cache-aware, pipeline-aware
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading boxscore data...")
def load_boxscore() -> pd.DataFrame:
    """Load the full-season boxscore (all players, all games).
    Tries cache parquets first, then EuroleaguePipeline API, then static CSV.
    """
    # 1) Try cache parquets — look for the one with ~9000+ rows (boxscore)
    if CACHE_DIR.exists():
        for p in sorted(CACHE_DIR.glob("*.parquet"), key=lambda x: x.stat().st_size, reverse=True):
            try:
                df = pd.read_parquet(p)
                if len(df) > 500 and "Player" in df.columns and "Team" in df.columns:
                    return df
            except Exception:
                continue

    # 2) Try API via pipeline
    try:
        from data_pipeline import EuroleaguePipeline, HAS_API
        if HAS_API:
            pipe = EuroleaguePipeline(season=_get_season_int())
            df = pipe.get_player_boxscore_season()
            if not df.empty:
                return df
    except Exception:
        pass

    return pd.DataFrame()


@st.cache_data(show_spinner="Loading standings...")
def load_standings() -> pd.DataFrame:
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                if "club.name" in df.columns and "gamesWon" in df.columns:
                    return df
            except Exception:
                continue
    try:
        from data_pipeline import EuroleaguePipeline, HAS_API
        if HAS_API:
            pipe = EuroleaguePipeline(season=_get_season_int())
            return pipe.get_standings()
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading team stats...")
def load_team_stats() -> pd.DataFrame:
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                if "team.name" in df.columns and "pointsScored" in df.columns:
                    return df
            except Exception:
                continue
    try:
        from data_pipeline import EuroleaguePipeline, HAS_API
        if HAS_API:
            pipe = EuroleaguePipeline(season=_get_season_int())
            return pipe.get_team_season_stats()
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading leaders...")
def load_leaders() -> pd.DataFrame:
    if CACHE_DIR.exists():
        for p in CACHE_DIR.glob("*.parquet"):
            try:
                df = pd.read_parquet(p)
                if "playerName" in df.columns and "rank" in df.columns:
                    return df
            except Exception:
                continue
    try:
        from data_pipeline import EuroleaguePipeline, HAS_API
        if HAS_API:
            pipe = EuroleaguePipeline(season=_get_season_int())
            return pipe.get_player_leaders(stat_category="Score", top_n=30)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    path = OUTPUTS_DIR / "config.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    path1 = OUTPUTS_DIR / "model_comparison.csv"
    path2 = OUTPUTS_DIR / "model_comparison_results.csv"
    df1 = pd.read_csv(path1) if path1.exists() else pd.DataFrame()
    df2 = pd.read_csv(path2) if path2.exists() else pd.DataFrame()
    if df1.empty and df2.empty:
        return pd.DataFrame()
    if df2.empty:
        return df1
    if df1.empty:
        return df2
    return df2 if len(df2) >= len(df1) else df1


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    path = OUTPUTS_DIR / "feature_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_scenarios() -> pd.DataFrame:
    path = OUTPUTS_DIR / "scenarios.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_enhanced_csv() -> pd.DataFrame:
    path = OUTPUTS_DIR / "player_enhanced.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    import joblib
    # Prefer root outputs (consistent with feature_importance.csv)
    for candidate in [
        OUTPUTS_DIR / "best_model.joblib",
        OUTPUTS_DIR / "models" / "best_model.joblib",
    ]:
        if candidate.exists():
            return joblib.load(candidate)
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    import joblib
    for candidate in [
        OUTPUTS_DIR / "scaler.joblib",
        OUTPUTS_DIR / "models" / "scaler.joblib",
    ]:
        if candidate.exists():
            return joblib.load(candidate)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_season_int() -> int:
    cfg = load_config()
    return int(cfg.get("season", "2025"))


def _parse_minutes(mp_str) -> float:
    try:
        parts = str(mp_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60
    except Exception:
        return np.nan


def _metric_card(title: str, value: str, sub: str = "", down: bool = False):
    cls = "down" if down else ""
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h1>{value}</h1>
        <p class="{cls}">{sub}</p>
    </div>
    """


@st.cache_data(show_spinner=False)
def engineer_player_df(player_games: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering on a player's boxscore games via the pipeline."""
    try:
        from data_pipeline import EuroleaguePipeline
        return EuroleaguePipeline.engineer_player_features(player_games)
    except Exception:
        return player_games


def get_team_name_map(standings_df: pd.DataFrame) -> dict:
    """Map team codes to full names from standings."""
    if standings_df.empty:
        return {}
    if "club.code" in standings_df.columns and "club.name" in standings_df.columns:
        return dict(zip(standings_df["club.code"], standings_df["club.name"]))
    return {}


# ---------------------------------------------------------------------------
# Sidebar — Season / Team / Player selectors
# ---------------------------------------------------------------------------
cfg = load_config()
season_int = _get_season_int()

st.sidebar.title("\U0001F3C0 Euroleague Analytics")
st.sidebar.caption(f"made by Yusuf Özden")
st.sidebar.caption(f"Season {season_int}-{season_int + 1}")

# Load core data
boxscore_df = load_boxscore()
standings_df = load_standings()

if boxscore_df.empty:
    st.sidebar.warning("No boxscore data available. Check cache/ or euroleague_api.")
    team_names = []
    team_code_map = {}
else:
    # Build team list sorted alphabetically by full name
    team_code_map = get_team_name_map(standings_df)
    raw_teams = sorted(boxscore_df["Team"].unique())
    if team_code_map:
        team_names = sorted(raw_teams, key=lambda c: team_code_map.get(c, c))
    else:
        team_names = raw_teams

# Team selector
def _team_label(code):
    name = team_code_map.get(code, "")
    return f"{code} — {name}" if name else code

selected_team = st.sidebar.selectbox(
    "Select Team",
    team_names,
    format_func=_team_label,
    index=0 if team_names else None,
)

# Filter boxscore to selected team
if selected_team and not boxscore_df.empty:
    team_box = boxscore_df[boxscore_df["Team"] == selected_team].copy()
    # Player list for this team
    team_players = sorted(team_box["Player"].unique())
    # Remove DNP-only entries
    active_players = []
    for p in team_players:
        p_games = team_box[team_box["Player"] == p]
        if (p_games["Minutes"] != "DNP").any() and (p_games["IsPlaying"] != 0).any():
            active_players.append(p)
    if not active_players:
        active_players = team_players

    selected_player = st.sidebar.selectbox("Select Player", active_players)
else:
    team_box = pd.DataFrame()
    selected_player = None

st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Team Overview",
        "Player Performance",
        "League Standings",
        "Feature Analysis",
        "Model Results",
        "Predictions & Forecast",
    ],
    index=0,
)

team_display = team_code_map.get(selected_team, selected_team) if selected_team else "—"
st.sidebar.markdown("---")
st.sidebar.info(f"**Team**: {team_display}\n\n**Player**: {selected_player or '—'}")


# ===================================================================
# PAGE: Team Overview
# ===================================================================
def page_team_overview():
    st.header(f"Team Overview — {team_display}")

    if team_box.empty:
        st.info("No data available for the selected team.")
        return

    # --- Team aggregate stats from team_stats cache ---
    ts_df = load_team_stats()
    if not ts_df.empty and selected_team:
        row = ts_df[ts_df["team.code"] == selected_team]
        if not row.empty:
            r = row.iloc[0]
            cols = st.columns(6)
            kpis = [
                ("PPG", f"{r.get('pointsScored', 0):.1f}", ""),
                ("2P%", str(r.get("twoPointersPercentage", "—")), ""),
                ("3P%", str(r.get("threePointersPercentage", "—")), ""),
                ("FT%", str(r.get("freeThrowsPercentage", "—")), ""),
                ("Rebounds", f"{r.get('totalRebounds', 0):.1f}", ""),
                ("PIR", f"{r.get('pir', 0):.1f}", ""),
            ]
            for col, (title, value, sub) in zip(cols, kpis):
                col.markdown(_metric_card(title, value, sub), unsafe_allow_html=True)
            st.markdown("---")

    # --- Standings row for this team ---
    if not standings_df.empty and selected_team:
        row = standings_df[standings_df["club.code"] == selected_team]
        if not row.empty:
            r = row.iloc[0]
            cols = st.columns(5)
            record_kpis = [
                ("Record", f"{r.get('gamesWon', 0)}-{r.get('gamesLost', 0)}", f"#{r.get('position', '?')}"),
                ("Win%", str(r.get("winPercentage", "—")), ""),
                ("Home", str(r.get("homeRecord", "—")), ""),
                ("Away", str(r.get("awayRecord", "—")), ""),
                ("Point Diff", str(r.get("pointsDifference", "—")), ""),
            ]
            for col, (title, value, sub) in zip(cols, record_kpis):
                col.markdown(_metric_card(title, value, sub), unsafe_allow_html=True)
            st.markdown("---")

    # --- Roster table: per-game averages ---
    st.subheader("Roster — Per-Game Averages")

    played = team_box[team_box["Minutes"] != "DNP"].copy()
    if played.empty:
        st.info("No games with playing time found for this team.")
        return

    numeric_cols = ["Points", "TotalRebounds", "Assistances", "Steals",
                    "Turnovers", "BlocksFavour", "Valuation", "Plusminus"]
    for c in numeric_cols:
        if c in played.columns:
            played[c] = pd.to_numeric(played[c], errors="coerce")

    roster = played.groupby("Player").agg(
        GP=("Round", "nunique"),
        PPG=("Points", "mean"),
        RPG=("TotalRebounds", "mean"),
        APG=("Assistances", "mean"),
        SPG=("Steals", "mean"),
        TPG=("Turnovers", "mean"),
        PIR=("Valuation", "mean"),
        PlusMinus=("Plusminus", "mean"),
    ).round(1).sort_values("PPG", ascending=False).reset_index()

    st.dataframe(roster, use_container_width=True, hide_index=True)

    # --- Team scoring per round ---
    st.subheader("Team Points Per Round")
    round_pts = played.groupby("Round")["Points"].sum().reset_index()
    round_pts = round_pts.sort_values("Round")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=round_pts["Round"], y=round_pts["Points"],
        marker_color="#636EFA", name="Points",
    ))
    avg_pts = round_pts["Points"].mean()
    fig.add_hline(y=avg_pts, line_dash="dash", line_color="white",
                  annotation_text=f"Avg: {avg_pts:.0f}")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Round", yaxis_title="Team Points",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Top scorers bar chart ---
    st.subheader("Top Scorers (Total Points)")
    top = roster.nlargest(10, "PPG")
    fig_top = px.bar(
        top, x="PPG", y="Player", orientation="h",
        template=PLOTLY_TEMPLATE, color="PPG",
        color_continuous_scale="Viridis",
    )
    fig_top.update_layout(yaxis=dict(autorange="reversed"), height=380,
                          xaxis_title="Points Per Game", yaxis_title="")
    st.plotly_chart(fig_top, use_container_width=True)


# ===================================================================
# PAGE: Player Performance
# ===================================================================
def page_player_performance():
    st.header(f"Player Performance — {selected_player}")

    if not selected_player or team_box.empty:
        st.info("Select a team and player from the sidebar.")
        return

    # Get this player's games
    player_games = team_box[team_box["Player"] == selected_player].copy()
    player_games = player_games[player_games["Minutes"] != "DNP"].copy()

    if player_games.empty:
        st.info(f"No games with playing time found for {selected_player}.")
        return

    # Sort by round
    player_games = player_games.sort_values("Round").reset_index(drop=True)

    # Engineer features via pipeline
    df = engineer_player_df(player_games)

    # Ensure key numeric columns
    for c in ["Points", "TotalRebounds", "Assistances", "Steals", "Turnovers",
              "BlocksFavour", "Valuation", "Plusminus", "FreeThrowsMade",
              "FreeThrowsAttempted", "FieldGoalsMade2", "FieldGoalsAttempted2",
              "FieldGoalsMade3", "FieldGoalsAttempted3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n_games = len(df)
    avg_pts = df["Points"].mean() if "Points" in df.columns else None
    avg_reb = df["TotalRebounds"].mean() if "TotalRebounds" in df.columns else None
    avg_ast = df["Assistances"].mean() if "Assistances" in df.columns else None
    avg_pir = df["Valuation"].mean() if "Valuation" in df.columns else None

    # GmSc (engineered or compute)
    if "GmSc" in df.columns:
        avg_gmsc = df["GmSc"].mean()
    else:
        avg_gmsc = None

    # FG%
    if "FG_pct" in df.columns:
        avg_fg = df["FG_pct"].mean()
    elif "FieldGoalsMade2" in df.columns and "FieldGoalsAttempted2" in df.columns:
        fgm = df["FieldGoalsMade2"].fillna(0) + df.get("FieldGoalsMade3", pd.Series(0, index=df.index)).fillna(0)
        fga = df["FieldGoalsAttempted2"].fillna(0) + df.get("FieldGoalsAttempted3", pd.Series(0, index=df.index)).fillna(0)
        df["FG_pct"] = (fgm / fga).replace([np.inf, -np.inf], 0).fillna(0)
        avg_fg = df["FG_pct"].mean()
    else:
        avg_fg = None

    # TS%
    avg_ts = df["TS_pct"].mean() if "TS_pct" in df.columns else None

    # Trend
    trend_arrow = ""
    trend_text = "N/A"
    pts_col = "Points" if "Points" in df.columns else None
    if pts_col and n_games >= 6:
        last5 = df[pts_col].iloc[-5:].mean()
        prior = df[pts_col].iloc[:-5].mean()
        trend_arrow = "\u2191" if last5 >= prior else "\u2193"
        trend_text = f"Last 5: {last5:.1f} vs prior: {prior:.1f}"

    # --- KPI Cards ---
    cols = st.columns(7)
    kpis = [
        ("Games", f"{n_games}", "", False),
        ("PPG", f"{avg_pts:.1f}" if avg_pts is not None else "N/A", "", False),
        ("RPG", f"{avg_reb:.1f}" if avg_reb is not None else "N/A", "", False),
        ("APG", f"{avg_ast:.1f}" if avg_ast is not None else "N/A", "", False),
        ("FG%", f"{avg_fg:.1%}" if avg_fg is not None else "N/A", "", False),
        ("PIR", f"{avg_pir:.1f}" if avg_pir is not None else "N/A", "", False),
        ("Trend", trend_arrow or "N/A", trend_text, "\u2193" in (trend_arrow or "")),
    ]
    for col, (title, value, sub, is_down) in zip(cols, kpis):
        col.markdown(_metric_card(title, value, sub, is_down), unsafe_allow_html=True)

    st.markdown("---")

    # Game number for x-axis
    if "game_num" not in df.columns:
        df["game_num"] = range(1, len(df) + 1)

    # --- Points & PIR Trend ---
    st.subheader("Points & PIR Trend")

    overlay_options = []
    overlays_map = {}

    # Rolling means on Points
    for w in [3, 5, 7]:
        col_name = f"PTS_roll_mean_{w}"
        if col_name in df.columns:
            label = f"PTS MA-{w}"
            overlay_options.append(label)
            overlays_map[label] = col_name
        elif "Points" in df.columns:
            df[f"_pts_ma_{w}"] = df["Points"].rolling(w, min_periods=1).mean()
            label = f"PTS MA-{w}"
            overlay_options.append(label)
            overlays_map[label] = f"_pts_ma_{w}"

    # EWMA
    for span in [3, 5]:
        col_name = f"PTS_ewma_{span}"
        if col_name in df.columns:
            label = f"PTS EWMA-{span}"
            overlay_options.append(label)
            overlays_map[label] = col_name

    selected_overlays = st.multiselect(
        "Overlay Lines", overlay_options,
        default=[overlay_options[1]] if len(overlay_options) > 1 else overlay_options[:1],
    )

    fig_trend = go.Figure()

    if "Points" in df.columns:
        fig_trend.add_trace(go.Scatter(
            x=df["game_num"], y=df["Points"],
            mode="lines+markers", name="Points",
            line=dict(color="#636EFA", width=2), marker=dict(size=5),
        ))
    if "Valuation" in df.columns:
        fig_trend.add_trace(go.Scatter(
            x=df["game_num"], y=df["Valuation"],
            mode="lines+markers", name="PIR",
            line=dict(color="#EF553B", width=2, dash="dot"), marker=dict(size=4),
        ))
    if "GmSc" in df.columns:
        fig_trend.add_trace(go.Scatter(
            x=df["game_num"], y=df["GmSc"],
            mode="lines+markers", name="GmSc",
            line=dict(color="#AB63FA", width=1.5, dash="dashdot"), marker=dict(size=3),
        ))

    overlay_colors = ["#00CC96", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"]
    for i, ov in enumerate(selected_overlays):
        cn = overlays_map.get(ov)
        if cn and cn in df.columns:
            fig_trend.add_trace(go.Scatter(
                x=df["game_num"], y=df[cn], mode="lines", name=ov,
                line=dict(color=overlay_colors[i % len(overlay_colors)], width=2, dash="dash"),
            ))

    fig_trend.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Game #", yaxis_title="Value",
        height=420, legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- FG% Bar & Distribution side-by-side ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("FG% Per Game")
        fg_col = "FG_pct" if "FG_pct" in df.columns else None
        if fg_col:
            fig_fg = go.Figure(go.Bar(
                x=df["game_num"], y=df[fg_col],
                marker_color=np.where(df[fg_col] >= df[fg_col].mean(), "#00CC96", "#EF553B"),
            ))
            fig_fg.add_hline(y=df[fg_col].mean(), line_dash="dash", line_color="white",
                             annotation_text=f"Avg: {df[fg_col].mean():.1%}")
            fig_fg.update_layout(
                template=PLOTLY_TEMPLATE, xaxis_title="Game #", yaxis_title="FG%",
                yaxis_tickformat=".0%", height=380,
            )
            st.plotly_chart(fig_fg, use_container_width=True)
        else:
            st.info("FG% data not available.")

    with col_right:
        st.subheader("Points Distribution")
        if "Points" in df.columns:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Violin(
                y=df["Points"], box_visible=True, meanline_visible=True,
                name="Points", fillcolor="#636EFA", opacity=0.7, line_color="white",
            ))
            fig_dist.update_layout(
                template=PLOTLY_TEMPLATE, yaxis_title="Points", height=380,
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Points data not available.")

    # --- Game log table ---
    with st.expander("Full Game Log"):
        display_cols = [c for c in ["Round", "Minutes", "Points", "TotalRebounds",
                                     "Assistances", "Steals", "Turnovers",
                                     "BlocksFavour", "Valuation", "Plusminus",
                                     "FG_pct", "TS_pct", "GmSc"]
                        if c in df.columns]
        st.dataframe(df[display_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: League Standings
# ===================================================================
def page_league_standings():
    st.header("League Standings")

    if standings_df.empty:
        st.info("Standings data not available. Run the notebook or ensure euroleague_api is accessible.")
        return

    # --- Standings bar chart ---
    st.subheader("Wins by Team")
    sdf = standings_df.sort_values("gamesWon", ascending=True).copy()
    colors = ["#636EFA"] * len(sdf)
    if selected_team and selected_team in sdf["club.code"].values:
        idx = sdf[sdf["club.code"] == selected_team].index
        for i in idx:
            pos = list(sdf.index).index(i)
            colors[pos] = "#00CC96"

    fig_st = go.Figure(go.Bar(
        x=sdf["gamesWon"], y=sdf["club.name"], orientation="h",
        marker_color=colors, text=sdf["gamesWon"], textposition="auto",
    ))
    fig_st.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Wins", yaxis_title="",
        height=max(450, len(sdf) * 30),
    )
    st.plotly_chart(fig_st, use_container_width=True)

    # --- Standings table ---
    st.subheader("Full Standings")
    table_cols = [c for c in ["position", "club.name", "gamesPlayed", "gamesWon",
                               "gamesLost", "winPercentage", "pointsDifference",
                               "homeRecord", "awayRecord", "last5Form"]
                  if c in standings_df.columns]
    display = standings_df[table_cols].sort_values("position").reset_index(drop=True)
    rename = {
        "position": "#", "club.name": "Team", "gamesPlayed": "GP",
        "gamesWon": "W", "gamesLost": "L", "winPercentage": "Win%",
        "pointsDifference": "+/-", "homeRecord": "Home", "awayRecord": "Away",
        "last5Form": "Last 5",
    }
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})
    st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Team stats comparison ---
    ts_df = load_team_stats()
    if not ts_df.empty:
        st.subheader("Team Stats Comparison")
        stat_col = st.selectbox("Stat to compare", [
            "pointsScored", "totalRebounds", "assists", "steals",
            "turnovers", "pir", "threePointersPercentage",
        ], format_func=lambda x: x.replace("Percentage", "%").replace("pointers", "P").title())

        if stat_col in ts_df.columns:
            cdf = ts_df[["team.name", stat_col]].sort_values(stat_col, ascending=True)
            colors2 = ["#636EFA"] * len(cdf)
            if selected_team:
                match = ts_df[ts_df["team.code"] == selected_team]
                if not match.empty:
                    team_name = match.iloc[0]["team.name"]
                    for i, tn in enumerate(cdf["team.name"]):
                        if tn == team_name:
                            colors2[i] = "#00CC96"

            fig_comp = go.Figure(go.Bar(
                x=cdf[stat_col], y=cdf["team.name"], orientation="h",
                marker_color=colors2, text=cdf[stat_col].round(1), textposition="auto",
            ))
            fig_comp.update_layout(
                template=PLOTLY_TEMPLATE, height=max(450, len(cdf) * 30),
                xaxis_title=stat_col.title(), yaxis_title="",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # --- Scoring leaders ---
    leaders_df = load_leaders()
    if not leaders_df.empty:
        st.subheader("Scoring Leaders")
        disp_cols = [c for c in ["rank", "playerName", "clubNames", "gamesPlayed",
                                  "averagePerGame", "total"]
                     if c in leaders_df.columns]
        ld = leaders_df[disp_cols].head(20).rename(columns={
            "rank": "#", "playerName": "Player", "clubNames": "Team",
            "gamesPlayed": "GP", "averagePerGame": "PPG", "total": "Total",
        })
        st.dataframe(ld, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: Feature Analysis
# ===================================================================
def page_feature_analysis():
    st.header(f"Feature Analysis — {selected_player}")

    if not selected_player or team_box.empty:
        st.info("Select a team and player from the sidebar.")
        return

    player_games = team_box[team_box["Player"] == selected_player].copy()
    player_games = player_games[player_games["Minutes"] != "DNP"].copy()

    if len(player_games) < 5:
        st.info(f"Not enough games ({len(player_games)}) for meaningful feature analysis. Need at least 5.")
        return

    player_games = player_games.sort_values("Round").reset_index(drop=True)
    df = engineer_player_df(player_games)

    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.dropna(axis=1, how="all")
    # Drop low-variance columns
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    key_cols = [c for c in [
        "Points", "TotalRebounds", "Assistances", "Steals", "Turnovers",
        "Valuation", "Plusminus", "GmSc", "FG_pct", "TS_pct", "eFG_pct",
        "MP_decimal", "PTS_per_min", "AST_per_min",
        "GmSc_roll_mean_3", "GmSc_roll_mean_5", "GmSc_ewma_3",
        "GmSc_momentum", "season_pct", "cum_PTS_mean",
    ] if c in numeric_df.columns]

    if len(key_cols) < 3:
        key_cols = list(numeric_df.columns[:20])

    if len(key_cols) >= 3:
        corr = numeric_df[key_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r",
            template=PLOTLY_TEMPLATE, aspect="auto", height=600,
        )
        fig_corr.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Feature importance (from saved outputs if same player, or compute MI) ---
    st.subheader("Feature Importance (Mutual Information)")
    fi_df = load_feature_importance()
    if not fi_df.empty:
        st.caption("Pre-computed from notebook analysis (original player)")
        fi_sorted = fi_df.sort_values("MI", ascending=True)
        fig_mi = px.bar(
            fi_sorted, x="MI", y="Feature", orientation="h",
            template=PLOTLY_TEMPLATE, color="MI", color_continuous_scale="Viridis",
            height=max(400, len(fi_sorted) * 22),
        )
        fig_mi.update_layout(yaxis_title="", xaxis_title="Mutual Information Score")
        st.plotly_chart(fig_mi, use_container_width=True)
    else:
        st.info("Feature importance file not found. Run the notebook to generate it.")

    # --- Distributions ---
    st.subheader("Feature Distributions")
    dist_feats = [c for c in ["Points", "Valuation", "FG_pct", "Assistances", "GmSc"]
                  if c in df.columns]
    if dist_feats:
        fig_hist = make_subplots(rows=1, cols=len(dist_feats), subplot_titles=dist_feats)
        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
        for i, feat in enumerate(dist_feats):
            vals = pd.to_numeric(df[feat], errors="coerce").dropna()
            fig_hist.add_trace(
                go.Histogram(x=vals, nbinsx=12, marker_color=colors[i % len(colors)],
                             name=feat, showlegend=False),
                row=1, col=i + 1,
            )
        fig_hist.update_layout(template=PLOTLY_TEMPLATE, height=350)
        st.plotly_chart(fig_hist, use_container_width=True)


# ===================================================================
# PAGE: Model Results
# ===================================================================
def page_model_results():
    st.header("Model Comparison & Results")
    st.caption("Models trained on notebook analysis. Results shown are from the original training run.")

    mc_df = load_model_comparison()
    if mc_df.empty:
        st.info("No model comparison files found in outputs/.")
        return

    col_map = {}
    for c in mc_df.columns:
        cl = c.lower().replace(" ", "_")
        if cl == "model":
            col_map[c] = "Model"
        elif "train_rmse" in cl:
            col_map[c] = "Train_RMSE"
        elif "test_rmse" in cl:
            col_map[c] = "Test_RMSE"
        elif "test_mae" in cl:
            col_map[c] = "Test_MAE"
        elif "test_r2" in cl:
            col_map[c] = "Test_R2"
        elif "train_r2" in cl:
            col_map[c] = "Train_R2"
        elif "train_mae" in cl:
            col_map[c] = "Train_MAE"
        elif "cv_rmse" in cl:
            col_map[c] = "CV_RMSE"
    mc_df = mc_df.rename(columns=col_map)

    # --- RMSE bar chart ---
    st.subheader("Train vs Test RMSE")
    if "Train_RMSE" in mc_df.columns and "Test_RMSE" in mc_df.columns:
        mc_df["Train_RMSE"] = pd.to_numeric(mc_df["Train_RMSE"], errors="coerce")
        mc_df["Test_RMSE"] = pd.to_numeric(mc_df["Test_RMSE"], errors="coerce")

        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(
            name="Train RMSE", x=mc_df["Model"], y=mc_df["Train_RMSE"],
            marker_color="#636EFA",
        ))
        fig_rmse.add_trace(go.Bar(
            name="Test RMSE", x=mc_df["Model"], y=mc_df["Test_RMSE"],
            marker_color="#EF553B",
        ))
        fig_rmse.update_layout(
            barmode="group", template=PLOTLY_TEMPLATE,
            xaxis_title="Model", yaxis_title="RMSE", height=420,
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    # --- Metrics table ---
    st.subheader("All Metrics")
    display_cols = [c for c in ["Model", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE",
                                 "Train_MAE", "Test_MAE", "CV_RMSE"] if c in mc_df.columns]
    st.dataframe(
        mc_df[display_cols].style.format(
            {c: "{:.4f}" for c in display_cols if c not in ("Model", "CV_RMSE")},
            na_rep="-",
        ),
        use_container_width=True,
    )

    # --- Diagnostics ---
    st.subheader("Model Diagnostics")
    model = load_model()
    if model is None:
        st.info("Trained model (best_model.joblib) not found.")
        return

    df = load_enhanced_csv()
    if df.empty:
        st.info("Enhanced data not available for diagnostics.")
        return

    target_col = "GmSc"
    if target_col not in df.columns:
        st.info("GmSc column not found.")
        return

    feature_names = None
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    elif hasattr(model, "feature_name_"):
        feature_names = list(model.feature_name_())
    if feature_names is None:
        fi_df = load_feature_importance()
        if not fi_df.empty:
            feature_names = fi_df["Feature"].tolist()
    if feature_names is None:
        st.info("Cannot determine model features.")
        return

    available_features = [f for f in feature_names if f in df.columns]
    if not available_features:
        st.info("Model features not found in enhanced data.")
        return

    X = df[available_features].copy()
    y = df[target_col].copy()
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if len(X) == 0:
        st.info("No complete data rows.")
        return

    scaler = load_scaler()
    try:
        X_input = scaler.transform(X) if scaler is not None else X.values
        y_pred = model.predict(X_input)
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
        return

    residuals = y.values - y_pred
    col_a, col_b = st.columns(2)

    with col_a:
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=y.values, y=y_pred, mode="markers",
            marker=dict(color="#636EFA", size=8, opacity=0.7), name="Predictions",
        ))
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        fig_avp.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="white", dash="dash"), name="Perfect Fit",
        ))
        fig_avp.update_layout(
            template=PLOTLY_TEMPLATE, title="Actual vs Predicted",
            xaxis_title="Actual", yaxis_title="Predicted", height=400,
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with col_b:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers",
            marker=dict(color="#EF553B", size=8, opacity=0.7), name="Residuals",
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="white")
        fig_res.update_layout(
            template=PLOTLY_TEMPLATE, title="Residuals",
            xaxis_title="Predicted", yaxis_title="Residual", height=400,
        )
        st.plotly_chart(fig_res, use_container_width=True)


# ===================================================================
# PAGE: Predictions & Forecast
# ===================================================================
def page_predictions():
    st.header(f"Predictions & Forecast — {selected_player}")

    # --- Scenario Analysis (from saved outputs) ---
    st.subheader("Scenario Analysis (Notebook Results)")
    scenarios_df = load_scenarios()
    if not scenarios_df.empty:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(
            x=scenarios_df["Scenario"], y=scenarios_df["Pred_GmSc"],
            error_y=dict(
                type="data", symmetric=False,
                array=scenarios_df["CI_high"] - scenarios_df["Pred_GmSc"],
                arrayminus=scenarios_df["Pred_GmSc"] - scenarios_df["CI_low"],
            ),
            marker_color="#636EFA", name="Predicted GmSc",
        ))
        fig_sc.update_layout(
            template=PLOTLY_TEMPLATE, yaxis_title="Predicted Game Score",
            xaxis_title="Scenario", height=420,
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        with st.expander("Scenario Details"):
            st.dataframe(scenarios_df, use_container_width=True)
    else:
        st.info("Scenarios file not found.")

    st.markdown("---")

    # --- Interactive Prediction ---
    st.subheader("Interactive Prediction")
    model = load_model()
    scaler = load_scaler()

    if model is None:
        st.info("Trained model not available.")
    else:
        # Resolve feature names: model attrs → scaler attrs → feature_importance.csv
        feature_names = None
        for src in [model, scaler]:
            if src is not None and hasattr(src, "feature_names_in_"):
                feature_names = list(src.feature_names_in_)
                break
        if feature_names is None:
            fi_df = load_feature_importance()
            if not fi_df.empty:
                feature_names = fi_df["Feature"].tolist()
        # Ensure feature count matches scaler expectation
        n_expected = getattr(scaler, "n_features_in_", None) if scaler else None
        if feature_names and n_expected and len(feature_names) != n_expected:
            # Feature list doesn't match scaler — try loading from enhanced CSV columns
            df_enh_cols = load_enhanced_csv()
            if not df_enh_cols.empty:
                # Use all numeric columns except target as candidates
                exclude = {"GmSc", "game_num", "MP", "high_scoring", "efficient_game"}
                candidates = [c for c in df_enh_cols.select_dtypes(include=[np.number]).columns
                              if c not in exclude]
                if len(candidates) >= n_expected:
                    feature_names = candidates[:n_expected]

        col1, col2 = st.columns(2)
        with col1:
            fg_pct = st.slider("FG%", 0.0, 1.0, 0.45, 0.01)
            pts_per_min = st.slider("PTS per Minute", 0.0, 2.0, 0.40, 0.01)
        with col2:
            ast_val = st.slider("Assists", 0, 10, 2, 1)
            mp_decimal = st.slider("Minutes Played", 10.0, 45.0, 30.0, 0.5)

        if st.button("Predict Game Score", type="primary"):
            df_enh = load_enhanced_csv()
            if feature_names and not df_enh.empty:
                mean_vals = {}
                for f in feature_names:
                    if f in df_enh.columns:
                        mean_vals[f] = pd.to_numeric(df_enh[f], errors="coerce").mean()
                    else:
                        mean_vals[f] = 0.0
                overrides = {
                    "FG%": fg_pct, "FG_pct": fg_pct,
                    "PTS_per_min": pts_per_min, "AST": float(ast_val),
                    "MP_decimal": mp_decimal, "PTS": pts_per_min * mp_decimal,
                }
                for key, val in overrides.items():
                    if key in mean_vals:
                        mean_vals[key] = val

                X_input = pd.DataFrame([mean_vals])[feature_names]
                try:
                    X_arr = scaler.transform(X_input) if scaler is not None else X_input.values
                    pred = model.predict(X_arr)[0]
                    st.success(f"Predicted Game Score: **{pred:.2f}**")
                    st.markdown(f"""
| Input | Value |
|-------|-------|
| FG% | {fg_pct:.0%} |
| PTS/min | {pts_per_min:.2f} |
| AST | {ast_val} |
| Minutes | {mp_decimal:.1f} |
| Est. PTS | {pts_per_min * mp_decimal:.1f} |
""")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.warning("Cannot determine model features or enhanced data missing.")

    st.markdown("---")

    # --- ARIMA Forecast on selected player ---
    st.subheader("Points Time Series Forecast")

    if not selected_player or team_box.empty:
        st.info("Select a player to generate forecast.")
        return

    player_games = team_box[team_box["Player"] == selected_player].copy()
    player_games = player_games[player_games["Minutes"] != "DNP"].copy()
    player_games = player_games.sort_values("Round").reset_index(drop=True)

    pts_col = "Points"
    if pts_col not in player_games.columns:
        st.info("Points data not available.")
        return

    pts_series = pd.to_numeric(player_games[pts_col], errors="coerce").dropna().reset_index(drop=True)

    if len(pts_series) < 10:
        st.info(f"Not enough games ({len(pts_series)}) for ARIMA forecast. Need at least 10.")
        return

    try:
        from statsmodels.tsa.arima.model import ARIMA

        forecast_steps = st.slider("Forecast games ahead", 1, 15, 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_model = ARIMA(pts_series, order=(2, 1, 2))
            arima_fit = arima_model.fit()

        forecast_result = arima_fit.get_forecast(steps=forecast_steps)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.10)

        hist_x = list(range(1, len(pts_series) + 1))
        fore_x_full = [hist_x[-1]] + list(range(len(pts_series) + 1,
                                                  len(pts_series) + 1 + forecast_steps))
        fore_x_only = fore_x_full[1:]

        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        mean_vals = forecast_mean.values

        # Monte Carlo simulation paths using ARIMA residuals
        residuals = arima_fit.resid.values
        n_sim = 30
        np.random.seed(42)
        sim_paths = []
        for _ in range(n_sim):
            path = [pts_series.values[-1]]
            for s in range(forecast_steps):
                noise = np.random.choice(residuals)
                path.append(max(mean_vals[s] + noise, 0))
            sim_paths.append(path)

        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=hist_x, y=pts_series.values,
            mode="lines+markers", name="Historical",
            line=dict(color="#636EFA", width=2), marker=dict(size=4),
        ))

        # Simulation paths (light, transparent)
        for i, path in enumerate(sim_paths):
            fig_fc.add_trace(go.Scatter(
                x=fore_x_full, y=path,
                mode="lines", showlegend=(i == 0),
                name="Simulated Paths" if i == 0 else None,
                line=dict(color="rgba(99, 110, 250, 0.12)", width=1),
                hoverinfo="skip",
            ))

        # CI band
        fig_fc.add_trace(go.Scatter(
            x=fore_x_only + fore_x_only[::-1],
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor="rgba(0, 204, 150, 0.12)",
            line=dict(color="rgba(0, 204, 150, 0)"), name="90% CI",
        ))

        # Mean forecast line (connected from last historical)
        fig_fc.add_trace(go.Scatter(
            x=fore_x_full, y=[pts_series.values[-1]] + list(mean_vals),
            mode="lines+markers", name="Forecast (mean)",
            line=dict(color="#00CC96", width=2.5, dash="dash"),
            marker=dict(size=6, symbol="diamond"),
        ))

        fig_fc.update_layout(
            template=PLOTLY_TEMPLATE, xaxis_title="Game #", yaxis_title="Points",
            height=450, legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        with st.expander("Forecast Values"):
            fc_table = pd.DataFrame({
                "Game #": fore_x_only,
                "Forecast PTS": mean_vals.round(2),
                "CI Low (90%)": lower.round(2),
                "CI High (90%)": upper.round(2),
            })
            st.dataframe(fc_table, use_container_width=True)

    except ImportError:
        st.info("Install statsmodels to enable ARIMA: `pip install statsmodels`")
    except Exception as e:
        st.warning(f"ARIMA forecast failed: {e}")


# ===================================================================
# Page Router
# ===================================================================
PAGES = {
    "Team Overview": page_team_overview,
    "Player Performance": page_player_performance,
    "League Standings": page_league_standings,
    "Feature Analysis": page_feature_analysis,
    "Model Results": page_model_results,
    "Predictions & Forecast": page_predictions,
}

PAGES[page]()

st.sidebar.markdown("---")
st.sidebar.caption("Euroleague Analytics v1.0")
st.sidebar.caption(f"Data: {season_int}-{season_int + 1} Season")
