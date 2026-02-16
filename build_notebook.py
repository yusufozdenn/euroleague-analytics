#!/usr/bin/env python3
"""
Build Script: euroleague_analytics_pro.ipynb
=============================================
Generates a production-quality Jupyter notebook for Euroleague basketball analytics.
Uses the EuroleaguePipeline from data_pipeline.py for data ingestion and feature engineering.

Usage:
    python build_notebook.py
"""

import json
import os

NOTEBOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "euroleague_analytics_pro.ipynb")


# ──────────────────────────────────────────────────────────────
# Cell Helpers
# ──────────────────────────────────────────────────────────────

def md(source: str) -> dict:
    """Create a markdown cell dict."""
    lines = source.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(source: str) -> dict:
    """Create a code cell dict."""
    lines = source.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


# ──────────────────────────────────────────────────────────────
# Notebook Cells
# ──────────────────────────────────────────────────────────────

cells = []

# ── 1. MARKDOWN: Title & Architecture ────────────────────────

cells.append(md(r"""
# Euroleague Analytics Pro

**Advanced Player Performance Modeling & Forecasting**

---

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    EUROLEAGUE ANALYTICS PRO                      │
 ├───────────┬──────────────┬──────────────┬───────────────────────┤
 │  DATA     │  FEATURES    │  MODELS      │  OUTPUTS              │
 │           │              │              │                       │
 │ API/CSV   │ Rolling Agg  │ Linear       │ SHAP Explanations     │
 │ Boxscore  │ EWMA         │ Tree-Based   │ Scenario Predictions  │
 │ Team Stats│ Shooting %   │ SVR / KNN    │ ARIMA Forecast        │
 │ Standings │ Per-Minute   │ Neural Net   │ Player Report         │
 │ Leaders   │ Lag / Moment │ Stacking     │ CSV / Model Export    │
 └───────────┴──────────────┴──────────────┴───────────────────────┘
```

**Pipeline:** `data_pipeline.py` → Feature Engineering → EDA → Modeling → Diagnostics → Forecast

---
"""))

# ── 2. CODE: Imports ─────────────────────────────────────────

cells.append(code(r"""
# ── Standard Library ──
import os, sys, json, warnings, time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

warnings.filterwarnings("ignore")

# ── Core Scientific Stack ──
import numpy as np
import pandas as pd

# ── Visualization ──
import matplotlib
matplotlib.use("Agg")  # safe backend, overridden in notebook
import matplotlib.pyplot as plt
import seaborn as sns

# ── Scikit-learn ──
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from scipy import stats

# ── Optional: XGBoost ──
HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    pass

# ── Optional: LightGBM ──
HAS_LGBM = False
try:
    import lightgbm as lgbm
    HAS_LGBM = True
except Exception:
    pass

# ── Optional: TensorFlow / Keras ──
HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers, callbacks
    HAS_TF = True
except Exception:
    pass

# ── Optional: SHAP ──
HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except Exception:
    pass

# ── Optional: Statsmodels ──
HAS_SM = False
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_SM = True
except Exception:
    pass

# ── Optional: Plotly ──
HAS_PLOTLY = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    pass

# ── Joblib ──
import joblib

# ── Pipeline ──
from data_pipeline import EuroleaguePipeline, HAS_API

# ── IPython Display ──
from IPython.display import display, HTML

# ── Version Summary ──
print("=" * 60)
print("LIBRARY AVAILABILITY")
print("=" * 60)
_libs = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": __import__("sklearn").__version__,
    "scipy": stats.scipy.__version__ if hasattr(stats, "scipy") else __import__("scipy").__version__,
    "matplotlib": matplotlib.__version__,
    "seaborn": sns.__version__,
}
if HAS_XGB:
    _libs["xgboost"] = xgb.__version__
if HAS_LGBM:
    _libs["lightgbm"] = lgbm.__version__
if HAS_TF:
    _libs["tensorflow"] = tf.__version__
if HAS_SHAP:
    _libs["shap"] = shap.__version__
if HAS_SM:
    _libs["statsmodels"] = sm.__version__ if hasattr(sm, "__version__") else "available"
if HAS_PLOTLY:
    _libs["plotly"] = __import__("plotly").__version__

for lib, ver in _libs.items():
    print(f"  {lib:15s} : {ver}")
print(f"\n  euroleague_api  : {'AVAILABLE' if HAS_API else 'NOT INSTALLED'}")
print(f"  XGBoost         : {'YES' if HAS_XGB else 'NO'}")
print(f"  LightGBM        : {'YES' if HAS_LGBM else 'NO'}")
print(f"  TensorFlow      : {'YES' if HAS_TF else 'NO'}")
print(f"  SHAP            : {'YES' if HAS_SHAP else 'NO'}")
print(f"  Statsmodels     : {'YES' if HAS_SM else 'NO'}")
print(f"  Plotly          : {'YES' if HAS_PLOTLY else 'NO'}")
print("=" * 60)

# ── Plotting defaults ──
PALETTE = sns.color_palette("husl", 12)
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({"figure.figsize": (12, 6), "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight"})
"""))

# ── 3. CODE: Interactive Configuration ───────────────────────

cells.append(code(r"""
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Season dropdown (2020-2025)
season_dd = widgets.Dropdown(
    options=[(f"{y}-{y+1}", y) for y in range(2020, 2026)],
    value=2024,
    description="Season:",
)

# Player name input
player_input = widgets.Text(
    value="Hayes",
    description="Player:",
    placeholder="e.g. Hayes, Tavares, Vezenkov",
)

# Test size slider
test_slider = widgets.IntSlider(
    value=7, min=3, max=15,
    description="Test Games:",
)

# Target variable dropdown
target_dd = widgets.Dropdown(
    options=["GmSc", "Valuation", "Points"],
    value="GmSc",
    description="Target:",
)

# Display config widgets
config_box = widgets.VBox([
    widgets.HTML("<h3>Configuration</h3>"),
    widgets.HBox([season_dd, player_input]),
    widgets.HBox([test_slider, target_dd]),
])
display(config_box)
"""))

# ── 4. CODE: Build CONFIG ────────────────────────────────────

cells.append(code(r"""
CONFIG = {
    "season": season_dd.value,
    "player_name": player_input.value,
    "target_preference": target_dd.value,
    "static_csv": "nhd_euroleague.csv",
    "output_dir": "outputs",
    "seed": 42,
    "test_size": test_slider.value,
}

SEED = CONFIG["seed"]
np.random.seed(SEED)

# Create output directories
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "models"), exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "plots"), exist_ok=True)

print("Configuration")
print("=" * 50)
for k, v in CONFIG.items():
    print(f"  {k:20s}: {v}")
print("=" * 50)
"""))

# ── 5. CODE: Data Ingestion ──────────────────────────────────

cells.append(code(r"""
# ── Initialize Pipeline ──
pipe = EuroleaguePipeline(season=CONFIG["season"])
pipe.info()

# ── Fetch Boxscore ──
boxscore_full = pd.DataFrame()

if HAS_API:
    print("\nFetching boxscore data from Euroleague API...")
    try:
        boxscore_full = pipe.get_player_boxscore_season()
        print(f"Boxscore fetched: {boxscore_full.shape}")
    except Exception as e:
        print(f"API fetch failed: {e}")

if boxscore_full.empty:
    print("\nFalling back to static CSV...")
    boxscore_full = pipe.load_static_csv(CONFIG["static_csv"])
    if not boxscore_full.empty:
        print(f"Static CSV loaded: {boxscore_full.shape}")
        print(f"Columns: {list(boxscore_full.columns)}")
    else:
        print("ERROR: No data available. Please provide nhd_euroleague.csv or install euroleague_api.")

print(f"\nDataset shape: {boxscore_full.shape}")
display(boxscore_full.head())
"""))

# ── 6. CODE: Player Selection (Interactive) ──────────────────

cells.append(code(r"""
# ── Player Selection ──
from IPython.display import display
import ipywidgets as widgets

player_dd = None  # define default

if not boxscore_full.empty:
    # Find player name column
    name_col = None
    for c in boxscore_full.columns:
        if "player" in c.lower() and "id" not in c.lower():
            name_col = c
            break

    if name_col is not None:
        all_players = sorted(boxscore_full[name_col].unique())
        matching = [p for p in all_players if CONFIG["player_name"].lower() in p.lower()]

        player_dd = widgets.Dropdown(
            options=matching if matching else all_players[:50],
            description="Select Player:",
        )

        search_input = widgets.Text(description="Search:", placeholder="Type to filter...")

        def on_search(change):
            filtered = [p for p in all_players if change["new"].lower() in p.lower()]
            player_dd.options = filtered if filtered else ["No match"]

        search_input.observe(on_search, names="value")
        display(widgets.VBox([
            widgets.HTML("<h3>Player Selection</h3>"),
            search_input,
            player_dd,
        ]))
        print(f"\nFound {len(matching)} matching players for '{CONFIG['player_name']}':")
        for p in matching:
            print(f"  - {p}")
    else:
        # Static CSV without player name column: treat entire dataset as the player
        print("No player name column found -- using entire dataset as single-player data.")
else:
    print("No data loaded.")
"""))

# ── 7. CODE: Extract Selected Player ─────────────────────────

cells.append(code(r"""
# ── Extract Selected Player Data ──
selected_player = CONFIG["player_name"]
if player_dd is not None and hasattr(player_dd, "value") and player_dd.value != "No match":
    selected_player = player_dd.value

print(f"Selected player: {selected_player}")

# Check if we have a name column (API data) or static CSV
name_col = None
for c in boxscore_full.columns:
    if "player" in c.lower() and "id" not in c.lower():
        name_col = c
        break

if name_col is not None:
    # API data: filter by player
    player_raw = pipe.get_player_game_stats(selected_player, boxscore_full)
else:
    # Static CSV: entire dataset IS the player
    player_raw = boxscore_full.copy()

if player_raw.empty:
    print(f"WARNING: No data found for '{selected_player}'. Using static CSV fallback.")
    player_raw = pipe.load_static_csv(CONFIG["static_csv"])

print(f"\nPlayer data shape: {player_raw.shape}")
display(player_raw.head(10))
"""))

# ── 8. CODE: Contextual Data ─────────────────────────────────

cells.append(code(r"""
# ── Fetch Contextual Data ──
team_stats = pd.DataFrame()
standings = pd.DataFrame()
leaders = pd.DataFrame()

if HAS_API:
    try:
        team_stats = pipe.get_team_season_stats()
        print(f"Team stats: {team_stats.shape}")
    except Exception as e:
        print(f"Team stats fetch failed: {e}")

    try:
        standings = pipe.get_standings()
        print(f"Standings: {standings.shape}")
    except Exception as e:
        print(f"Standings fetch failed: {e}")

    try:
        leaders = pipe.get_player_leaders()
        print(f"Leaders: {leaders.shape}")
    except Exception as e:
        print(f"Leaders fetch failed: {e}")
else:
    print("API not available -- contextual data skipped.")

print("\nContextual data summary:")
print(f"  Team stats : {team_stats.shape if not team_stats.empty else 'N/A'}")
print(f"  Standings  : {standings.shape if not standings.empty else 'N/A'}")
print(f"  Leaders    : {leaders.shape if not leaders.empty else 'N/A'}")
"""))

# ── 9. MARKDOWN: Data Profiling ──────────────────────────────

cells.append(md(r"""
---
## Data Profiling
"""))

# ── 10. CODE: Data Profiling ─────────────────────────────────

cells.append(code(r"""
print("DATA PROFILING")
print("=" * 60)
print(f"\nShape: {player_raw.shape[0]} rows x {player_raw.shape[1]} columns")
print(f"\nColumn dtypes:")
display(player_raw.dtypes.to_frame("dtype"))

print(f"\nMissing values:")
missing = player_raw.isnull().sum()
missing_pct = (missing / len(player_raw) * 100).round(2)
miss_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
miss_df = miss_df[miss_df["missing"] > 0]
if miss_df.empty:
    print("  No missing values.")
else:
    display(miss_df)

print(f"\nDuplicate rows: {player_raw.duplicated().sum()}")

print(f"\nDescriptive Statistics:")
display(player_raw.describe().round(3))
"""))

# ── 11. CODE: Normality Tests ────────────────────────────────

cells.append(code(r"""
# ── Normality Tests (Shapiro-Wilk) ──
print("NORMALITY TESTS (Shapiro-Wilk)")
print("=" * 60)

numeric_cols = player_raw.select_dtypes(include=[np.number]).columns.tolist()
norm_results = []

for col in numeric_cols:
    data = player_raw[col].dropna()
    if len(data) >= 8 and data.std() > 0:
        stat, p = stats.shapiro(data)
        norm_results.append({
            "Feature": col,
            "W-statistic": round(stat, 4),
            "p-value": round(p, 4),
            "Normal (p>0.05)": "Yes" if p > 0.05 else "No",
        })

if norm_results:
    norm_df = pd.DataFrame(norm_results)
    display(norm_df)
else:
    print("Not enough data for normality tests.")
"""))

# ── 12. MARKDOWN: Feature Engineering ────────────────────────

cells.append(md(r"""
---
## Feature Engineering
"""))

# ── 13. CODE: Apply Feature Engineering ──────────────────────

cells.append(code(r"""
# ── Apply Feature Engineering ──
old_cols = set(player_raw.columns)

# Detect FG% column for static CSV
fg_pct_col = None
for c in player_raw.columns:
    if c.strip() in ("FG%", "FG_pct", "fg_pct"):
        fg_pct_col = c.strip()
        break

player_df = EuroleaguePipeline.engineer_player_features(player_raw, fg_pct_col=fg_pct_col)
new_cols = set(player_df.columns) - old_cols

print(f"Feature engineering complete.")
print(f"  Original columns : {len(old_cols)}")
print(f"  New columns      : {len(new_cols)}")
print(f"  Total columns    : {len(player_df.columns)}")
print(f"\nNew features:")
for c in sorted(new_cols):
    print(f"  + {c}")

display(player_df.head())
"""))

# ── 14. CODE: Prepare Modeling Dataset ────────────────────────

cells.append(code(r"""
# ── Prepare Modeling Dataset ──

# Determine target column
TARGET = None
preference_order = [CONFIG["target_preference"], "GmSc", "Valuation", "Points", "PTS"]
for candidate in preference_order:
    for col in player_df.columns:
        if col.lower() == candidate.lower():
            TARGET = col
            break
    if TARGET:
        break

if TARGET is None:
    # Last resort: first numeric column
    numeric_cols = player_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        TARGET = numeric_cols[-1]

print(f"TARGET variable: {TARGET}")

# Select numeric features, exclude identifiers
exclude_patterns = ["id", "player", "team", "phase", "round", "season", "game_code"]
feature_cols = []
for c in player_df.select_dtypes(include=[np.number]).columns:
    if c == TARGET:
        continue
    if any(pat in c.lower() for pat in exclude_patterns):
        continue
    feature_cols.append(c)

# Build modeling dataframe
model_df = player_df[feature_cols + [TARGET]].dropna().reset_index(drop=True)

print(f"\nModeling dataset: {model_df.shape}")
print(f"Features ({len(feature_cols)}):")
for i, c in enumerate(feature_cols, 1):
    print(f"  {i:2d}. {c}")
print(f"\nTarget distribution:")
display(model_df[TARGET].describe().to_frame().T)
"""))

# ── 15. MARKDOWN: EDA ────────────────────────────────────────

cells.append(md(r"""
---
## Exploratory Data Analysis
"""))

# ── 16. CODE: Distribution Plots ─────────────────────────────

cells.append(code(r"""
# ── Distribution Plots ──
key_features = [c for c in [TARGET, "FG_pct", "MP_decimal", "PTS_per_min", "TS_pct",
                             "eFG_pct", "GmSc_roll_mean_5", "season_pct"]
                if c in model_df.columns][:8]

n_plots = len(key_features)
if n_plots > 0:
    n_cols_plot = min(4, n_plots)
    n_rows_plot = (n_plots + n_cols_plot - 1) // n_cols_plot
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(5 * n_cols_plot, 4 * n_rows_plot))
    axes = np.array(axes).flatten() if n_plots > 1 else [axes]

    for i, feat in enumerate(key_features):
        ax = axes[i]
        data = model_df[feat].dropna()
        ax.hist(data, bins=15, alpha=0.7, color=PALETTE[i % len(PALETTE)], edgecolor="white", density=True)
        try:
            data.plot.kde(ax=ax, color="black", linewidth=1.5)
        except Exception:
            pass
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("Density")

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "distributions.png"))
    plt.show()
else:
    print("No features available for distribution plots.")
"""))

# ── 17. CODE: Correlation Matrix ─────────────────────────────

cells.append(code(r"""
# ── Correlation Matrix with Significance ──
corr_features = [c for c in feature_cols if c in model_df.columns][:20]  # limit for readability
if len(corr_features) > 2:
    corr_data = model_df[corr_features + [TARGET]].dropna()
    corr_matrix = corr_data.corr()

    # Compute p-values
    n = len(corr_data)
    p_matrix = pd.DataFrame(np.ones((len(corr_matrix), len(corr_matrix))),
                             index=corr_matrix.index, columns=corr_matrix.columns)
    for i, ci in enumerate(corr_matrix.columns):
        for j, cj in enumerate(corr_matrix.columns):
            if i != j:
                _, p = stats.pearsonr(corr_data[ci], corr_data[cj])
                p_matrix.iloc[i, j] = p

    # Annotation with significance stars
    annot = corr_matrix.round(2).astype(str)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            p = p_matrix.iloc[i, j]
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            annot.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}{stars}"

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Matrix (*** p<0.001, ** p<0.01, * p<0.05)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "correlation_matrix.png"))
    plt.show()
else:
    print("Not enough features for correlation matrix.")
"""))

# ── 18. CODE: Performance Trend ──────────────────────────────

cells.append(code(r"""
# ── Performance Trend Over Season ──
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Panel 1: Target trend with MA/EWMA
ax = axes[0, 0]
if TARGET in player_df.columns:
    target_series = pd.to_numeric(player_df[TARGET], errors="coerce")
    x_range = range(1, len(target_series) + 1)
    ax.plot(x_range, target_series.values, "o-", alpha=0.5, label="Raw", markersize=4)
    ma5 = target_series.rolling(5, min_periods=1).mean()
    ewma5 = target_series.ewm(span=5).mean()
    ax.plot(x_range, ma5.values, "-", linewidth=2, label="MA(5)", color=PALETTE[1])
    ax.plot(x_range, ewma5.values, "--", linewidth=2, label="EWMA(5)", color=PALETTE[2])
    ax.set_title(f"{TARGET} Trend")
    ax.set_xlabel("Game #")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax.transAxes)

# Panel 2: FG%
ax = axes[0, 1]
fg_col = "FG_pct" if "FG_pct" in player_df.columns else ("FG%" if "FG%" in player_df.columns else None)
if fg_col:
    fg_series = pd.to_numeric(player_df[fg_col], errors="coerce")
    ax.bar(range(1, len(fg_series) + 1), fg_series.values, alpha=0.7, color=PALETTE[3])
    ax.axhline(fg_series.mean(), color="red", linestyle="--", label=f"Mean={fg_series.mean():.3f}")
    ax.set_title("Field Goal %")
    ax.set_xlabel("Game #")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No FG% data", ha="center", va="center", transform=ax.transAxes)

# Panel 3: PTS per minute
ax = axes[1, 0]
if "PTS_per_min" in player_df.columns:
    ppm = player_df["PTS_per_min"]
    ax.plot(range(1, len(ppm) + 1), ppm.values, "s-", alpha=0.6, color=PALETTE[4], markersize=4)
    ax.fill_between(range(1, len(ppm) + 1), 0, ppm.values, alpha=0.15, color=PALETTE[4])
    ax.set_title("Points per Minute")
    ax.set_xlabel("Game #")
else:
    ax.text(0.5, 0.5, "No PTS/min data", ha="center", va="center", transform=ax.transAxes)

# Panel 4: Target distribution
ax = axes[1, 1]
if TARGET in player_df.columns:
    target_vals = pd.to_numeric(player_df[TARGET], errors="coerce").dropna()
    ax.hist(target_vals, bins=12, alpha=0.7, color=PALETTE[5], edgecolor="white", density=True)
    try:
        target_vals.plot.kde(ax=ax, color="black", linewidth=1.5)
    except Exception:
        pass
    ax.axvline(target_vals.mean(), color="red", linestyle="--", label=f"Mean={target_vals.mean():.2f}")
    ax.axvline(target_vals.median(), color="orange", linestyle=":", label=f"Median={target_vals.median():.2f}")
    ax.set_title(f"{TARGET} Distribution")
    ax.legend()
else:
    ax.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax.transAxes)

plt.suptitle(f"Performance Dashboard: {selected_player}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "performance_trend.png"))
plt.show()
"""))

# ── 19. CODE: Box/Swarm + Decomposition ──────────────────────

cells.append(code(r"""
# ── Box + Swarm + Time Series Decomposition ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Box + Swarm for target
ax = axes[0]
if TARGET in model_df.columns:
    target_vals = model_df[TARGET].dropna()
    sns.boxplot(y=target_vals, ax=ax, color=PALETTE[0], width=0.3)
    sns.swarmplot(y=target_vals, ax=ax, color="black", alpha=0.5, size=4)
    ax.set_title(f"{TARGET} Box + Swarm")
else:
    ax.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax.transAxes)

# Outlier detection: IQR
ax = axes[1]
if TARGET in model_df.columns:
    Q1 = target_vals.quantile(0.25)
    Q3 = target_vals.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = target_vals[(target_vals < lower) | (target_vals > upper)]

    ax.scatter(range(len(target_vals)), target_vals, c="steelblue", alpha=0.6, label="Normal")
    if len(outliers) > 0:
        outlier_idx = target_vals.index[target_vals.isin(outliers)]
        ax.scatter(outlier_idx, outliers, c="red", s=80, zorder=5, label=f"Outliers ({len(outliers)})")
    ax.axhline(upper, color="red", linestyle="--", alpha=0.5, label=f"Upper={upper:.1f}")
    ax.axhline(lower, color="red", linestyle="--", alpha=0.5, label=f"Lower={lower:.1f}")
    ax.set_title("Outlier Detection (IQR)")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No target data", ha="center", va="center", transform=ax.transAxes)

# Seasonal Decomposition
ax = axes[2]
if HAS_SM and TARGET in player_df.columns:
    target_ts = pd.to_numeric(player_df[TARGET], errors="coerce").dropna()
    if len(target_ts) >= 8:
        period = min(max(4, len(target_ts) // 4), len(target_ts) // 2)
        try:
            decomp = seasonal_decompose(target_ts.values, model="additive", period=period)
            ax.plot(decomp.trend, label="Trend", color=PALETTE[0])
            ax.plot(decomp.seasonal, label="Seasonal", color=PALETTE[1], alpha=0.5)
            ax.set_title("Time Series Decomposition")
            ax.legend(fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Decomposition failed:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
    else:
        ax.text(0.5, 0.5, "Not enough data\nfor decomposition", ha="center", va="center",
                transform=ax.transAxes)
else:
    ax.text(0.5, 0.5, "statsmodels not available", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "box_swarm_decomp.png"))
plt.show()
"""))

# ── 20. CODE: League Context ─────────────────────────────────

cells.append(code(r"""
# ── League Context: Standings ──
if not standings.empty:
    # Find team and wins columns
    team_col = None
    wins_col = None
    for c in standings.columns:
        cl = c.lower()
        if "team" in cl or "club" in cl:
            team_col = c
        if cl in ("wins", "w", "won"):
            wins_col = c

    if team_col and wins_col:
        plot_data = standings.sort_values(wins_col, ascending=True).tail(18)
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(plot_data[team_col].astype(str), pd.to_numeric(plot_data[wins_col], errors="coerce"),
                       color=PALETTE[:len(plot_data)], edgecolor="white")
        ax.set_xlabel("Wins")
        ax.set_title(f"Euroleague Standings {CONFIG['season']}-{CONFIG['season']+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "standings.png"))
        plt.show()
    else:
        print(f"Cannot identify team/wins columns in standings. Columns: {list(standings.columns)}")
else:
    print("No standings data available.")
"""))

# ── 21. MARKDOWN: Feature Selection ──────────────────────────

cells.append(md(r"""
---
## Feature Selection
"""))

# ── 22. CODE: Collinearity Removal ───────────────────────────

cells.append(code(r"""
# ── Collinearity Removal (threshold > 0.95) ──
CORR_THRESHOLD = 0.95

if len(feature_cols) > 1:
    corr_abs = model_df[feature_cols].corr().abs()
    upper_tri = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > CORR_THRESHOLD)]

    print(f"Features before: {len(feature_cols)}")
    print(f"Dropped (r > {CORR_THRESHOLD}): {len(to_drop)}")
    if to_drop:
        for c in to_drop:
            print(f"  - {c}")

    feature_cols_filtered = [c for c in feature_cols if c not in to_drop]
    print(f"Features after: {len(feature_cols_filtered)}")
else:
    feature_cols_filtered = feature_cols[:]
    print("Not enough features for collinearity analysis.")
"""))

# ── 23. CODE: Mutual Information ──────────────────────────────

cells.append(code(r"""
# ── Mutual Information Scores ──
if len(feature_cols_filtered) > 0 and TARGET in model_df.columns:
    mi_data = model_df[feature_cols_filtered + [TARGET]].dropna()
    X_mi = mi_data[feature_cols_filtered]
    y_mi = mi_data[TARGET]

    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=SEED)
    mi_df = pd.DataFrame({"Feature": feature_cols_filtered, "MI_Score": mi_scores})
    mi_df = mi_df.sort_values("MI_Score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(mi_df) * 0.35)))
    ax.barh(mi_df["Feature"], mi_df["MI_Score"], color=PALETTE[0])
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(f"Feature Importance (MI with {TARGET})")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "mutual_information.png"))
    plt.show()

    display(mi_df.sort_values("MI_Score", ascending=False).head(15))
else:
    print("Cannot compute MI scores.")
"""))

# ── 24. CODE: PCA ────────────────────────────────────────────

cells.append(code(r"""
# ── PCA Analysis ──
if len(feature_cols_filtered) >= 3:
    pca_data = model_df[feature_cols_filtered].dropna()
    scaler_pca = StandardScaler()
    X_pca_scaled = scaler_pca.fit_transform(pca_data)

    n_components = min(len(feature_cols_filtered), len(pca_data), 10)
    pca = PCA(n_components=n_components)
    pca.fit(X_pca_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scree plot
    ax = axes[0]
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    ax.bar(range(1, len(explained) + 1), explained, alpha=0.7, color=PALETTE[0], label="Individual")
    ax.step(range(1, len(cumulative) + 1), cumulative, where="mid", color=PALETTE[1], linewidth=2, label="Cumulative")
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    ax.legend()

    # Biplot (PC1 vs PC2)
    ax = axes[1]
    X_pca_transformed = pca.transform(X_pca_scaled)
    ax.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], alpha=0.6, c=PALETTE[0])

    # Loading vectors (top features)
    loadings = pca.components_[:2].T
    n_arrows = min(8, len(feature_cols_filtered))
    importance = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
    top_idx = np.argsort(importance)[-n_arrows:]

    scale = max(abs(X_pca_transformed[:, 0]).max(), abs(X_pca_transformed[:, 1]).max()) * 0.8
    for idx in top_idx:
        ax.arrow(0, 0, loadings[idx, 0] * scale, loadings[idx, 1] * scale,
                 head_width=scale * 0.03, head_length=scale * 0.02, fc=PALETTE[2], ec=PALETTE[2])
        ax.text(loadings[idx, 0] * scale * 1.1, loadings[idx, 1] * scale * 1.1,
                feature_cols_filtered[idx], fontsize=8, color=PALETTE[2])

    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title("PCA Biplot")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "pca_analysis.png"))
    plt.show()

    print(f"\nComponents to explain 95% variance: {np.argmax(cumulative >= 0.95) + 1}")
else:
    print("Not enough features for PCA.")
"""))

# ── 25. MARKDOWN: Preprocessing ──────────────────────────────

cells.append(md(r"""
---
## Preprocessing & Train/Test Split
"""))

# ── 26. CODE: Temporal Split ─────────────────────────────────

cells.append(code(r"""
# ── Temporal Train/Test Split ──
# IMPORTANT: Temporal split, NOT random. Last N games as test set.

final_features = feature_cols_filtered[:]
X = model_df[final_features].values
y = model_df[TARGET].values

test_size = min(CONFIG["test_size"], len(X) - 3)  # ensure at least 3 training samples
train_size = len(X) - test_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Temporal Split (last {test_size} games as test)")
print(f"  Train: {X_train.shape} | y_train range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"  Test:  {X_test.shape} | y_test  range: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"\nFeatures: {len(final_features)}")
print(f"Scaler: StandardScaler (fit on train only)")
"""))

# ── 27. MARKDOWN: Model Training ─────────────────────────────

cells.append(md(r"""
---
## Model Training & Evaluation
"""))

# ── 28. CODE: Evaluation Helper ──────────────────────────────

cells.append(code(r"""
# ── Evaluation Helper ──
results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te, store=True):
    'Evaluate a model and store results.'
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)

    # Handle Keras models returning 2D arrays
    if hasattr(y_pred_train, "ndim") and y_pred_train.ndim > 1:
        y_pred_train = y_pred_train.flatten()
    if hasattr(y_pred_test, "ndim") and y_pred_test.ndim > 1:
        y_pred_test = y_pred_test.flatten()

    train_rmse = np.sqrt(mean_squared_error(y_tr, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred_test))
    test_mae = mean_absolute_error(y_te, y_pred_test)
    test_r2 = r2_score(y_te, y_pred_test) if len(y_te) > 1 else float("nan")

    res = {
        "Model": name,
        "Train_RMSE": round(train_rmse, 4),
        "Test_RMSE": round(test_rmse, 4),
        "Test_MAE": round(test_mae, 4),
        "Test_R2": round(test_r2, 4),
        "y_pred_test": y_pred_test,
    }

    if store:
        results[name] = res

    print(f"  {name:25s} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | "
          f"MAE: {test_mae:.4f} | R2: {test_r2:.4f}")
    return res
"""))

# ── 29. CODE: Linear Models ──────────────────────────────────

cells.append(code(r"""
# ── Linear Models ──
print("LINEAR MODELS")
print("=" * 80)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
evaluate("LinearRegression", lr, X_train_scaled, y_train, X_test_scaled, y_test)

# RidgeCV
ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=min(5, train_size - 1))
ridge.fit(X_train_scaled, y_train)
print(f"    Ridge alpha: {ridge.alpha_:.4f}")
evaluate("RidgeCV", ridge, X_train_scaled, y_train, X_test_scaled, y_test)

# LassoCV
lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=min(5, train_size - 1), random_state=SEED, max_iter=5000)
lasso.fit(X_train_scaled, y_train)
print(f"    Lasso alpha: {lasso.alpha_:.4f}")
evaluate("LassoCV", lasso, X_train_scaled, y_train, X_test_scaled, y_test)

# ElasticNetCV
enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], alphas=np.logspace(-3, 1, 10),
                    cv=min(5, train_size - 1), random_state=SEED, max_iter=5000)
enet.fit(X_train_scaled, y_train)
print(f"    ElasticNet alpha: {enet.alpha_:.4f}, l1_ratio: {enet.l1_ratio_:.2f}")
evaluate("ElasticNetCV", enet, X_train_scaled, y_train, X_test_scaled, y_test)
"""))

# ── 30. CODE: Tree-Based Models ──────────────────────────────

cells.append(code(r"""
# ── Tree-Based Models ──
print("\nTREE-BASED MODELS")
print("=" * 80)

# Random Forest with GridSearchCV
rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5],
}
rf_cv = GridSearchCV(
    RandomForestRegressor(random_state=SEED),
    rf_params,
    cv=min(3, train_size - 1),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
rf_cv.fit(X_train_scaled, y_train)
rf_best = rf_cv.best_estimator_
print(f"    RF best params: {rf_cv.best_params_}")
evaluate("RandomForest", rf_best, X_train_scaled, y_train, X_test_scaled, y_test)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)
gb.fit(X_train_scaled, y_train)
evaluate("GradientBoosting", gb, X_train_scaled, y_train, X_test_scaled, y_test)

# XGBoost
if HAS_XGB:
    xgb_model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=SEED, verbosity=0, n_jobs=-1,
    )
    xgb_model.fit(X_train_scaled, y_train)
    evaluate("XGBoost", xgb_model, X_train_scaled, y_train, X_test_scaled, y_test)
else:
    print("  XGBoost: SKIPPED (not installed)")

# LightGBM
if HAS_LGBM:
    lgbm_model = lgbm.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=SEED, verbose=-1, n_jobs=-1,
    )
    lgbm_model.fit(X_train_scaled, y_train)
    evaluate("LightGBM", lgbm_model, X_train_scaled, y_train, X_test_scaled, y_test)
else:
    print("  LightGBM: SKIPPED (not installed)")
"""))

# ── 31. CODE: SVR & KNN ──────────────────────────────────────

cells.append(code(r"""
# ── SVR & KNN ──
print("\nSVR & KNN")
print("=" * 80)

# SVR
svr = SVR(kernel="rbf", C=10.0, gamma="scale")
svr.fit(X_train_scaled, y_train)
evaluate("SVR_RBF", svr, X_train_scaled, y_train, X_test_scaled, y_test)

# KNN
k = min(5, train_size - 1)
knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
knn.fit(X_train_scaled, y_train)
evaluate(f"KNN(k={k})", knn, X_train_scaled, y_train, X_test_scaled, y_test)
"""))

# ── 32. CODE: Neural Network ─────────────────────────────────

cells.append(code(r"""
# ── Neural Network (TensorFlow / Keras) ──
print("\nNEURAL NETWORK")
print("=" * 80)

if HAS_TF:
    tf.random.set_seed(SEED)

    n_features_nn = X_train_scaled.shape[1]

    nn_model = keras.Sequential([
        layers.Input(shape=(n_features_nn,)),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])

    nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    nn_callbacks = [
        callbacks.EarlyStopping(patience=30, restore_best_weights=True, monitor="val_loss"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=10, monitor="val_loss"),
    ]

    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=max(4, train_size // 4),
        validation_split=0.2,
        callbacks=nn_callbacks,
        verbose=0,
    )

    evaluate("NeuralNetwork", nn_model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="Train MAE")
    if "val_mae" in history.history:
        axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("MAE Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()

    plt.suptitle("Neural Network Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "nn_training.png"))
    plt.show()
else:
    print("  TensorFlow: SKIPPED (not installed)")
"""))

# ── 33. CODE: Stacking Ensemble ──────────────────────────────

cells.append(code(r"""
# ── Stacking Ensemble ──
print("\nSTACKING ENSEMBLE")
print("=" * 80)

estimators = [
    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 10))),
    ("rf", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=SEED)),
    ("gb", GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=SEED)),
]

if HAS_XGB:
    estimators.append(("xgb", xgb.XGBRegressor(n_estimators=50, max_depth=3, verbosity=0, random_state=SEED)))

if HAS_LGBM:
    estimators.append(("lgbm", lgbm.LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1, random_state=SEED)))

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10)),
    cv=min(3, train_size - 1),
    n_jobs=-1,
)
stack.fit(X_train_scaled, y_train)
evaluate("StackingEnsemble", stack, X_train_scaled, y_train, X_test_scaled, y_test)
"""))

# ── 34. MARKDOWN: Model Comparison ───────────────────────────

cells.append(md(r"""
---
## Model Comparison
"""))

# ── 35. CODE: Results Table & Chart ──────────────────────────

cells.append(code(r"""
# ── Results Table & Chart ──
results_list = []
for name, res in results.items():
    results_list.append({
        "Model": res["Model"],
        "Train_RMSE": res["Train_RMSE"],
        "Test_RMSE": res["Test_RMSE"],
        "Test_MAE": res["Test_MAE"],
        "Test_R2": res["Test_R2"],
    })

results_df = pd.DataFrame(results_list).sort_values("Test_RMSE")
print("MODEL COMPARISON (sorted by Test RMSE)")
print("=" * 80)
display(results_df.reset_index(drop=True))

# Save results
results_df.to_csv(os.path.join(CONFIG["output_dir"], "model_comparison_results.csv"), index=False)

# Chart
fig, ax = plt.subplots(figsize=(12, max(4, len(results_df) * 0.5)))
y_pos = range(len(results_df))
ax.barh(y_pos, results_df["Test_RMSE"], color=PALETTE[0], alpha=0.8, label="Test RMSE")
ax.barh(y_pos, results_df["Train_RMSE"], color=PALETTE[1], alpha=0.4, label="Train RMSE")
ax.set_yticks(y_pos)
ax.set_yticklabels(results_df["Model"])
ax.set_xlabel("RMSE")
ax.set_title("Model Comparison: RMSE")
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "model_comparison.png"))
plt.show()

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name} (Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f})")
"""))

# ── 36. MARKDOWN: Best Model Deep Dive ───────────────────────

cells.append(md(r"""
---
## Best Model Deep Dive
"""))

# ── 37. CODE: Diagnostics ────────────────────────────────────

cells.append(code(r"""
# ── Best Model Diagnostics ──
# Retrieve the best model object
best_name = results_df.iloc[0]["Model"]
best_preds = results[best_name]["y_pred_test"]

# Map model names to objects
model_objects = {
    "LinearRegression": lr,
    "RidgeCV": ridge,
    "LassoCV": lasso,
    "ElasticNetCV": enet,
    "RandomForest": rf_best,
    "GradientBoosting": gb,
    "SVR_RBF": svr,
    "StackingEnsemble": stack,
}
if HAS_XGB and "XGBoost" in results:
    model_objects["XGBoost"] = xgb_model
if HAS_LGBM and "LightGBM" in results:
    model_objects["LightGBM"] = lgbm_model
if HAS_TF and "NeuralNetwork" in results:
    model_objects["NeuralNetwork"] = nn_model
for k in list(results.keys()):
    if k.startswith("KNN"):
        model_objects[k] = knn

best_model_obj = model_objects.get(best_name, lr)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, best_preds, alpha=0.7, color=PALETTE[0], s=60, edgecolors="white")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, "r--", alpha=0.7, label="Perfect")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title(f"Actual vs Predicted ({best_name})")
ax.legend()

# 2. Residuals
ax = axes[0, 1]
residuals = y_test - best_preds
ax.scatter(best_preds, residuals, alpha=0.7, color=PALETTE[1], s=60, edgecolors="white")
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.set_title("Residual Plot")

# 3. Residual Distribution
ax = axes[1, 0]
ax.hist(residuals, bins=max(5, len(residuals) // 2), alpha=0.7, color=PALETTE[2], edgecolor="white", density=True)
try:
    pd.Series(residuals).plot.kde(ax=ax, color="black", linewidth=1.5)
except Exception:
    pass
ax.set_title("Residual Distribution")
ax.set_xlabel("Residual")

# 4. Q-Q Plot
ax = axes[1, 1]
if len(residuals) >= 3:
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
else:
    ax.text(0.5, 0.5, "Not enough data for Q-Q", ha="center", va="center", transform=ax.transAxes)

plt.suptitle(f"Diagnostics: {best_name}", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "diagnostics.png"))
plt.show()
"""))

# ── 38. CODE: Learning Curve ─────────────────────────────────

cells.append(code(r"""
# ── Learning Curve ──
if train_size >= 6:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a simple model for learning curve (avoid NN)
    lc_model = best_model_obj
    if best_name == "NeuralNetwork":
        lc_model = RidgeCV(alphas=np.logspace(-3, 3, 10))

    try:
        train_sizes_arr, train_scores, test_scores = learning_curve(
            lc_model, X_train_scaled, y_train,
            cv=min(3, train_size - 1),
            scoring="neg_mean_squared_error",
            train_sizes=np.linspace(0.3, 1.0, min(5, train_size)),
            n_jobs=-1,
        )

        train_rmse_lc = np.sqrt(-train_scores)
        test_rmse_lc = np.sqrt(-test_scores)

        ax.plot(train_sizes_arr, train_rmse_lc.mean(axis=1), "o-", color=PALETTE[0], label="Train RMSE")
        ax.fill_between(train_sizes_arr,
                        train_rmse_lc.mean(axis=1) - train_rmse_lc.std(axis=1),
                        train_rmse_lc.mean(axis=1) + train_rmse_lc.std(axis=1),
                        alpha=0.15, color=PALETTE[0])

        ax.plot(train_sizes_arr, test_rmse_lc.mean(axis=1), "o-", color=PALETTE[1], label="CV RMSE")
        ax.fill_between(train_sizes_arr,
                        test_rmse_lc.mean(axis=1) - test_rmse_lc.std(axis=1),
                        test_rmse_lc.mean(axis=1) + test_rmse_lc.std(axis=1),
                        alpha=0.15, color=PALETTE[1])

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Learning Curve ({best_name if best_name != 'NeuralNetwork' else 'RidgeCV'})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "learning_curve.png"))
        plt.show()
    except Exception as e:
        print(f"Learning curve computation failed: {e}")
else:
    print("Not enough training data for learning curve analysis.")
"""))

# ── 39. MARKDOWN: SHAP ───────────────────────────────────────

cells.append(md(r"""
---
## SHAP Explainability
"""))

# ── 40. CODE: SHAP Analysis ──────────────────────────────────

cells.append(code(r"""
# ── SHAP Analysis ──
if HAS_SHAP:
    print("SHAP ANALYSIS")
    print("=" * 60)

    # Pick a tree-based model for TreeExplainer
    shap_model = None
    shap_model_name = None
    for candidate_name in ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"]:
        if candidate_name in model_objects:
            shap_model = model_objects[candidate_name]
            shap_model_name = candidate_name
            break

    if shap_model is not None:
        print(f"Using {shap_model_name} for SHAP analysis")

        try:
            explainer = shap.TreeExplainer(shap_model)
            shap_values = explainer.shap_values(X_test_scaled)

            # Summary plot
            fig, ax = plt.subplots(figsize=(12, max(4, len(final_features) * 0.3)))
            shap.summary_plot(shap_values, X_test_scaled, feature_names=final_features, show=False)
            plt.title(f"SHAP Summary Plot ({shap_model_name})")
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "shap_summary.png"))
            plt.show()

            # Bar plot
            fig, ax = plt.subplots(figsize=(10, max(4, len(final_features) * 0.3)))
            shap.summary_plot(shap_values, X_test_scaled, feature_names=final_features,
                              plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance ({shap_model_name})")
            plt.tight_layout()
            plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "shap_bar.png"))
            plt.show()

            # Waterfall for first test sample
            if len(X_test_scaled) > 0:
                try:
                    explanation = shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
                        data=X_test_scaled[0],
                        feature_names=final_features,
                    )
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.waterfall_plot(explanation, show=False)
                    plt.title("SHAP Waterfall (First Test Game)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "shap_waterfall.png"))
                    plt.show()
                except Exception as e:
                    print(f"Waterfall plot failed: {e}")
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    else:
        print("No tree-based model available for SHAP analysis.")
else:
    print("SHAP not installed. Skipping explainability analysis.")
"""))

# ── 41. MARKDOWN: ARIMA ──────────────────────────────────────

cells.append(md(r"""
---
## ARIMA Time Series Forecasting
"""))

# ── 42. CODE: ARIMA ──────────────────────────────────────────

cells.append(code(r"""
# ── ARIMA Forecasting ──
if HAS_SM:
    print("ARIMA FORECASTING")
    print("=" * 60)

    target_ts = pd.to_numeric(player_df[TARGET] if TARGET in player_df.columns else pd.Series(dtype=float),
                               errors="coerce").dropna().reset_index(drop=True)

    if len(target_ts) >= 10:
        # Auto order selection: test a few combinations
        best_aic = float("inf")
        best_order = (1, 0, 1)

        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        model_arima = ARIMA(target_ts.values, order=(p, d, q))
                        fit = model_arima.fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")

        # Fit best model
        final_arima = ARIMA(target_ts.values, order=best_order)
        arima_fit = final_arima.fit()
        print(arima_fit.summary())

        # Forecast next 5 games
        n_forecast = 5
        forecast_result = arima_fit.get_forecast(steps=n_forecast)
        forecast_mean = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int(alpha=0.05)

        fig, ax = plt.subplots(figsize=(14, 6))
        x_hist = range(1, len(target_ts) + 1)
        x_fc = range(len(target_ts) + 1, len(target_ts) + n_forecast + 1)

        ax.plot(x_hist, target_ts.values, "o-", alpha=0.6, label="Historical", color=PALETTE[0])
        ax.plot(x_fc, forecast_mean, "s--", color=PALETTE[2], linewidth=2, label="Forecast")
        ax.fill_between(x_fc, forecast_ci[:, 0], forecast_ci[:, 1],
                        alpha=0.2, color=PALETTE[2], label="95% CI")
        ax.axvline(len(target_ts) + 0.5, color="grey", linestyle=":", alpha=0.5)
        ax.set_xlabel("Game #")
        ax.set_ylabel(TARGET)
        ax.set_title(f"ARIMA{best_order} Forecast: {selected_player}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "arima_forecast.png"))
        plt.show()

        print(f"\nForecast for next {n_forecast} games:")
        for i, (mean, lo, hi) in enumerate(zip(forecast_mean, forecast_ci[:, 0], forecast_ci[:, 1]), 1):
            print(f"  Game +{i}: {mean:.2f}  [{lo:.2f}, {hi:.2f}]")
    else:
        print(f"Not enough data for ARIMA (need >= 10, have {len(target_ts)}).")
else:
    print("Statsmodels not installed. Skipping ARIMA analysis.")
"""))

# ── 43. MARKDOWN: Scenarios ──────────────────────────────────

cells.append(md(r"""
---
## Scenario Predictions with Bootstrap CI
"""))

# ── 44. CODE: Scenario Predictions ───────────────────────────

cells.append(code(r"""
# ── Scenario Predictions with Bootstrap CI ──
print("SCENARIO PREDICTIONS")
print("=" * 60)

# Use the best model for predictions
pred_model = best_model_obj

# Build scenarios based on feature statistics from training data
train_df = model_df.iloc[:train_size]
feature_means = train_df[final_features].mean()
feature_stds = train_df[final_features].std()

scenarios = {
    "Poor Game": feature_means - 1.0 * feature_stds,
    "Below Average": feature_means - 0.5 * feature_stds,
    "Average Game": feature_means,
    "Above Average": feature_means + 0.5 * feature_stds,
    "Elite Game": feature_means + 1.0 * feature_stds,
}

# Bootstrap CI
n_bootstrap = 500
scenario_results = {}

for name, scenario_raw in scenarios.items():
    scenario_scaled = scaler.transform(scenario_raw.values.reshape(1, -1))

    # Point prediction
    if best_name == "NeuralNetwork" and HAS_TF:
        point_pred = pred_model.predict(scenario_scaled, verbose=0).flatten()[0]
    else:
        point_pred = pred_model.predict(scenario_scaled)[0]

    # Bootstrap: add noise to features
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        noise = np.random.normal(0, 0.05, scenario_raw.shape)
        noisy = scenario_raw.values + noise * feature_stds.values
        noisy_scaled = scaler.transform(noisy.reshape(1, -1))
        if best_name == "NeuralNetwork" and HAS_TF:
            bp = pred_model.predict(noisy_scaled, verbose=0).flatten()[0]
        else:
            bp = pred_model.predict(noisy_scaled)[0]
        bootstrap_preds.append(bp)

    ci_lo = np.percentile(bootstrap_preds, 2.5)
    ci_hi = np.percentile(bootstrap_preds, 97.5)

    scenario_results[name] = {"Prediction": round(point_pred, 2),
                               "CI_Low": round(ci_lo, 2), "CI_High": round(ci_hi, 2)}
    print(f"  {name:20s}: {point_pred:7.2f}  [{ci_lo:.2f}, {ci_hi:.2f}]")

# Chart
fig, ax = plt.subplots(figsize=(10, 5))
names_list = list(scenario_results.keys())
preds_list = [scenario_results[n]["Prediction"] for n in names_list]
ci_lo_list = [scenario_results[n]["CI_Low"] for n in names_list]
ci_hi_list = [scenario_results[n]["CI_High"] for n in names_list]
errors = [[p - lo for p, lo in zip(preds_list, ci_lo_list)],
          [hi - p for p, hi in zip(preds_list, ci_hi_list)]]

ax.barh(names_list, preds_list, xerr=errors, color=PALETTE[:len(names_list)],
        edgecolor="white", capsize=5, alpha=0.8)
ax.set_xlabel(f"Predicted {TARGET}")
ax.set_title(f"Scenario Predictions for {selected_player} (95% Bootstrap CI)")
plt.tight_layout()
plt.savefig(os.path.join(CONFIG["output_dir"], "plots", "scenarios.png"))
plt.show()
"""))

# ── 45. MARKDOWN: Player Report ──────────────────────────────

cells.append(md(r"""
---
## Player Performance Report
"""))

# ── 46. CODE: Player Report ──────────────────────────────────

cells.append(code(r"""
# ── Player Performance Report ──
print("=" * 70)
print(f"  PLAYER PERFORMANCE REPORT: {selected_player.upper()}")
print(f"  Season: {CONFIG['season']}-{CONFIG['season']+1}")
print("=" * 70)

# Basic stats
n_games = len(player_df)
print(f"\n  Games Analyzed: {n_games}")

if TARGET in player_df.columns:
    target_vals = pd.to_numeric(player_df[TARGET], errors="coerce").dropna()
    print(f"\n  {TARGET} Summary:")
    print(f"    Mean:   {target_vals.mean():.2f}")
    print(f"    Median: {target_vals.median():.2f}")
    print(f"    Std:    {target_vals.std():.2f}")
    print(f"    Min:    {target_vals.min():.2f}")
    print(f"    Max:    {target_vals.max():.2f}")

# Additional stats
for stat_col in ["FG_pct", "TS_pct", "eFG_pct", "PTS_per_min", "MP_decimal"]:
    if stat_col in player_df.columns:
        vals = pd.to_numeric(player_df[stat_col], errors="coerce").dropna()
        if len(vals) > 0:
            print(f"\n  {stat_col}:")
            print(f"    Mean: {vals.mean():.3f} | Std: {vals.std():.3f}")

# Model performance
print(f"\n  Best Model: {best_name}")
print(f"    Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"    Test MAE:  {results_df.iloc[0]['Test_MAE']:.4f}")
print(f"    Test R2:   {results_df.iloc[0]['Test_R2']:.4f}")

# Trend
if TARGET in player_df.columns and len(target_vals) >= 5:
    recent_5 = target_vals.tail(5).mean()
    overall = target_vals.mean()
    trend = "IMPROVING" if recent_5 > overall else "DECLINING" if recent_5 < overall else "STABLE"
    print(f"\n  Trend: {trend}")
    print(f"    Recent 5-game avg: {recent_5:.2f}")
    print(f"    Season avg:        {overall:.2f}")
    print(f"    Delta:             {recent_5 - overall:+.2f}")

# Scenarios
if scenario_results:
    print(f"\n  Scenario Predictions ({TARGET}):")
    for name, vals in scenario_results.items():
        print(f"    {name:20s}: {vals['Prediction']:7.2f}  [{vals['CI_Low']:.2f}, {vals['CI_High']:.2f}]")

print("\n" + "=" * 70)
"""))

# ── 47. MARKDOWN: Export ─────────────────────────────────────

cells.append(md(r"""
---
## Export & Persistence
"""))

# ── 48. CODE: Save Everything ─────────────────────────────────

cells.append(code(r"""
# ── Save Everything ──
print("EXPORTING ARTIFACTS")
print("=" * 60)

# 1. Player data CSV
player_csv_path = os.path.join(CONFIG["output_dir"], "player_data_engineered.csv")
player_df.to_csv(player_csv_path, index=False)
print(f"  Player data: {player_csv_path}")

# 2. Model comparison CSV
comp_csv_path = os.path.join(CONFIG["output_dir"], "model_comparison_results.csv")
results_df.to_csv(comp_csv_path, index=False)
print(f"  Model comparison: {comp_csv_path}")

# 3. Feature importance CSV
if len(final_features) > 0 and TARGET in model_df.columns:
    mi_data_export = model_df[final_features + [TARGET]].dropna()
    mi_scores_export = mutual_info_regression(mi_data_export[final_features], mi_data_export[TARGET],
                                               random_state=SEED)
    fi_df = pd.DataFrame({"Feature": final_features, "MI_Score": mi_scores_export})
    fi_df = fi_df.sort_values("MI_Score", ascending=False)
    fi_csv_path = os.path.join(CONFIG["output_dir"], "feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"  Feature importance: {fi_csv_path}")

# 4. Model persistence (best model)
if best_name != "NeuralNetwork":
    model_path = os.path.join(CONFIG["output_dir"], "models", "best_model.joblib")
    joblib.dump(best_model_obj, model_path)
    print(f"  Best model ({best_name}): {model_path}")
elif HAS_TF and best_name == "NeuralNetwork":
    nn_path = os.path.join(CONFIG["output_dir"], "models", "nn_model.keras")
    try:
        nn_model.save(nn_path)
        print(f"  Neural network: {nn_path}")
    except Exception as e:
        print(f"  NN save failed: {e}")

# 5. Scaler
scaler_path = os.path.join(CONFIG["output_dir"], "models", "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"  Scaler: {scaler_path}")

# 6. Config JSON
config_path = os.path.join(CONFIG["output_dir"], "config.json")
config_export = CONFIG.copy()
config_export["features"] = final_features
config_export["target"] = TARGET
config_export["best_model"] = best_name
with open(config_path, "w") as f:
    json.dump(config_export, f, indent=2, default=str)
print(f"  Config: {config_path}")

# 7. Scenario predictions
scenario_csv_path = os.path.join(CONFIG["output_dir"], "scenario_predictions.csv")
pd.DataFrame(scenario_results).T.to_csv(scenario_csv_path)
print(f"  Scenarios: {scenario_csv_path}")

print("\nAll artifacts saved successfully.")
"""))

# ── 49. MARKDOWN: Conclusions ─────────────────────────────────

cells.append(md(r"""
---
## Conclusions
"""))

# ── 50. CODE: Summary ────────────────────────────────────────

cells.append(code(r"""
# ── Project Summary ──
print("=" * 70)
print("  EUROLEAGUE ANALYTICS PRO - PROJECT SUMMARY")
print("=" * 70)

_data_src = 'Euroleague API' if HAS_API else 'Static CSV'
_xgb_st = 'Enabled' if HAS_XGB else 'Not available'
_lgbm_st = 'Enabled' if HAS_LGBM else 'Not available'
_tf_st = 'Enabled' if HAS_TF else 'Not available'
_shap_st = 'Enabled' if HAS_SHAP else 'Not available'
_sm_st = 'Enabled' if HAS_SM else 'Not available'
_pl_st = 'Enabled' if HAS_PLOTLY else 'Not available'

print(f"  Player:           {selected_player}")
print(f"  Season:           {CONFIG['season']}-{CONFIG['season']+1}")
print(f"  Games Analyzed:   {len(player_df)}")
print(f"  Target Variable:  {TARGET}")
print(f"  Features Used:    {len(final_features)}")
print(f"  Models Trained:   {len(results)}")
print(f"  Best Model:       {best_name}")
print(f"  Best Test RMSE:   {results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"  Best Test R2:     {results_df.iloc[0]['Test_R2']:.4f}")
print(f"  Data Source:      {_data_src}")
print(f"  Train/Test Split: Temporal ({train_size} / {test_size})")
print(f"  Optional Libraries:")
print(f"    XGBoost:      {_xgb_st}")
print(f"    LightGBM:     {_lgbm_st}")
print(f"    TensorFlow:   {_tf_st}")
print(f"    SHAP:         {_shap_st}")
print(f"    Statsmodels:  {_sm_st}")
print(f"    Plotly:       {_pl_st}")
print(f"  Output Directory: {CONFIG['output_dir']}/")
print("=" * 70)
print("  Analysis complete.")
print("=" * 70)
"""))


# ──────────────────────────────────────────────────────────────
# Build Notebook
# ──────────────────────────────────────────────────────────────

def build_notebook():
    """Assemble and write the notebook JSON."""
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "cells": cells,
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Notebook written: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(cells)}")

    md_count = sum(1 for c in cells if c["cell_type"] == "markdown")
    code_count = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"  Markdown cells: {md_count}")
    print(f"  Code cells:     {code_count}")


if __name__ == "__main__":
    build_notebook()
