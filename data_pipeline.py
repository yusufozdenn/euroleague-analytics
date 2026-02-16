"""
Euroleague Dynamic Data Pipeline
=================================
Fetches live data from euroleague_api for player, team, and league analysis.
Includes local caching to avoid redundant API calls.

Usage:
    from data_pipeline import EuroleaguePipeline
    pipe = EuroleaguePipeline(season=2024)
    player_df = pipe.get_player_game_stats("Hayes")
    team_df = pipe.get_team_stats()
    standings = pipe.get_standings()
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Check euroleague_api availability ───
try:
    from euroleague_api.player_stats import PlayerStats
    from euroleague_api.team_stats import TeamStats
    from euroleague_api.standings import Standings
    from euroleague_api.shot_data import ShotData
    from euroleague_api.boxscore_data import BoxScoreData
    from euroleague_api.EuroLeagueData import EuroLeagueData
    HAS_API = True
    logger.info("euroleague_api loaded successfully.")
except ImportError:
    HAS_API = False
    logger.warning("euroleague_api not installed. Install: pip install euroleague-api")


class EuroleaguePipeline:
    """End-to-end data pipeline for Euroleague analytics."""

    COMPETITION = "E"  # E=Euroleague, U=Eurocup

    def __init__(self, season: int = 2024, cache_dir: str = "cache",
                 cache_ttl_hours: int = 6):
        """
        Args:
            season: Season start year (e.g. 2024 for 2024-25 season)
            cache_dir: Directory for cached API responses
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.season = season
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        if HAS_API:
            self.player_stats = PlayerStats(competition=self.COMPETITION)
            self.team_stats_api = TeamStats(competition=self.COMPETITION)
            self.standings_api = Standings(competition=self.COMPETITION)
            self.shot_data_api = ShotData(competition=self.COMPETITION)
            self.boxscore_api = BoxScoreData(competition=self.COMPETITION)
            self.base_api = EuroLeagueData(competition=self.COMPETITION)

        logger.info(f"Pipeline initialized: season={season}, cache={cache_dir}")

    # ─── Caching ───────────────────────────────────────────────
    def _cache_key(self, name: str, **kwargs) -> str:
        params = f"{name}_{self.season}_{kwargs}"
        return hashlib.md5(params.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mtime < self.cache_ttl:
                logger.info(f"Cache hit: {key}")
                return pd.read_parquet(path)
        return None

    def _cache_set(self, key: str, df: pd.DataFrame):
        df.to_parquet(self._cache_path(key), index=False)
        logger.info(f"Cached: {key} ({len(df)} rows)")

    # ─── Player Stats ─────────────────────────────────────────
    def get_player_season_stats(self, endpoint: str = "traditional",
                                 statistic_mode: str = "PerGame") -> pd.DataFrame:
        """Fetch aggregated player stats for the season.

        Args:
            endpoint: 'traditional', 'advanced', 'misc', or 'scoring'
            statistic_mode: 'PerGame', 'Accumulated', or 'Per100Possessions'
        """
        if not HAS_API:
            logger.error("euroleague_api not available")
            return pd.DataFrame()

        key = self._cache_key("player_season", endpoint=endpoint, mode=statistic_mode)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        logger.info(f"Fetching player stats: {endpoint} / {statistic_mode}")
        try:
            df = self.player_stats.get_player_stats_single_season(
                endpoint=endpoint,
                season=self.season,
                statistic_mode=statistic_mode
            )
            self._cache_set(key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch player stats: {e}")
            return pd.DataFrame()

    def get_player_boxscore_season(self) -> pd.DataFrame:
        """Fetch game-by-game boxscore for ALL players in the season."""
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("player_boxscore_season")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        logger.info(f"Fetching full season boxscore data (season={self.season})...")
        try:
            df = self.boxscore_api.get_player_boxscore_stats_single_season(
                season=self.season
            )
            self._cache_set(key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch boxscore: {e}")
            return pd.DataFrame()

    def get_player_game_stats(self, player_name: str,
                               boxscore_df: pd.DataFrame = None) -> pd.DataFrame:
        """Extract a specific player's game-by-game stats from boxscore data.

        Args:
            player_name: Partial name match (case-insensitive)
            boxscore_df: Pre-fetched boxscore df (avoids re-fetching)
        """
        if boxscore_df is None:
            boxscore_df = self.get_player_boxscore_season()

        if boxscore_df.empty:
            logger.warning("No boxscore data available")
            return pd.DataFrame()

        # Find player name column
        name_col = None
        for col in boxscore_df.columns:
            if "player" in col.lower() and "id" not in col.lower():
                name_col = col
                break

        if name_col is None:
            # Try common column names
            for candidate in ["Player", "playerName", "player_name", "PLAYER"]:
                if candidate in boxscore_df.columns:
                    name_col = candidate
                    break

        if name_col is None:
            logger.error(f"Cannot find player name column. Columns: {list(boxscore_df.columns)}")
            return pd.DataFrame()

        mask = boxscore_df[name_col].str.contains(player_name, case=False, na=False)
        player_df = boxscore_df[mask].copy()

        if player_df.empty:
            logger.warning(f"Player '{player_name}' not found. Available players sample:")
            unique = boxscore_df[name_col].unique()[:20]
            for p in unique:
                logger.info(f"  - {p}")

        logger.info(f"Found {len(player_df)} games for '{player_name}'")
        return player_df

    # ─── Team Stats ───────────────────────────────────────────
    def get_team_season_stats(self, endpoint: str = "traditional",
                               statistic_mode: str = "PerGame") -> pd.DataFrame:
        """Fetch team stats for the season."""
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("team_season", endpoint=endpoint, mode=statistic_mode)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        logger.info(f"Fetching team stats: {endpoint}")
        try:
            df = self.team_stats_api.get_team_stats_single_season(
                endpoint=endpoint,
                season=self.season,
                statistic_mode=statistic_mode
            )
            self._cache_set(key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch team stats: {e}")
            return pd.DataFrame()

    # ─── Standings ────────────────────────────────────────────
    def get_standings(self, round_number: int = 34,
                      endpoint: str = "basicstandings") -> pd.DataFrame:
        """Fetch league standings.

        Args:
            round_number: Round to get standings for (max=34 for regular season).
                          Auto-retries with lower rounds if the requested round
                          doesn't exist yet (mid-season).
            endpoint: 'basicstandings', 'streaks', 'aheadbehind', 'margins'
        """
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("standings", round=round_number, endpoint=endpoint)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # Try requested round, then fall back to lower rounds (for mid-season)
        for rnd in range(round_number, 0, -1):
            logger.info(f"Fetching standings: round={rnd}")
            try:
                df = self.standings_api.get_standings(
                    season=self.season,
                    round_number=rnd,
                    endpoint=endpoint
                )
                if df is not None and not df.empty:
                    self._cache_set(key, df)
                    return df
            except Exception as e:
                if rnd == 1:
                    logger.error(f"Failed to fetch standings: {e}")
                continue
        return pd.DataFrame()

    # ─── Shot Data ────────────────────────────────────────────
    def get_shot_data_season(self) -> pd.DataFrame:
        """Fetch shot-level data for the entire season."""
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("shots_season")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        logger.info(f"Fetching shot data for season {self.season} (this may take a while)...")
        try:
            df = self.shot_data_api.get_game_shot_data_single_season(
                season=self.season
            )
            self._cache_set(key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch shot data: {e}")
            return pd.DataFrame()

    # ─── Game Metadata ────────────────────────────────────────
    def get_season_games(self) -> pd.DataFrame:
        """Fetch all game metadata for the season (scores, dates, teams)."""
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("season_games")
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        logger.info(f"Fetching game metadata for season {self.season}")
        try:
            df = self.base_api.get_gamecodes_season(season=self.season)
            self._cache_set(key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch game metadata: {e}")
            return pd.DataFrame()

    # ─── Multi-Season ─────────────────────────────────────────
    def get_player_boxscore_multi_season(self, start_season: int,
                                          end_season: int) -> pd.DataFrame:
        """Fetch boxscore data across multiple seasons."""
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("boxscore_multi", start=start_season, end=end_season)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        logger.info(f"Fetching boxscore: seasons {start_season}-{end_season}")
        try:
            df = self.boxscore_api.get_player_boxscore_stats_multiple_seasons(
                start_season=start_season,
                end_season=end_season
            )
            self._cache_set(key, df)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch multi-season boxscore: {e}")
            return pd.DataFrame()

    # ─── Feature Engineering ──────────────────────────────────
    @staticmethod
    def engineer_player_features(df: pd.DataFrame,
                                  pts_col: str = "Points",
                                  min_col: str = "Minutes",
                                  fg_pct_col: str = None) -> pd.DataFrame:
        """Apply advanced feature engineering to player game-by-game data.

        Automatically detects column names from euroleague_api format.
        """
        d = df.copy()
        d = d.sort_values(by=[c for c in d.columns if "round" in c.lower() or
                               "game" in c.lower() or "date" in c.lower()][:1] or
                          d.columns[:1]).reset_index(drop=True)

        # ─── Auto-detect columns ───
        cols = {c.lower(): c for c in d.columns}

        def find_col(*candidates):
            for c in candidates:
                if c in cols:
                    return cols[c]
                for k, v in cols.items():
                    if c in k:
                        return v
            return None

        pts = find_col("points", "pts", "score")
        mins = find_col("minutes", "timeplayed", "min")
        ast = find_col("assistances", "assists", "ast")
        reb = find_col("totalrebounds", "rebounds", "reb")
        stl = find_col("steals", "stl")
        tov = find_col("turnovers", "tov", "to")
        blk = find_col("blocksfavour", "blocks", "blk")
        fgm2 = find_col("fieldgoalsmade2", "twopointsmade", "fgm2")
        fga2 = find_col("fieldgoalsattempted2", "twopointsattempted", "fga2")
        fgm3 = find_col("fieldgoalsmade3", "threepointsmade", "fgm3")
        fga3 = find_col("fieldgoalsattempted3", "threepointsattempted", "fga3")
        ftm = find_col("freethrowsmade", "ftm")
        fta = find_col("freethrowsattempted", "fta")
        pir = find_col("valuation", "pir")

        d["game_num"] = range(1, len(d) + 1)

        # ─── Minutes to decimal ───
        if mins and d[mins].dtype == object:
            def _parse_min(x):
                try:
                    parts = str(x).split(":")
                    return int(parts[0]) + int(parts[1]) / 60
                except:
                    return np.nan
            d["MP_decimal"] = d[mins].apply(_parse_min)
        elif mins:
            d["MP_decimal"] = pd.to_numeric(d[mins], errors="coerce")
        else:
            d["MP_decimal"] = np.nan

        # ─── Shooting percentages ───
        if fgm2 and fga2:
            total_fgm = pd.to_numeric(d[fgm2], errors="coerce").fillna(0)
            total_fga = pd.to_numeric(d[fga2], errors="coerce").fillna(0)
            if fgm3 and fga3:
                total_fgm += pd.to_numeric(d[fgm3], errors="coerce").fillna(0)
                total_fga += pd.to_numeric(d[fga3], errors="coerce").fillna(0)
            d["FG_pct"] = (total_fgm / total_fga).replace([np.inf, -np.inf], 0).fillna(0)
        elif fg_pct_col and fg_pct_col in d.columns:
            d["FG_pct"] = pd.to_numeric(d[fg_pct_col], errors="coerce")

        # True Shooting %
        if pts and fga2 and fta:
            total_fga_all = pd.to_numeric(d.get(fga2, pd.Series(0)), errors="coerce").fillna(0)
            if fga3:
                total_fga_all += pd.to_numeric(d[fga3], errors="coerce").fillna(0)
            fta_val = pd.to_numeric(d[fta], errors="coerce").fillna(0)
            pts_val = pd.to_numeric(d[pts], errors="coerce").fillna(0)
            d["TS_pct"] = (pts_val / (2 * (total_fga_all + 0.44 * fta_val))).replace(
                [np.inf, -np.inf], 0).fillna(0)

        # Effective FG%
        if fgm2 and fga2 and fgm3:
            fgm_all = pd.to_numeric(d[fgm2], errors="coerce").fillna(0)
            fga_all = pd.to_numeric(d[fga2], errors="coerce").fillna(0)
            fgm3_val = pd.to_numeric(d[fgm3], errors="coerce").fillna(0)
            if fga3:
                fga_all += pd.to_numeric(d[fga3], errors="coerce").fillna(0)
            fgm_all += fgm3_val
            d["eFG_pct"] = ((fgm_all + 0.5 * fgm3_val) / fga_all).replace(
                [np.inf, -np.inf], 0).fillna(0)

        # ─── Per-minute stats ───
        if pts:
            pts_num = pd.to_numeric(d[pts], errors="coerce").fillna(0)
            d["PTS_per_min"] = (pts_num / d["MP_decimal"]).replace([np.inf, -np.inf], 0).fillna(0)
        if ast:
            ast_num = pd.to_numeric(d[ast], errors="coerce").fillna(0)
            d["AST_per_min"] = (ast_num / d["MP_decimal"]).replace([np.inf, -np.inf], 0).fillna(0)

        # ─── Game Score (GmSc) ───
        # GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*ORB + 0.3*DRB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV
        if pts:
            gmsc = pd.to_numeric(d[pts], errors="coerce").fillna(0)
            if fgm2:
                fgm_t = pd.to_numeric(d[fgm2], errors="coerce").fillna(0)
                if fgm3:
                    fgm_t += pd.to_numeric(d[fgm3], errors="coerce").fillna(0)
                gmsc += 0.4 * fgm_t
            if fga2:
                fga_t = pd.to_numeric(d[fga2], errors="coerce").fillna(0)
                if fga3:
                    fga_t += pd.to_numeric(d[fga3], errors="coerce").fillna(0)
                gmsc -= 0.7 * fga_t
            if fta and ftm:
                gmsc -= 0.4 * (pd.to_numeric(d[fta], errors="coerce").fillna(0) -
                               pd.to_numeric(d[ftm], errors="coerce").fillna(0))
            if reb:
                gmsc += 0.7 * pd.to_numeric(d[reb], errors="coerce").fillna(0)
            if ast:
                gmsc += 0.7 * pd.to_numeric(d[ast], errors="coerce").fillna(0)
            if stl:
                gmsc += pd.to_numeric(d[stl], errors="coerce").fillna(0)
            if blk:
                gmsc += 0.7 * pd.to_numeric(d[blk], errors="coerce").fillna(0)
            if tov:
                gmsc -= pd.to_numeric(d[tov], errors="coerce").fillna(0)
            d["GmSc"] = gmsc

        # ─── Rolling & EWMA ───
        target = "GmSc" if "GmSc" in d.columns else (pts if pts else None)
        pts_series = pd.to_numeric(d[pts], errors="coerce") if pts else None

        if target and target in d.columns:
            target_series = d[target]
            for w in [3, 5, 7]:
                d[f"{target}_roll_mean_{w}"] = target_series.rolling(w, min_periods=1).mean()
                d[f"{target}_roll_std_{w}"] = target_series.rolling(w, min_periods=1).std().fillna(0)
            for span in [3, 5]:
                d[f"{target}_ewma_{span}"] = target_series.ewm(span=span).mean()
            # Momentum
            d[f"{target}_momentum"] = target_series - target_series.rolling(5, min_periods=1).mean()

        if pts_series is not None:
            for w in [3, 5]:
                d[f"PTS_roll_mean_{w}"] = pts_series.rolling(w, min_periods=1).mean()
            for span in [3, 5]:
                d[f"PTS_ewma_{span}"] = pts_series.ewm(span=span).mean()
            # Lag features
            for lag in [1, 2, 3]:
                d[f"PTS_lag_{lag}"] = pts_series.shift(lag)
                if target and target in d.columns:
                    d[f"{target}_lag_{lag}"] = d[target].shift(lag)
            d["PTS_momentum"] = pts_series - pts_series.rolling(5, min_periods=1).mean()

        # ─── Season progress ───
        d["season_pct"] = d["game_num"] / d["game_num"].max()

        # ─── Cumulative stats ───
        if pts_series is not None:
            d["cum_PTS_mean"] = pts_series.expanding().mean()
        if "FG_pct" in d.columns:
            d["cum_FG_mean"] = d["FG_pct"].expanding().mean()

        return d

    # ─── Fallback: Load Static CSV ────────────────────────────
    @staticmethod
    def load_static_csv(path: str = "nhd_euroleague.csv") -> pd.DataFrame:
        """Load the static NHD dataset as fallback."""
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            logger.info(f"Loaded static CSV: {path} ({len(df)} rows)")
            return df
        logger.error(f"Static CSV not found: {path}")
        return pd.DataFrame()

    # ─── Player Leaders ───────────────────────────────────────
    def get_player_leaders(self, stat_category: str = "Score",
                            top_n: int = 20) -> pd.DataFrame:
        """Fetch player stat leaders."""
        if not HAS_API:
            return pd.DataFrame()

        key = self._cache_key("leaders", stat=stat_category)
        cached = self._cache_get(key)
        if cached is not None:
            return cached.head(top_n)

        logger.info(f"Fetching player leaders: {stat_category}")
        try:
            df = self.player_stats.get_player_stats_leaders_single_season(
                season=self.season,
                stat_category=stat_category
            )
            self._cache_set(key, df)
            return df.head(top_n)
        except Exception as e:
            logger.error(f"Failed to fetch leaders: {e}")
            return pd.DataFrame()

    # ─── Summary Info ─────────────────────────────────────────
    def info(self):
        """Print pipeline configuration."""
        print("=" * 50)
        print("EUROLEAGUE DATA PIPELINE")
        print("=" * 50)
        print(f"Season       : {self.season}-{self.season + 1}")
        print(f"Competition  : {'Euroleague' if self.COMPETITION == 'E' else 'EuroCup'}")
        print(f"Cache dir    : {self.cache_dir}")
        print(f"Cache TTL    : {self.cache_ttl}")
        print(f"API available: {HAS_API}")
        print("=" * 50)


# ─── CLI Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Euroleague Data Pipeline")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--player", type=str, default="Hayes")
    parser.add_argument("--action", choices=["player", "team", "standings", "info"],
                        default="info")
    args = parser.parse_args()

    pipe = EuroleaguePipeline(season=args.season)

    if args.action == "info":
        pipe.info()
    elif args.action == "player":
        df = pipe.get_player_boxscore_season()
        if not df.empty:
            player = pipe.get_player_game_stats(args.player, df)
            print(f"\n{args.player} - {len(player)} games found")
            print(player.head())
    elif args.action == "team":
        df = pipe.get_team_season_stats()
        print(df.head(20))
    elif args.action == "standings":
        df = pipe.get_standings()
        print(df)
