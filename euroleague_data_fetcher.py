"""
Euroleague Data Fetcher
Fetch player statistics from Euroleague API and Basketball Reference

Usage:
    python euroleague_data_fetcher.py --player "Nigel Hayes-Davis" --season 2023-24
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

try:
    from euroleague_api.player_stats import PlayerStats
    from euroleague_api.game_stats import GameStats
    EUROLEAGUE_API_AVAILABLE = True
except ImportError:
    print("⚠️  euroleague-api not installed. Install with: pip install euroleague-api")
    EUROLEAGUE_API_AVAILABLE = False

def fetch_euroleague_player_data(player_name, season="E2023"):
    """
    Fetch player data from Euroleague API

    Args:
        player_name (str): Player name (e.g., "HAYES-DAVIS, NIGEL")
        season (str): Season code (e.g., "E2023" for 2023-24)

    Returns:
        pd.DataFrame: Player game-by-game statistics
    """
    if not EUROLEAGUE_API_AVAILABLE:
        print("❌ Euroleague API not available. Please install it first.")
        return None

    try:
        print(f"🔍 Fetching data for {player_name} - Season {season}...")

        # Initialize API
        player_stats = PlayerStats(season)

        # Get all players to find the player code
        all_players = player_stats.get_player_stats()

        # Search for player
        player_data = all_players[
            all_players['Player'].str.contains(player_name.upper(), case=False, na=False)
        ]

        if player_data.empty:
            print(f"❌ Player '{player_name}' not found in season {season}")
            print("Available players sample:")
            print(all_players['Player'].head(20))
            return None

        player_code = player_data.iloc[0]['Player_ID']
        print(f"✅ Found player: {player_data.iloc[0]['Player']} (ID: {player_code})")

        # Get game-by-game stats
        game_stats = player_stats.get_player_game_stats(player_code)

        print(f"✅ Retrieved {len(game_stats)} games")

        return game_stats

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def process_euroleague_data(df):
    """
    Process and clean Euroleague API data to match our analysis format

    Args:
        df (pd.DataFrame): Raw data from Euroleague API

    Returns:
        pd.DataFrame: Processed data
    """
    if df is None or df.empty:
        return None

    # Map Euroleague API columns to our format
    column_mapping = {
        'Minutes': 'MP',
        'FieldGoalsMade2': 'FGM2',
        'FieldGoalsAttempted2': 'FGA2',
        'FieldGoalsMade3': 'FGM3',
        'FieldGoalsAttempted3': 'FGA3',
        'FreeThrowsMade': 'FTM',
        'FreeThrowsAttempted': 'FTA',
        'TotalRebounds': 'REB',
        'Assists': 'AST',
        'Steals': 'STL',
        'Turnovers': 'TOV',
        'BlocksFavour': 'BLK',
        'FoulsCommited': 'PF',
        'ValuationRating': 'PIR',
        'Points': 'PTS'
    }

    # Select and rename columns
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df_processed = df[available_cols].copy()
    df_processed.rename(columns=column_mapping, inplace=True)

    # Calculate FG%
    if 'FGM2' in df_processed.columns and 'FGA2' in df_processed.columns:
        total_fgm = df_processed['FGM2'] + df_processed.get('FGM3', 0)
        total_fga = df_processed['FGA2'] + df_processed.get('FGA3', 0)
        df_processed['FG%'] = (total_fgm / total_fga).fillna(0)

    # Calculate Game Score (simplified version)
    # GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA - FTM) + 0.7*REB + 0.7*AST + STL + 0.7*BLK - 0.4*PF - TOV
    if all(col in df_processed.columns for col in ['PTS', 'REB', 'AST']):
        df_processed['GmSc'] = (
            df_processed['PTS'] +
            0.7 * df_processed.get('REB', 0) +
            0.7 * df_processed.get('AST', 0) +
            df_processed.get('STL', 0) +
            0.7 * df_processed.get('BLK', 0) -
            0.4 * df_processed.get('PF', 0) -
            df_processed.get('TOV', 0)
        )

    return df_processed

def create_sample_data():
    """
    Create sample data for demonstration if API is not available
    """
    print("📝 Creating sample data for demonstration...")

    # Enhanced version of the existing data with additional features
    data = {
        'MP': ['30:42', '33:22', '27:17', '27:53', '31:44', '40:00', '39:58', '27:45'],
        'FG%': [0.000, 0.400, 0.250, 0.500, 0.273, 0.500, 0.889, 0.500],
        'AST': [0, 2, 1, 4, 2, 4, 3, 1],
        'PTS': [0, 11, 7, 14, 9, 9, 23, 12],
        'GmSc': [-5.1, 8.0, 2.3, 10.1, 6.3, 11.3, 23.4, 7.2],
        'REB': [5, 7, 4, 6, 5, 8, 9, 5],  # Added rebounds
        'STL': [0, 1, 0, 2, 1, 1, 2, 0],  # Added steals
        'TOV': [2, 1, 1, 0, 2, 1, 0, 1],  # Added turnovers
    }

    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser(description='Fetch Euroleague player statistics')
    parser.add_argument('--player', type=str, default='HAYES-DAVIS',
                       help='Player last name (default: HAYES-DAVIS)')
    parser.add_argument('--season', type=str, default='E2023',
                       help='Season code (default: E2023 for 2023-24)')
    parser.add_argument('--output', type=str, default='euroleague_data_fetched.csv',
                       help='Output CSV filename')
    parser.add_argument('--sample', action='store_true',
                       help='Use sample data instead of API')

    args = parser.parse_args()

    print("=" * 70)
    print("EUROLEAGUE DATA FETCHER")
    print("=" * 70)

    if args.sample or not EUROLEAGUE_API_AVAILABLE:
        df = create_sample_data()
    else:
        df = fetch_euroleague_player_data(args.player, args.season)
        if df is not None:
            df = process_euroleague_data(df)

    if df is not None and not df.empty:
        # Save to CSV
        output_path = os.path.join(os.path.dirname(__file__), args.output)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Data saved to: {output_path}")
        print(f"📊 Total games: {len(df)}")
        print(f"📋 Columns: {', '.join(df.columns.tolist())}")
        print(f"\n{df.head()}")
    else:
        print("\n❌ No data retrieved")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
