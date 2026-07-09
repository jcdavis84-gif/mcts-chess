#!/usr/bin/env python3
"""
Convert expert chess games (PGN format) to replay buffer format for behavioral cloning.

Downloads games from PGNMentor or uses a local PGN file, then converts positions to training data:
- Board tensor: (18, 8, 8) encoding
- Policy: One-hot vector for expert move (not MCTS distribution)
- Value: Game outcome
- Material: Material balance

Creates separate train/eval splits.

Usage:
    # Download Kasparov games and create train/eval split
    python prepare_expert_games.py \\
        --player kasparov \\
        --train-games 200 \\
        --eval-games 20 \\
        --year-min 1980 \\
        --year-max 1990

    # Download from multiple players
    python prepare_expert_games.py \\
        --player kasparov karpov fischer \\
        --train-games 200 \\
        --eval-games 20

    # Or use local PGN file(s)
    python prepare_expert_games.py \\
        --pgn-file Kasparov.pgn Karpov.pgn Fischer.pgn \\
        --train-games 200 \\
        --eval-games 20

PGN Sources:
    - PGNMentor: https://www.pgnmentor.com/files.html#players
    - ChessGames: https://www.chessgames.com/
    - TWIC: https://theweekinchess.com/twic
"""

import argparse
import sys
import chess
import chess.pgn
import numpy as np
import requests
import io
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from board_encoder import BoardEncoder, calculate_material_balance
from action_converter import ActionConverter
from replay_buffer import ReplayBuffer


def download_pgnmentor_games(player_name: str):
    """
    Download games from PGNMentor.com.

    Args:
        player_name: Player name (e.g., 'kasparov', 'karpov', 'fischer')

    Returns:
        String containing PGN data, or None if download failed
    """
    # PGNMentor naming convention: lowercase, e.g., "kasparov.zip"
    player_lower = player_name.lower()
    url = f"https://www.pgnmentor.com/players/{player_lower}.zip"

    print(f"Attempting to download {player_name} games from PGNMentor...")
    print(f"URL: {url}")

    try:
        import zipfile
        import tempfile

        # Add headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Download zip file
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        # Extract PGN from zip
        with tempfile.TemporaryFile() as tmp:
            tmp.write(response.content)
            tmp.seek(0)

            with zipfile.ZipFile(tmp) as z:
                # Find PGN file in zip (usually player.pgn)
                pgn_files = [f for f in z.namelist() if f.endswith('.pgn')]

                if not pgn_files:
                    print("Error: No PGN file found in zip archive")
                    return None

                # Read first PGN file
                with z.open(pgn_files[0]) as f:
                    pgn_data = f.read().decode('utf-8', errors='ignore')
                    return pgn_data

    except requests.exceptions.RequestException as e:
        print(f"\nAutomatic download failed: {e}")
        print("\n" + "=" * 80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 80)
        print("\nPGNMentor blocks automated downloads. Please download manually:")
        print()
        print(f"1. Visit: https://www.pgnmentor.com/files.html#players")
        print(f"2. Find '{player_name.title()}' in the list")
        print(f"3. Click to download the ZIP file")
        print(f"4. Extract the PGN file")
        print(f"5. Run this script again with: --pgn-file {player_lower}.pgn")
        print()
        print("=" * 80)
        return None
    except Exception as e:
        print(f"Error processing download: {e}")
        return None


def parse_pgn_file(pgn_path: str):
    """Parse games from a PGN file."""
    games = []

    print(f"Reading PGN file: {pgn_path}")

    with open(pgn_path, encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)

    print(f"Found {len(games)} games")
    return games


def stream_pgn_file(pgn_path: str):
    """Stream games from a PGN file one at a time (memory efficient)."""
    with open(pgn_path, encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield game


def parse_pgn_string(pgn_string: str):
    """Parse games from a PGN string."""
    games = []
    pgn_io = io.StringIO(pgn_string)

    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)

    return games


def stream_pgn_string(pgn_string: str):
    """Stream games from a PGN string one at a time (memory efficient)."""
    pgn_io = io.StringIO(pgn_string)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        yield game


def filter_games_by_year(games, year_min=None, year_max=None):
    """Filter games by year range."""
    if year_min is None and year_max is None:
        return games

    filtered = []
    for game in games:
        date = game.headers.get("Date", "????.??.??")
        try:
            year = int(date.split(".")[0])
            if year_min and year < year_min:
                continue
            if year_max and year > year_max:
                continue
            filtered.append(game)
        except (ValueError, IndexError):
            # Can't parse year - skip this game
            continue

    return filtered


def stream_filter_games_by_year(game_iterator, year_min=None, year_max=None):
    """Filter streamed games by year range (memory efficient)."""
    for game in game_iterator:
        if year_min is None and year_max is None:
            yield game
            continue

        date = game.headers.get("Date", "????.??.??")
        try:
            year = int(date.split(".")[0])
            if year_min and year < year_min:
                continue
            if year_max and year > year_max:
                continue
            yield game
        except (ValueError, IndexError):
            # Can't parse year - skip this game
            continue


def convert_game_to_training_data(game, encoder, converter):
    """
    Convert a single game to training data.

    For each position in the game:
    - Encodes board state (our encoder handles chess.Board objects)
    - Creates one-hot policy for the expert move
    - Assigns game outcome as value
    - Calculates material balance

    Returns:
        List of (board_tensor, policy_one_hot, value, material) tuples
    """
    result = game.headers.get("Result", "*")

    if result == "1-0":
        white_outcome = 1.0
    elif result == "0-1":
        white_outcome = -1.0
    elif result == "1/2-1/2":
        white_outcome = 0.0
    else:
        # Unknown/unfinished game - skip
        return []

    training_data = []
    board = game.board()

    for move in game.mainline_moves():
        # Encode current position
        board_tensor = encoder.encode(board)

        # Create one-hot policy vector for expert move
        try:
            action_idx = converter.move_to_action(move, board)
            policy_one_hot = np.zeros(4672, dtype=np.float32)
            policy_one_hot[action_idx] = 1.0
        except (ValueError, AssertionError):
            # Invalid move conversion - skip position
            board.push(move)
            continue

        # Calculate material balance
        material = calculate_material_balance(board)

        # Value from current player's perspective
        if board.turn == chess.WHITE:
            value = white_outcome
        else:
            value = -white_outcome

        training_data.append((board_tensor, policy_one_hot, value, material))

        # Make the move
        board.push(move)

    return training_data


def create_replay_buffer_from_games(games, encoder, converter, buffer_capacity, split_name):
    """Convert games to replay buffer."""
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    total_positions = 0

    print(f"\nProcessing {len(games)} games for {split_name} split...")

    for i, game in enumerate(games, 1):
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        result = game.headers.get("Result", "*")
        date = game.headers.get("Date", "????.??.??")

        game_data = convert_game_to_training_data(game, encoder, converter)

        if not game_data:
            continue

        for board_tensor, policy, value, material in game_data:
            replay_buffer.add(board_tensor, policy, value, material)

        total_positions += len(game_data)

        if i % 10 == 0 or i == len(games):
            print(f"  {i}/{len(games)} games | {total_positions} positions")

    return replay_buffer, total_positions


def create_replay_buffer_from_stream(game_iterator, encoder, converter, buffer_capacity, split_name, max_games):
    """Convert streamed games to replay buffer (memory efficient)."""
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    total_positions = 0
    games_processed = 0

    print(f"\nProcessing up to {max_games} games for {split_name} split (streaming mode)...")

    for game in game_iterator:
        if games_processed >= max_games:
            break

        games_processed += 1

        game_data = convert_game_to_training_data(game, encoder, converter)

        if not game_data:
            continue

        for board_tensor, policy, value, material in game_data:
            replay_buffer.add(board_tensor, policy, value, material)

        total_positions += len(game_data)

        if games_processed % 10 == 0 or games_processed == max_games:
            print(f"  {games_processed}/{max_games} games | {total_positions} positions")

    return replay_buffer, total_positions


def show_buffer_stats(replay_buffer, split_name):
    """Display statistics for a replay buffer."""
    if len(replay_buffer) == 0:
        return

    all_values = np.array([entry[2] for entry in replay_buffer.buffer])
    all_materials = np.array([entry[3] for entry in replay_buffer.buffer])

    print(f"\n{split_name} Split Statistics:")
    print(f"  Total positions: {len(replay_buffer)}")
    print()

    print("  Value distribution:")
    white_wins = np.sum(all_values > 0.5)
    draws = np.sum(np.abs(all_values) < 0.5)
    black_wins = np.sum(all_values < -0.5)
    print(f"    White wins: {white_wins} ({white_wins/len(all_values)*100:.1f}%)")
    print(f"    Draws:      {draws} ({draws/len(all_values)*100:.1f}%)")
    print(f"    Black wins: {black_wins} ({black_wins/len(all_values)*100:.1f}%)")
    print()

    print("  Material balance:")
    print(f"    Mean:   {all_materials.mean():+.2f}")
    print(f"    Median: {np.median(all_materials):+.2f}")
    print(f"    Range:  [{all_materials.min():+.0f}, {all_materials.max():+.0f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Convert expert PGN games to replay buffer for behavioral cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Kasparov 1980-1990 games from PGNMentor
  python prepare_expert_games.py --player kasparov --year-min 1980 --year-max 1990

  # Download from multiple players
  python prepare_expert_games.py --player kasparov karpov fischer --train-games 200 --eval-games 20

  # Use multiple local PGN files
  python prepare_expert_games.py --pgn-file Kasparov.pgn Karpov.pgn Fischer.pgn --train-games 500 --eval-games 50

PGN Sources:
  - PGNMentor: https://www.pgnmentor.com/files.html#players
  - ChessGames: https://www.chessgames.com/
  - TWIC: https://theweekinchess.com/twic
        """
    )

    # Input source
    parser.add_argument("--player", type=str, nargs='+', help="Player name(s) to download from PGNMentor (e.g., 'kasparov' or 'kasparov karpov fischer')")
    parser.add_argument("--pgn-file", type=str, nargs='+', help="Path(s) to local PGN file(s) (can specify multiple)")

    # Train/eval split
    parser.add_argument("--train-games", type=int, default=200, help="Number of games for training split (default: 200)")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of games for eval split (default: 20)")

    # Filtering
    parser.add_argument("--year-min", type=int, help="Minimum year (e.g., 1980)")
    parser.add_argument("--year-max", type=int, help="Maximum year (e.g., 1990)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./", help="Output directory (default: ./)")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix (default: 'expert' or player names)")

    # Memory efficiency
    parser.add_argument("--stream", action="store_true", help="Stream games one-at-a-time (low memory mode for large PGN files)")

    args = parser.parse_args()

    if not args.player and not args.pgn_file:
        parser.error("Must specify either --player or --pgn-file")

    print("=" * 80)
    print("Expert Games → Replay Buffer Converter")
    print("Behavioral Cloning / Supervised Learning")
    print("=" * 80)
    print()

    # Initialize encoder/converter
    print("Initializing components...")
    encoder = BoardEncoder()
    converter = ActionConverter()
    print("✓ BoardEncoder and ActionConverter ready")
    print()

    # STREAMING MODE (memory efficient)
    if args.stream:
        print("=" * 80)
        print("Streaming Mode (Memory Efficient)")
        print("=" * 80)
        print(f"Mode: Stream games one-at-a-time")
        print(f"Memory usage: ~constant (doesn't load all games)")
        print()

        # Convert train set (first pass through PGN)
        print("=" * 80)
        print("Converting Training Games")
        print("=" * 80)

        if args.pgn_file:
            # Stream from multiple PGN files
            def train_game_generator():
                for pgn_path in args.pgn_file:
                    yield from stream_filter_games_by_year(
                        stream_pgn_file(pgn_path),
                        args.year_min,
                        args.year_max
                    )
            train_buffer, train_positions = create_replay_buffer_from_stream(
                train_game_generator(), encoder, converter,
                buffer_capacity=1000000, split_name="Training", max_games=args.train_games
            )
        else:
            # Download and stream from players
            def train_game_generator():
                for player_name in args.player:
                    print(f"\nDownloading {player_name}...")
                    pgn_string = download_pgnmentor_games(player_name)
                    if pgn_string:
                        yield from stream_filter_games_by_year(
                            stream_pgn_string(pgn_string),
                            args.year_min,
                            args.year_max
                        )
            train_buffer, train_positions = create_replay_buffer_from_stream(
                train_game_generator(), encoder, converter,
                buffer_capacity=1000000, split_name="Training", max_games=args.train_games
            )

        # Convert eval set (second pass through PGN)
        print()
        print("=" * 80)
        print("Converting Eval Games")
        print("=" * 80)

        if args.pgn_file:
            # Stream from multiple PGN files, skip first N games
            def eval_game_generator():
                games_seen = 0
                for pgn_path in args.pgn_file:
                    for game in stream_filter_games_by_year(
                        stream_pgn_file(pgn_path),
                        args.year_min,
                        args.year_max
                    ):
                        if games_seen < args.train_games:
                            games_seen += 1
                            continue
                        yield game

            eval_buffer, eval_positions = create_replay_buffer_from_stream(
                eval_game_generator(), encoder, converter,
                buffer_capacity=100000, split_name="Eval", max_games=args.eval_games
            )
        else:
            # Download and stream from players, skip first N games
            def eval_game_generator():
                games_seen = 0
                for player_name in args.player:
                    print(f"\nDownloading {player_name}...")
                    pgn_string = download_pgnmentor_games(player_name)
                    if pgn_string:
                        for game in stream_filter_games_by_year(
                            stream_pgn_string(pgn_string),
                            args.year_min,
                            args.year_max
                        ):
                            if games_seen < args.train_games:
                                games_seen += 1
                                continue
                            yield game

            eval_buffer, eval_positions = create_replay_buffer_from_stream(
                eval_game_generator(), encoder, converter,
                buffer_capacity=100000, split_name="Eval", max_games=args.eval_games
            )

    # NORMAL MODE (load all into memory)
    else:
        print("=" * 80)
        print("Loading Games")
        print("=" * 80)

        games = []

        if args.pgn_file:
            # Load from multiple PGN files
            for pgn_path in args.pgn_file:
                print(f"\nLoading {pgn_path}...")
                file_games = parse_pgn_file(pgn_path)
                print(f"✓ Loaded {len(file_games)} games from {pgn_path}")
                games.extend(file_games)
        else:
            # Download games from multiple players
            for player_name in args.player:
                print(f"\nDownloading {player_name}...")
                pgn_string = download_pgnmentor_games(player_name)
                if not pgn_string:
                    print(f"⚠ Skipping {player_name} (download failed)")
                    continue

                player_games = parse_pgn_string(pgn_string)
                print(f"✓ Loaded {len(player_games)} games from {player_name}")
                games.extend(player_games)

        if not games:
            print("Error: No games found")
            return 1

        print(f"\n✓ Total games loaded: {len(games)}")

        # Filter by year
        if args.year_min or args.year_max:
            print(f"\nFiltering: {args.year_min or 'any'} - {args.year_max or 'any'}")
            games = filter_games_by_year(games, args.year_min, args.year_max)
            print(f"✓ {len(games)} games after filtering")

        # Create train/eval split
        total_needed = args.train_games + args.eval_games

        if len(games) < total_needed:
            print(f"\nWarning: Only {len(games)} games available, need {total_needed}")
            print(f"Adjusting split proportionally...")
            ratio = len(games) / total_needed
            train_games = int(args.train_games * ratio)
            eval_games = len(games) - train_games
        else:
            train_games = args.train_games
            eval_games = args.eval_games

        train_set = games[:train_games]
        eval_set = games[train_games:train_games + eval_games]

        print()
        print("=" * 80)
        print("Train/Eval Split")
        print("=" * 80)
        print(f"Training games: {len(train_set)}")
        print(f"Eval games:     {len(eval_set)}")

        # Convert train set
        print()
        print("=" * 80)
        print("Converting Training Games")
        print("=" * 80)
        train_buffer, train_positions = create_replay_buffer_from_games(
            train_set, encoder, converter, buffer_capacity=1000000, split_name="Training"
        )

        # Convert eval set
        print()
        print("=" * 80)
        print("Converting Eval Games")
        print("=" * 80)
        eval_buffer, eval_positions = create_replay_buffer_from_games(
            eval_set, encoder, converter, buffer_capacity=100000, split_name="Eval"
        )

    # Show statistics
    print()
    print("=" * 80)
    print("Statistics")
    print("=" * 80)
    show_buffer_stats(train_buffer, "Training")
    show_buffer_stats(eval_buffer, "Eval")

    # Save buffers
    print()
    print("=" * 80)
    print("Saving Replay Buffers")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate prefix if not provided
    if args.prefix is None:
        if args.player:
            if len(args.player) == 1:
                prefix = args.player[0].lower()
            else:
                prefix = "_".join(args.player).lower()
        elif args.pgn_file:
            # Use PGN filenames (without extension)
            basenames = [Path(f).stem.lower() for f in args.pgn_file]
            if len(basenames) == 1:
                prefix = basenames[0]
            else:
                prefix = "_".join(basenames)
        else:
            prefix = "expert"
    else:
        prefix = args.prefix

    train_path = output_dir / f"{prefix}_train.pkl"
    eval_path = output_dir / f"{prefix}_eval.pkl"

    train_buffer.save(str(train_path))
    eval_buffer.save(str(eval_path))

    train_size_mb = os.path.getsize(train_path) / (1024 * 1024)
    eval_size_mb = os.path.getsize(eval_path) / (1024 * 1024)

    print(f"✓ Training buffer: {train_path} ({train_size_mb:.2f} MB)")
    print(f"✓ Eval buffer:     {eval_path} ({eval_size_mb:.2f} MB)")

    # Next steps
    print()
    print("=" * 80)
    print("Next Steps: Behavioral Cloning")
    print("=" * 80)
    print()
    print("Train with supervised learning:")
    print(f"  python train.py \\")
    print(f"      --load-buffer {train_path} \\")
    print(f"      --cycles 10 \\")
    print(f"      --train-steps-per-cycle 200 \\")
    print(f"      --batch-size 256 \\")
    print(f"      --learning-rate 1e-3")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
