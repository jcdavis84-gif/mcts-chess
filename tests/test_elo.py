#!/usr/bin/env python3
"""
ELO Rating Evaluation Script

Tests an AlphaZero checkpoint against Stockfish at various skill levels
to estimate the model's ELO rating.

Usage:
    python test_elo.py --checkpoint checkpoints/model_final.pt --simulations 400 --skill-level 5
    python test_elo.py --checkpoint checkpoints/model_final.pt --simulations 800 --skill-level 10 --num-games 100
"""

import argparse
import chess
import chess.engine
import torch
import numpy as np
import time
from pathlib import Path
from model import AlphaZeroNetwork
from board_encoder import BoardEncoder
from action_converter import ActionConverter
from mcts import MCTS


# Stockfish skill level to approximate ELO mapping
# Based on: https://github.com/official-stockfish/Stockfish/wiki/UCI-&-Commands
STOCKFISH_ELO = {
    0: 1350,
    1: 1450,
    2: 1550,
    3: 1650,
    4: 1750,
    5: 1850,
    6: 1950,
    7: 2050,
    8: 2150,
    9: 2250,
    10: 2350,
    11: 2450,
    12: 2550,
    13: 2650,
    14: 2750,
    15: 2850,
    16: 2950,
    17: 3050,
    18: 3150,
    19: 3250,
    20: 3350,
}


def estimate_elo(opponent_elo: int, wins: int, losses: int, draws: int) -> tuple:
    """
    Estimate ELO rating based on match results using the ELO formula.

    Args:
        opponent_elo: ELO rating of the opponent
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws

    Returns:
        (estimated_elo, margin_of_error): Estimated ELO and 95% confidence interval
    """
    total_games = wins + losses + draws
    if total_games == 0:
        return opponent_elo, 0

    score = (wins + 0.5 * draws) / total_games

    # ELO difference formula: Diff = 400 * log10(score / (1 - score))
    if score == 0:
        elo_diff = -800  # Crushed
    elif score == 1:
        elo_diff = 800  # Perfect score
    else:
        import math
        elo_diff = 400 * math.log10(score / (1 - score))

    estimated_elo = opponent_elo + elo_diff

    # Calculate margin of error (95% confidence interval)
    # Standard error: sqrt(p * (1-p) / n) where p is win probability
    if total_games > 1:
        std_error = np.sqrt(score * (1 - score) / total_games)
        # Convert score error to ELO error (approximate derivative at score)
        if 0.01 < score < 0.99:
            elo_error = 400 * 1.96 * std_error / (score * (1 - score) * np.log(10))
        else:
            elo_error = 200  # Large uncertainty at extremes
    else:
        elo_error = 400

    return estimated_elo, elo_error


def load_model(checkpoint_path: str, device: str = "cpu") -> AlphaZeroNetwork:
    """
    Load AlphaZero model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded AlphaZeroNetwork
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters (use defaults for backward compatibility)
    if "model_state_dict" in checkpoint:
        # New format with hyperparameters
        d_model = checkpoint.get("d_model", 128)
        num_blocks = checkpoint.get("num_blocks", 4)
        num_heads = checkpoint.get("num_heads", 2)
        d_policy = checkpoint.get("d_policy", 64)
        state_dict = checkpoint["model_state_dict"]
    else:
        # Old format (raw state_dict) - use defaults
        d_model = 128
        num_blocks = 4
        num_heads = 2
        d_policy = 64
        state_dict = checkpoint

    # Create model with extracted/default hyperparameters
    model = AlphaZeroNetwork(
        d_model=d_model,
        num_blocks=num_blocks,
        num_heads=num_heads,
        d_policy=d_policy,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model architecture: d_model={d_model}, num_blocks={num_blocks}, num_heads={num_heads}, d_policy={d_policy}")

    return model


def play_game(
    mcts: MCTS,
    encoder: BoardEncoder,
    converter: ActionConverter,
    stockfish_engine: chess.engine.SimpleEngine,
    alphazero_is_white: bool,
    time_limit: float = 1.0,
    node_limit: int = None,
    verbose: bool = False,
) -> int:
    """
    Play a single game between AlphaZero and Stockfish.

    Args:
        mcts: MCTS instance for AlphaZero
        encoder: Board encoder
        converter: Action converter
        stockfish_engine: Stockfish engine
        alphazero_is_white: If True, AlphaZero plays as White
        time_limit: Time limit for Stockfish (seconds per move)
        node_limit: If set, limit Stockfish to this many nodes per move
            (overrides time_limit; very low values make Stockfish much weaker)
        verbose: If True, print moves

    Returns:
        result: 1 for AlphaZero win, 0 for draw, -1 for Stockfish win
    """
    board = chess.Board()
    mcts.reset_tree()

    if verbose:
        print(f"\n{'='*60}")
        print(f"AlphaZero: {'White' if alphazero_is_white else 'Black'}")
        print(f"{'='*60}\n")

    move_num = 0
    while not board.is_game_over():
        move_num += 1

        if (board.turn == chess.WHITE) == alphazero_is_white:
            # AlphaZero's turn
            if verbose:
                print(f"Move {move_num} - AlphaZero thinking...", end=" ", flush=True)

            start_time = time.time()
            policy = mcts.search(board)
            action_idx = mcts.select_action(policy=policy)
            move = converter.action_to_move(action_idx, board)
            elapsed = time.time() - start_time

            if verbose:
                print(f"{board.san(move)} ({elapsed:.1f}s)")
        else:
            # Stockfish's turn
            if verbose:
                print(f"Move {move_num} - Stockfish thinking...", end=" ", flush=True)

            start_time = time.time()
            if node_limit is not None:
                limit = chess.engine.Limit(nodes=node_limit)
            else:
                limit = chess.engine.Limit(time=time_limit)
            result = stockfish_engine.play(board, limit)
            move = result.move
            elapsed = time.time() - start_time

            if verbose:
                print(f"{board.san(move)} ({elapsed:.1f}s)")

        board.push(move)

    # Determine result
    outcome = board.outcome()
    if outcome.winner is None:
        result = 0  # Draw
        result_str = "Draw"
    elif (outcome.winner == chess.WHITE) == alphazero_is_white:
        result = 1  # AlphaZero win
        result_str = "AlphaZero wins"
    else:
        result = -1  # Stockfish win
        result_str = "Stockfish wins"

    if verbose:
        print(f"\nResult: {result_str}")
        print(f"Reason: {outcome.termination.name}")
        print(f"Moves: {move_num}\n")

    return result


def run_match(
    checkpoint_path: str,
    stockfish_path: str,
    skill_level: int,
    num_games: int,
    num_simulations: int,
    stockfish_time: float = 1.0,
    stockfish_nodes: int = None,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """
    Run a match between AlphaZero and Stockfish.

    Args:
        checkpoint_path: Path to AlphaZero checkpoint
        stockfish_path: Path to Stockfish binary
        skill_level: Stockfish skill level (0-20)
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        stockfish_time: Time limit for Stockfish (seconds per move)
        stockfish_nodes: If set, limit Stockfish to this many nodes per move
            (overrides stockfish_time; Elo mapping no longer applies)
        device: Device to run on
        verbose: If True, print detailed game info

    Returns:
        results: Dictionary with match statistics
    """
    print(f"\n{'='*70}")
    print(f"Loading AlphaZero checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    # Load model
    model = load_model(checkpoint_path, device)
    encoder = BoardEncoder()
    converter = ActionConverter()

    # Create MCTS instance
    mcts = MCTS(
        model=model,
        encoder=encoder,
        converter=converter,
        num_simulations=num_simulations,
        c_puct=1.5,
        temperature=0.0,  # Deterministic (argmax) for evaluation
        add_dirichlet_noise=False,
    )

    # Initialize Stockfish
    if stockfish_nodes is not None:
        print(f"Starting Stockfish (skill level {skill_level}, {stockfish_nodes} nodes/move — Elo mapping does not apply)")
    else:
        print(f"Starting Stockfish (skill level {skill_level}, ~{STOCKFISH_ELO[skill_level]} ELO)")
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    stockfish.configure({"Skill Level": skill_level})

    # Play games
    print(f"Playing {num_games} games ({num_simulations} simulations/move)...\n")

    wins = 0
    draws = 0
    losses = 0

    start_time = time.time()

    for game_num in range(num_games):
        alphazero_is_white = (game_num % 2 == 0)  # Alternate colors

        if not verbose:
            print(f"Game {game_num + 1}/{num_games}...", end=" ", flush=True)

        result = play_game(
            mcts=mcts,
            encoder=encoder,
            converter=converter,
            stockfish_engine=stockfish,
            alphazero_is_white=alphazero_is_white,
            time_limit=stockfish_time,
            node_limit=stockfish_nodes,
            verbose=verbose,
        )

        if result == 1:
            wins += 1
            result_str = "WIN"
        elif result == 0:
            draws += 1
            result_str = "DRAW"
        else:
            losses += 1
            result_str = "LOSS"

        if not verbose:
            print(f"{result_str} (W:{wins} D:{draws} L:{losses})")

    elapsed_time = time.time() - start_time

    # Close engine
    stockfish.quit()

    # Calculate ELO estimate
    opponent_elo = STOCKFISH_ELO[skill_level]
    estimated_elo, margin = estimate_elo(opponent_elo, wins, losses, draws)

    # Print summary
    print(f"\n{'='*70}")
    print(f"MATCH RESULTS vs Stockfish Level {skill_level} ({opponent_elo} ELO)")
    print(f"{'='*70}")
    print(f"Games:      {num_games}")
    print(f"Wins:       {wins} ({100*wins/num_games:.1f}%)")
    print(f"Draws:      {draws} ({100*draws/num_games:.1f}%)")
    print(f"Losses:     {losses} ({100*losses/num_games:.1f}%)")
    print(f"Score:      {wins + 0.5*draws:.1f}/{num_games} ({100*(wins + 0.5*draws)/num_games:.1f}%)")
    print(f"\nEstimated AlphaZero ELO: {estimated_elo:.0f} ± {margin:.0f}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"{'='*70}\n")

    return {
        "skill_level": skill_level,
        "opponent_elo": opponent_elo,
        "num_games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": wins + 0.5 * draws,
        "estimated_elo": estimated_elo,
        "margin_of_error": margin,
        "time": elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test AlphaZero ELO rating against Stockfish",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (200 sims, 20 games)
  python test_elo.py --checkpoint checkpoints/model_final.pt --simulations 200 --skill-level 5 --num-games 20

  # Standard evaluation (400 sims, 50 games)
  python test_elo.py --checkpoint checkpoints/model_final.pt --simulations 400 --skill-level 5

  # Strong play (800 sims, 100 games)
  python test_elo.py --checkpoint checkpoints/model_final.pt --simulations 800 --skill-level 10 --num-games 100

  # Test against beginner
  python test_elo.py --checkpoint checkpoints/model_final.pt --simulations 400 --skill-level 1
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to AlphaZero checkpoint file (.pt)",
    )

    parser.add_argument(
        "--simulations",
        type=int,
        required=True,
        help="Number of MCTS simulations per move (100=quick, 400=standard, 800=strong)",
    )

    parser.add_argument(
        "--skill-level",
        type=int,
        default=5,
        choices=range(0, 21),
        help="Stockfish skill level (0-20). Level 5 ≈ 1850 ELO, Level 10 ≈ 2350 ELO",
    )

    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games to play (default: 50, recommend 50+ for accuracy)",
    )

    parser.add_argument(
        "--stockfish",
        type=str,
        default="stockfish",
        help="Path to Stockfish binary (default: 'stockfish' from PATH)",
    )

    parser.add_argument(
        "--stockfish-time",
        type=float,
        default=1.0,
        help="Time limit for Stockfish per move in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--stockfish-nodes",
        type=int,
        default=None,
        help="Node limit for Stockfish per move (overrides --stockfish-time). "
        "Very low values (e.g. 50-500) make Stockfish far weaker than skill level 0; "
        "the skill-level Elo mapping does not apply when this is set.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on (default: cuda if available, else cpu)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game information",
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    # Run match
    results = run_match(
        checkpoint_path=args.checkpoint,
        stockfish_path=args.stockfish,
        skill_level=args.skill_level,
        num_games=args.num_games,
        num_simulations=args.simulations,
        stockfish_time=args.stockfish_time,
        stockfish_nodes=args.stockfish_nodes,
        device=args.device,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    exit(main())
