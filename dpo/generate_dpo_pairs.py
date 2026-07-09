#!/usr/bin/env python3
"""
Generate DPO preference pairs from master games using Stockfish evaluation.

For each position in master games:
1. Get top-k moves from policy head (no MCTS)
2. Evaluate each move with Stockfish
3. If best and worst differ by >= threshold, create preference pair:
   - Preferred: (position, best_move)
   - Rejected: (position, worst_move)

Usage:
    # Generate from 100 master games, top-5 policy moves, 0.5 pawn threshold
    python generate_dpo_pairs.py \
        --pgn-file games.pgn \
        --num-games 100 \
        --model model_latest.pt \
        --top-k 5 \
        --threshold 0.5 \
        --output dpo_pairs.pkl

    # Use Stockfish at specific path
    python generate_dpo_pairs.py \
        --pgn-file games.pgn \
        --stockfish-path /usr/local/bin/stockfish \
        --depth 15
"""

import argparse
import chess
import chess.pgn
import chess.engine
import numpy as np
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import AlphaZeroNetwork
from board_encoder import BoardEncoder
from action_converter import ActionConverter


class StockfishEvaluator:
    """Wrapper for Stockfish evaluation."""

    def __init__(self, stockfish_path="/opt/homebrew/bin/stockfish", depth=15, time_limit=0.1):
        """
        Args:
            stockfish_path: Path to Stockfish binary
            depth: Search depth (15 is ~instant, 20 is good balance)
            time_limit: Time limit in seconds (backup to depth)
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.depth = depth
        self.time_limit = time_limit

    def evaluate_move(self, board, move):
        """
        Evaluate a move from current position.

        Returns:
            Centipawn evaluation after the move (from side-to-move perspective)
            Positive = good for side to move, negative = bad
        """
        # Make the move
        board_copy = board.copy()
        board_copy.push(move)

        # Evaluate resulting position
        info = self.engine.analyse(
            board_copy,
            chess.engine.Limit(depth=self.depth, time=self.time_limit)
        )

        # Score from the mover's perspective. info["score"].relative would be
        # from the side to move in the *resulting* position (the opponent),
        # which inverts the preference labels.
        score = info["score"].pov(board.turn)

        # Convert to centipawns (handle mate scores)
        if score.is_mate():
            # Mate in N moves - assign large value
            mate_in = score.mate()
            if mate_in > 0:
                cp = 10000 - mate_in * 100  # Favor faster mates
            else:
                cp = -10000 - mate_in * 100
        else:
            cp = score.score()

        return cp

    def close(self):
        """Clean up engine."""
        self.engine.quit()


def load_games_from_pgn(pgn_path, num_games, skip_games=0):
    """Load games from PGN file, optionally skipping the first N decisive games."""
    games = []
    skipped = 0

    print(f"Loading games from {pgn_path}..." + (f" (skipping first {skip_games} decisive games)" if skip_games else ""))
    with open(pgn_path, encoding='utf-8', errors='ignore') as f:
        while len(games) < num_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            # Only use decisive games (better signal)
            result = game.headers.get("Result", "*")
            if result in ["1-0", "0-1"]:
                if skipped < skip_games:
                    skipped += 1
                    continue
                games.append(game)

    print(f"✓ Loaded {len(games)} games")
    return games


def get_top_k_policy_moves(model, board, encoder, converter, k=5):
    """
    Get top-k moves from policy head (no MCTS).

    Returns:
        List of (move, log_prob) tuples, sorted by probability (highest first)
    """
    # Encode board
    board_tensor = encoder.encode(board)

    # Get legal moves mask
    legal_moves = list(board.legal_moves)
    legal_mask = np.zeros(4672, dtype=bool)
    for move in legal_moves:
        try:
            action_idx = converter.move_to_action(move, board)
            legal_mask[action_idx] = True
        except (ValueError, AssertionError):
            continue

    # Get policy distribution
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device

        board_tensor = torch.from_numpy(board_tensor).float().unsqueeze(0).to(device)
        legal_mask_tensor = torch.from_numpy(legal_mask).bool().unsqueeze(0).to(device)

        policy_logits, _, _ = model(board_tensor, legal_mask_tensor)
        log_probs = torch.log_softmax(policy_logits, dim=-1)[0]  # (4672,)

    # Get top-k actions
    top_k_values, top_k_indices = torch.topk(log_probs, min(k, len(legal_moves)))

    # Convert to moves
    top_moves = []
    for idx, log_prob in zip(top_k_indices, top_k_values):
        try:
            move = converter.action_to_move(idx.item(), board)
            if move in legal_moves:
                top_moves.append((move, log_prob.item()))
        except (ValueError, AssertionError):
            continue

    return top_moves


def generate_preference_pairs(games, model, encoder, converter, evaluator, top_k=5, threshold=0.5):
    """
    Generate preference pairs from games.

    Returns:
        List of preference pairs: [
            {
                'board_tensor': (18, 8, 8),
                'preferred_action': int,
                'rejected_action': int,
                'preferred_eval': float,
                'rejected_eval': float,
            },
            ...
        ]
    """
    pairs = []
    total_positions = 0

    print(f"\nGenerating preference pairs from {len(games)} games...")
    print(f"Settings: top-{top_k} policy moves, threshold={threshold} pawns")

    for game_idx, game in enumerate(tqdm(games, desc="Processing games")):
        board = game.board()

        for move in game.mainline_moves():
            total_positions += 1

            # Get top-k moves from policy
            try:
                top_moves = get_top_k_policy_moves(model, board, encoder, converter, k=top_k)
            except Exception as e:
                board.push(move)
                continue

            if len(top_moves) < 2:
                board.push(move)
                continue

            # Evaluate each move with Stockfish
            evaluations = []
            for chess_move, log_prob in top_moves:
                try:
                    cp_eval = evaluator.evaluate_move(board, chess_move)
                    action_idx = converter.move_to_action(chess_move, board)
                    evaluations.append((chess_move, action_idx, log_prob, cp_eval))
                except Exception as e:
                    continue

            if len(evaluations) < 2:
                board.push(move)
                continue

            # Find best and worst moves
            evaluations.sort(key=lambda x: x[3], reverse=True)  # Sort by eval (descending)
            best_move, best_action, best_log_prob, best_eval = evaluations[0]
            worst_move, worst_action, worst_log_prob, worst_eval = evaluations[-1]

            # Check if difference meets threshold
            eval_diff = (best_eval - worst_eval) / 100.0  # Convert centipawns to pawns

            if eval_diff >= threshold:
                # Create preference pair
                board_tensor = encoder.encode(board)

                pair = {
                    'board_tensor': board_tensor,
                    'fen': board.fen(),  # Store FEN for easy reconstruction
                    'preferred_action': best_action,
                    'rejected_action': worst_action,
                    'preferred_log_prob': best_log_prob,
                    'rejected_log_prob': worst_log_prob,
                    'preferred_eval': best_eval / 100.0,  # Convert to pawns
                    'rejected_eval': worst_eval / 100.0,
                    'eval_diff': eval_diff,
                }

                pairs.append(pair)

            # Make the actual game move
            board.push(move)

    print(f"\n✓ Generated {len(pairs)} preference pairs from {total_positions} positions")
    print(f"  Acceptance rate: {len(pairs)/total_positions*100:.2f}%")

    if pairs:
        eval_diffs = [p['eval_diff'] for p in pairs]
        print(f"  Eval difference: mean={np.mean(eval_diffs):.2f}, median={np.median(eval_diffs):.2f}, max={np.max(eval_diffs):.2f}")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs from master games",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input
    parser.add_argument("--pgn-file", type=str, required=True, help="Path to PGN file with master games")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to process (default: 100)")
    parser.add_argument("--skip-games", type=int, default=0, help="Skip the first N decisive games (for chunked/parallel generation)")
    parser.add_argument("--model", type=str, default="model_latest.pt", help="Path to model checkpoint (default: model_latest.pt)")

    # Policy settings
    parser.add_argument("--top-k", type=int, default=5, help="Number of top policy moves to evaluate (default: 5)")

    # Stockfish settings
    parser.add_argument("--stockfish-path", type=str, default="/opt/homebrew/bin/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish search depth (default: 15)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum eval difference in pawns to create pair (default: 0.5)")

    # Output
    parser.add_argument("--output", type=str, default="dpo_pairs.pkl", help="Output path for preference pairs (default: dpo_pairs.pkl)")

    args = parser.parse_args()

    print("=" * 80)
    print("DPO Preference Pair Generator")
    print("=" * 80)
    print()

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        # Detect architecture from state dict
        state_dict = checkpoint['model_state_dict']
        d_model = state_dict['token_proj.weight'].shape[0]

        # Detect number of blocks by finding max block index
        block_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.')]
        num_blocks = max(block_indices) + 1 if block_indices else 4

        num_heads = 2  # Default
        d_policy = state_dict['policy_proj.weight'].shape[0]

        model = AlphaZeroNetwork(
            d_model=d_model,
            num_blocks=num_blocks,
            num_heads=num_heads,
            d_policy=d_policy
        )
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Checkpoint format not recognized")

    model.eval()
    print(f"✓ Model loaded (d_model={d_model}, blocks={num_blocks}, d_policy={d_policy})")

    # Initialize components
    print("\nInitializing components...")
    encoder = BoardEncoder()
    converter = ActionConverter()
    print("✓ BoardEncoder and ActionConverter ready")

    # Initialize Stockfish
    print(f"\nInitializing Stockfish (depth={args.depth})...")
    evaluator = StockfishEvaluator(
        stockfish_path=args.stockfish_path,
        depth=args.depth
    )
    print("✓ Stockfish ready")

    # Load games
    games = load_games_from_pgn(args.pgn_file, args.num_games, args.skip_games)

    if not games:
        print("Error: No games loaded")
        return 1

    # Generate preference pairs
    try:
        pairs = generate_preference_pairs(
            games, model, encoder, converter, evaluator,
            top_k=args.top_k,
            threshold=args.threshold
        )
    finally:
        evaluator.close()

    if not pairs:
        print("\nWarning: No preference pairs generated!")
        print("Try:")
        print("  - Lowering --threshold (current: {})".format(args.threshold))
        print("  - Increasing --top-k (current: {})".format(args.top_k))
        return 1

    # Save pairs
    print(f"\nSaving preference pairs to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(pairs, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(pairs)} pairs ({file_size_mb:.2f} MB)")

    # Next steps
    print()
    print("=" * 80)
    print("Next Steps: Train with DPO")
    print("=" * 80)
    print()
    print("Train the model:")
    print(f"  python train_dpo.py \\")
    print(f"      --pairs {args.output} \\")
    print(f"      --model {args.model} \\")
    print(f"      --beta 0.1 \\")
    print(f"      --lr 5e-5 \\")
    print(f"      --epochs 3")
    print()


if __name__ == "__main__":
    main()
