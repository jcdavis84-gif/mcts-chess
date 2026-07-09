#!/usr/bin/env python3
"""
Compare checkpoints by % match with Stockfish's best move on held-out positions.

Positions come from games AFTER the BC training window in the concatenated
PGN stream (Kasparov, Karpov, Capablanca — the order used by
bc/prepare_expert_games.py with --train-games 6200 --eval-games 20), so they
were never seen by BC training or DPO pair generation.

For each sampled position, Stockfish (fixed depth) provides the reference
best move. Each checkpoint is scored on:
  - top-1 match: move choice == Stockfish best move
  - top-5 match: Stockfish best move in the top-5
By default the raw policy head picks the moves (one forward pass, no MCTS).
With an @SIMS suffix on the checkpoint spec, moves are ranked by MCTS visit
counts instead (temperature 0, no Dirichlet noise — play-time conditions).
A per-position chance floor (mean of 1/num_legal_moves) is reported for context.

Usage:
    python compare_best_move_match.py \
        --checkpoint bc=checkpoints/model_latest.pt \
        --checkpoint dpo=checkpoints/model_dpo_epoch2.pt \
        --checkpoint dpo_mcts=checkpoints/model_dpo_epoch2.pt@400 \
        --num-positions 400 \
        --output logs/best_move_match.json
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from model import AlphaZeroNetwork
from board_encoder import BoardEncoder
from action_converter import ActionConverter
from mcts import MCTS


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint, reading hyperparameters if stored, else inferring
    from the state dict (num_heads is not inferrable; falls back to 2)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{path}: no model_state_dict")
    sd = ckpt["model_state_dict"]

    d_model = ckpt.get("d_model", sd["token_proj.weight"].shape[0])
    block_indices = [int(k.split(".")[1]) for k in sd if k.startswith("blocks.")]
    num_blocks = ckpt.get("num_blocks", max(block_indices) + 1 if block_indices else 4)
    num_heads = ckpt.get("num_heads", 2)
    d_policy = ckpt.get("d_policy", sd["policy_proj.weight"].shape[0])

    model = AlphaZeroNetwork(d_model=d_model, num_blocks=num_blocks,
                             num_heads=num_heads, d_policy=d_policy)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    arch = f"d_model={d_model}, blocks={num_blocks}, heads={num_heads}, d_policy={d_policy}"
    return model, arch


def collect_holdout_positions(pgn_files, skip_games, num_games):
    """Stream games across PGN files (all games, file order — matching
    prepare_expert_games.py's split), skip the first `skip_games`, take the
    next `num_games`, and return deduped (fen, num_legal) positions."""
    positions = {}
    games_seen = 0
    games_used = 0
    for pgn_path in pgn_files:
        if games_used >= num_games:
            break
        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while games_used < num_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games_seen += 1
                if games_seen <= skip_games:
                    continue
                games_used += 1
                board = game.board()
                for move in game.mainline_moves():
                    legal = board.legal_moves.count()
                    if legal >= 2:
                        key = " ".join(board.fen().split()[:4])  # ignore clocks
                        positions.setdefault(key, (board.fen(), legal))
                    board.push(move)
    return games_used, list(positions.values())


def policy_top_moves(model, board, encoder, converter, k=5):
    """Top-k legal moves from the raw policy head (no MCTS)."""
    legal_moves = list(board.legal_moves)
    legal_mask = np.zeros(4672, dtype=bool)
    for mv in legal_moves:
        try:
            legal_mask[converter.move_to_action(mv, board)] = True
        except (ValueError, AssertionError):
            continue

    with torch.no_grad():
        bt = torch.from_numpy(encoder.encode(board)).float().unsqueeze(0)
        mask = torch.from_numpy(legal_mask).bool().unsqueeze(0)
        logits, _, _ = model(bt, mask)
        log_probs = torch.log_softmax(logits, dim=-1)[0]

    top_vals, top_idx = torch.topk(log_probs, min(k, len(legal_moves)))
    moves = []
    for idx in top_idx:
        try:
            mv = converter.action_to_move(idx.item(), board)
            if mv in legal_moves:
                moves.append(mv)
        except (ValueError, AssertionError):
            continue
    return moves


def mcts_top_moves(mcts, board, converter, k=5):
    """Top-k legal moves by MCTS visit count. Moves with zero visits are
    never returned, so fewer than k moves is possible at low sim counts."""
    mcts.reset_tree()
    pi = mcts.search(board)
    moves = []
    for idx in np.argsort(pi)[::-1]:
        if pi[idx] <= 0 or len(moves) >= k:
            break
        try:
            moves.append(converter.action_to_move(int(idx), board))
        except (ValueError, AssertionError):
            continue
    return moves


def main():
    parser = argparse.ArgumentParser(description="Score checkpoints on Stockfish best-move match")
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="LABEL=PATH[@SIMS]",
                        help="Checkpoint to score (repeatable). With @SIMS, "
                        "score through MCTS with that many simulations "
                        "instead of the raw policy head")
    parser.add_argument("--pgn-files", nargs="+",
                        default=["bc/Kasparov.pgn", "bc/Karpov.pgn", "bc/Capablanca.pgn"],
                        help="PGN files in prepare_expert_games.py order")
    parser.add_argument("--skip-games", type=int, default=6200,
                        help="Games to skip (BC training window; default 6200)")
    parser.add_argument("--num-games", type=int, default=20,
                        help="Held-out games to draw positions from (default 20)")
    parser.add_argument("--num-positions", type=int, default=400,
                        help="Positions to sample (default 400)")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish depth (default 15)")
    parser.add_argument("--stockfish-path", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=str, default="logs/best_move_match.json")
    args = parser.parse_args()

    checkpoints = []
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        if not path:
            parser.error(f"--checkpoint must be LABEL=PATH[@SIMS], got: {spec}")
        path, _, sims = path.partition("@")
        checkpoints.append((label, path, int(sims) if sims else 0))

    print("=" * 70)
    print("Best-move match vs Stockfish on held-out positions")
    print("=" * 70)

    games_used, positions = collect_holdout_positions(
        args.pgn_files, args.skip_games, args.num_games)
    print(f"Held-out games used: {games_used} (after skipping {args.skip_games})")
    print(f"Unique positions available: {len(positions)}")

    rng = random.Random(args.seed)
    if len(positions) > args.num_positions:
        positions = rng.sample(positions, args.num_positions)
    print(f"Sampled positions: {len(positions)}")

    chance = float(np.mean([1.0 / n for _, n in positions]))
    print(f"Chance floor (mean 1/legal): {chance*100:.2f}%")

    # Annotate with Stockfish best move
    print(f"\nAnnotating with Stockfish depth {args.depth}...")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    annotated = []
    t0 = time.time()
    try:
        for i, (fen, n_legal) in enumerate(positions):
            board = chess.Board(fen)
            result = engine.play(board, chess.engine.Limit(depth=args.depth))
            annotated.append((fen, n_legal, result.move.uci()))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(positions)} ({time.time() - t0:.0f}s)")
    finally:
        engine.quit()
    print(f"✓ Annotated {len(annotated)} positions in {time.time() - t0:.0f}s")

    encoder = BoardEncoder()
    converter = ActionConverter()
    results = {}
    for label, path, sims in checkpoints:
        model, arch = load_checkpoint(path)
        mode = f"MCTS {sims} sims" if sims else "raw policy"
        print(f"\nScoring '{label}' ({path}, {mode})\n  {arch}")
        mcts = MCTS(model=model, encoder=encoder, converter=converter,
                    num_simulations=sims, c_puct=1.5, temperature=0.0,
                    add_dirichlet_noise=False) if sims else None
        top1 = top5 = 0
        top1_moves = []
        t0 = time.time()
        for i, (fen, n_legal, best_uci) in enumerate(annotated):
            board = chess.Board(fen)
            if mcts is not None:
                moves = mcts_top_moves(mcts, board, converter, k=5)
            else:
                moves = policy_top_moves(model, board, encoder, converter, k=5)
            ucis = [m.uci() for m in moves]
            top1_moves.append(ucis[0] if ucis else None)
            if ucis and ucis[0] == best_uci:
                top1 += 1
            if best_uci in ucis:
                top5 += 1
            if sims and (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(annotated)} ({time.time() - t0:.0f}s)")
        n = len(annotated)
        results[label] = {
            "path": path, "arch": arch, "mcts_sims": sims,
            "top1_match": top1 / n, "top5_match": top5 / n,
            "top1_count": top1, "top5_count": top5,
            "top1_moves": top1_moves,
        }
        print(f"  top-1 match: {top1}/{n} ({top1 / n * 100:.1f}%)")
        print(f"  top-5 match: {top5}/{n} ({top5 / n * 100:.1f}%)")
        print(f"  scoring time: {time.time() - t0:.0f}s")

    output = {
        "config": {
            "pgn_files": args.pgn_files, "skip_games": args.skip_games,
            "num_games": games_used, "num_positions": len(annotated),
            "stockfish_depth": args.depth, "seed": args.seed,
        },
        "chance_floor": chance,
        "positions": [{"fen": fen, "stockfish_best": best_uci}
                      for fen, _, best_uci in annotated],
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'Model':<12} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 30)
    print(f"{'(chance)':<12} {chance*100:>7.1f}% {'—':>8}")
    for label, r in results.items():
        print(f"{label:<12} {r['top1_match']*100:>7.1f}% {r['top5_match']*100:>7.1f}%")
    print("=" * 70)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
