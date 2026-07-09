#!/usr/bin/env python3
"""
Score checkpoints on preference accuracy over the DPO validation holdout.

Preference accuracy: fraction of holdout pairs where the model's own policy
log-prob ranks the Stockfish-preferred move above the rejected one — the same
model-only 'accuracy' metric train_dpo.py reports during validation (all-True
legal masks; the shared softmax normalization cancels when comparing two moves).

The holdout is reproduced exactly as train_dpo.py builds it: the last
--val-split fraction of the pairs file, in file order, no shuffle.

Usage:
    python dpo/eval_preference_accuracy.py \
        --pairs dpo_pairs.pkl \
        --checkpoint bc=checkpoints/model_latest.pt \
        --checkpoint dpo=checkpoints/model_dpo_epoch2.pt \
        --output logs/preference_accuracy.json
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.model import AlphaZeroNetwork


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint, reading hyperparameters if stored, else inferring
    from the state dict (num_heads is not inferrable; falls back to 2)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    d_model = ckpt.get("d_model", state_dict["token_proj.weight"].shape[0])
    block_indices = [int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")]
    num_blocks = ckpt.get("num_blocks", max(block_indices) + 1 if block_indices else 4)
    num_heads = ckpt.get("num_heads", 2)
    d_policy = ckpt.get("d_policy", state_dict["policy_proj.weight"].shape[0])

    model = AlphaZeroNetwork(
        d_model=d_model, num_blocks=num_blocks, num_heads=num_heads, d_policy=d_policy
    )
    model.load_state_dict(state_dict)
    model.eval()
    arch = f"d_model={d_model}, blocks={num_blocks}, heads={num_heads}, d_policy={d_policy}"
    return model, arch


def main():
    parser = argparse.ArgumentParser(description="Score checkpoints on DPO holdout preference accuracy")
    parser.add_argument("--pairs", type=str, default="dpo_pairs.pkl", help="Preference pairs pickle")
    parser.add_argument("--checkpoint", action="append", required=True, metavar="LABEL=PATH",
                        help="Checkpoint to score (repeatable)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Holdout fraction, matching train_dpo.py (default: 0.1)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None, help="Write results JSON here")
    args = parser.parse_args()

    with open(args.pairs, "rb") as f:
        pairs = pickle.load(f)
    num_val = int(len(pairs) * args.val_split)
    val_pairs = pairs[len(pairs) - num_val:]
    print(f"Loaded {len(pairs)} pairs from {args.pairs}; holdout = last {len(val_pairs)}")

    boards = torch.stack([torch.from_numpy(p["board_tensor"]).float() for p in val_pairs])
    preferred = torch.tensor([p["preferred_action"] for p in val_pairs], dtype=torch.long)
    rejected = torch.tensor([p["rejected_action"] for p in val_pairs], dtype=torch.long)

    results = {}
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        model, arch = load_checkpoint(path)
        print(f"\nScoring '{label}' ({path})")
        print(f"  {arch}")

        correct, margin_sum = 0, 0.0
        start = time.time()
        with torch.no_grad():
            for i in range(0, len(val_pairs), args.batch_size):
                b = boards[i:i + args.batch_size]
                masks = torch.ones(b.shape[0], 4672, dtype=torch.bool)
                logits, _, _ = model(b, masks)
                log_probs = F.log_softmax(logits, dim=-1)
                lp_pref = log_probs.gather(1, preferred[i:i + args.batch_size].unsqueeze(1)).squeeze(1)
                lp_rej = log_probs.gather(1, rejected[i:i + args.batch_size].unsqueeze(1)).squeeze(1)
                correct += (lp_pref > lp_rej).sum().item()
                margin_sum += (lp_pref - lp_rej).sum().item()

        accuracy = correct / len(val_pairs)
        mean_margin = margin_sum / len(val_pairs)
        print(f"  preference accuracy: {correct}/{len(val_pairs)} ({accuracy:.1%})")
        print(f"  mean log-prob margin: {mean_margin:+.2f}")
        print(f"  time: {time.time() - start:.1f}s")

        results[label] = {
            "path": path, "arch": arch,
            "accuracy": accuracy, "correct": correct,
            "num_pairs": len(val_pairs), "mean_margin": mean_margin,
        }

    print("\n" + "=" * 60)
    print(f"{'Model':<16} {'Accuracy':>10} {'Margin':>10}")
    print("-" * 38)
    for label, r in results.items():
        print(f"{label:<16} {r['accuracy']:>9.1%} {r['mean_margin']:>+10.2f}")
    print("=" * 60)

    if args.output:
        out = {
            "config": {"pairs": args.pairs, "val_split": args.val_split, "num_pairs": len(val_pairs)},
            "results": results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
