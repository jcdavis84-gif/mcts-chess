#!/usr/bin/env python3
"""
Train chess model using Direct Preference Optimization (DPO).

Trains the policy head to prefer better moves (as judged by Stockfish)
over worse moves, using preference pairs generated from master games.

Usage:
    # Basic training
    python train_dpo.py \
        --pairs dpo_pairs.pkl \
        --model model_latest.pt \
        --output model_dpo.pt

    # Custom hyperparameters
    python train_dpo.py \
        --pairs dpo_pairs.pkl \
        --model model_latest.pt \
        --beta 0.1 \
        --lr 5e-5 \
        --batch-size 64 \
        --epochs 3 \
        --warmup-steps 100

    # With eval-difference weighting
    python train_dpo.py \
        --pairs dpo_pairs.pkl \
        --model model_latest.pt \
        --use-margin \
        --margin-weight 1.0
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import copy
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.model import AlphaZeroNetwork
from engine.board_encoder import BoardEncoder
from engine.action_converter import ActionConverter
from dpo_loss import dpo_loss, dpo_loss_with_margin


class PreferencePairDataset(Dataset):
    """Dataset of preference pairs for DPO training."""

    def __init__(self, pairs):
        """
        Args:
            pairs: List of preference pair dicts from generate_dpo_pairs.py
        """
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        return {
            'board_tensor': torch.from_numpy(pair['board_tensor']).float(),
            'preferred_action': pair['preferred_action'],
            'rejected_action': pair['rejected_action'],
            'eval_diff': pair['eval_diff'],
        }


def collate_fn(batch):
    """Collate batch of preference pairs."""
    board_tensors = torch.stack([item['board_tensor'] for item in batch])
    preferred_actions = torch.tensor([item['preferred_action'] for item in batch], dtype=torch.long)
    rejected_actions = torch.tensor([item['rejected_action'] for item in batch], dtype=torch.long)
    eval_diffs = torch.tensor([item['eval_diff'] for item in batch], dtype=torch.float32)

    return board_tensors, preferred_actions, rejected_actions, eval_diffs


def get_legal_masks(board_tensors, converter):
    """
    Get legal move masks for a batch of boards.

    This is a placeholder - in practice we'd need to reconstruct boards from tensors.
    For now, we'll use all-True masks (DPO doesn't strictly need legal masks since
    we're only computing log probs for specific actions).
    """
    batch_size = board_tensors.shape[0]
    return torch.ones(batch_size, 4672, dtype=torch.bool)


def create_reference_model(model):
    """Create a frozen copy of the model for reference policy."""
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


def linear_warmup_schedule(optimizer, warmup_steps):
    """Learning rate warmup schedule."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, ref_model, dataloader, optimizer, scheduler, device, beta, use_margin, margin_weight):
    """Train for one epoch."""
    model.train()
    ref_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_kl = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for board_tensors, preferred_actions, rejected_actions, eval_diffs in progress_bar:
        # Move to device
        board_tensors = board_tensors.to(device)
        preferred_actions = preferred_actions.to(device)
        rejected_actions = rejected_actions.to(device)
        eval_diffs = eval_diffs.to(device)

        # Get legal masks (all-True for simplicity)
        legal_masks = torch.ones(board_tensors.shape[0], 4672, dtype=torch.bool, device=device)

        # Forward pass - policy model
        policy_logits, _, _ = model(board_tensors, legal_masks)

        # Forward pass - reference model (frozen)
        with torch.no_grad():
            ref_logits, _, _ = ref_model(board_tensors, legal_masks)

        # Compute DPO loss
        if use_margin:
            loss, metrics = dpo_loss_with_margin(
                policy_logits, ref_logits,
                preferred_actions, rejected_actions,
                legal_masks, eval_diffs,
                beta=beta, margin_weight=margin_weight
            )
        else:
            loss, metrics = dpo_loss(
                policy_logits, ref_logits,
                preferred_actions, rejected_actions,
                legal_masks, beta=beta
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent instability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += metrics['loss']
        total_accuracy += metrics['accuracy']
        total_kl += metrics['kl_divergence']
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['accuracy']:.3f}",
            'kl': f"{metrics['kl_divergence']:.4f}"
        })

    avg_metrics = {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'kl_divergence': total_kl / num_batches,
    }

    return avg_metrics


def evaluate(model, ref_model, dataloader, device, beta, use_margin, margin_weight):
    """Evaluate on validation set."""
    model.eval()
    ref_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for board_tensors, preferred_actions, rejected_actions, eval_diffs in tqdm(dataloader, desc="Evaluating"):
            board_tensors = board_tensors.to(device)
            preferred_actions = preferred_actions.to(device)
            rejected_actions = rejected_actions.to(device)
            eval_diffs = eval_diffs.to(device)

            legal_masks = torch.ones(board_tensors.shape[0], 4672, dtype=torch.bool, device=device)

            policy_logits, _, _ = model(board_tensors, legal_masks)
            ref_logits, _, _ = ref_model(board_tensors, legal_masks)

            if use_margin:
                loss, metrics = dpo_loss_with_margin(
                    policy_logits, ref_logits,
                    preferred_actions, rejected_actions,
                    legal_masks, eval_diffs,
                    beta=beta, margin_weight=margin_weight
                )
            else:
                loss, metrics = dpo_loss(
                    policy_logits, ref_logits,
                    preferred_actions, rejected_actions,
                    legal_masks, beta=beta
                )

            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            total_kl += metrics['kl_divergence']
            num_batches += 1

    avg_metrics = {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'kl_divergence': total_kl / num_batches,
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train chess model with DPO",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input
    parser.add_argument("--pairs", type=str, required=True, help="Path to preference pairs pickle")
    parser.add_argument("--model", type=str, default="model_latest.pt", help="Path to initial model checkpoint")

    # DPO hyperparameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter (default: 0.1)")
    parser.add_argument("--use-margin", action="store_true", help="Weight loss by eval difference")
    parser.add_argument("--margin-weight", type=float, default=1.0, help="Margin weighting strength (default: 1.0)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps (default: 100)")

    # Data split
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")

    # Output
    parser.add_argument("--output", type=str, default="model_dpo.pt", help="Output path for trained model")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs (default: 1)")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")

    args = parser.parse_args()

    print("=" * 80)
    print("DPO Training for Chess Policy")
    print("=" * 80)
    print()

    # Load preference pairs
    print(f"Loading preference pairs from {args.pairs}...")
    with open(args.pairs, 'rb') as f:
        pairs = pickle.load(f)
    print(f"✓ Loaded {len(pairs)} preference pairs")

    # Train/val split
    num_val = int(len(pairs) * args.val_split)
    num_train = len(pairs) - num_val

    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:]

    print(f"  Training: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")

    # Create datasets
    train_dataset = PreferencePairDataset(train_pairs)
    val_dataset = PreferencePairDataset(val_pairs) if val_pairs else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid pickling issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    ) if val_dataset else None

    # Load model
    print(f"\nLoading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        d_model = state_dict['token_proj.weight'].shape[0]

        # Detect number of blocks by finding max block index
        block_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.')]
        num_blocks = max(block_indices) + 1 if block_indices else 4

        num_heads = 2
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

    print(f"✓ Model loaded (d_model={d_model}, blocks={num_blocks})")

    # Create reference model (frozen copy)
    print("\nCreating reference model (frozen copy)...")
    ref_model = create_reference_model(model)
    print("✓ Reference model created")

    # Move to device
    device = torch.device(args.device)
    model = model.to(device)
    ref_model = ref_model.to(device)
    print(f"✓ Using device: {device}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = linear_warmup_schedule(optimizer, args.warmup_steps)

    # Training loop
    print()
    print("=" * 80)
    print("Training")
    print("=" * 80)
    print(f"Hyperparameters:")
    print(f"  Beta: {args.beta}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Use margin weighting: {args.use_margin}")
    if args.use_margin:
        print(f"  Margin weight: {args.margin_weight}")
    print()

    best_val_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(
            model, ref_model, train_loader, optimizer, scheduler,
            device, args.beta, args.use_margin, args.margin_weight
        )

        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  KL divergence: {train_metrics['kl_divergence']:.4f}")

        # Validate
        if val_loader:
            val_metrics = evaluate(
                model, ref_model, val_loader,
                device, args.beta, args.use_margin, args.margin_weight
            )

            print(f"\nValidation metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  KL divergence: {val_metrics['kl_divergence']:.4f}")

            # Track best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                print(f"  ✓ New best validation accuracy!")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = args.output
            if args.epochs > 1 and epoch != args.epochs:
                checkpoint_path = args.output.replace('.pt', f'_epoch{epoch}.pt')

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics if val_loader else None,
            }, checkpoint_path)

            print(f"\n✓ Saved checkpoint: {checkpoint_path}")

    # Final save
    print()
    print("=" * 80)
    print("Training Complete")
    print("=" * 80)
    print(f"\n✓ Final model saved: {args.output}")
    print(f"  Best validation accuracy: {best_val_accuracy:.4f}")

    print()
    print("Next steps:")
    print(f"  1. Evaluate against baseline: python evaluate.py --model1 {args.output} --model2 {args.model}")
    print(f"  2. Test in self-play: python self_play.py --model {args.output}")
    print()


if __name__ == "__main__":
    main()
