#!/usr/bin/env python3
"""
AlphaZero Chess Training Script

Trains an AlphaZero-style chess engine using:
- Transformer policy-value network
- Monte Carlo Tree Search (MCTS)
- Self-play data generation
- Replay buffer

Usage:
    python train.py
"""

import torch
import numpy as np
import random
import time
import argparse
import os
import shutil
import glob
from pathlib import Path

import wandb

from model import AlphaZeroNetwork
from board_encoder import BoardEncoder
from action_converter import ActionConverter
from replay_buffer import ReplayBuffer
from trainer import AlphaZeroTrainer



def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_utilization():
    """Get GPU utilization percentage (if available)."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.utilization()
        except:
            return 0
    return 0


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"




def get_disk_space_gb(path: str = ".") -> float:
    """Get available disk space in GB for the given path."""
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_latest: int = 1):
    """
    Delete old checkpoints, keeping only the N most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_latest: Number of latest checkpoints to keep (default: 1)
    """
    # Find all cycle checkpoints (exclude model_final.pt)
    pattern = os.path.join(checkpoint_dir, "model_cycle_*.pt")
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_latest:
        return  # Nothing to clean up

    # Sort by cycle number (extract from filename)
    def get_cycle_num(path):
        basename = os.path.basename(path)
        # Extract number from "model_cycle_N.pt"
        try:
            return int(basename.replace("model_cycle_", "").replace(".pt", ""))
        except ValueError:
            return 0

    checkpoints.sort(key=get_cycle_num)

    # Delete all but the latest N
    to_delete = checkpoints[:-keep_latest] if keep_latest > 0 else checkpoints

    for ckpt_path in to_delete:
        try:
            os.remove(ckpt_path)
            print(f"  Deleted old checkpoint: {os.path.basename(ckpt_path)}")
        except OSError as e:
            print(f"  Warning: Could not delete {ckpt_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero Chess")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--cycles", type=int, default=20, help="Number of training cycles")
    parser.add_argument("--games-per-cycle", type=int, default=20, help="Self-play games per cycle (G)")
    parser.add_argument("--train-steps-per-cycle", type=int, default=200, help="Training steps per cycle (T)")
    parser.add_argument("--mcts-simulations", type=int, default=20, help="MCTS simulations per move")
    parser.add_argument("--buffer-capacity", type=int, default=7000, help="Replay buffer capacity")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N cycles")

    # Model hyperparameters
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--num-heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--d-policy", type=int, default=64, help="Policy head dimension")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="alphazero-chess", help="W&B project name")

    # Replay buffer persistence
    parser.add_argument("--load-buffer", type=str, default=None, help="Path to load existing replay buffer (skips self-play)")
    parser.add_argument("--save-buffer-interval", type=int, default=0, help="Save buffer every N cycles (0=disabled)")

    # Resume training
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoints/model_cycle_4.pt)")

    # Self-play filtering
    parser.add_argument("--discard-draws", action="store_true", help="Discard all positions from drawn games during self-play")
    parser.add_argument("--last-n-plies", type=int, default=None, help="Keep only last N plies from each self-play game")

    # Training verbosity
    parser.add_argument("--log-training-steps", action="store_true", help="Log loss for each training step (batch)")

    # Parallelization
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers for self-play (None=auto-detect, 1=sequential)")

    # Material balance loss
    parser.add_argument("--material-loss-weight", type=float, default=0.1, help="Weight for material balance prediction loss (default: 0.1)")
    parser.add_argument("--test-material", action="store_true", help="Test material balance: run 1 cycle with detailed material stats")

    args = parser.parse_args()

    # Override settings if testing material balance
    if args.test_material:
        args.cycles = 1
        args.games_per_cycle = 1
        args.train_steps_per_cycle = 50
        args.mcts_simulations = 5
        args.log_training_steps = True
        print("=" * 80)
        print("MATERIAL BALANCE TEST MODE")
        print("=" * 80)
        print("Overriding settings for testing:")
        print(f"  cycles = 1")
        print(f"  games_per_cycle = 5")
        print(f"  train_steps_per_cycle = 50")
        print(f"  mcts_simulations = 20")
        print(f"  log_training_steps = True")
        print("=" * 80)
        print()

    # Set seed
    set_seed(args.seed)

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Initialize W&B (only if explicitly requested)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "seed": args.seed,
                "device": device,
                "cycles": args.cycles,
                "games_per_cycle": args.games_per_cycle,
                "train_steps_per_cycle": args.train_steps_per_cycle,
                "mcts_simulations": args.mcts_simulations,
                "buffer_capacity": args.buffer_capacity,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "d_model": args.d_model,
                "num_blocks": args.num_blocks,
                "num_heads": args.num_heads,
                "d_policy": args.d_policy,
            }
        )
        use_wandb = True
    else:
        use_wandb = False

    print("=" * 80)
    print("AlphaZero Chess Training")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Training cycles: {args.cycles}")
    print(f"Games per cycle: {args.games_per_cycle}")
    print(f"Training steps per cycle: {args.train_steps_per_cycle}")
    print(f"MCTS simulations: {args.mcts_simulations}")
    print(f"Replay buffer capacity: {args.buffer_capacity}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"W&B logging: {'enabled' if use_wandb else 'disabled'}")
    print()
    print(f"Model: d_model={args.d_model}, blocks={args.num_blocks}, heads={args.num_heads}")
    print("=" * 80)
    print()

    # Initialize components
    print("Initializing components...")
    encoder = BoardEncoder()
    converter = ActionConverter()

    # Load or create replay buffer
    if args.load_buffer:
        print(f"Loading replay buffer from: {args.load_buffer}")
        replay_buffer = ReplayBuffer.load(args.load_buffer)
        print(f"Loaded {len(replay_buffer)} positions")
    else:
        replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

    # Create model
    model = AlphaZeroNetwork(
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        d_policy=args.d_policy,
    )

    # Create trainer
    trainer = AlphaZeroTrainer(
        model=model,
        encoder=encoder,
        converter=converter,
        replay_buffer=replay_buffer,
        device=device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        material_loss_weight=args.material_loss_weight,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
        print("✓ Checkpoint loaded successfully")
        print(f"  Total training steps so far: {trainer.total_steps}")
        print()

    # Display worker information
    if device == "cuda" and torch.cuda.is_available():
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Model size: {model_size_mb:.1f} MB")
        print(f"GPU VRAM: {total_vram_gb:.1f} GB")

        # Estimate workers if auto-detect
        if args.num_workers is None:
            from self_play import _estimate_safe_workers
            estimated_workers = _estimate_safe_workers(model, None, device)
            print(f"Self-play workers: {estimated_workers} (auto-detected)")
        elif args.num_workers == 1:
            print(f"Self-play workers: 1 (sequential, no multiprocessing)")
        else:
            print(f"Self-play workers: {args.num_workers} (user-specified)")
    else:
        if args.num_workers and args.num_workers > 1:
            print(f"Warning: Multi-worker self-play only supported on CUDA. Using sequential.")
        print(f"Self-play workers: 1 (CPU mode)")

    print()

    # Training loop
    total_start_time = time.time()

    for cycle in range(1, args.cycles + 1):
        print(f"{'=' * 80}")
        print(f"Cycle {cycle}/{args.cycles}")
        print(f"{'=' * 80}")

        # 1. Update inference model (snapshot current model)
        trainer.update_inference_model()

        # 2. Generate self-play data (skip if buffer was loaded)
        if not args.load_buffer:
            print(f"\nSelf-Play ({args.games_per_cycle} games):")
            selfplay_stats = trainer.generate_self_play_data(
                num_games=args.games_per_cycle,
                num_simulations=args.mcts_simulations,
                discard_draws=args.discard_draws,
                last_n_plies=args.last_n_plies,
                num_workers=args.num_workers,
            )

            white_pct = selfplay_stats["white_wins"] / args.games_per_cycle * 100
            draw_pct = selfplay_stats["draws"] / args.games_per_cycle * 100
            black_pct = selfplay_stats["black_wins"] / args.games_per_cycle * 100

            print(f"  Outcomes: W: {white_pct:.1f}% | D: {draw_pct:.1f}% | L: {black_pct:.1f}%")
            print(f"  Avg game length: {selfplay_stats['avg_game_length']:.1f} plies")
            print(f"  Time: {format_time(selfplay_stats['time'])} ({selfplay_stats['games_per_sec']:.2f} games/s)")
            print(f"  Positions added: {selfplay_stats['num_positions']}")

            # Log self-play stats to W&B
            if use_wandb:
                wandb.log({
                    "cycle": cycle,
                    "selfplay/white_wins": selfplay_stats["white_wins"],
                    "selfplay/draws": selfplay_stats["draws"],
                    "selfplay/black_wins": selfplay_stats["black_wins"],
                    "selfplay/white_win_pct": white_pct,
                    "selfplay/draw_pct": draw_pct,
                    "selfplay/black_win_pct": black_pct,
                    "selfplay/avg_game_length": selfplay_stats["avg_game_length"],
                    "selfplay/num_positions": selfplay_stats["num_positions"],
                    "selfplay/games_per_sec": selfplay_stats["games_per_sec"],
                    "selfplay/time": selfplay_stats["time"],
                }, step=cycle)
        else:
            print(f"\nSkipping self-play (using loaded buffer data)")

        # 3. Train model
        print(f"\nTraining ({args.train_steps_per_cycle} steps):")
        train_stats = trainer.train_steps(
            num_steps=args.train_steps_per_cycle,
            log_interval=10 if args.log_training_steps else 0,
        )

        print(f"  Total Loss: {train_stats['total_loss']:.4f}")
        print(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
        print(f"  Value Loss: {train_stats['value_loss']:.4f}")
        print(f"  Material Loss: {train_stats['material_loss']:.4f}")
        print(f"  Learning Rate: {train_stats['learning_rate']:.6f}")
        print(f"  Time: {format_time(train_stats['time'])} ({train_stats['steps_per_sec']:.2f} steps/s)")

        # Show detailed material balance stats if testing
        if args.test_material:
            print(f"\n{'=' * 80}")
            print("Material Balance Test Results")
            print(f"{'=' * 80}")

            # Sample a batch and show material predictions
            if len(replay_buffer) > 0:
                batch_size = min(8, len(replay_buffer))
                board_batch, policy_batch, value_batch, material_batch = replay_buffer.sample(batch_size)

                # Get model predictions
                model.eval()
                with torch.no_grad():
                    board_tensor = torch.from_numpy(board_batch).to(device)
                    _, _, material_pred = model(board_tensor)
                    material_pred = material_pred.cpu().numpy().flatten()

                material_targets = material_batch

                print(f"\nMaterial Balance Predictions (sample of {batch_size} positions):")
                print(f"  {'#':<4} {'Target':<10} {'Predicted':<10} {'Error':<10} {'Abs Error':<10}")
                print(f"  {'-'*4} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

                errors = []
                for i in range(batch_size):
                    target = material_targets[i]
                    pred = material_pred[i]
                    error = pred - target
                    abs_error = abs(error)
                    errors.append(abs_error)

                    print(f"  {i:<4} {target:+10.2f} {pred:+10.2f} {error:+10.2f} {abs_error:10.2f}")

                print(f"\nMaterial Prediction Statistics:")
                print(f"  Mean Absolute Error: {np.mean(errors):.2f}")
                print(f"  Max Absolute Error:  {np.max(errors):.2f}")
                print(f"  Min Absolute Error:  {np.min(errors):.2f}")

                # Show material distribution in buffer
                all_materials = np.array([entry[3] for entry in replay_buffer.buffer])
                print(f"\nMaterial Distribution in Replay Buffer:")
                print(f"  Mean:   {all_materials.mean():+.2f}")
                print(f"  Median: {np.median(all_materials):+.2f}")
                print(f"  Std:    {all_materials.std():.2f}")
                print(f"  Min:    {all_materials.min():+.2f}")
                print(f"  Max:    {all_materials.max():+.2f}")
                print(f"{'=' * 80}")

        # Log training stats to W&B
        if use_wandb:
            wandb.log({
                "train/total_loss": train_stats["total_loss"],
                "train/policy_loss": train_stats["policy_loss"],
                "train/value_loss": train_stats["value_loss"],
                "train/material_loss": train_stats["material_loss"],
                "train/learning_rate": train_stats["learning_rate"],
                "train/steps_per_sec": train_stats["steps_per_sec"],
                "train/time": train_stats["time"],
            }, step=cycle)

        # 4. Buffer and system stats
        print(f"\nBuffer: {len(replay_buffer):,} / {args.buffer_capacity:,} positions")

        gpu_util = get_gpu_utilization()
        if gpu_util > 0:
            print(f"GPU Utilization: {gpu_util}%")

        # Log system stats to W&B
        if use_wandb:
            wandb.log({
                "system/buffer_size": len(replay_buffer),
                "system/buffer_utilization_pct": len(replay_buffer) / args.buffer_capacity * 100,
                "system/gpu_utilization": gpu_util if gpu_util > 0 else None,
            }, step=cycle)

        # 5. Save checkpoint
        if cycle % args.checkpoint_interval == 0:
            # Always save to the same file (overwrites previous)
            checkpoint_path = f"{args.checkpoint_dir}/model_latest.pt"

            try:
                trainer.save_checkpoint(checkpoint_path)
                checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"\nCheckpoint saved: {checkpoint_path} ({checkpoint_size_mb:.1f} MB)")
                print(f"  Cycle: {cycle}/{args.cycles}")
                print(f"  Total training steps: {trainer.total_steps:,}")

            except RuntimeError as e:
                print(f"\n❌ Error saving checkpoint: {e}")
                print(f"Free disk space: {get_disk_space_gb(args.checkpoint_dir):.1f} GB")
                print("Training will continue, but checkpoint was not saved.")

        # 6. Save replay buffer if requested
        if args.save_buffer_interval > 0 and cycle % args.save_buffer_interval == 0:
            buffer_path = f"{args.checkpoint_dir}/buffer_cycle_{cycle}.pkl"
            replay_buffer.save(buffer_path)
            file_size_mb = os.path.getsize(buffer_path) / (1024 * 1024)
            print(f"Replay buffer saved: {buffer_path} ({file_size_mb:.2f} MB)")

        # Estimate time remaining
        elapsed_total = time.time() - total_start_time
        avg_time_per_cycle = elapsed_total / cycle
        remaining_cycles = args.cycles - cycle
        estimated_remaining = avg_time_per_cycle * remaining_cycles

        print(f"\nElapsed: {format_time(elapsed_total)} | Estimated remaining: {format_time(estimated_remaining)}")
        print()

    # Final checkpoint
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)

    final_checkpoint = f"{args.checkpoint_dir}/model_final.pt"

    try:
        trainer.save_checkpoint(final_checkpoint)
        checkpoint_size_mb = os.path.getsize(final_checkpoint) / (1024 * 1024)
        print(f"Final checkpoint saved: {final_checkpoint} ({checkpoint_size_mb:.1f} MB)")
        print(f"Total training steps: {trainer.total_steps:,}")
    except Exception as e:
        print(f"❌ Error saving final checkpoint: {e}")
        print(f"Free disk space: {get_disk_space_gb(args.checkpoint_dir):.1f} GB")
        print("Warning: Training completed but final checkpoint was not saved!")

    # Save final replay buffer
    if args.save_buffer_interval > 0 or args.load_buffer:
        final_buffer_path = f"{args.checkpoint_dir}/buffer_final.pkl"
        replay_buffer.save(final_buffer_path)
        file_size_mb = os.path.getsize(final_buffer_path) / (1024 * 1024)
        print(f"Final replay buffer saved: {final_buffer_path} ({file_size_mb:.2f} MB)")

    print(f"Total training time: {format_time(time.time() - total_start_time)}")

    print()

    # Finish W&B run
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
