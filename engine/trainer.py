import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
from pathlib import Path
from typing import Dict
from .model import AlphaZeroNetwork
from .replay_buffer import ReplayBuffer
from .self_play import play_games
from .board_encoder import BoardEncoder
from .action_converter import ActionConverter


class AlphaZeroTrainer:
    """
    Manages the training loop for AlphaZero.

    Implements continuous update strategy:
    1. Snapshot current model to frozen inference_model
    2. Generate G games using frozen model
    3. Train main model for T steps
    4. Repeat

    This prevents prediction instability during data generation.
    """

    def __init__(
        self,
        model: AlphaZeroNetwork,
        encoder: BoardEncoder,
        converter: ActionConverter,
        replay_buffer: ReplayBuffer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        batch_size: int = 128,
        num_warmup_steps: int = 500,
        gradient_clip: float = 1.0,
        material_loss_weight: float = 0.1,
    ):
        self.model = model.to(device)
        self.encoder = encoder
        self.converter = converter
        self.replay_buffer = replay_buffer
        self.device = device
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        self.material_loss_weight = material_loss_weight

        # Create frozen inference model (copy of current model)
        self.inference_model = copy.deepcopy(self.model)
        self.inference_model.eval()

        # Optimizer with AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler: warmup + cosine decay
        self.num_warmup_steps = num_warmup_steps
        self.base_lr = learning_rate
        self.total_steps = 0

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.material_loss_fn = nn.MSELoss()

    def update_inference_model(self):
        """
        Update the frozen inference model with current training model weights.
        This is called at the start of each cycle.
        """
        self.inference_model.load_state_dict(self.model.state_dict())
        self.inference_model.eval()

    def generate_self_play_data(
        self,
        num_games: int,
        num_simulations: int = 20,
        discard_draws: bool = False,
        last_n_plies: int = None,
        num_workers: int = None,
    ) -> Dict[str, float]:
        """
        Generate self-play games using the frozen inference model.

        Args:
            num_games: Number of games to play
            num_simulations: MCTS simulations per move
            discard_draws: If True, discard all positions from drawn games
            last_n_plies: If set, only keep last N plies from each game
            num_workers: Number of parallel workers (None = auto-detect, 1 = sequential)

        Returns:
            stats: Dictionary with game statistics (based on original games before filtering)
        """
        self.inference_model.eval()

        start_time = time.time()

        # Play games (returns filtered data and pre-filter statistics)
        game_data, game_stats = play_games(
            model=self.inference_model,
            encoder=self.encoder,
            converter=self.converter,
            num_games=num_games,
            num_simulations=num_simulations,
            discard_draws=discard_draws,
            last_n_plies=last_n_plies,
            num_workers=num_workers,
        )

        # Add to replay buffer
        for board_tensor, policy, value, material in game_data:
            self.replay_buffer.add(board_tensor, policy, value, material)

        elapsed_time = time.time() - start_time

        # Use pre-filter statistics from play_games
        stats = {
            "white_wins": game_stats["white_wins"],
            "draws": game_stats["draws"],
            "black_wins": game_stats["black_wins"],
            "avg_game_length": game_stats["avg_game_length"],
            "num_positions": len(game_data),
            "time": elapsed_time,
            "games_per_sec": num_games / elapsed_time,
        }

        return stats

    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.

        Returns:
            losses: Dictionary with loss values
        """
        if self.replay_buffer.is_empty():
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "material_loss": 0.0}

        self.model.train()

        # Sample batch from replay buffer
        board_tensors, policy_targets, value_targets, material_targets = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        board_tensors = torch.from_numpy(board_tensors).to(self.device)
        policy_targets = torch.from_numpy(policy_targets).to(self.device)
        value_targets = torch.from_numpy(value_targets).to(self.device)
        material_targets = torch.from_numpy(material_targets).to(self.device)

        # Update learning rate
        self._update_learning_rate()

        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                policy_logits, value_pred, material_pred = self.model(board_tensors)

                # Policy loss: cross-entropy
                policy_loss = self.policy_loss_fn(policy_logits, policy_targets)

                # Value loss: MSE
                value_loss = self.value_loss_fn(value_pred, value_targets)

                # Material loss: MSE
                material_loss = self.material_loss_fn(material_pred, material_targets)

                # Total loss (with weighted material loss)
                total_loss = policy_loss + value_loss + self.material_loss_weight * material_loss

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # CPU training (no mixed precision)
            policy_logits, value_pred, material_pred = self.model(board_tensors)

            # Policy loss: cross-entropy
            policy_loss = self.policy_loss_fn(policy_logits, policy_targets)

            # Value loss: MSE
            value_loss = self.value_loss_fn(value_pred, value_targets)

            # Material loss: MSE
            material_loss = self.material_loss_fn(material_pred, material_targets)

            # Total loss (with weighted material loss)
            total_loss = policy_loss + value_loss + self.material_loss_weight * material_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Optimizer step
            self.optimizer.step()

        self.total_steps += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "material_loss": material_loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def train_steps(self, num_steps: int, log_interval: int = 0) -> Dict[str, float]:
        """
        Perform multiple training steps and return average losses.

        Args:
            num_steps: Number of training steps to perform
            log_interval: If > 0, print loss every N steps (0 = disabled)

        Returns:
            avg_losses: Dictionary with average loss values
        """
        start_time = time.time()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        material_loss_sum = 0.0
        lr_sum = 0.0

        for step in range(num_steps):
            losses = self.train_step()
            total_loss_sum += losses["total_loss"]
            policy_loss_sum += losses["policy_loss"]
            value_loss_sum += losses["value_loss"]
            material_loss_sum += losses["material_loss"]
            lr_sum += losses["learning_rate"]

            if log_interval > 0 and (step + 1) % log_interval == 0:
                print(f"  Step {step+1}/{num_steps}: Total={losses['total_loss']:.4f}, Policy={losses['policy_loss']:.4f}, Value={losses['value_loss']:.4f}, Material={losses['material_loss']:.4f}")

        elapsed_time = time.time() - start_time

        return {
            "total_loss": total_loss_sum / num_steps,
            "policy_loss": policy_loss_sum / num_steps,
            "value_loss": value_loss_sum / num_steps,
            "material_loss": material_loss_sum / num_steps,
            "learning_rate": lr_sum / num_steps,
            "time": elapsed_time,
            "steps_per_sec": num_steps / elapsed_time,
        }

    def _update_learning_rate(self):
        """
        Update learning rate with warmup + cosine decay.

        Warmup: Linear from 0 to base_lr over num_warmup_steps
        Decay: Cosine decay (not implemented in this minimal version)
        """
        if self.total_steps < self.num_warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.total_steps + 1) / self.num_warmup_steps
        else:
            # Constant LR (for simplicity; could add cosine decay)
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                # Model hyperparameters (for proper loading)
                "d_model": self.model.d_model,
                "num_blocks": self.model.num_blocks,
                "num_heads": self.model.num_heads,
                "d_policy": self.model.d_policy,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)

        # Update inference model
        self.update_inference_model()
