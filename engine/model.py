import torch
import torch.nn as nn
import numpy as np


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block with self-attention and MLP.
    """

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # x shape: (batch, 64, d_model)

        # Pre-LayerNorm attention
        h1 = self.ln1(x)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + attn_out

        # Pre-LayerNorm MLP
        h2 = self.ln2(x)
        ff_out = self.mlp(h2)
        x = x + ff_out

        return x


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero-style transformer network with policy, value, and material heads.

    Architecture:
    - Input: 18x8x8 board encoding
    - Tokenization: Flatten to 64x18, project to d_model
    - Transformer: N blocks of self-attention
    - Policy head: Projects to 4672 action logits
    - Value head: Mean-pool -> MLP -> Tanh (scalar in [-1, 1])
    - Material head: Mean-pool -> MLP -> scalar (unbounded, White - Black material balance)
    """

    def __init__(self, d_model=128, num_blocks=4, num_heads=2, d_policy=64):
        super().__init__()

        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_policy = d_policy

        # Tokenization: project 18-dim input to d_model
        self.token_proj = nn.Linear(18, d_model)

        # Positional embedding: learned 64 x d_model
        self.pos_embedding = nn.Parameter(torch.randn(64, d_model) * 0.02)

        # Transformer blocks
        d_ff = 4 * d_model  # MLP hidden dim = 4x d_model
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_proj = nn.Linear(d_model, d_policy)  # Per-token projection
        self.policy_head = nn.Linear(64 * d_policy, 4672)  # Flatten and project to action space

        # Value head
        self.value_fc1 = nn.Linear(d_model, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Material balance head
        self.material_fc1 = nn.Linear(d_model, 256)
        self.material_fc2 = nn.Linear(256, 1)

    def forward(self, x, legal_moves_mask=None):
        """
        Args:
            x: (batch, 18, 8, 8) board encoding
            legal_moves_mask: (batch, 4672) boolean mask, True for legal moves

        Returns:
            policy_logits: (batch, 4672) raw logits (illegal moves masked to -1e9)
            value: (batch,) scalar in [-1, 1]
            material: (batch,) scalar material balance (unbounded)
        """
        batch_size = x.shape[0]

        # Flatten spatial dimensions: (batch, 18, 8, 8) -> (batch, 64, 18)
        # Rearrange: rank * 8 + file indexing
        x = x.view(batch_size, 18, 64).transpose(1, 2)  # (batch, 64, 18)

        # Project to d_model
        x = self.token_proj(x)  # (batch, 64, d_model)

        # Add positional embeddings
        x = x + self.pos_embedding.unsqueeze(0)  # (batch, 64, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)  # (batch, 64, d_model)

        # Policy head
        policy_tokens = self.policy_proj(x)  # (batch, 64, d_policy)
        policy_flat = policy_tokens.reshape(batch_size, 64 * self.d_policy)
        policy_logits = self.policy_head(policy_flat)  # (batch, 4672)

        # Mask illegal moves
        if legal_moves_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_moves_mask, -1e9)

        # Value head: mean-pool across tokens
        value_pooled = x.mean(dim=1)  # (batch, d_model)
        value = self.value_fc1(value_pooled)
        value = torch.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value).squeeze(-1)  # (batch,)

        # Material balance head: mean-pool across tokens
        material_pooled = x.mean(dim=1)  # (batch, d_model)
        material = self.material_fc1(material_pooled)
        material = torch.relu(material)
        material = self.material_fc2(material)
        material = material.squeeze(-1)  # (batch,) - unbounded

        return policy_logits, value, material

    def predict(self, board_tensor, legal_moves_mask=None):
        """
        Convenience method for single-board prediction (used in MCTS).

        Args:
            board_tensor: (18, 8, 8) numpy array or tensor
            legal_moves_mask: (4672,) boolean mask

        Returns:
            policy: (4672,) numpy array of probabilities
            value: float
        Note: Material prediction is computed but not returned (MCTS doesn't need it)
        """
        self.eval()
        with torch.no_grad():
            # Get model device
            device = next(self.parameters()).device

            # Convert to tensor if needed
            if isinstance(board_tensor, np.ndarray):
                board_tensor = torch.from_numpy(board_tensor).float()

            # Add batch dimension and move to device
            x = board_tensor.unsqueeze(0).to(device)  # (1, 18, 8, 8)

            if legal_moves_mask is not None:
                if isinstance(legal_moves_mask, np.ndarray):
                    legal_moves_mask = torch.from_numpy(legal_moves_mask).bool()
                legal_moves_mask = legal_moves_mask.unsqueeze(0).to(device)  # (1, 4672)

            # Forward pass
            policy_logits, value, material = self.forward(x, legal_moves_mask)

            # Convert to probabilities
            policy = torch.softmax(policy_logits, dim=-1)

            # Return numpy arrays (material not returned for MCTS compatibility)
            return policy.cpu().numpy()[0], value.cpu().item()
