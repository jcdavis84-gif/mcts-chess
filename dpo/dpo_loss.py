"""
Direct Preference Optimization (DPO) loss for chess policy learning.

DPO optimizes the policy to prefer better moves over worse moves,
using a reference policy (frozen) as a baseline.

Loss formula:
    L_DPO = -log(sigmoid(beta * (log π_θ(preferred) - log π_θ(rejected)
                                 - log π_ref(preferred) + log π_ref(rejected))))

where:
    - π_θ: Current policy (being optimized)
    - π_ref: Reference policy (frozen copy of initial model)
    - beta: Temperature parameter (controls deviation from reference)

Intuition:
    - When preferred move has higher probability than rejected, loss is low
    - When rejected move has higher probability, loss is high
    - Reference policy provides regularization (prevents collapse)

Reference:
    Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    https://arxiv.org/abs/2305.18290
"""

import torch
import torch.nn.functional as F


def dpo_loss(policy_logits, ref_logits, preferred_actions, rejected_actions, legal_masks, beta=0.1):
    """
    Compute DPO loss for a batch of preference pairs.

    Args:
        policy_logits: (batch, 4672) - Logits from current policy
        ref_logits: (batch, 4672) - Logits from reference policy (frozen)
        preferred_actions: (batch,) - Action indices for preferred moves
        rejected_actions: (batch,) - Action indices for rejected moves
        legal_masks: (batch, 4672) - Boolean mask of legal moves
        beta: Temperature parameter (default: 0.1)

    Returns:
        loss: Scalar DPO loss
        metrics: Dict with logging metrics
    """
    batch_size = policy_logits.shape[0]

    # Compute log probabilities
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)  # (batch, 4672)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # (batch, 4672)

    # Gather log probs for preferred and rejected actions
    policy_log_preferred = policy_log_probs.gather(1, preferred_actions.unsqueeze(1)).squeeze(1)  # (batch,)
    policy_log_rejected = policy_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)  # (batch,)

    ref_log_preferred = ref_log_probs.gather(1, preferred_actions.unsqueeze(1)).squeeze(1)  # (batch,)
    ref_log_rejected = ref_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)  # (batch,)

    # Compute DPO loss
    # log_ratio = log π_θ(preferred) - log π_θ(rejected) - log π_ref(preferred) + log π_ref(rejected)
    log_ratio = (policy_log_preferred - policy_log_rejected) - (ref_log_preferred - ref_log_rejected)

    # Loss = -log(sigmoid(beta * log_ratio))
    # Numerically stable: -log(sigmoid(x)) = log(1 + exp(-x))
    loss = -F.logsigmoid(beta * log_ratio).mean()

    # Compute metrics for logging
    with torch.no_grad():
        # Accuracy: how often does policy prefer the preferred action?
        policy_prefers_preferred = (policy_log_preferred > policy_log_rejected).float().mean()

        # Reference policy's preference (for comparison)
        ref_prefers_preferred = (ref_log_preferred > ref_log_rejected).float().mean()

        # Implicit reward (how much better is preferred vs rejected?)
        implicit_reward = (log_ratio / beta).mean()

        # KL divergence from reference policy
        kl_div = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1).mean()

    metrics = {
        'loss': loss.item(),
        'accuracy': policy_prefers_preferred.item(),
        'ref_accuracy': ref_prefers_preferred.item(),
        'implicit_reward': implicit_reward.item(),
        'kl_divergence': kl_div.item(),
    }

    return loss, metrics


def dpo_loss_with_margin(policy_logits, ref_logits, preferred_actions, rejected_actions,
                         legal_masks, eval_diffs, beta=0.1, margin_weight=1.0):
    """
    DPO loss with eval-difference weighting.

    Larger eval differences get more weight (stronger signal).

    Args:
        policy_logits: (batch, 4672) - Logits from current policy
        ref_logits: (batch, 4672) - Logits from reference policy (frozen)
        preferred_actions: (batch,) - Action indices for preferred moves
        rejected_actions: (batch,) - Action indices for rejected moves
        legal_masks: (batch, 4672) - Boolean mask of legal moves
        eval_diffs: (batch,) - Eval difference in pawns (preferred - rejected)
        beta: Temperature parameter (default: 0.1)
        margin_weight: How much to weight by eval difference (default: 1.0)

    Returns:
        loss: Scalar DPO loss
        metrics: Dict with logging metrics
    """
    batch_size = policy_logits.shape[0]

    # Compute log probabilities
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

    # Gather log probs
    policy_log_preferred = policy_log_probs.gather(1, preferred_actions.unsqueeze(1)).squeeze(1)
    policy_log_rejected = policy_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)

    ref_log_preferred = ref_log_probs.gather(1, preferred_actions.unsqueeze(1)).squeeze(1)
    ref_log_rejected = ref_log_probs.gather(1, rejected_actions.unsqueeze(1)).squeeze(1)

    # Compute DPO loss per sample
    log_ratio = (policy_log_preferred - policy_log_rejected) - (ref_log_preferred - ref_log_rejected)
    per_sample_loss = -F.logsigmoid(beta * log_ratio)

    # Weight by eval difference (larger differences = more important)
    # Normalize weights so mean weight = 1.0
    weights = 1.0 + margin_weight * (eval_diffs - eval_diffs.mean()) / (eval_diffs.std() + 1e-8)
    weights = weights.clamp(min=0.1)  # Prevent zero weights

    # Weighted loss
    loss = (per_sample_loss * weights).mean()

    # Metrics
    with torch.no_grad():
        policy_prefers_preferred = (policy_log_preferred > policy_log_rejected).float().mean()
        ref_prefers_preferred = (ref_log_preferred > ref_log_rejected).float().mean()
        implicit_reward = (log_ratio / beta).mean()
        kl_div = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1).mean()

    metrics = {
        'loss': loss.item(),
        'accuracy': policy_prefers_preferred.item(),
        'ref_accuracy': ref_prefers_preferred.item(),
        'implicit_reward': implicit_reward.item(),
        'kl_divergence': kl_div.item(),
        'mean_eval_diff': eval_diffs.mean().item(),
    }

    return loss, metrics


if __name__ == "__main__":
    # Simple test
    print("Testing DPO loss...")

    batch_size = 16
    num_actions = 4672

    # Create dummy data
    policy_logits = torch.randn(batch_size, num_actions)
    ref_logits = torch.randn(batch_size, num_actions)

    preferred_actions = torch.randint(0, num_actions, (batch_size,))
    rejected_actions = torch.randint(0, num_actions, (batch_size,))

    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

    # Compute loss
    loss, metrics = dpo_loss(policy_logits, ref_logits, preferred_actions, rejected_actions, legal_masks, beta=0.1)

    print(f"\nDummy batch test:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  KL divergence: {metrics['kl_divergence']:.4f}")

    # Test with margin
    eval_diffs = torch.rand(batch_size) * 2.0 + 0.5  # 0.5-2.5 pawns

    loss_margin, metrics_margin = dpo_loss_with_margin(
        policy_logits, ref_logits, preferred_actions, rejected_actions,
        legal_masks, eval_diffs, beta=0.1, margin_weight=1.0
    )

    print(f"\nWith eval-difference weighting:")
    print(f"  Loss: {loss_margin.item():.4f}")
    print(f"  Mean eval diff: {metrics_margin['mean_eval_diff']:.4f}")

    print("\n✓ DPO loss implementation working!")
