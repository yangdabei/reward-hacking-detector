"""Policy divergence metrics for reward-hacking detection."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def soften_policy(action: int, n_actions: int = 4, epsilon: float = 0.05) -> np.ndarray:
    """Convert a deterministic action to a softened probability distribution.

    Places (1 - epsilon) mass on the chosen action and distributes epsilon
    uniformly across the remaining (n_actions - 1) actions.

    Args:
        action: Deterministic action index in [0, n_actions).
        n_actions: Total number of actions (default 4).
        epsilon: Smoothing factor (default 0.05).

    Returns:
        1-D numpy array of length n_actions summing to 1.
    """
    probs = np.full(n_actions, epsilon / (n_actions - 1))
    probs[action] = 1.0 - epsilon
    return probs


def compute_kl_divergence(
    learned_policy: dict,
    reference_policy: dict,
    visited_states: list[tuple[int, int]],
    n_actions: int = 4,
    epsilon: float = 0.05,
) -> float:
    """Compute D_KL(reference || learned) averaged over visited states.

    States not present in learned_policy are skipped.

    Args:
        learned_policy: Dict mapping (row, col) -> action int for the learned agent.
        reference_policy: Dict mapping (row, col) -> action int for the reference agent.
        visited_states: States to include in the divergence estimate.
        n_actions: Number of actions (default 4).
        epsilon: Softening parameter to avoid log(0) (default 0.05).

    Returns:
        Mean KL divergence (scalar float) across all comparable states.
    """
    comparable = [s for s in visited_states if s in learned_policy]
    n = len(comparable)

    ref_actions = np.array([reference_policy[s] for s in comparable])
    learned_actions = np.array([learned_policy[s] for s in comparable])

    # Build (n, n_actions) probability matrices in one shot
    fill = epsilon / (n_actions - 1)
    ref_probs = np.full((n, n_actions), fill)
    learned_probs = np.full((n, n_actions), fill)
    ref_probs[np.arange(n), ref_actions] = 1.0 - epsilon
    learned_probs[np.arange(n), learned_actions] = 1.0 - epsilon

    kl_per_state = np.sum(ref_probs * np.log(ref_probs / learned_probs), axis=1)
    return float(kl_per_state.mean())


def compute_policy_divergence_report(
    learned_policy: dict,
    reference_policy: dict,
    visited_states: list,
    n_actions: int = 4,
) -> dict:
    """Compute a full policy divergence report.

    Args:
        learned_policy: Dict mapping (row, col) -> action int for the learned agent.
        reference_policy: Dict mapping (row, col) -> action int for the reference agent.
        visited_states: States visited during evaluation.
        n_actions: Number of actions (default 4).

    Returns:
        Dict with keys:
          - "kl_divergence": mean KL divergence (float or None if unavailable)
          - "n_states_compared": number of states included in the computation
          - "n_states_skipped": number of states skipped (not in learned_policy)
    """
    n_compared = sum(1 for s in visited_states if s in learned_policy)
    n_skipped = len(visited_states) - n_compared

    if n_compared == 0:
        logger.warning("No comparable states found; returning None for kl_divergence.")
        return {"kl_divergence": None, "n_states_compared": 0, "n_states_skipped": n_skipped}

    kl = compute_kl_divergence(learned_policy, reference_policy, visited_states, n_actions)
    logger.debug(f"KL divergence: {kl:.4f} over {n_compared} states ({n_skipped} skipped)")

    return {"kl_divergence": kl, "n_states_compared": n_compared, "n_states_skipped": n_skipped}
