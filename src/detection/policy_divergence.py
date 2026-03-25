"""Policy divergence metrics. TODO: Implement KL divergence (ML Exercise 4)."""

import numpy as np
import logging
from typing import Any

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

    Args:
        learned_policy: Dict mapping (row, col) -> action int for the learned agent.
        reference_policy: Dict mapping (row, col) -> action int for the reference agent.
        visited_states: States to include in the divergence estimate.
        n_actions: Number of actions (default 4).
        epsilon: Softening parameter (default 0.05).

    Returns:
        Mean KL divergence (scalar float) across all comparable states.
    """
    # TODO [ML EXERCISE 4 — KL Divergence Between Policies]:
    #
    # Compute: D_KL(reference || learned) = sum_a ref(a|s) * log(ref(a|s) / learned(a|s))
    # averaged over all visited_states.
    #
    # Steps:
    # 1. For each state in visited_states:
    #    a. Get learned action from learned_policy[state] (int)
    #    b. Get reference action from reference_policy[state] (int)
    #    c. Convert both to probability distributions using soften_policy()
    #    d. Compute KL divergence for this state:
    #       kl = sum(ref_probs * np.log(ref_probs / learned_probs + 1e-10))
    # 2. Return the mean KL divergence across all states
    # 3. Skip states not in learned_policy (they haven't been visited during training)
    #
    # Think about: why use KL(ref || learned) rather than KL(learned || ref)?
    # What happens when learned_probs is near 0 for an action the reference policy uses?
    raise NotImplementedError("Implement compute_kl_divergence() — see TODO above")


def compute_policy_divergence_report(
    learned_policy: dict,
    reference_policy: dict,
    visited_states: list,
    n_actions: int = 4,
) -> dict:
    """Compute a full policy divergence report.

    Compares the learned policy against the reference policy over the
    supplied visited states and returns summary statistics.

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
    # TODO [ML EXERCISE 4 — Policy Divergence Report]:
    # Implement compute_policy_divergence_report():
    # Steps:
    # 1. Count how many visited_states are present in learned_policy
    #    (n_states_compared) and how many are absent (n_states_skipped).
    # 2. Filter visited_states to only those in both policies.
    # 3. If no comparable states: return {"kl_divergence": None, ...}
    # 4. Call compute_kl_divergence() on the filtered state list.
    # 5. Return the result dict with all three keys.
    raise NotImplementedError(
        "Implement compute_policy_divergence_report() — see TODO above"
    )
