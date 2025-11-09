"""Token-return mapping utilities."""

import numpy as np
from numpy.typing import NDArray

from .discretizer import ReturnDiscretizer


def expected_return(
    probs: NDArray[np.float64],
    discretizer: ReturnDiscretizer | None = None
) -> NDArray[np.float64]:
    """Calculate expected return from probability distribution over tokens.

    Following the paper's protocol: E[r] is computed as the weighted average
    of bin midpoints using the predicted probability distribution.

    Args:
        probs: Probability distribution over tokens.
               Shape: (..., 402) where last dimension is token probabilities
        discretizer: ReturnDiscretizer instance. If None, creates a new one.

    Returns:
        Expected returns as decimals.
        Shape: (...) - same as input without last dimension
    """
    if discretizer is None:
        discretizer = ReturnDiscretizer()

    # Get midpoints for all tokens
    midpoints = discretizer.get_all_midpoints()

    # Calculate weighted average: E[r] = sum(p_i * midpoint_i)
    return np.sum(probs * midpoints, axis=-1)



def sample_return(
    probs: NDArray[np.float64],
    discretizer: ReturnDiscretizer | None = None,
    rng: np.random.Generator | None = None
) -> NDArray[np.float64]:
    """Sample a return from probability distribution over tokens.

    Args:
        probs: Probability distribution over tokens.
               Shape: (..., 402)
        discretizer: ReturnDiscretizer instance. If None, creates a new one.
        rng: Random number generator. If None, uses default RNG.

    Returns:
        Sampled returns (using bin midpoints).
        Shape: (...) - same as input without last dimension
    """
    if discretizer is None:
        discretizer = ReturnDiscretizer()

    if rng is None:
        rng = np.random.default_rng()

    # Flatten all but last dimension for sampling
    original_shape = probs.shape[:-1]
    probs_flat = probs.reshape(-1, probs.shape[-1])

    # Sample tokens
    sampled_tokens = np.array([
        rng.choice(discretizer.n_tokens, p=p)
        for p in probs_flat
    ])

    # Convert to returns
    sampled_returns = discretizer.tokens_to_midpoints(sampled_tokens)

    # Reshape back to original shape
    return sampled_returns.reshape(original_shape)
