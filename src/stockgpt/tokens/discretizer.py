"""Return discretizer following StockGPT protocol.

This module implements the discretization scheme described in the paper:
- 50 basis points (bps) bins
- 402 tokens (indices 0-401)
- Closed-right intervals
- Cap at ±100% (±10,000 bps)

Reference: docs/ssrn-4787199.pdf, Section on Tokenization
"""

import numpy as np
from numpy.typing import NDArray


class ReturnDiscretizer:
    """Discretizes continuous returns into tokens.

    Following the paper's protocol:
    - Bins of 50 bps width
    - 402 total tokens (0-401)
    - Closed-right intervals: (-inf, b1], (b1, b2], ..., (b_n-1, inf)
    - Capped at ±10,000 bps (±100%)

    Token mapping:
    - Token 0: (-inf, -10000] (extreme negative)
    - Token 1-200: (-10000, 0] in 50 bps increments
    - Token 200: Exactly 0%
    - Token 201-400: (0, 10000] in 50 bps increments
    - Token 401: (10000, inf) (extreme positive)

    Examples:
        >>> disc = ReturnDiscretizer()
        >>> # Paper example: -2.4%, 0%, 0%, 5%, 4.8%
        >>> returns = np.array([-0.024, 0.0, 0.0, 0.05, 0.048])
        >>> tokens = disc.returns_to_tokens(returns)
        >>> # Expected: [196, 200, 200, 210, 210]
        >>> print(tokens)
        [196 200 200 210 210]
    """

    BIN_WIDTH_BPS: int = 50
    MAX_BPS: int = 10000
    N_TOKENS: int = 402

    def __init__(self) -> None:
        """Initialize the discretizer with bin boundaries."""
        self.bin_width_bps = self.BIN_WIDTH_BPS
        self.max_bps = self.MAX_BPS
        self.n_tokens = self.N_TOKENS

        # Create bin edges
        # Range from -10000 to +10000 in 50 bps steps
        # Total: 400 bins + 2 extreme bins
        self.bin_edges = self._create_bin_edges()

    def _create_bin_edges(self) -> NDArray[np.float64]:
        """Create bin edges for discretization.

        Returns:
            Array of bin edges in basis points.
            Shape: (400,) for 401 interior bins (not counting extremes)
        """
        # Create edges from -10000 to +10000 in steps of 50
        # This creates 400 edges for 401 interior bins
        # Token 0: < -10000
        # Tokens 1-200: (-10000, 0] in 50 bps increments
        # Tokens 201-400: (0, 10000] in 50 bps increments
        # Token 401: > 10000
        return np.arange(
            -self.max_bps,
            self.max_bps + 1,
            self.bin_width_bps,
            dtype=np.float64
        )

    def returns_to_tokens(self, returns: NDArray[np.float64]) -> NDArray[np.int64]:
        """Convert returns to token indices.

        Args:
            returns: Array of returns as decimals (e.g., 0.05 for 5%).
                    Shape: (n,)

        Returns:
            Array of token indices in range [0, 401].
            Shape: (n,)
        """
        # Convert returns to basis points
        returns_bps = returns * 10000.0

        # Cap at ±10,000 bps
        returns_bps = np.clip(returns_bps, -self.max_bps, self.max_bps)

        # Use digitize with right=True for closed-right intervals
        # digitize returns indices where returns_bps[i] would be inserted
        # With right=True: bin edges are right-inclusive
        tokens = np.digitize(returns_bps, self.bin_edges, right=True)

        # Ensure tokens are in valid range [0, 401]
        return np.clip(tokens, 0, self.n_tokens - 1).astype(np.int64)


    def tokens_to_midpoints(self, tokens: NDArray[np.int64]) -> NDArray[np.float64]:
        """Convert token indices to bin midpoint returns.

        Args:
            tokens: Array of token indices in range [0, 401].
                   Shape: (n,)

        Returns:
            Array of midpoint returns as decimals.
            Shape: (n,)
        """
        midpoints = np.zeros(len(tokens), dtype=np.float64)

        for i, token in enumerate(tokens):
            if token == 0:
                # Extreme negative: use -10000 bps
                midpoints[i] = -1.0
            elif token == self.n_tokens - 1:
                # Extreme positive: use +10000 bps
                midpoints[i] = 1.0
            else:
                # Regular bin: calculate midpoint
                # Token k (k=1..400) corresponds to bin (edges[k-1], edges[k]]
                left_edge = self.bin_edges[token - 1]
                right_edge = self.bin_edges[token]
                midpoint_bps = (left_edge + right_edge) / 2.0
                midpoints[i] = midpoint_bps / 10000.0

        return midpoints

    def get_all_midpoints(self) -> NDArray[np.float64]:
        """Get midpoints for all 402 tokens.

        Returns:
            Array of midpoint returns for all tokens.
            Shape: (402,)
        """
        tokens = np.arange(self.n_tokens, dtype=np.int64)
        return self.tokens_to_midpoints(tokens)
