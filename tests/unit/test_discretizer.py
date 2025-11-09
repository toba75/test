"""Unit tests for return discretizer."""

import numpy as np
import pytest

from stockgpt.tokens.discretizer import ReturnDiscretizer


class TestReturnDiscretizer:
    """Tests for ReturnDiscretizer class."""
    
    def test_initialization(self) -> None:
        """Test discretizer initializes with correct parameters."""
        disc = ReturnDiscretizer()
        assert disc.bin_width_bps == 50
        assert disc.max_bps == 10000
        assert disc.n_tokens == 402
        assert len(disc.bin_edges) == 401  # 400 interior bins + 1
    
    def test_paper_example(self) -> None:
        """Test the example from the paper.
        
        Paper example (AGENT.md line 8):
        Returns: -2.4%, 0%, 0%, 5%, 4.8%
        Expected tokens: 196, 200, 200, 210, 210
        """
        disc = ReturnDiscretizer()
        returns = np.array([-0.024, 0.0, 0.0, 0.05, 0.048])
        tokens = disc.returns_to_tokens(returns)
        expected = np.array([196, 200, 200, 210, 210])
        np.testing.assert_array_equal(tokens, expected)
    
    def test_zero_return(self) -> None:
        """Test that 0% return maps to token 200.
        
        Token 200 covers interval (-50, 0] bps, so 0.0 is at the right edge.
        """
        disc = ReturnDiscretizer()
        returns = np.array([0.0])
        tokens = disc.returns_to_tokens(returns)
        assert tokens[0] == 200
    
    def test_extreme_negative(self) -> None:
        """Test extreme negative returns are capped."""
        disc = ReturnDiscretizer()
        returns = np.array([-2.0, -1.5, -1.0])  # -200%, -150%, -100%
        tokens = disc.returns_to_tokens(returns)
        # All should be in the extreme negative range (token 0 or 1)
        assert np.all(tokens <= 1)
    
    def test_extreme_positive(self) -> None:
        """Test extreme positive returns are capped."""
        disc = ReturnDiscretizer()
        returns = np.array([1.0, 1.5, 2.0])  # 100%, 150%, 200%
        tokens = disc.returns_to_tokens(returns)
        # All should be in the extreme positive range (token 400 or 401)
        # 100% = 10000 bps exactly, which is at right edge of token 400
        assert np.all(tokens >= 400)
    
    def test_symmetric_returns(self) -> None:
        """Test that positive and negative returns are symmetric."""
        disc = ReturnDiscretizer()
        test_return = 0.025  # 2.5% = 250 bps = 5 bins from zero
        returns = np.array([test_return, -test_return])
        tokens = disc.returns_to_tokens(returns)
        
        # Token 200 is zero, so +2.5% and -2.5% should be equidistant
        assert abs(tokens[0] - 200) == abs(tokens[1] - 200)
    
    def test_bin_boundaries(self) -> None:
        """Test returns at bin boundaries."""
        disc = ReturnDiscretizer()
        
        # Token 200 = (-50, 0] bps
        # Token 201 = (0, 50] bps
        returns = np.array([0.0, -0.0001, 0.0001])
        tokens = disc.returns_to_tokens(returns)
        assert tokens[0] == 200  # 0 bps at right edge of (-50, 0]
        assert tokens[1] == 200  # -1 bps in (-50, 0]
        assert tokens[2] == 201  # 1 bps in (0, 50]
        
        # Test exact boundary at 50 bps
        returns2 = np.array([0.005, 0.0049])
        tokens2 = disc.returns_to_tokens(returns2)
        assert tokens2[0] == 201  # 50 bps exactly, at right edge of (0, 50]
        assert tokens2[1] == 201  # 49 bps in (0, 50]
    
    def test_tokens_to_midpoints(self) -> None:
        """Test converting tokens back to midpoint returns."""
        disc = ReturnDiscretizer()
        
        # Test token 200: midpoint of (-50, 0] = -25 bps = -0.0025
        tokens = np.array([200])
        midpoints = disc.tokens_to_midpoints(tokens)
        assert abs(midpoints[0] - (-0.0025)) < 1e-10
        
        # Test extreme tokens
        tokens_extreme = np.array([0, 401])
        midpoints_extreme = disc.tokens_to_midpoints(tokens_extreme)
        assert midpoints_extreme[0] == -1.0
        assert midpoints_extreme[1] == 1.0
    
    def test_get_all_midpoints(self) -> None:
        """Test getting all midpoints."""
        disc = ReturnDiscretizer()
        midpoints = disc.get_all_midpoints()
        
        assert len(midpoints) == 402
        assert midpoints[0] == -1.0  # Extreme negative
        assert midpoints[401] == 1.0  # Extreme positive
        # Token 200 midpoint is -25 bps = -0.0025
        assert abs(midpoints[200] - (-0.0025)) < 1e-10
    
    def test_roundtrip_consistency(self) -> None:
        """Test that returns -> tokens -> midpoints is consistent."""
        disc = ReturnDiscretizer()
        
        # Generate test returns
        returns = np.linspace(-0.5, 0.5, 100)
        
        # Convert to tokens and back
        tokens = disc.returns_to_tokens(returns)
        midpoints = disc.tokens_to_midpoints(tokens)
        
        # Midpoints should be close to original (within bin width)
        differences = np.abs(returns - midpoints)
        max_diff = 0.005  # 50 bps / 2 = 0.25 bps, but allow 0.5% tolerance
        assert np.all(differences <= max_diff)
    
    def test_token_range(self) -> None:
        """Test that all tokens are in valid range [0, 401]."""
        disc = ReturnDiscretizer()
        
        # Generate many random returns
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.1, 10000)
        
        tokens = disc.returns_to_tokens(returns)
        
        assert np.all(tokens >= 0)
        assert np.all(tokens <= 401)
