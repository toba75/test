"""Unit tests for token-return mapping."""

import numpy as np
import pytest

from stockgpt.tokens.discretizer import ReturnDiscretizer
from stockgpt.tokens.mapping import expected_return, sample_return


class TestMapping:
    """Tests for token-return mapping functions."""
    
    def test_expected_return_deterministic(self) -> None:
        """Test expected return with deterministic probability."""
        disc = ReturnDiscretizer()
        
        # All probability on token 200 (midpoint at -25 bps = -0.0025)
        probs = np.zeros(402)
        probs[200] = 1.0
        
        exp_ret = expected_return(probs[np.newaxis, :], disc)
        assert abs(exp_ret[0] - (-0.0025)) < 1e-10
    
    def test_expected_return_positive(self) -> None:
        """Test expected return with positive skew."""
        disc = ReturnDiscretizer()
        
        # All probability on token 210 (5% return from paper example)
        probs = np.zeros(402)
        probs[210] = 1.0
        
        exp_ret = expected_return(probs[np.newaxis, :], disc)
        # Should be close to 5% (0.05)
        assert 0.045 <= exp_ret[0] <= 0.055
    
    def test_expected_return_uniform(self) -> None:
        """Test expected return with uniform distribution."""
        disc = ReturnDiscretizer()
        
        # Uniform probability over all tokens
        probs = np.ones(402) / 402.0
        
        exp_ret = expected_return(probs[np.newaxis, :], disc)
        # Should be close to 0 due to symmetry
        assert abs(exp_ret[0]) < 0.01
    
    def test_expected_return_batch(self) -> None:
        """Test expected return with batch of distributions."""
        disc = ReturnDiscretizer()
        
        # Create batch of 3 distributions
        batch_probs = np.zeros((3, 402))
        batch_probs[0, 200] = 1.0  # Token 200: midpoint -0.0025
        batch_probs[1, 210] = 1.0  # Positive return
        batch_probs[2, 190] = 1.0  # Negative return
        
        exp_ret = expected_return(batch_probs, disc)
        
        assert len(exp_ret) == 3
        assert abs(exp_ret[0] - (-0.0025)) < 1e-10  # Token 200 midpoint
        assert exp_ret[1] > 0  # Positive
        assert exp_ret[2] < 0  # Negative
    
    def test_sample_return_deterministic(self) -> None:
        """Test sampling with deterministic probability."""
        disc = ReturnDiscretizer()
        rng = np.random.default_rng(42)
        
        # All probability on token 200 (midpoint -0.0025)
        probs = np.zeros(402)
        probs[200] = 1.0
        
        sampled = sample_return(probs[np.newaxis, :], disc, rng)
        assert abs(sampled[0] - (-0.0025)) < 1e-10
    
    def test_sample_return_distribution(self) -> None:
        """Test sampling follows distribution."""
        disc = ReturnDiscretizer()
        rng = np.random.default_rng(42)
        
        # 50% on token 200, 50% on token 210
        probs = np.zeros(402)
        probs[200] = 0.5
        probs[210] = 0.5
        
        # Sample many times
        n_samples = 1000
        probs_repeated = np.tile(probs, (n_samples, 1))
        samples = sample_return(probs_repeated, disc, rng)
        
        # Check that we get both values
        unique_samples = np.unique(np.round(samples, 4))
        assert len(unique_samples) <= 2  # Should only get 2 different values
        
        # Mean should be between the two midpoints
        midpoint_200 = disc.tokens_to_midpoints(np.array([200]))[0]
        midpoint_210 = disc.tokens_to_midpoints(np.array([210]))[0]
        mean_sample = np.mean(samples)
        assert midpoint_200 <= mean_sample <= midpoint_210
    
    def test_expected_vs_sampled(self) -> None:
        """Test that sampled mean approximates expected return."""
        disc = ReturnDiscretizer()
        rng = np.random.default_rng(42)
        
        # Create a non-trivial distribution
        probs = np.ones(402) / 402.0
        probs[200:220] *= 2.0  # More weight on slightly positive returns
        probs /= probs.sum()  # Renormalize
        
        # Calculate expected return
        exp_ret = expected_return(probs[np.newaxis, :], disc)[0]
        
        # Sample many times
        n_samples = 10000
        probs_repeated = np.tile(probs, (n_samples, 1))
        samples = sample_return(probs_repeated, disc, rng)
        mean_sample = np.mean(samples)
        
        # Sampled mean should be close to expected return
        assert abs(mean_sample - exp_ret) < 0.01
