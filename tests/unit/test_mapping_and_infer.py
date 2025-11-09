import numpy as np
from stockgpt.tokens.mapping import expected_return, sample_return
from stockgpt.tokens.discretizer import ReturnDiscretizer


def test_expected_return_uniform():
    disc = ReturnDiscretizer()
    # create uniform probs for single example
    probs = np.ones((1, disc.n_tokens), dtype=float) / disc.n_tokens
    exp = expected_return(probs, disc)
    assert exp.shape == (1,)
    assert isinstance(exp[0], float)


def test_sample_return_distribution():
    disc = ReturnDiscretizer()
    rng = np.random.default_rng(123)
    probs = np.ones((5, disc.n_tokens), dtype=float) / disc.n_tokens
    samples = sample_return(probs, disc, rng)
    assert samples.shape == (5,)

