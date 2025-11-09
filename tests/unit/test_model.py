"""Unit tests for GPT model architecture."""

import torch

from stockgpt.model.gpt import StockGPT, create_model
from stockgpt.model.mask import create_causal_mask


class TestCausalMask:
    """Tests for causal mask creation."""

    def test_mask_shape(self) -> None:
        """Test mask has correct shape."""
        seq_len = 10
        mask = create_causal_mask(seq_len)
        assert mask.shape == (seq_len, seq_len)

    def test_mask_is_lower_triangular(self) -> None:
        """Test mask is lower triangular."""
        seq_len = 5
        mask = create_causal_mask(seq_len)

        # Check diagonal and below are True
        for i in range(seq_len):
            for j in range(i + 1):
                assert mask[i, j].item() is True

        # Check above diagonal is False
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert mask[i, j].item() is False

    def test_mask_device(self) -> None:
        """Test mask can be created on different devices."""
        seq_len = 10
        mask_cpu = create_causal_mask(seq_len, device=torch.device('cpu'))
        assert mask_cpu.device.type == 'cpu'


class TestStockGPT:
    """Tests for StockGPT model."""

    def test_model_creation(self) -> None:
        """Test model can be created with default parameters."""
        model = create_model()
        assert isinstance(model, StockGPT)
        assert model.vocab_size == 402
        assert model.seq_len == 256
        assert model.d_model == 128
        assert model.n_layers == 4

    def test_parameter_count(self) -> None:
        """Test model has approximately 0.93M parameters."""
        model = create_model()
        n_params = model.count_parameters()

        # Should be around 930k parameters (allow 10% tolerance)
        target = 930_000
        tolerance = 0.15  # 15% tolerance
        assert abs(n_params - target) / target < tolerance

        print(f"Model has {n_params:,} parameters (target: ~930,000)")

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        batch_size = 8
        seq_len = 256
        vocab_size = 402

        model = create_model()
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(tokens)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_forward_pass_shorter_sequence(self) -> None:
        """Test forward pass with sequence shorter than max length."""
        batch_size = 4
        seq_len = 100
        vocab_size = 402

        model = create_model()
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(tokens)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_model_output_range(self) -> None:
        """Test model output logits are in reasonable range."""
        batch_size = 2
        seq_len = 10
        vocab_size = 402

        model = create_model()
        model.eval()

        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits = model(tokens)

        # Logits should not be too extreme
        assert torch.isfinite(logits).all()
        assert logits.abs().max() < 100.0

    def test_model_gradients(self) -> None:
        """Test model can compute gradients."""
        batch_size = 2
        seq_len = 10
        vocab_size = 402

        model = create_model()
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = model(tokens)
        loss = logits.mean()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_custom_parameters(self) -> None:
        """Test model can be created with custom parameters."""
        model = create_model(
            vocab_size=500,
            seq_len=128,
            d_model=64,
            n_layers=2,
            n_heads=2,
            dropout=0.1,
        )

        assert model.vocab_size == 500
        assert model.seq_len == 128
        assert model.d_model == 64
        assert model.n_layers == 2

    def test_causal_property(self) -> None:
        """Test that model respects causality (future doesn't affect past)."""
        model = create_model()
        model.eval()

        batch_size = 1
        seq_len = 10
        vocab_size = 402

        # Create two sequences that differ only in the last position
        tokens1 = torch.randint(0, vocab_size, (batch_size, seq_len))
        tokens2 = tokens1.clone()
        tokens2[:, -1] = (tokens1[:, -1] + 1) % vocab_size

        with torch.no_grad():
            logits1 = model(tokens1)
            logits2 = model(tokens2)

        # Predictions for all positions except the last should be identical
        torch.testing.assert_close(logits1[:, :-1, :], logits2[:, :-1, :])

        # Prediction for the last position should differ
        assert not torch.allclose(logits1[:, -1, :], logits2[:, -1, :])
