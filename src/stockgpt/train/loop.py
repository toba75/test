"""Training loop for StockGPT."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..model.gpt import StockGPT
from .optim import create_cosine_schedule_with_warmup, create_optimizer


class Trainer:
    """Trainer for StockGPT model.

    Args:
        model: StockGPT model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration dictionary
        device: Device to train on
    """

    def __init__(
        self,
        model: StockGPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device setup
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model.to(device)

        # Training parameters
        self.num_steps = config.get('num_steps', 10000)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.log_interval = config.get('log_interval', 100)
        self.eval_interval = config.get('eval_interval', 500)
        self.save_interval = config.get('save_interval', 1000)
        self.use_amp = config.get('use_amp', True)

        # Optimizer and scheduler
        self.optimizer = create_optimizer(
            model,
            learning_rate=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01),
        )

        self.scheduler = create_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 500),
            num_training_steps=self.num_steps,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

    def train_step(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor) -> float:
        """Single training step.

        Args:
            input_tokens: Input token sequences (batch, seq_len)
            target_tokens: Target token sequences (batch, seq_len)

        Returns:
            Loss value
        """
        self.model.train()

        input_tokens = input_tokens.to(self.device)
        target_tokens = target_tokens.to(self.device)

        # Forward pass with automatic mixed precision
        if self.use_amp and self.scaler is not None:
            with autocast(dtype=torch.bfloat16):
                logits = self.model(input_tokens)
                # Compute loss: predict next token at each position
                # Shift logits and targets for next-token prediction
                logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets_flat = target_tokens[:, 1:].contiguous().view(-1)
                loss = self.criterion(logits_flat, targets_flat)
        else:
            logits = self.model(input_tokens)
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets_flat = target_tokens[:, 1:].contiguous().view(-1)
            loss = self.criterion(logits_flat, targets_flat)

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for input_tokens, target_tokens in self.val_loader:
            input_tokens = input_tokens.to(self.device)
            target_tokens = target_tokens.to(self.device)

            if self.use_amp:
                with autocast(dtype=torch.bfloat16):
                    logits = self.model(input_tokens)
                    logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                    targets_flat = target_tokens[:, 1:].contiguous().view(-1)
                    loss = self.criterion(logits_flat, targets_flat)
            else:
                logits = self.model(input_tokens)
                logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets_flat = target_tokens[:, 1:].contiguous().view(-1)
                loss = self.criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

            # Limit evaluation to speed up
            if num_batches >= 50:
                break

        return total_loss / num_batches if num_batches > 0 else 0.0


    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.parent / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def train(self, checkpoint_dir: Path) -> None:
        """Run full training loop.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        print(f"Starting training for {self.num_steps} steps")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")

        train_iter = iter(self.train_loader)

        start_time = time.time()

        while self.step < self.num_steps:
            try:
                input_tokens, target_tokens = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                input_tokens, target_tokens = next(train_iter)

            # Training step
            loss = self.train_step(input_tokens, target_tokens)
            self.train_losses.append(loss)
            self.step += 1

            # Logging
            if self.step % self.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed
                print(
                    f"Step {self.step}/{self.num_steps} | "
                    f"Loss: {loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Steps/s: {steps_per_sec:.2f}"
                )

            # Evaluation
            if self.step % self.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append((self.step, val_loss))
                print(f"Validation loss at step {self.step}: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        checkpoint_dir / f'checkpoint_step_{self.step}.pt',
                        is_best=True
                    )
                    print(f"New best model saved (val_loss: {val_loss:.4f})")

            # Periodic checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'checkpoint_step_{self.step}.pt'
                )

        print("\nTraining completed!")
        print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
