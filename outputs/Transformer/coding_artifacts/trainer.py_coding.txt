"""trainer.py

This module defines the Trainer class, which manages the training loop for the Transformer model.
It uses the Adam optimizer with a custom learning rate scheduler (including warmup and inverse square-root decay),
applies gradient clipping, logs training metrics, and periodically saves checkpoints.

All hyperparameters are read from the centralized configuration (config.yaml via config.py).
The Trainer class is used by the main experiment driver (main.py) to launch training.
"""

import time
import math
import itertools
from typing import Any, Dict, List

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from config import CONFIG  # Global configuration (instance of Config)
from model import TransformerModel
from utils import LOGGER, save_checkpoint, log_metrics, create_dir_if_not_exists


class Trainer:
    """
    Trainer handles the training loop of the Transformer model.
    It receives the model, training data, development data, and configuration parameters.
    """

    def __init__(
        self,
        model: TransformerModel,
        train_data: List[Dict[str, Any]],
        dev_data: List[Dict[str, Any]],
        config: Any = CONFIG,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (TransformerModel): An instantiated Transformer model.
            train_data (List[Dict[str, Any]]): Training dataset batches.
            dev_data (List[Dict[str, Any]]): Development/validation dataset batches.
            config (Any): Configuration object loaded from config.yaml via config.py.
        """
        self.model: TransformerModel = model
        self.train_data: List[Dict[str, Any]] = train_data
        self.dev_data: List[Dict[str, Any]] = dev_data
        self.config = config

        # Set device: use GPU if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set up the Adam optimizer with hyperparameters from configuration.
        initial_lr: float = self.get_learning_rate(1)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=initial_lr,
            betas=(self.config.training.adam.beta1, self.config.training.adam.beta2),
            eps=self.config.training.adam.epsilon,
        )

        # Training hyperparameters from configuration.
        self.total_steps: int = self.config.training.total_steps
        self.warmup_steps: int = self.config.training.warmup_steps
        self.d_model: int = self.config.model.d_model

        # Global training step counter starting at 1.
        self.current_step: int = 1

        # Set checkpoint saving parameters.
        self.checkpoint_interval: int = 10000  # Save checkpoint every 10000 steps.
        self.checkpoint_dir: str = "checkpoints"
        create_dir_if_not_exists(self.checkpoint_dir)

        # Create a cyclic iterator for training batches.
        self.train_iter = itertools.cycle(self.train_data)

        # Set gradient clipping norm (optional safeguard).
        self.grad_clip: float = 1.0

        # Define logging interval (in steps).
        self.log_interval: int = 100

        LOGGER.info(
            f"Trainer initialized on device {self.device}. Total training steps: {self.total_steps}."
        )

    def get_learning_rate(self, step: int) -> float:
        """
        Compute the learning rate based on the current training step using the formula:
            lr = (d_model)^(-0.5) * min(step^(-0.5), step * (warmup_steps)^(-1.5))

        Args:
            step (int): Current training step.

        Returns:
            float: The computed learning rate.
        """
        lr = (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )
        return lr

    def save_checkpoint(self) -> None:
        """
        Saves the current state of the model, optimizer, and training step to a checkpoint file.
        Uses the save_checkpoint helper function from utils.py.
        """
        save_checkpoint(
            self.model, self.optimizer, self.current_step, self.checkpoint_dir
        )

    def pad_batch(self, sequences: List[List[int]]) -> torch.Tensor:
        """
        Pads a batch of sequences to have uniform length using zero as the padding value.

        Args:
            sequences (List[List[int]]): List of sequences (each sequence is a list of token IDs).

        Returns:
            torch.Tensor: Padded tensor of shape (batch_size, max_seq_len).
        """
        tensor_seqs = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        padded_batch = pad_sequence(tensor_seqs, batch_first=True, padding_value=0)
        return padded_batch

    def train(self) -> None:
        """
        Runs the complete training loop until the global training step reaches total_steps.

        For each training step:
            - Retrieves a batch from the training data.
            - Moves the batch to the appropriate device.
            - Performs a forward pass and computes label-smoothed loss via model.train_step().
            - Backpropagates the loss and updates model parameters.
            - Dynamically updates the learning rate according to the scheduling formula.
            - Logs training metrics periodically.
            - Saves checkpoints at predefined intervals.
        """
        self.model.train()  # Enable training mode.
        start_time = time.time()
        progress_bar = tqdm(total=self.total_steps, desc="Training Steps", unit="step")

        while self.current_step <= self.total_steps:
            # Retrieve next batch from cyclic iterator.
            batch = next(self.train_iter)

            # Convert batch lists to padded tensors and move to device.
            # For translation tasks, expect keys "src" and "tgt".
            # For parsing tasks, expect key "inputs".
            if "src" in batch and "tgt" in batch:
                src_tensor = self.pad_batch(batch["src"]).to(self.device)
                tgt_tensor = self.pad_batch(batch["tgt"]).to(self.device)
                batch = {"src": src_tensor, "tgt": tgt_tensor}
            elif "inputs" in batch:
                inputs_tensor = self.pad_batch(batch["inputs"]).to(self.device)
                batch = {"inputs": inputs_tensor}
            else:
                LOGGER.error(
                    "Batch format not recognized. Expected keys 'src'/'tgt' or 'inputs'."
                )
                continue

            # Forward pass: compute logits and loss with label smoothing.
            output: Dict[str, torch.Tensor] = self.model.train_step(batch)
            loss: torch.Tensor = output.get("loss")
            if loss is None:
                LOGGER.error("Loss not computed during training step.")
                continue

            # Backpropagation.
            self.optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping.
            clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Optimizer step.
            self.optimizer.step()

            # Update learning rate according to schedule.
            current_lr: float = self.get_learning_rate(self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

            # Logging: log every self.log_interval steps.
            if self.current_step % self.log_interval == 0:
                elapsed_time = time.time() - start_time
                metrics = {
                    "loss": loss.item(),
                    "lr": current_lr,
                    "step": self.current_step,
                    "elapsed_time_sec": elapsed_time,
                }
                log_metrics(self.current_step, metrics)

            # Checkpointing: save checkpoint every checkpoint_interval steps.
            if self.current_step % self.checkpoint_interval == 0:
                self.save_checkpoint()

            self.current_step += 1
            progress_bar.update(1)

        progress_bar.close()
        LOGGER.info("Training completed.")
