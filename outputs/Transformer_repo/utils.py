"""
utils.py

This module provides utility functions for logging, checkpoint management, tokenization, 
and miscellaneous file/directory operations. All functions rely on configuration 
parameters sourced from the central config.py to ensure consistency and reproducibility.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

# Import configuration from config.py
from config import CONFIG


def initialize_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Initialize and configure a logger instance.

    Args:
        log_level (str): Logging level as a string (e.g., "INFO", "DEBUG"). Default is "INFO".
        log_file (Optional[str]): Optional file path for file logging. If provided,
                                  logs will also be written to this file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("transformer_logger")
    # Set logging level based on provided log_level or default to INFO.
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Prevent adding duplicate handlers if logger is already configured.
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        
        # Console handler for standard output.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler, if a log file path is provided.
        if log_file:
            # Ensure the directory for the log file exists.
            log_dir = os.path.dirname(log_file)
            if log_dir:
                create_dir_if_not_exists(log_dir)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger


# Global logger instance to be used by all modules.
LOGGER: logging.Logger = initialize_logger()


def log_metrics(step: int, metrics: Dict[str, Any], logger: logging.Logger = LOGGER) -> None:
    """
    Log training progress metrics in a standardized format.

    Args:
        step (int): The current training step/epoch.
        metrics (Dict[str, Any]): Dictionary containing metric names and their corresponding values.
        logger (logging.Logger): Logger instance to use. Defaults to the global LOGGER.
    """
    metrics_str = ", ".join(
        [f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items()]
    )
    logger.info(f"Step {step}: {metrics_str}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    additional_state: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save the current training state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model whose state is to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state is to be saved.
        step (int): The current training step.
        checkpoint_dir (str): Directory where checkpoint files are to be saved.
        additional_state (Optional[Dict[str, Any]]): Any additional state information (e.g., scheduler state).
    """
    try:
        create_dir_if_not_exists(checkpoint_dir)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step
        }
        if additional_state:
            checkpoint.update(additional_state)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pth")
        torch.save(checkpoint, checkpoint_path)
        LOGGER.info(f"Checkpoint saved at step {step} to {checkpoint_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save checkpoint at step {step}: {str(e)}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Load a checkpoint from the disk and update the model and optimizer states.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): Model whose state is to be loaded.
        optimizer (torch.optim.Optimizer): Optimizer whose state is to be loaded.

    Returns:
        Dict[str, Any]: The checkpoint dictionary containing additional state information (if any).
    """
    try:
        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint.get("step", 0)
        LOGGER.info(f"Checkpoint loaded from {filepath} at step {step}")
        return checkpoint
    except Exception as e:
        LOGGER.error(f"Error loading checkpoint from {filepath}: {str(e)}")
        return {}


def load_tokenizer(tokenizer_type: str, vocab_file: Optional[str] = None) -> Any:
    """
    Load and initialize a tokenizer based on the specified tokenizer type.

    Args:
        tokenizer_type (str): Type of tokenizer to load ("BPE" or "wordpiece").
        vocab_file (Optional[str]): Optional path to a tokenizer/vocabulary file.
                                    If None, a default file is used based on the tokenizer type.

    Returns:
        Any: An initialized tokenizer instance.
    """
    default_vocab_file: str = ""
    if tokenizer_type.lower() == "bpe":
        default_vocab_file = "bpe_tokenizer.json"
    elif tokenizer_type.lower() == "wordpiece":
        default_vocab_file = "wordpiece_tokenizer.json"
    else:
        LOGGER.error(f"Unsupported tokenizer type: {tokenizer_type}. Falling back to basic whitespace tokenizer.")
    
    # Use the provided vocabulary file if available; otherwise, use the default.
    vocab_file_to_use = vocab_file if vocab_file is not None else default_vocab_file
    
    try:
        if vocab_file_to_use and os.path.isfile(vocab_file_to_use):
            tokenizer = Tokenizer.from_file(vocab_file_to_use)
            LOGGER.info(f"Loaded tokenizer from {vocab_file_to_use}")
        else:
            # If a vocabulary file is not found, initialize a basic whitespace tokenizer.
            LOGGER.warning(f"Tokenizer file '{vocab_file_to_use}' not found. Initializing basic whitespace tokenizer.")
            from tokenizers.models import WordLevel
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        return tokenizer
    except Exception as e:
        LOGGER.error(f"Failed to load tokenizer: {str(e)}")
        raise e


def encode_text(text: str, tokenizer: Any) -> List[int]:
    """
    Encode a text string into a sequence of token IDs using the provided tokenizer.

    Args:
        text (str): Input text to encode.
        tokenizer (Any): Tokenizer instance that supports the encode() method.

    Returns:
        List[int]: The list of token IDs produced by the tokenizer.
    """
    try:
        encoding = tokenizer.encode(text)
        # The tokenizer's encode method typically returns an object with an 'ids' attribute.
        if hasattr(encoding, "ids"):
            return encoding.ids
        return encoding
    except Exception as e:
        LOGGER.error(f"Error encoding text: {str(e)}")
        return []


def decode_tokens(token_ids: List[int], tokenizer: Any) -> str:
    """
    Decode a sequence of token IDs back into a text string using the provided tokenizer.

    Args:
        token_ids (List[int]): List of token IDs to decode.
        tokenizer (Any): Tokenizer instance that supports the decode() method.

    Returns:
        str: The decoded text string.
    """
    try:
        decoded_text = tokenizer.decode(token_ids)
        return decoded_text
    except Exception as e:
        LOGGER.error(f"Error decoding tokens: {str(e)}")
        return ""


def create_dir_if_not_exists(path: str) -> None:
    """
    Ensure that a directory exists; if not, create it.

    Args:
        path (str): Path of the directory to create.
    """
    try:
        if path and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)
            LOGGER.info(f"Directory created at: {path}")
    except Exception as e:
        LOGGER.error(f"Error creating directory '{path}': {str(e)}")


def format_time(seconds: float) -> str:
    """
    Format a time duration in seconds to a string in HH:MM:SS format.

    Args:
        seconds (float): Number of seconds.

    Returns:
        str: Time formatted as a string (HH:MM:SS).
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))
