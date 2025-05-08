## utils.py
"""Generic utility helpers shared by the SEQA-reproduction code-base.

The module is intentionally lightweight and **stateless** (apart from
global logging handlers and PRNG seeds).  No other file should replicate
the functionality implemented here.

Public API
----------
seed_everything(seed:int) -> int
    Sets Python `random`, NumPy and PyTorch seeds (CPU & CUDA)
    *and* `PYTHONHASHSEED` for deterministic behaviour.

flatten_dict(nested:dict, parent_key:str = "", sep:str = "/") -> dict
    Converts arbitrarily nested dictionaries (and lists / tuples) into a
    flattened mapping where keys are concatenated with `sep`.

length_limit(tokenizer, text:str, max_tokens:int) -> str
    Ensures `text` is no longer than `max_tokens` when tokenised with the
    supplied Hugging-Face tokenizer; truncates from the **end** if
    necessary whilst trying to keep whole-line boundaries.

setup_logging(cfg:Config, name:str|None = None) -> logging.Logger
    Configures the Python logging subsystem once and returns a module
    specific logger.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# --------------------------------------------------------------------------- #
# Reproducibility utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed: int | None) -> int:  # noqa: WPS231 (complexity is fine)
    """Seed *all* relevant PRNGs.

    Parameters
    ----------
    seed:
        Non-negative integer seed.  If ``None`` an exception is raised:
        callers **must** pass an explicit seed to guarantee reproducibility.

    Returns
    -------
    int
        The seed actually set (useful for logging).
    """
    if seed is None:
        raise ValueError("`seed` must be an explicit integer (not None).")

    if not isinstance(seed, int):
        raise TypeError(f"Seed must be int, not {type(seed)}.")

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("`seed` must be in the range [0, 2**32 - 1].")

    # Python hash seed ------------------------------------------------------- #
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Standard library random ------------------------------------------------ #
    random.seed(seed)
    # NumPy ------------------------------------------------------------------ #
    np.random.seed(seed)
    # PyTorch ---------------------------------------------------------------- #
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN behaviour (may impact speed) ----------------------- #
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    return seed


# --------------------------------------------------------------------------- #
# Dictionary flattening helper
# --------------------------------------------------------------------------- #
def flatten_dict(nested: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten *nested* dictionaries into a single-level dict.

    Keys from deeper levels are concatenated with *sep*.

    Example
    -------
    >>> flatten_dict({"a": {"b": 1, "c": 2}})
    {'a/b': 1, 'a/c': 2}

    Lists and tuples are supported by using the **index** as the next
    key component:

    >>> flatten_dict({"metrics": ["bleu", "rouge"]})
    {'metrics/0': 'bleu', 'metrics/1': 'rouge'}
    """
    items: Dict[str, Any] = {}

    def _rec(obj: Any, prefix: str = "") -> None:  # noqa: WPS430
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                _rec(v, key)
        elif isinstance(obj, (list, tuple)):
            for idx, v in enumerate(obj):
                key = f"{prefix}{sep}{idx}" if prefix else str(idx)
                _rec(v, key)
        else:
            items[prefix] = _safe_val(obj)

    def _safe_val(val: Any) -> Any:
        # Convert non JSON-serialisable values to string representation.
        basic_types = (str, int, float, bool, type(None))
        return val if isinstance(val, basic_types) else str(val)

    _rec(nested, parent_key)
    return items


# --------------------------------------------------------------------------- #
# Token length guard
# --------------------------------------------------------------------------- #
def length_limit(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_tokens: int,
    *,
    newline_token: str = "\n",
) -> str:
    """Ensure *text* encodes to no more than *max_tokens* tokens.

    If *text* already complies, it's returned unmodified.  Otherwise we
    iteratively drop lines from the end (split on ``newline_token``) until
    token budget fits.  As a final fallback, we truncate by tokens.

    Parameters
    ----------
    tokenizer:
        Hugging-Face tokenizer configured for the *generator* model.
    text:
        The prompt string to be constrained.
    max_tokens:
        Maximum number of encoder tokens allowed.
    newline_token:
        Line delimiter used when pruning whole lines (default ``\\n``).

    Returns
    -------
    str
        The possibly truncated text no longer than *max_tokens* tokens.

    Raises
    ------
    RuntimeError
        If the function fails to shorten the text below the token budget.
    """
    if max_tokens <= 0:
        raise ValueError("`max_tokens` must be positive.")

    def _n_tokens(t: str) -> int:
        # Fast token counting without tensor allocation
        return len(tokenizer(t, add_special_tokens=False, return_attention_mask=False).input_ids)

    # Fast path -------------------------------------------------------------- #
    if _n_tokens(text) <= max_tokens:
        return text

    # Attempt to remove whole lines first ----------------------------------- #
    lines: List[str] = text.split(newline_token)
    while lines and _n_tokens(newline_token.join(lines)) > max_tokens:
        lines.pop()  # drop from end (least important context)
    truncated = newline_token.join(lines)
    if lines and _n_tokens(truncated) <= max_tokens:
        return truncated

    # Fallback: token level truncation -------------------------------------- #
    # Directly cut the token list and re-decode; this guarantees compliance.
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False).input_ids
    if not ids:
        raise RuntimeError("Cannot truncate an empty prompt.")
    limited_ids = ids[:max_tokens]
    limited_text = tokenizer.decode(limited_ids, clean_up_tokenization_spaces=False)

    if _n_tokens(limited_text) > max_tokens:
        # Should never happen, but double-check to be safe.
        raise RuntimeError("Failed to truncate prompt within the token limit.")
    return limited_text


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Any, name: str | None = None) -> logging.Logger:
    """Initialise and return a module-specific logger.

    Parameters
    ----------
    cfg:
        A config object that **must** expose attribute `random_seed`
        (only used for early logging).  Using *Any* to avoid importing
        `Config` here and creating circular dependencies.
    name:
        The logger name; defaults to the caller's module ``__name__``.

    Returns
    -------
    logging.Logger
        A fully configured logger instance.
    """
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # ------------------------------------------------------------------- #
        # Handler initialisation runs only ONCE regardless of how many
        # modules import utils.setup_logging
        # ------------------------------------------------------------------- #
        root_logger.setLevel(logging.DEBUG)  # capture all; handlers filter
        log_level = logging.DEBUG if getattr(cfg, "debug", False) else logging.INFO

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

        # Console handler ---------------------------------------------------- #
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        console.setFormatter(formatter)
        root_logger.addHandler(console)

        # File handler ------------------------------------------------------- #
        logs_dir = Path("logs")
        _ensure_dir(logs_dir)
        timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        file_path = logs_dir / f"seqa_{timestamp}.log"
        file_handler = logging.FileHandler(file_path, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # First message ------------------------------------------------------ #
        root_logger.info("Logging initialised.  Log level: %s", logging.getLevelName(log_level))
        root_logger.debug("Random seed set to %s.", getattr(cfg, "random_seed", "N/A"))

    return logging.getLogger(name or __name__)
