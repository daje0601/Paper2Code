## generator.py
"""
generator.py

Encapsulates loading of the sequence-to-sequence generator
(`facebook/bart-large` by default) and provides a single public method
`answer(prompt: str) -> str` that turns a fully-formed prompt into a
natural-language answer, strictly following the hyper-parameters laid out
in *config.yaml* and the reproduction design document.

Key Features
------------
•  Stateless *w.r.t.* Bayesian filtering or dataset specifics.
•  Deterministic thanks to `utils.seed_everything(cfg.random_seed)`.
•  Transparent device placement (CUDA if available; otherwise CPU with
   a performance warning).
•  Hard enforcement of the encoder token budget to avoid position-
   embedding overflow in BART (default 1024 tokens).
•  Minimal set of generation arguments – **only** those explicitly
   mentioned in the paper/config are used.

Public Interface
----------------
class Generator:
    __init__(cfg: Config)
    answer(prompt: str) -> str
    (optional) answer_batch(prompts: List[str]) -> List[str]
"""

from __future__ import annotations

import warnings
from typing import List, Sequence

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from config import Config
from utils import seed_everything, setup_logging

__all__ = ["Generator"]


class Generator:  # noqa: WPS110 (domain term)
    """Wrapper around a Hugging-Face seq-to-seq *generator* model."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Config) -> None:
        """
        Parameters
        ----------
        cfg:
            Immutable configuration singleton loaded via `Config.load`.
        """
        # ---------------------------------------------------------------- #
        # Seeding for deterministic generation
        # ---------------------------------------------------------------- #
        seed_everything(cfg.random_seed)

        # ---------------------------------------------------------------- #
        # Logging
        # ---------------------------------------------------------------- #
        self._logger = setup_logging(cfg, name=self.__class__.__name__)

        # ---------------------------------------------------------------- #
        # Hyper-parameters pulled from config
        # ---------------------------------------------------------------- #
        self._model_name: str = getattr(cfg, "model_name")
        self._num_beams: int = int(getattr(cfg, "beam"))
        # Private attribute in Config (leading underscore); fall back safe.
        self._max_gen_len: int = int(getattr(cfg, "_max_gen_length", 128))
        self._enc_max_tokens: int = int(getattr(cfg, "max_prompt_tokens"))

        # ---------------------------------------------------------------- #
        # Device selection
        # ---------------------------------------------------------------- #
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self._logger.info("Using GPU device for generation.")
        else:
            self.device = torch.device("cpu")
            warnings.warn(
                "CUDA not available – running BART-Large on CPU. "
                "This will be considerably slower.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._logger.info("Using CPU device for generation.")

        # ---------------------------------------------------------------- #
        # Tokeniser
        # ---------------------------------------------------------------- #
        try:
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self._model_name,
                use_fast=True,
            )
        except OSError as exc:  # network issues / wrong id
            self._logger.exception("Failed to load tokenizer '%s'.", self._model_name)
            raise

        # Safety: ensure `pad_token_id` is set (important for batching)
        if self.tokenizer.pad_token_id is None:
            self._logger.warning(
                "Tokenizer '%s' lacks pad_token_id – setting it to eos_token_id.",
                self._model_name,
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ---------------------------------------------------------------- #
        # Model
        # ---------------------------------------------------------------- #
        try:
            self.model: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else None,
            )
        except OSError:
            self._logger.exception("Failed to load model '%s'.", self._model_name)
            raise

        # Move to device and switch to inference mode
        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------- #
        # Log summary
        # ---------------------------------------------------------------- #
        n_params = sum(p.numel() for p in self.model.parameters()) / 1_000_000
        self._logger.info(
            "Generator ready – %s (%.1fM params, beams=%d, "
            "enc_max=%d, gen_max=%d)",
            self._model_name,
            n_params,
            self._num_beams,
            self._enc_max_tokens,
            self._max_gen_len,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def answer(self, prompt: str) -> str:  # noqa: WPS231 (complexity acceptable)
        """Generate an answer string for a *single* prompt.

        Parameters
        ----------
        prompt:
            Fully-formed textual prompt (including question, context,
            instructions).  Must not be empty.

        Returns
        -------
        str
            Model-generated answer, stripped of surrounding whitespace.

        Raises
        ------
        ValueError
            If *prompt* is empty or contains only whitespace.
        RuntimeError
            If generation fails due to model/tokeniser mis-configuration.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("`prompt` must be a non-empty string.")

        # ---------------------------------------------------------------- #
        # Tokenisation with hard truncation (safeguard)
        # ---------------------------------------------------------------- #
        inputs = self.tokenizer(
            prompt,
            max_length=self._enc_max_tokens,
            truncation=True,
            return_tensors="pt",
            padding=False,  # prompt builder already fits budget
        )

        # Explicitly check truncation happened and warn once
        if inputs.input_ids.shape[1] == self._enc_max_tokens and len(
            self.tokenizer(prompt, add_special_tokens=False).input_ids
        ) > self._enc_max_tokens:
            self._logger.warning(
                "Prompt was truncated to %d tokens to fit encoder budget.",
                self._enc_max_tokens,
            )

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ---------------------------------------------------------------- #
        # Generation
        # ---------------------------------------------------------------- #
        generated_ids = self.model.generate(
            **inputs,
            max_length=self._max_gen_len,
            num_beams=self._num_beams,
            early_stopping=True,
            length_penalty=1.0,  # paper does not modify
            repetition_penalty=1.0,  # paper does not mention
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if generated_ids.ndim != 2 or generated_ids.size(0) == 0:
            raise RuntimeError("Model.generate() returned an unexpected shape.")

        answer_str: str = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return answer_str or ""  # never return None

    # -------------------- OPTIONAL: batched generation ------------------- #
    @torch.inference_mode()
    def answer_batch(self, prompts: Sequence[str]) -> List[str]:
        """Batch version of `answer` for improved throughput.

        Not used directly in the reproduction pipeline but provided for
        convenience (e.g., during evaluation).

        Parameters
        ----------
        prompts:
            Iterable of prompt strings.

        Returns
        -------
        List[str]
            Generated answers in the same order.
        """
        if not prompts:
            return []

        # Tokenise with *individual* truncation to maintain alignment
        encodings = self.tokenizer(
            list(prompts),
            max_length=self._enc_max_tokens,
            truncation=True,
            padding=True,  # pad to longest in batch
            return_tensors="pt",
        ).to(self.device)

        generated = self.model.generate(
            input_ids=encodings.input_ids,
            attention_mask=encodings.attention_mask,
            max_length=self._max_gen_len,
            num_beams=self._num_beams,
            early_stopping=True,
            length_penalty=1.0,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        answers = [
            self.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            for ids in generated
        ]
        return answers

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"Generator(model='{self._model_name}', device='{self.device.type}', "
            f"beams={self._num_beams}, enc_max={self._enc_max_tokens}, "
            f"gen_max={self._max_gen_len})"
        )
