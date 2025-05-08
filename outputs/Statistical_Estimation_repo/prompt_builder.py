## prompt_builder.py
"""
prompt_builder.py

Assembly of the textual *prompt* fed into the generative model
(`facebook/bart-large`).  The class adheres to the public interface
specified in the project design:

    class PromptBuilder:
        + __init__(cfg: Config, tokenizer: PreTrainedTokenizerBase)
        + build(question: str, accepted_chunks: List[str]) -> str

Responsibilities
----------------
1. Combine system role, user question, *accepted* context chunks and an
   explicit instruction line into a single string.

2. Ensure the total number of **encoder tokens** does **not** exceed the
   budget defined by ``cfg.max_prompt_tokens``.  If required, the context
   section (i.e. accepted chunks) is truncated *deterministically* by
   discarding **later** chunks (keeping earlier chunks first).

3. Provide sensible fall-backs for edge-cases (e.g., zero accepted
   chunks) without breaking downstream generation.

Implementation notes
--------------------
* Token counting uses the exact same *tokenizer* instance as the
  generator to guarantee consistency.
* A tiny utility function `_n_tokens` performs fast counting with
  ``add_special_tokens=False`` because full special-token handling is
  applied only once inside the generator itself.
"""

from __future__ import annotations

from typing import List

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from config import Config
from utils import length_limit, setup_logging

__all__ = ["PromptBuilder"]


class PromptBuilder:  # noqa: WPS110 (domain term)
    """Construct textual prompts within a fixed token budget."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Config, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Parameters
        ----------
        cfg:
            Immutable configuration singleton.
        tokenizer:
            Hugging-Face tokenizer *identical* to the one used by the
            generator (BART).  Required for accurate token counting.
        """
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._logger = setup_logging(cfg, name=self.__class__.__name__)

        self._max_prompt_tokens: int = int(getattr(cfg, "max_prompt_tokens"))
        if self._max_prompt_tokens <= 0:
            raise ValueError("`max_prompt_tokens` must be a positive integer.")

        # Pre-compute token count of a single newline for efficiency.
        self._newline_toklen: int = self._count_tokens("\n")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self, question: str, accepted_chunks: List[str]) -> str:
        """Return a full prompt string respecting the token budget.

        Parameters
        ----------
        question:
            User natural-language question.
        accepted_chunks:
            Ordered list of context chunks *already* selected by the
            Bayesian filter.  May be an empty list.

        Returns
        -------
        str
            The final prompt ready to be fed into the generator.
        """
        if not isinstance(question, str):
            raise TypeError("`question` must be of type str.")
        if not isinstance(accepted_chunks, list):
            raise TypeError("`accepted_chunks` must be List[str].")

        # ---------------------------------------------------------------- #
        # 1) Build *static* segments (role, question, header, ending)      #
        # ---------------------------------------------------------------- #
        role_line = "You are a helpful QA system."
        question_line = f"The user asks: {question}"
        context_header = "Below is relevant context filtered from a larger document:"
        ending_line = "Please provide the best possible answer."

        header_text = "\n".join((role_line, question_line, context_header)) + "\n"
        tail_text = "\n" + ending_line  # leading newline to separate from context

        header_tokens = self._count_tokens(header_text)
        tail_tokens = self._count_tokens(tail_text)

        # ---------------------------------------------------------------- #
        # 2) Determine token budget for *context*                          #
        # ---------------------------------------------------------------- #
        budget = self._max_prompt_tokens - header_tokens - tail_tokens
        if budget <= 0:
            # Extremely narrow budget – fallback to length-limited header.
            self._logger.warning(
                "Header+tail exceed or equal total token budget (%d ≥ %d). "
                "Prompt will be truncated aggressively.",
                header_tokens + tail_tokens,
                self._max_prompt_tokens,
            )
            raw_prompt = header_text + tail_text.lstrip("\n")
            return length_limit(self._tokenizer, raw_prompt, self._max_prompt_tokens)

        # ---------------------------------------------------------------- #
        # 3) Greedily append chunks until budget exhausted                 #
        # ---------------------------------------------------------------- #
        selected_chunks: List[str] = []
        used_tokens = 0
        for chunk in accepted_chunks:
            if not isinstance(chunk, str):
                self._logger.warning("Encountered non-string chunk – skipping.")
                continue

            # Token length of chunk plus preceding newline *if* not first.
            addition_tokens = (
                self._newline_toklen + self._count_tokens(chunk)
                if selected_chunks
                else self._count_tokens(chunk)
            )
            if used_tokens + addition_tokens > budget:
                break  # cannot fit this (or any further) chunk
            selected_chunks.append(chunk)
            used_tokens += addition_tokens

        if not selected_chunks:
            selected_chunks.append("No additional context was deemed novel.")

        context_text = "\n".join(selected_chunks)

        # ---------------------------------------------------------------- #
        # 4) Assemble full prompt & final safeguard                        #
        # ---------------------------------------------------------------- #
        prompt = header_text + context_text + tail_text

        # Final safety net – truncate last resort (should rarely trigger)
        total_tokens = self._count_tokens(prompt)
        if total_tokens > self._max_prompt_tokens:
            self._logger.warning(
                "Prompt length %d exceeds budget %d after assembly. "
                "Applying fallback truncation.",
                total_tokens,
                self._max_prompt_tokens,
            )
            prompt = length_limit(self._tokenizer, prompt, self._max_prompt_tokens)

        return prompt

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens for *text* (no special tokens)."""
        return len(
            self._tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids
        )

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"PromptBuilder(max_prompt_tokens={self._max_prompt_tokens}, "
            f"newline_toklen={self._newline_toklen})"
        )
