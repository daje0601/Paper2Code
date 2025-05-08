## chunker.py
"""
chunker.py

Implements the `Chunker` class described in the project design.

The class converts a *single* very-long document string into an **ordered
list** of overlapping chunk strings whose *token* length does **not**
exceed the maximum configured value (defaults: 300 tokens with 30-token
overlap).  Chunk boundaries are computed **in token space** to guarantee
faithfulness with respect to the downstream `facebook/bart-large`
tokeniser, which is passed to the constructor.

The implementation follows the methodology detailed in Section 3.1 of the
paper:

    • Fixed-size sliding-window   (size = M,  overlap = O)
    • Overlap is preserved *verbatim* – no loss of information.
    • Edge-cases (short docs, tail residue) handled gracefully.

The module is **stateless** beyond holding references to `Config`
hyper-parameters and the tokenizer, making it safe for concurrent use.
"""

from __future__ import annotations

from typing import List

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from config import Config
from utils import setup_logging

__all__ = ["Chunker"]


class Chunker:  # noqa: WPS110  (name chosen intentionally; domain term)
    """Split long documents into fixed-size overlapping chunks."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Config, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        Parameters
        ----------
        cfg:
            Global configuration singleton containing `chunk_size` and
            `overlap` values amongst others.
        tokenizer:
            *Initialised* Hugging-Face tokenizer **identical** to the one
            later used by the generator (BART).  It determines token
            boundaries for chunking.
        """
        self._logger = setup_logging(cfg, name=self.__class__.__name__)
        self._tokenizer = tokenizer

        # Read hyper-parameters from config (fail fast on invalid combos)
        self._chunk_size: int = int(getattr(cfg, "chunk_size"))
        self._overlap: int = int(getattr(cfg, "overlap"))
        if self._chunk_size <= 0:
            raise ValueError("`chunk_size` must be positive.")
        if not (0 <= self._overlap < self._chunk_size):
            raise ValueError("`overlap` must satisfy 0 ≤ overlap < chunk_size.")

        # Derived constant: stride length
        self._step: int = self._chunk_size - self._overlap

        self._logger.debug(
            "Initialised Chunker (chunk_size=%d, overlap=%d, step=%d).",
            self._chunk_size,
            self._overlap,
            self._step,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def split(self, doc: str) -> List[str]:
        """Split *doc* into an ordered list of overlapping chunks.

        The *token* length of each returned chunk, when re-tokenised with
        the **same** tokenizer, is guaranteed to be ≤ `chunk_size`.

        Parameters
        ----------
        doc:
            Raw document text (can be arbitrarily long).

        Returns
        -------
        List[str]
            List of chunk strings; may be empty if *doc* is empty or
            whitespace-only.
        """
        if not isinstance(doc, str):
            raise TypeError(f"`doc` must be str, not {type(doc)}.")

        stripped = doc.strip()
        if not stripped:
            # Nothing to do – avoid useless tokeniser calls.
            self._logger.debug("Received empty/whitespace document; returning [].")
            return []

        # Tokenise the *entire* document once (fast path)
        input_ids = self._tokenizer.encode(
            stripped,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        n_tokens = len(input_ids)
        if n_tokens == 0:
            # Highly unlikely, but guard against pathological tokenisers.
            self._logger.warning("Tokeniser returned 0 tokens for non-empty doc.")
            return []

        # Sliding-window segmentation
        chunks: List[str] = []
        for start in range(0, n_tokens, self._step):
            end = min(start + self._chunk_size, n_tokens)
            chunk_ids = input_ids[start:end]

            # Decode back to text; remove leading/trailing spaces.
            chunk_text = self._tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk_text:  # guard against potential empty decodes
                chunks.append(chunk_text)

            # Early exit if we've reached the end (avoids an extra loop)
            if end == n_tokens:
                break

        # Sanity check: no chunk should exceed budget
        for idx, chunk in enumerate(chunks):
            length = len(
                self._tokenizer(
                    chunk,
                    add_special_tokens=False,
                    return_attention_mask=False,
                ).input_ids
            )
            if length > self._chunk_size:
                # This should never happen – indicates either a bug or
                # the tokenizer behaving unexpectedly.
                raise RuntimeError(
                    f"Chunk {idx} exceeds token budget "
                    f"({length} > {self._chunk_size})."
                )

        self._logger.debug(
            "Split document (%d tok) into %d chunk(s).", n_tokens, len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401  (single-line description)
        return (
            f"Chunker(chunk_size={self._chunk_size}, "
            f"overlap={self._overlap}, step={self._step})"
        )
