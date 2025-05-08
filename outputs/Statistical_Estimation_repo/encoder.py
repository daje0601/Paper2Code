## encoder.py
"""
encoder.py

Vector encoder for document chunks used by the Bayesian novelty filter.

The Encoder class is intentionally minimal and **stateless** with respect
to gradients: the underlying SentenceTransformer is *frozen* (we never
call `.train()` nor compute gradients).  Its sole purpose is to map lists
of strings into L2-normalised float32 vectors suitable for cosine
similarity computations.

Public Interface
----------------
class Encoder:
    __init__(cfg: Config)
        Loads the sentence-level embedding model specified in *cfg* (or a
        sensible default) onto the appropriate device.

    embed(chunks: List[str]) -> torch.Tensor
        Returns a 2-D tensor of shape ``(len(chunks), dim)`` where each row
        has unit L2-norm.  The tensor lives on the same device as the
        underlying model (GPU if available, else CPU).

Notes
-----
*   The chosen default model is
    ``sentence-transformers/all-MiniLM-L6-v2`` (dim = 384).  This can be
    overridden by adding the key ``sentence_encoder_name`` to *config.yaml*.
*   A conservative default batch size of **64** is used unless the config
    provides ``encoder_batch_size``.
*   Encoder exposes its internal **tokenizer** via the attribute
    ``self.tokenizer`` so that the `Chunker` can reuse the exact same
    tokenisation rules if desired.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer  # type: ignore

from config import Config
from utils import seed_everything, setup_logging

__all__ = ["Encoder"]


class Encoder:  # noqa: WPS110 (name chosen deliberately – domain term)
    """L2-normalised sentence embedding generator."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Config) -> None:
        """
        Parameters
        ----------
        cfg:
            Global, *frozen* configuration instance.  Only **read access**
            is performed; the object is never mutated.
        """
        # Ensure deterministic weight initialisation, dropout, etc.
        seed_everything(cfg.random_seed)

        self._logger = setup_logging(cfg, name=self.__class__.__name__)
        self._cfg = cfg

        # -------------------- Model selection --------------------------- #
        default_model = "sentence-transformers/all-MiniLM-L6-v2"
        self._model_name: str = getattr(cfg, "sentence_encoder_name", default_model)
        if not isinstance(self._model_name, str) or not self._model_name:
            self._logger.warning(
                "Invalid or empty sentence_encoder_name in config; "
                "falling back to '%s'.",
                default_model,
            )
            self._model_name = default_model

        # -------------------- Device selection -------------------------- #
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._logger.info("Loading sentence encoder '%s' on %s …", self._model_name, self.device)

        # -------------------- Model loading ----------------------------- #
        try:
            self._model = SentenceTransformer(self._model_name, device=str(self.device))
        except RuntimeError as exc:
            # Graceful degradation when CUDA OOM or other GPU issues occur.
            if "CUDA" in str(exc) and self.device.type == "cuda":
                self._logger.error(
                    "Failed to load model on GPU due to: %s. "
                    "Retrying on CPU — this will be slower.",
                    exc,
                )
                self.device = torch.device("cpu")
                self._model = SentenceTransformer(self._model_name, device=str(self.device))
            else:
                raise

        self.dim: int = self._model.get_sentence_embedding_dimension()
        # Expose tokenizer for `Chunker` reuse (SentenceTransformer has `.tokenizer`)
        self.tokenizer = getattr(self._model, "tokenizer", None)

        self._batch_size: int = int(getattr(cfg, "encoder_batch_size", 64))
        if self._batch_size <= 0:
            self._logger.warning(
                "encoder_batch_size must be > 0; defaulting to 64."
            )
            self._batch_size = 64

        self._logger.info(
            "Sentence encoder ready (dim=%d, batch_size=%d).",
            self.dim,
            self._batch_size,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _l2_normalise(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Return L2-normalised copy of *vectors* along dim=1."""
        return F.normalize(vectors, p=2, dim=1, eps=eps)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def embed(self, chunks: List[str]) -> torch.Tensor:
        """Embed *chunks* into unit-length vectors.

        Parameters
        ----------
        chunks:
            List of chunk strings.  Order is preserved.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_chunks, dim)`` on *self.device* with dtype
            ``torch.float32``.  If *chunks* is empty, an empty tensor with
            zero rows is returned.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"`chunks` must be List[str], not {type(chunks)}.")
        if not chunks:
            # Return an *empty* tensor with the correct dimensionality.
            return torch.empty((0, self.dim), dtype=torch.float32, device=self.device)

        show_progress_bar = len(chunks) >= 1000
        self._logger.debug(
            "Encoding %d chunk(s) into %d-dim vectors (progress bar: %s).",
            len(chunks),
            self.dim,
            show_progress_bar,
        )

        with torch.inference_mode():  # alias for torch.no_grad() in modern PyTorch
            embeddings: torch.Tensor = self._model.encode(
                chunks,
                batch_size=self._batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=True,
                device=str(self.device),
                normalize_embeddings=False,  # we'll normalise ourselves
            )

        if embeddings.dtype != torch.float32:
            embeddings = embeddings.float()

        embeddings = self._l2_normalise(embeddings)

        # Sanity check: each row should have ||x||₂ ≈ 1.
        if torch.any(torch.isnan(embeddings)):
            raise RuntimeError("NaN encountered in embeddings.")

        return embeddings

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401 (short description acceptable)
        return (
            f"Encoder(model='{self._model_name}', device='{self.device.type}', "
            f"dim={self.dim}, batch_size={self._batch_size})"
        )
