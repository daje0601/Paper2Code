## evaluation.py
"""
evaluation.py

Compute evaluation metrics for generative Question-Answering outputs
exactly as described in Section 4.4 of the paper *“Statistical Estimation
for Large-Scale Question Answering”*.

Supported metrics (case-insensitive keys)
----------------------------------------
•  BLEU           – SacreBLEU corpus score (tokenise='13a')
•  ROUGE_L        – Mean F1 of ROUGE-L (Porter stemmer)
•  BERTScore      – Mean F1 using *roberta-large*
•  Perplexity     – Average perplexity measured by
                    *EleutherAI/gpt-neo-125M*

The list of metrics to compute is read from `config.yaml`
(`evaluation.metrics`).  If the configuration is missing or empty, **all
four** metrics are computed.

Public API
----------
class Evaluator:
    __init__(cfg: Config)
    score(preds: List[str], refs: List[str]) -> dict
"""

from __future__ import annotations

import collections
import warnings
from typing import Dict, List

import numpy as np
import torch

from config import Config
from utils import seed_everything, setup_logging

__all__ = ["Evaluator"]


class Evaluator:  # noqa: WPS110  (domain term)
    """Metric computation wrapper adhering to design specification."""

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: Config) -> None:
        """
        Parameters
        ----------
        cfg:
            Global, *immutable* configuration instance.
        """
        # ---------------------------------------------------------------- #
        # Reproducibility
        # ---------------------------------------------------------------- #
        seed_everything(cfg.random_seed)

        # ---------------------------------------------------------------- #
        # Logging
        # ---------------------------------------------------------------- #
        self._logger = setup_logging(cfg, name=self.__class__.__name__)

        # ---------------------------------------------------------------- #
        # Device selection (shared by heavy metrics)
        # ---------------------------------------------------------------- #
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cpu":
            warnings.warn(
                "CUDA not available – BERTScore and Perplexity will run on CPU "
                "and may be slow.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ---------------------------------------------------------------- #
        # Determine which metrics to compute
        # ---------------------------------------------------------------- #
        requested = cfg.get("evaluation.metrics", [])
        if not requested:
            # Default to ALL metrics when list is missing / empty
            requested = ["BLEU", "ROUGE_L", "BERTScore", "Perplexity"]

        # Normalise keys to upper for internal flag handling
        normalised = [m.upper() for m in requested]
        self._flags: Dict[str, bool] = {
            "BLEU": "BLEU" in normalised,
            "ROUGE_L": "ROUGE_L" in normalised,
            "BERTSCORE": "BERTSCORE" in normalised or "BERTSCORE" in normalised,
            "PERPLEXITY": "PERPLEXITY" in normalised,
        }

        # ---------------------------------------------------------------- #
        # Lazy-loaded heavy resources
        # ---------------------------------------------------------------- #
        self._ppl_tokenizer = None  # type: ignore[assignment]
        self._ppl_model = None  # type: ignore[assignment]

        self._logger.info(
            "Evaluator initialised (device=%s).  Metrics enabled: %s",
            self.device.type,
            ", ".join(k for k, v in self._flags.items() if v),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def score(self, preds: List[str], refs: List[str]) -> Dict[str, float]:
        """Compute the configured metrics for `preds` vs `refs`.

        Parameters
        ----------
        preds:
            List of model-generated answers.
        refs:
            List of reference answers (ground-truth).

        Returns
        -------
        Dict[str, float]
            Mapping metric-name → scalar value.  The order follows the
            sequence provided in *config.yaml* (or the default order).
        """
        if len(preds) != len(refs):
            raise ValueError(
                f"Pred/ref length mismatch ({len(preds)} vs {len(refs)})."
            )
        if not preds:
            raise ValueError("Input lists must contain at least one prediction.")

        results = collections.OrderedDict()

        # Compute metrics in deterministic order
        if self._flags["BLEU"]:
            results["BLEU"] = self._compute_bleu(preds, refs)
        if self._flags["ROUGE_L"]:
            results["ROUGE_L"] = self._compute_rouge_l(preds, refs)
        if self._flags["BERTSCORE"]:
            results["BERTScore"] = self._compute_bertscore(preds, refs)
        if self._flags["PERPLEXITY"]:
            results["Perplexity"] = self._compute_perplexity(preds)

        self._logger.info(
            "Evaluation complete: %s",
            ", ".join(f"{k}={v:.4f}" for k, v in results.items()),
        )
        return results

    # ------------------------------------------------------------------ #
    # Metric helpers                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_bleu(preds: List[str], refs: List[str]) -> float:
        """Corpus BLEU (SacreBLEU, tokenise='13a')."""
        from sacrebleu import corpus_bleu  # lazy import

        bleu = corpus_bleu(preds, [refs], tokenize="13a", lowercase=False)
        return float(bleu.score)

    @staticmethod
    def _compute_rouge_l(preds: List[str], refs: List[str]) -> float:
        """Mean F1 of ROUGE-L using stemming (Porter)."""
        from rouge_score import rouge_scorer  # lazy import

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(preds, refs)
        ]
        return float(np.mean(scores))

    def _compute_bertscore(self, preds: List[str], refs: List[str]) -> float:
        """Mean BERTScore F1 (roberta-large, english baseline)."""
        from bert_score import score as bert_score  # lazy import

        # The function internally handles batching & GPU
        _, _, f1 = bert_score(
            preds,
            refs,
            model_type="roberta-large",
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
            device=str(self.device),
        )
        return float(f1.mean().item())

    # ------------------------- Perplexity --------------------------------- #
    def _lazy_load_ppl_model(self) -> None:
        """Load GPT-Neo-125M only once (lazy)."""
        if self._ppl_model is not None:
            return  # already loaded

        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

        model_name = "EleutherAI/gpt-neo-125M"
        self._logger.info("Loading perplexity model '%s' on %s …", model_name, self.device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(self.device)
        model.eval()

        self._ppl_tokenizer = tokenizer
        self._ppl_model = model

    def _compute_perplexity(self, preds: List[str]) -> float:
        """Average perplexity of *preds* measured by GPT-Neo-125M."""
        self._lazy_load_ppl_model()
        assert self._ppl_model is not None and self._ppl_tokenizer is not None  # mypy

        batch_size = 8  # sensible default; small memory footprint
        ppl_values: List[torch.Tensor] = []

        for start in range(0, len(preds), batch_size):
            batch = preds[start : start + batch_size]  # noqa: WPS362

            # Replace empty predictions with single <eos> to avoid NaNs
            batch = [txt if txt.strip() else self._ppl_tokenizer.eos_token for txt in batch]

            enc = self._ppl_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Standard causal-LM perplexity: teacher-forcing with shifted labels
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100  # ignore padding in loss

            with torch.inference_mode():
                loss = self._ppl_model(**enc, labels=labels).loss  # mean-token NLL
            ppl_values.append(torch.exp(loss).detach().cpu())

        mean_ppl = torch.mean(torch.stack(ppl_values)).item()
        return float(mean_ppl)

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                     #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # noqa: D401
        enabled = ", ".join(k for k, v in self._flags.items() if v)
        return f"Evaluator(device='{self.device.type}', metrics=[{enabled}])"
