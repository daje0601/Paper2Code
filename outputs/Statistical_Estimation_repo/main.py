## main.py
"""Command-line entry point for reproducing SEQA experiments.

This script orchestrates *all* high-level operations:

1.  Parse CLI arguments (config path, dataset key, output directory).
2.  Load the immutable `Config` singleton from *config.yaml*.
3.  Seed every PRNG for reproducibility (NumPy / random / PyTorch).
4.  Instantiate the **SEQAPipeline** that wires together
      Chunker → Encoder → BayesianFilter → PromptBuilder → Generator
5.  Iterate over the chosen dataset, generate answers, stream-write
    JSONL results and keep predictions / references in memory.
6.  Compute evaluation metrics using `Evaluator` and persist them.
7.  Exit with status code 0 on success.

The implementation follows the design and interfaces specified in the
project documentation; no additional public methods are introduced.

Example
-------
$ python main.py \
      --config   config.yaml \
      --dataset  hotpotqa_distractor \
      --out_dir  results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm

from bayesian_filter import BayesianFilter
from chunker import Chunker
from config import Config
from dataset_loader import DatasetLoader
from encoder import Encoder
from evaluation import Evaluator
from generator import Generator
from prompt_builder import PromptBuilder
from utils import seed_everything, setup_logging

# --------------------------------------------------------------------------- #
# ------------------------------- CLI PARSING --------------------------------#
# --------------------------------------------------------------------------- #
def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="SEQA-Reproduction",
        description="Reproduce experiments from "
        "'Statistical Estimation for Large-Scale Question Answering'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("hotpotqa_distractor", "triviaqa_rc"),
        required=True,
        help="Dataset key as defined in DatasetLoader registry.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Directory where answers/metrics files are written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device e.g. 'cpu', 'cuda:0'. "
        "If omitted the script auto-detects.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# ----------------------------- SEQA PIPELINE --------------------------------#
# --------------------------------------------------------------------------- #
class SEQAPipeline:  # noqa: WPS110  (domain-specific name)
    """End-to-end pipeline for a *single* QA example."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._logger = setup_logging(cfg, name=self.__class__.__name__)

        # ------ Generator (owns tokenizer – must come FIRST) ------------- #
        self._generator = Generator(cfg)
        tokenizer = self._generator.tokenizer  # share across components

        # ------ Remaining components ------------------------------------ #
        self._encoder = Encoder(cfg)
        self._chunker = Chunker(cfg, tokenizer=tokenizer)
        self._filter = BayesianFilter(cfg)
        self._prompt_builder = PromptBuilder(cfg, tokenizer=tokenizer)

        # Retrieval is optional; not required for “Chunk-Only” variant.
        self._retrieval_enabled = bool(getattr(cfg, "retrieval_enabled", False))
        if self._retrieval_enabled:
            from retrieval import Retrieval  # local import to avoid cost if unused

            self._retrieval = Retrieval(cfg, encoder=self._encoder)  # type: ignore
            # NOTE: index_corpus() must be called by the user *before*
            #       pipeline.run() if retrieval is enabled.  Not handled here.
        else:
            self._retrieval = None  # type: ignore

    # ------------------------------------------------------------------ #
    # Per-document execution
    # ------------------------------------------------------------------ #
    def run(self, sample: Dict[str, str]) -> Dict[str, str]:  # noqa: WPS231
        """Process **one** QA sample end-to-end and return result dict."""
        qid: str = sample["id"]
        question: str = sample["question"]
        context: str = sample["context"]

        # -------------------- Optional retrieval ------------------------ #
        if self._retrieval_enabled and self._retrieval is not None:
            passages = self._retrieval.retrieve(question)
            context = "\n\n".join(passages) + "\n\n" + context

        # -------------------- Chunk → Embed ---------------------------- #
        chunks: List[str] = self._chunker.split(context)
        embeds = self._encoder.embed(chunks)  # (K, dim)

        # -------------------- Bayesian filtering ----------------------- #
        accepted_chunks: List[str] = self._filter.filter(chunks, embeds)

        # -------------------- Prompt construction ---------------------- #
        prompt = self._prompt_builder.build(question, accepted_chunks)

        # -------------------- Generation ------------------------------- #
        answer = self._generator.answer(prompt)

        # Package result
        return {"id": qid, "pred": answer, "ref": sample["answer"]}


# --------------------------------------------------------------------------- #
# ---------------------------------- main -----------------------------------#
# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: WPS231 (acceptable complexity for entrypoint)
    """Entry point executed when the module is run as a script."""
    args = _parse_cli()

    # 1) Configuration ---------------------------------------------------- #
    cfg = Config.load(args.config)

    # 2) Early seeding for full reproducibility --------------------------- #
    seed_everything(cfg.random_seed)

    # 3) Logger (root) ---------------------------------------------------- #
    logger = setup_logging(cfg, name="main")
    logger.info("Loaded configuration: %s", cfg)

    # 4) Device override (optional) -------------------------------------- #
    if args.device is not None:
        # Simple validation of user-provided device string
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            logger.error("CUDA not available but --device requests CUDA.")
            sys.exit(1)
        torch_device = torch.device(args.device)
        # monkey-patch torch.cuda.* calls by setting environment
        if torch_device.type == "cpu":
            torch.cuda.is_available = lambda: False  # type: ignore  # pragma: no cover
        logger.info("Overriding device to %s via CLI.", torch_device)

    # 5) Dataset ---------------------------------------------------------- #
    loader = DatasetLoader(cfg, dataset_key=args.dataset)
    samples = loader.load()  # eager load; datasets are moderate in size
    logger.info("Dataset '%s' loaded (%d samples).", args.dataset, len(samples))

    # 6) Output directories / files --------------------------------------- #
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / f"answers_{args.dataset}.jsonl"
    metrics_path = out_dir / f"metrics_{args.dataset}.json"
    logger.info("Answers will be written to %s", answers_path)
    logger.info("Metrics will be written to %s", metrics_path)

    # 7) Build pipeline --------------------------------------------------- #
    pipeline = SEQAPipeline(cfg)

    # 8) Main loop -------------------------------------------------------- #
    preds: List[str] = []
    refs: List[str] = []

    with answers_path.open("w", encoding="utf-8") as fp:
        for sample in tqdm(samples, desc="Processing", unit="sample"):
            result = pipeline.run(sample)
            preds.append(result["pred"])
            refs.append(result["ref"])

            # Stream-write to JSONL
            json.dump(
                {"id": result["id"], "prediction": result["pred"]},
                fp,
                ensure_ascii=False,
            )
            fp.write("\n")
            fp.flush()

    # 9) Evaluate --------------------------------------------------------- #
    evaluator = Evaluator(cfg)
    metrics = evaluator.score(preds, refs)

    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    logger.info("Saved metrics to %s", metrics_path)

    logger.info("Finished successfully.")
    sys.exit(0)


# --------------------------------------------------------------------------- #
# Entry-point guard                                                           #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
