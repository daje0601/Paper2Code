"""main.py

Entry point for the Transformer experiment workflow.
This file orchestrates configuration loading, data preparation, model initialization,
training, and evaluation for both machine translation and constituency parsing tasks,
following the "Attention Is All You Need" paper's methodology and experimental setup.

Usage:
    python main.py --task [translation|parsing|both]

Author: [Your Name]
Date: [Current Date]
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import logging

# Import centralized configuration and global instance.
from config import CONFIG, Config

# Import DatasetLoader, TransformerModel, Trainer, and Evaluation classes.
from dataset_loader import DatasetLoader
from model import TransformerModel
from trainer import Trainer
from evaluation import Evaluation

# Import global logger from utils.
from utils import LOGGER


def run_translation_experiment() -> None:
    """
    Run the machine translation experiment using the transformer base model.
    Loads WMT data (using the translation configuration), trains the model,
    and evaluates performance using BLEU score.
    """
    LOGGER.info("Starting Machine Translation experiment...")

    # Initialize DatasetLoader for translation.
    translation_loader = DatasetLoader(CONFIG, task="translation")
    train_data, dev_data, test_data = translation_loader.load_data()
    
    # Instantiate the Transformer model using the configured (base) settings.
    model = TransformerModel(CONFIG)
    LOGGER.info("Transformer model for translation created.")

    # Initialize Trainer with model, training data and dev data.
    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, config=CONFIG)
    LOGGER.info("Starting training for machine translation...")
    trainer.train()
    
    # After training, evaluate using the Evaluation module.
    evaluator = Evaluation(model=model, test_data=test_data, config=CONFIG, task="translation")
    metrics = evaluator.evaluate_mt()
    LOGGER.info(f"Machine Translation Evaluation Metrics: {metrics}")


def run_parsing_experiment() -> None:
    """
    Run the constituency parsing experiment using a transformer configured for parsing.
    Loads WSJ parsing data, trains the model with parsing-specific hyperparameters,
    and evaluates performance using F1, precision, and recall metrics.
    """
    LOGGER.info("Starting Constituency Parsing experiment...")
    
    # Initialize DatasetLoader for parsing.
    parsing_loader = DatasetLoader(CONFIG, task="parsing")
    train_data, dev_data, test_data = parsing_loader.load_data()
    
    # Since the global CONFIG is set for translation (transformer_base), create a new configuration
    # for parsing by overriding the model configuration with parsing-specific settings.
    new_model_config = replace(CONFIG.model, type="transformer_parsing", d_model=CONFIG.parsing.model.d_model, num_layers=CONFIG.parsing.model.num_layers)
    # Create a new Config instance for parsing experiments.
    parsing_config = Config(
        training=CONFIG.training,
        model=new_model_config,
        inference=CONFIG.inference,
        parsing=CONFIG.parsing,
        data=CONFIG.data,
        hardware=CONFIG.hardware
    )
    LOGGER.info("Parsing configuration created with model type 'transformer_parsing'.")

    # Instantiate Transformer model with parsing configuration.
    model = TransformerModel(parsing_config)
    LOGGER.info("Transformer model for parsing created.")

    # Initialize Trainer for parsing.
    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data, config=parsing_config)
    LOGGER.info("Starting training for constituency parsing...")
    trainer.train()
    
    # Evaluate the parsing model.
    evaluator = Evaluation(model=model, test_data=test_data, config=parsing_config, task="parsing")
    metrics = evaluator.evaluate_parsing()
    LOGGER.info(f"Constituency Parsing Evaluation Metrics: {metrics}")


def main() -> None:
    """
    Main function to run experiments. It parses command-line arguments to decide
    which experiment to run: machine translation, constituency parsing, or both.
    """
    parser = argparse.ArgumentParser(description="Transformer Experiment Runner")
    parser.add_argument(
        "--task",
        type=str,
        choices=["translation", "parsing", "both"],
        default="translation",
        help="Specify which experiment to run: translation, parsing, or both (default: translation)."
    )
    args = parser.parse_args()

    LOGGER.info("Experiment started with configuration from config.yaml")
    
    if args.task == "translation":
        run_translation_experiment()
    elif args.task == "parsing":
        run_parsing_experiment()
    elif args.task == "both":
        run_translation_experiment()
        run_parsing_experiment()
    else:
        LOGGER.error(f"Unsupported task type provided: {args.task}")


if __name__ == "__main__":
    main()
