"""dataset_loader.py

This module defines the DatasetLoader class, responsible for reading raw text data
(from both machine translation and constituency parsing sources), performing tokenization 
(using Hugging Face tokenizers), building vocabularies, splitting data into training, 
development, and test sets, and grouping tokenized sequences into batches based on an 
approximate token count (as specified in the configuration).

The DatasetLoader class follows the design interface:
    + __init__(config: dict)
    + load_data() -> Tuple[Dataset, Dataset, Dataset]

The returned dataset for each split is a list of batch dictionaries, where each batch 
contains tokenized sequences (and their lengths) appropriate for subsequent training steps.
"""

import os
from typing import Any, Dict, List, Tuple, Optional

import torch

# Import configuration and utility functions.
from config import CONFIG, Config
from utils import LOGGER, load_tokenizer, encode_text, create_dir_if_not_exists


class DatasetLoader:
    """
    DatasetLoader is responsible for loading, tokenizing, splitting, and batching data for both
    machine translation and constituency parsing experiments based on the provided configuration.
    """

    def __init__(self, config: Config, task: str = "translation") -> None:
        """
        Initialize the DatasetLoader with configuration and task type.

        Args:
            config (Config): Configuration instance loaded from config.yaml via config.py.
            task (str): Task type; either "translation" or "parsing". Defaults to "translation".
        """
        self.config: Config = config
        self.task: str = task.lower()
        self.batch_tokens: int = self.config.training.batch_tokens

        # Set default file paths based on task.
        if self.task == "translation":
            # For machine translation, use the WMT 2014 English-to-German dataset by default.
            # Expected file paths (one for source and one for target) are in the 'data' directory.
            self.translation_dataset: str = "wmt_2014_en_de"
            self.src_file: str = os.path.join("data", "wmt_2014_en_de.src")
            self.tgt_file: str = os.path.join("data", "wmt_2014_en_de.tgt")
        elif self.task == "parsing":
            # For constituency parsing, use the WSJ dataset by default.
            self.parsing_dataset: str = "wsj"
            self.wsj_file: str = os.path.join("data", "wsj.txt")
        else:
            LOGGER.error(f"Unsupported task type: {self.task}. Defaulting to 'translation'.")
            self.task = "translation"
            self.translation_dataset = "wmt_2014_en_de"
            self.src_file = os.path.join("data", "wmt_2014_en_de.src")
            self.tgt_file = os.path.join("data", "wmt_2014_en_de.tgt")

        LOGGER.info(f"DatasetLoader initialized for task '{self.task}' with batch token limit {self.batch_tokens}.")

    def _read_lines(self, file_path: str) -> List[str]:
        """
        Reads a file from the given file path and returns a list of stripped text lines.

        Args:
            file_path (str): Path to the text file.

        Returns:
            List[str]: List of text lines.
        """
        if not os.path.isfile(file_path):
            LOGGER.error(f"File not found: {file_path}")
            return []
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                lines: List[str] = [line.strip() for line in f if line.strip()]
            LOGGER.info(f"Read {len(lines)} lines from {file_path}.")
            return lines
        except Exception as e:
            LOGGER.error(f"Error reading file {file_path}: {str(e)}")
            return []

    def _tokenize_translation(
        self, src_lines: List[str], tgt_lines: List[str]
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Tokenizes parallel source and target sentence lists using the configured tokenizer.

        If the model configuration indicates shared embeddings (share_embedding = True),
        the same tokenizer is used for both source and target; otherwise, separate tokenizers 
        are loaded based on the dataset configuration.

        Args:
            src_lines (List[str]): List of source language sentences.
            tgt_lines (List[str]): List of target language sentences.

        Returns:
            List[Tuple[List[int], List[int]]]: List of tuples containing tokenized source and
                                                target sentences as lists of token IDs.
        """
        # Select dataset configuration for translation.
        # Here we use the wmt_2014_en_de configuration.
        dataset_config = self.config.data.wmt_2014_en_de
        tokenizer_type: str = dataset_config.tokenizer  # e.g., "BPE"
        vocab_size: int = dataset_config.vocab_size

        # Load tokenizer. If shared embedding is True, use one tokenizer for both language sides.
        tokenizer = load_tokenizer(tokenizer_type)
        try:
            # Attempt to get vocabulary size from tokenizer.
            vocab = tokenizer.get_vocab()  # Assumes the tokenizer provides get_vocab()
            LOGGER.info(f"Tokenizer loaded with vocabulary size: {len(vocab)}. (Expected target: {vocab_size})")
        except Exception as e:
            LOGGER.warning(f"Unable to obtain vocabulary from tokenizer: {str(e)}")

        tokenized_pairs: List[Tuple[List[int], List[int]]] = []
        num_sentences: int = min(len(src_lines), len(tgt_lines))
        for idx in range(num_sentences):
            src_sentence: str = src_lines[idx]
            tgt_sentence: str = tgt_lines[idx]
            src_tokens: List[int] = encode_text(src_sentence, tokenizer)
            tgt_tokens: List[int] = encode_text(tgt_sentence, tokenizer)
            tokenized_pairs.append((src_tokens, tgt_tokens))
            if idx < 3:  # Log first few examples for debugging.
                LOGGER.debug(f"Example {idx}: src tokens ({len(src_tokens)} tokens), tgt tokens ({len(tgt_tokens)} tokens).")
        LOGGER.info(f"Tokenized {len(tokenized_pairs)} translation sentence pairs.")
        return tokenized_pairs

    def _tokenize_parsing(self, lines: List[str]) -> List[List[int]]:
        """
        Tokenizes a list of sentences for constituency parsing experiments.
        For parsing, a default tokenizer (e.g., BPE) is used.

        Args:
            lines (List[str]): List of raw sentences.

        Returns:
            List[List[int]]: List of tokenized sentences (each sentence is a list of token IDs).
        """
        # For parsing, we assume default tokenization using BPE.
        tokenizer = load_tokenizer("BPE")
        try:
            vocab = tokenizer.get_vocab()
            LOGGER.info(f"Parsing tokenizer loaded with vocabulary size: {len(vocab)}.")
        except Exception as e:
            LOGGER.warning(f"Unable to obtain vocabulary for parsing tokenizer: {str(e)}")

        tokenized_sentences: List[List[int]] = []
        for idx, sentence in enumerate(lines):
            tokens: List[int] = encode_text(sentence, tokenizer)
            tokenized_sentences.append(tokens)
            if idx < 3:
                LOGGER.debug(f"Parsing example {idx}: {len(tokens)} tokens.")
        LOGGER.info(f"Tokenized {len(tokenized_sentences)} parsing sentences.")
        return tokenized_sentences

    def _split_data(self, examples: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Splits the list of examples into training, development, and test sets using a simple
        80/10/10 ratio.

        Args:
            examples (List[Any]): List of tokenized examples (either sentence pairs or single sentences).

        Returns:
            Tuple[List[Any], List[Any], List[Any]]: (train_set, dev_set, test_set)
        """
        total: int = len(examples)
        train_end: int = int(total * 0.8)
        dev_end: int = train_end + int(total * 0.1)
        train_set: List[Any] = examples[:train_end]
        dev_set: List[Any] = examples[train_end:dev_end]
        test_set: List[Any] = examples[dev_end:]
        LOGGER.info(f"Data split into {len(train_set)} training, {len(dev_set)} dev, and {len(test_set)} test examples.")
        return train_set, dev_set, test_set

    def _create_batches(self, examples: List[Any], is_pair: bool = True) -> List[Dict[str, Any]]:
        """
        Groups tokenized examples into batches based on the approximate token count defined in
        the configuration. For translation tasks (is_pair=True), both source and target tokens are
        considered; for parsing (is_pair=False), only one sequence per example is used.

        Args:
            examples (List[Any]): List of tokenized examples. For translation, each example is a tuple
                                  (src_tokens, tgt_tokens). For parsing, each example is a list of token IDs.
            is_pair (bool): Indicates if examples contain paired sequences (True for translation). Defaults to True.

        Returns:
            List[Dict[str, Any]]: List of batch dictionaries containing the tokenized sequences.
                                  For translation: {"src": List[List[int]], "tgt": List[List[int]]}.
                                  For parsing: {"inputs": List[List[int]]}.
        """
        batches: List[Dict[str, Any]] = []
        if is_pair:
            current_batch: Dict[str, List[List[int]]] = {"src": [], "tgt": []}
            current_src_count: int = 0
            current_tgt_count: int = 0
            for src_tokens, tgt_tokens in examples:
                src_len: int = len(src_tokens)
                tgt_len: int = len(tgt_tokens)
                # If adding this example exceeds the batch token limit for either side, start a new batch.
                if (current_src_count + src_len > self.batch_tokens) or (current_tgt_count + tgt_len > self.batch_tokens):
                    if current_batch["src"]:
                        batches.append(current_batch)
                    current_batch = {"src": [], "tgt": []}
                    current_src_count = 0
                    current_tgt_count = 0
                current_batch["src"].append(src_tokens)
                current_batch["tgt"].append(tgt_tokens)
                current_src_count += src_len
                current_tgt_count += tgt_len
            if current_batch["src"]:
                batches.append(current_batch)
            LOGGER.info(f"Created {len(batches)} batches for translation data.")
        else:
            # For parsing tasks.
            current_batch: Dict[str, List[List[int]]] = {"inputs": []}
            current_token_count: int = 0
            for tokens in examples:
                token_len: int = len(tokens)
                if current_token_count + token_len > self.batch_tokens:
                    if current_batch["inputs"]:
                        batches.append(current_batch)
                    current_batch = {"inputs": []}
                    current_token_count = 0
                current_batch["inputs"].append(tokens)
                current_token_count += token_len
            if current_batch["inputs"]:
                batches.append(current_batch)
            LOGGER.info(f"Created {len(batches)} batches for parsing data.")
        return batches

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Loads raw text data from files, tokenizes the data, splits it into training, development,
        and test sets, and constructs batches based on token count.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
                (train_batches, dev_batches, test_batches)
        """
        if self.task == "translation":
            # Read source and target files.
            src_lines: List[str] = self._read_lines(self.src_file)
            tgt_lines: List[str] = self._read_lines(self.tgt_file)
            if not src_lines or not tgt_lines:
                LOGGER.error("Translation data files are empty or could not be read.")
                return [], [], []
            if len(src_lines) != len(tgt_lines):
                LOGGER.warning(
                    f"Source and target file line counts differ: {len(src_lines)} vs. {len(tgt_lines)}. "
                    "Proceeding with minimum count."
                )
            tokenized_examples: List[Tuple[List[int], List[int]]] = self._tokenize_translation(src_lines, tgt_lines)
            train_examples, dev_examples, test_examples = self._split_data(tokenized_examples)
            train_batches: List[Dict[str, Any]] = self._create_batches(train_examples, is_pair=True)
            dev_batches: List[Dict[str, Any]] = self._create_batches(dev_examples, is_pair=True)
            test_batches: List[Dict[str, Any]] = self._create_batches(test_examples, is_pair=True)
            LOGGER.info("Translation data loaded and batched successfully.")
            return train_batches, dev_batches, test_batches

        elif self.task == "parsing":
            # Read WSJ file.
            wsj_lines: List[str] = self._read_lines(self.wsj_file)
            if not wsj_lines:
                LOGGER.error("WSJ parsing data file is empty or could not be read.")
                return [], [], []
            # Note: Here we simply tokenize sentences. In practice, tree structures should be
            # linearized (e.g., using a bracket-based representation). This implementation uses raw sentences.
            tokenized_examples: List[List[int]] = self._tokenize_parsing(wsj_lines)
            train_examples, dev_examples, test_examples = self._split_data(tokenized_examples)
            train_batches: List[Dict[str, Any]] = self._create_batches(train_examples, is_pair=False)
            dev_batches: List[Dict[str, Any]] = self._create_batches(dev_examples, is_pair=False)
            test_batches: List[Dict[str, Any]] = self._create_batches(test_examples, is_pair=False)
            LOGGER.info("Parsing data loaded and batched successfully.")
            return train_batches, dev_batches, test_batches

        else:
            LOGGER.error(f"Unsupported task type during data loading: {self.task}")
            return [], [], []


# For debugging and testing purposes.
if __name__ == "__main__":
    # Create a DatasetLoader for translation and print dataset statistics.
    loader_translation = DatasetLoader(CONFIG, task="translation")
    train_data, dev_data, test_data = loader_translation.load_data()
    LOGGER.info(f"Translation: {len(train_data)} training batches, {len(dev_data)} dev batches, {len(test_data)} test batches.")

    # Create a DatasetLoader for parsing and print dataset statistics.
    loader_parsing = DatasetLoader(CONFIG, task="parsing")
    p_train, p_dev, p_test = loader_parsing.load_data()
    LOGGER.info(f"Parsing: {len(p_train)} training batches, {len(p_dev)} dev batches, {len(p_test)} test batches.")
