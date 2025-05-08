"""evaluation.py

This module implements the Evaluation class for assessing the performance of the trained 
Transformer model. It provides two primary evaluation methods:
  • evaluate_mt() for computing BLEU scores on machine translation tasks.
  • evaluate_parsing() for computing F1 (and associated precision and recall) scores 
    on constituency parsing tasks.

The evaluation routines use beam search decoding with parameters (beam size, length penalty,
and maximum output length offset) specified in the central configuration (config.yaml). 
Token decoding and other utility functions are imported from utils.py. For constituency parsing,
linearized tree outputs are converted to bracketed tree structures and evaluated by comparing 
their constituent spans.

All configuration values and hyperparameters are read from the central config.yaml file 
(via config.py). Default values are set where necessary.
"""

import math
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Set, Tuple
from tqdm import tqdm

# Import configuration and utilities.
from config import CONFIG
from model import TransformerModel
from utils import LOGGER, decode_tokens, load_tokenizer

# ---------------------------------------------------------------------------
# Helper Functions for Beam Search Decoding and Metric Computation
# ---------------------------------------------------------------------------

def _beam_search_decode(
    model: TransformerModel,
    src: torch.Tensor,
    device: torch.device,
    beam_size: int,
    length_penalty: float,
    max_output_length: int,
    start_token: int = 1,
    end_token: int = 2,
) -> List[int]:
    """
    Performs beam search decoding for a single source instance.
    
    Args:
        model: Trained TransformerModel instance.
        src: Source tensor of shape (1, src_seq_len).
        device: Torch device.
        beam_size: Beam search width.
        length_penalty: Exponent for length penalty.
        max_output_length: Maximum target sequence length (source length + offset).
        start_token: Token ID for begin-of-sentence (default: 1).
        end_token: Token ID for end-of-sentence (default: 2).
    
    Returns:
        A list of token IDs representing the best decoded sequence.
    """
    # Each beam candidate is a tuple: (sequence, cumulative_log_prob)
    beam: List[Tuple[List[int], float]] = [([start_token], 0.0)]
    completed_candidates: List[Tuple[List[int], float]] = []
    
    # Set model to evaluation and disable gradient computation.
    model.eval()
    with torch.no_grad():
        for _ in range(max_output_length):
            new_beam: List[Tuple[List[int], float]] = []
            for seq, cum_log_prob in beam:
                # If last token is end_token, add candidate to new beam unchanged.
                if seq[-1] == end_token:
                    new_beam.append((seq, cum_log_prob))
                    continue

                # Prepare target tensor for current candidate.
                tgt_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
                # Generate causal mask for target.
                tgt_len = tgt_tensor.size(1)
                # The model has a helper _generate_subsequent_mask, so we call it.
                tgt_mask = model._generate_subsequent_mask(tgt_len).to(device)
                
                # Forward pass: compute logits. Note: We only need the last time-step.
                logits = model(src, tgt_tensor, src_mask=None, tgt_mask=tgt_mask)  # (1, seq_len, vocab_size)
                logits_last = logits[0, -1, :]  # (vocab_size)
                log_probs = F.log_softmax(logits_last, dim=-1)  # (vocab_size)
                
                # Get top beam_size candidates.
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                topk_log_probs = topk_log_probs.cpu().tolist()
                topk_indices = topk_indices.cpu().tolist()

                for token_id, token_log_prob in zip(topk_indices, topk_log_probs):
                    new_seq = seq + [token_id]
                    new_log_prob = cum_log_prob + token_log_prob
                    new_beam.append((new_seq, new_log_prob))
            
            # If no candidates found, break.
            if not new_beam:
                break

            # Apply length penalty and select top beam_size candidates.
            def score_fn(candidate: Tuple[List[int], float]) -> float:
                seq, log_prob = candidate
                # Length penalty: (len(seq))**length_penalty
                return log_prob / (len(seq) ** length_penalty)
            
            new_beam = sorted(new_beam, key=score_fn, reverse=True)[:beam_size]
            beam = new_beam

            # Check if all candidates have ended.
            if all(seq[-1] == end_token for seq, _ in beam):
                break
        
        # Select the candidate with the highest score from the final beam.
        best_candidate = max(beam, key=score_fn)[0]
    
    return best_candidate

def compute_corpus_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Computes a corpus-level BLEU score for machine translation.
    This is a simplified BLEU implementation using up to 4-gram matching.
    
    Args:
        references: List of reference sentences (strings).
        hypotheses: List of hypothesis sentences (strings).
    
    Returns:
        BLEU score as a float.
    """
    import math

    def ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        counts = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            counts[ngram] = counts.get(ngram, 0) + 1
        return counts
    
    total_matches = {n: 0 for n in range(1, 5)}
    total_ngrams = {n: 0 for n in range(1, 5)}
    ref_length = 0
    hyp_length = 0
    
    for ref_sentence, hyp_sentence in zip(references, hypotheses):
        ref_tokens = ref_sentence.split()
        hyp_tokens = hyp_sentence.split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)
        for n in range(1, 5):
            ref_counts = ngram_counts(ref_tokens, n)
            hyp_counts = ngram_counts(hyp_tokens, n)
            total_ngrams[n] += max(len(hyp_tokens) - n + 1, 0)
            for ngram in hyp_counts:
                total_matches[n] += min(hyp_counts[ngram], ref_counts.get(ngram, 0))
    
    precisions = []
    for n in range(1, 5):
        if total_ngrams[n] == 0:
            precisions.append(0)
        else:
            precisions.append(total_matches[n] / total_ngrams[n])
    
    # Avoid log(0)
    if min(precisions) == 0:
        geo_mean = 0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    
    # Brevity penalty
    bp = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length) if hyp_length > 0 else 0.0
    bleu = bp * geo_mean
    return bleu

def _parse_tree(tokens: List[str], index: int, word_index: int) -> Tuple[Set[Tuple[int, int]], int, int]:
    """
    Recursively parses a bracketed tree represented as a list of tokens and returns 
    the set of constituent spans along with the updated index and the current word position.
    
    Args:
        tokens: List of tokens (strings) from the linearized tree.
        index: Current index in the tokens list.
        word_index: Current word position (integer).
    
    Returns:
        A tuple (spans, new_index, new_word_index) where:
            spans: Set of tuples indicating constituent spans (start, end).
            new_index: Updated index after parsing this node.
            new_word_index: Updated word position after processing subtree.
    """
    spans: Set[Tuple[int, int]] = set()
    assert tokens[index] == '(', "Expected '(' at token index {}.".format(index)
    index += 1  # Skip '('
    # Skip the label token.
    label = tokens[index]
    index += 1
    start = word_index
    # Process children until encountering ')'
    while tokens[index] != ')':
        if tokens[index] == '(':
            child_spans, index, word_index = _parse_tree(tokens, index, word_index)
            spans.update(child_spans)
        else:
            # Leaf word encountered.
            word_index += 1
            index += 1
    # Current node span from start to current word_index
    end = word_index
    spans.add((start, end))
    index += 1  # Skip ')'
    return spans, index, word_index

def tree_to_spans(tree_str: str) -> Set[Tuple[int, int]]:
    """
    Converts a bracketed tree string into a set of constituent spans.
    A span is defined as a tuple (start, end) with start inclusive and end exclusive.
    
    Args:
        tree_str: String representing the linearized parse tree in bracketed notation.
    
    Returns:
        A set of tuples representing the spans in the tree.
    """
    tokens = tree_str.replace('(', ' ( ').replace(')', ' ) ').split()
    spans, _, _ = _parse_tree(tokens, 0, 0)
    return spans

def compute_parsing_f1(gold_trees: List[str], pred_trees: List[str]) -> Dict[str, float]:
    """
    Computes F1 score (with precision and recall) for constituency parsing by comparing 
    the spans extracted from the gold-standard and predicted bracketed tree strings.
    
    Args:
        gold_trees: List of gold parse trees as bracketed strings.
        pred_trees: List of predicted parse trees as bracketed strings.
    
    Returns:
        A dictionary with keys "F1", "precision", and "recall".
    """
    total_gold = 0
    total_pred = 0
    total_overlap = 0
    
    for gold_tree, pred_tree in zip(gold_trees, pred_trees):
        gold_spans = tree_to_spans(gold_tree)
        pred_spans = tree_to_spans(pred_tree)
        total_gold += len(gold_spans)
        total_pred += len(pred_spans)
        total_overlap += len(gold_spans.intersection(pred_spans))
    
    precision = total_overlap / total_pred if total_pred > 0 else 0.0
    recall = total_overlap / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"F1": f1, "precision": precision, "recall": recall}

def linearization_to_tree(linear_str: str) -> str:
    """
    Converts a linearized tree string into a bracketed tree string.
    Placeholder implementation: This function currently assumes the linearized representation
    is already in bracketed form. Update this function if a different conversion strategy is needed.
    
    Args:
        linear_str: Linearized string representation of the parse tree.
    
    Returns:
        A bracketed tree string.
    """
    return linear_str  # Identity transformation (placeholder)

# ---------------------------------------------------------------------------
# Evaluation Class
# ---------------------------------------------------------------------------

class Evaluation:
    """
    Evaluation class for assessing Transformer model performance. It supports
    evaluation for both machine translation (BLEU score) and constituency parsing (F1 score).

    Attributes:
        model: Trained TransformerModel instance.
        test_data: Test dataset (list of batch dictionaries).
        config: Configuration parameters loaded from config.yaml.
        task: Task type ("translation" or "parsing").
        tokenizer: Tokenizer instance for decoding token IDs into text.
        device: Torch device.
    """

    def __init__(self, model: TransformerModel, test_data: List[Dict[str, Any]], config: Any, task: str = "translation") -> None:
        """
        Initializes the Evaluation instance.
        
        Args:
            model: Trained TransformerModel instance.
            test_data: Test dataset (list of batches). For translation, each batch should have keys "src" and "tgt".
                       For parsing, each batch should have key "inputs".
            config: Central configuration object loaded from config.yaml.
            task: Task type; either "translation" or "parsing". Defaults to "translation".
        """
        self.model = model
        self.test_data = test_data
        self.config = config
        self.task = task.lower()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Load tokenizer based on task settings.
        if self.task == "translation":
            tokenizer_type = self.config.data.wmt_2014_en_de.tokenizer
        else:
            tokenizer_type = "BPE"
        self.tokenizer = load_tokenizer(tokenizer_type)
        LOGGER.info(f"Evaluation initialized for task '{self.task}' using device {self.device}.")

    def evaluate_mt(self) -> Dict[str, Any]:
        """
        Evaluates the Transformer model on a machine translation test dataset.
        Uses beam search decoding with parameters:
          - beam_size from config.inference.beam_size (default 4)
          - length_penalty from config.inference.length_penalty (default 0.6)
          - max_output_length set to (input_length + config.inference.max_output_length_offset, default offset 50)
        
        Returns:
            A dictionary with the final BLEU score and additional statistics.
        """
        # Retrieve inference parameters from configuration.
        beam_size: int = self.config.inference.beam_size
        length_penalty: float = self.config.inference.length_penalty
        max_offset: int = self.config.inference.max_output_length_offset

        all_references: List[str] = []
        all_hypotheses: List[str] = []
        total_sentences: int = 0

        LOGGER.info("Starting machine translation evaluation...")
        for batch in tqdm(self.test_data, desc="Evaluating MT", unit="batch"):
            # Expect batch to have keys "src" and "tgt", each a list of token ID lists.
            src_batch: List[List[int]] = batch.get("src", [])
            tgt_batch: List[List[int]] = batch.get("tgt", [])
            for src_tokens, tgt_tokens in zip(src_batch, tgt_batch):
                # Prepare source tensor (batch size = 1).
                src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                # Determine maximum output length.
                max_output_length = src_tensor.size(1) + max_offset

                # Perform beam search decoding.
                pred_token_ids = _beam_search_decode(
                    self.model,
                    src_tensor,
                    self.device,
                    beam_size,
                    length_penalty,
                    max_output_length
                )
                # Decode predicted and reference tokens to strings.
                hypothesis: str = decode_tokens(pred_token_ids, self.tokenizer)
                reference: str = decode_tokens(tgt_tokens, self.tokenizer)

                all_hypotheses.append(hypothesis)
                all_references.append(reference)
                total_sentences += 1

        bleu_score: float = compute_corpus_bleu(all_references, all_hypotheses)
        LOGGER.info(f"Machine Translation Evaluation: {total_sentences} sentences evaluated, BLEU = {bleu_score:.4f}")
        return {"BLEU": bleu_score, "num_sentences": total_sentences}

    def evaluate_parsing(self) -> Dict[str, Any]:
        """
        Evaluates the Transformer model on a constituency parsing test dataset.
        Uses beam search decoding with parsing-specific parameters:
          - beam_size from config.parsing.inference.beam_size (default 21)
          - length_penalty from config.parsing.inference.length_penalty (default 0.3)
          - max_output_length set to (input_length + config.parsing.inference.max_output_length_offset, default offset 300)
        
        After decoding, the linearized outputs are converted to bracketed tree strings and compared
        to the gold-standard trees to compute F1, precision, and recall.
        
        Returns:
            A dictionary with F1, precision, recall, and sentence count.
        """
        # Retrieve parsing inference parameters.
        beam_size: int = self.config.parsing.inference.beam_size
        length_penalty: float = self.config.parsing.inference.length_penalty
        max_offset: int = self.config.parsing.inference.max_output_length_offset

        gold_trees: List[str] = []
        pred_trees: List[str] = []
        total_sentences: int = 0

        LOGGER.info("Starting constituency parsing evaluation...")
        for batch in tqdm(self.test_data, desc="Evaluating Parsing", unit="batch"):
            # Expect batch to have key "inputs": list of token ID lists.
            inputs_batch: List[List[int]] = batch.get("inputs", [])
            for input_tokens in inputs_batch:
                # For parsing, the same sequence is used as gold reference.
                # In practice, the gold tree string should be available.
                gold_linear: str = decode_tokens(input_tokens, self.tokenizer)
                gold_tree: str = linearization_to_tree(gold_linear)

                # Prepare source tensor (batch size = 1).
                src_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                max_output_length = src_tensor.size(1) + max_offset

                # Beam search decoding.
                pred_token_ids = _beam_search_decode(
                    self.model,
                    src_tensor,
                    self.device,
                    beam_size,
                    length_penalty,
                    max_output_length
                )
                pred_linear: str = decode_tokens(pred_token_ids, self.tokenizer)
                pred_tree: str = linearization_to_tree(pred_linear)

                gold_trees.append(gold_tree)
                pred_trees.append(pred_tree)
                total_sentences += 1

        parsing_metrics: Dict[str, float] = compute_parsing_f1(gold_trees, pred_trees)
        LOGGER.info(
            f"Parsing Evaluation: {total_sentences} sentences evaluated, F1 = {parsing_metrics['F1']:.4f}, "
            f"Precision = {parsing_metrics['precision']:.4f}, Recall = {parsing_metrics['recall']:.4f}"
        )
        result: Dict[str, Any] = {
            "F1": parsing_metrics["F1"],
            "precision": parsing_metrics["precision"],
            "recall": parsing_metrics["recall"],
            "num_sentences": total_sentences,
        }
        return result
