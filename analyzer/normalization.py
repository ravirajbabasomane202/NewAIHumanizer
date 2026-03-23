"""
analyzer/normalization.py
-------------------------
Normalization utilities applied post-feature-extraction.

Features already output [0,1] scores internally, but this module
provides a final pass to:
 - Clamp any out-of-range values
 - Apply optional calibration corrections
 - Document each feature's normalization strategy
"""

from typing import Dict, Any


# Normalization strategies documentation (for explainability)
NORMALIZATION_STRATEGIES: Dict[str, str] = {
    "perplexity":             "Sigmoid-inverse on log(perplexity); low PP → score≈1",
    "burstiness":             "Sigmoid-inverse on CoV; low CoV → score≈1",
    "sentence_diversity":     "1 - (unique_POS_seqs / total_sents); clipped to [0,1]",
    "lexical_diversity":      "1 - MTTR; low vocabulary richness → score≈1",
    "repetition":             "Repeated n-gram ratio; already in [0,1]",
    "semantic_predictability":"Avg TF-IDF cosine similarity; already in [0,1]",
    "syntactic_complexity":   "Sigmoid-inverse on clause density; low complexity → score≈1",
    "function_word_dist":     "Sigmoid on |SW_ratio - baseline|; deviation → score≈1",
    "entropy":                "1 - normalized Shannon entropy; low entropy → score≈1",
    "ngram_frequency_bias":   "Sigmoid on common-word ratio; high ratio → score≈1",
    "over_coherence":         "Sigmoid-inverse on similarity std; low std → score≈1",
    "emotional_variability":  "Sigmoid-inverse on sentiment std; flat emotion → score≈1",
    "error_patterns":         "Sigmoid-inverse on error density; few errors → score≈1",
    "contextual_depth":       "Sigmoid-inverse on personal/subjective ratio; low → score≈1",
    "stopword_patterning":    "Sigmoid-inverse on positional SW std; uniform → score≈1",
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Ensure value is within [lo, hi]."""
    return max(lo, min(hi, value))


def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Final normalization pass.

    Clamps all feature scores to [0, 1].
    Features are expected to already be in [0,1] range from their
    individual compute() methods.

    Parameters
    ----------
    raw_scores : dict mapping feature_name → raw_score (float)

    Returns
    -------
    dict mapping feature_name → normalized_score in [0,1]
    """
    return {name: round(clamp(score), 4) for name, score in raw_scores.items()}


def normalize_sentence_scores(
    per_sentence: Dict[str, list]
) -> Dict[str, list]:
    """
    Clamp per-sentence scores to [0, 1].

    Parameters
    ----------
    per_sentence : dict mapping feature_name → List[float]

    Returns
    -------
    dict mapping feature_name → List[float] all in [0, 1]
    """
    return {
        name: [round(clamp(s), 4) for s in scores]
        for name, scores in per_sentence.items()
    }
