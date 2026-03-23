"""
analyzer/highlighting.py
------------------------
Maps feature signals to specific sentences/tokens with reason tags.

Highlight logic:
  - Each sentence gets an aggregate AI-likelihood score from all features
  - Sentences above threshold are flagged with reason tags
  - Returns list of highlight dicts with:
    {text, start, end, label, reasons, score}
"""

import re
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# Human-readable reason messages for each feature
FEATURE_REASONS: Dict[str, str] = {
    "perplexity":             "low perplexity (predictable word choices)",
    "burstiness":             "uniform sentence length",
    "sentence_diversity":     "repetitive sentence structure",
    "lexical_diversity":      "limited vocabulary",
    "repetition":             "repeated phrases",
    "semantic_predictability":"high sentence-to-sentence similarity",
    "syntactic_complexity":   "simple flat sentence structure",
    "function_word_dist":     "unusual stopword distribution",
    "entropy":                "concentrated word frequency",
    "ngram_frequency_bias":   "over-reliance on common words",
    "over_coherence":         "over-coherent (unnaturally consistent)",
    "emotional_variability":  "flat emotional tone",
    "error_patterns":         "near-perfect grammar (no errors)",
    "contextual_depth":       "impersonal / lacks subjective voice",
    "stopword_patterning":    "uniform stopword positioning",
}


def find_sentence_positions(text: str, sentences: List[str]) -> List[Tuple[int, int]]:
    """
    Find the character start/end positions of each sentence in the original text.

    Uses a left-to-right search so overlapping/similar sentences are matched
    in order of appearance.

    Parameters
    ----------
    text      : original full text
    sentences : list of sentence strings (in order)

    Returns
    -------
    List of (start, end) tuples (end is exclusive)
    """
    positions = []
    search_start = 0

    for sent in sentences:
        # Escape for regex and find first occurrence from search_start
        escaped = re.escape(sent.strip())
        match = re.search(escaped, text[search_start:])
        if match:
            start = search_start + match.start()
            end = search_start + match.end()
            positions.append((start, end))
            search_start = end
        else:
            # Fallback: approximate position
            positions.append((search_start, search_start + len(sent)))
            search_start += len(sent)

    return positions


def build_highlights(
    text: str,
    sentences: List[str],
    per_sentence_scores: Dict[str, List[float]],
    weights: Dict[str, float],
    thresholds: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Build the list of highlight objects for each sentence.

    For each sentence:
      1. Collect feature scores for this sentence index
      2. Compute weighted aggregate score
      3. Determine which features are above their individual thresholds
      4. Assign label: "AI", "Mixed", or "Human"
      5. Attach reason tags

    Parameters
    ----------
    text                : original text
    sentences           : list of sentence strings
    per_sentence_scores : dict  feature_name → List[float] per-sentence scores
    weights             : dict  feature_name → weight
    thresholds          : dict  feature_name → threshold value (above = AI signal)

    Returns
    -------
    List of highlight dicts
    """
    positions = find_sentence_positions(text, sentences)
    highlights = []

    for i, (sent, (start, end)) in enumerate(zip(sentences, positions)):
        # Gather per-sentence scores for this index
        sentence_feature_scores: Dict[str, float] = {}
        for feat_name, scores in per_sentence_scores.items():
            if i < len(scores):
                sentence_feature_scores[feat_name] = scores[i]
            else:
                sentence_feature_scores[feat_name] = 0.5  # neutral fallback

        # Compute weighted aggregate for this sentence
        total_w = 0.0
        weighted_sum = 0.0
        for feat_name, score in sentence_feature_scores.items():
            w = weights.get(feat_name, 0.0)
            weighted_sum += w * score
            total_w += w

        agg_score = (weighted_sum / total_w) if total_w > 0 else 0.5

        # Find which features fire for this sentence
        reasons = []
        for feat_name, score in sentence_feature_scores.items():
            threshold = thresholds.get(feat_name, 0.6)
            if score >= threshold:
                reason = FEATURE_REASONS.get(feat_name, feat_name)
                reasons.append(reason)

        # Assign label
        if agg_score >= 0.70:
            label = "AI"
        elif agg_score >= 0.40:
            label = "Mixed"
        else:
            label = "Human"

        highlights.append({
            "text": sent,
            "start": start,
            "end": end,
            "label": label,
            "score": round(agg_score * 100, 1),
            "reasons": reasons,
        })

        logger.debug(
            f"Sentence {i}: label={label}, score={agg_score:.3f}, "
            f"reasons={reasons[:3]}"
        )

    return highlights


def build_token_highlights(
    sentence: str,
    sentence_score: float,
    reasons: List[str],
) -> List[Dict[str, Any]]:
    """
    Break a flagged sentence into token-level highlights.

    Flags tokens that are particularly "AI-like" based on simple heuristics:
      - Very common words (function words in a string)
      - Repeated tokens within sentence

    Parameters
    ----------
    sentence       : sentence text
    sentence_score : overall AI-likelihood score for this sentence (0–1)
    reasons        : list of reason strings for this sentence

    Returns
    -------
    List of token highlight dicts {token, start, end, flagged}
    """
    tokens = sentence.split()
    token_highlights = []
    offset = 0

    # Find repeated tokens
    token_counts = {}
    for t in tokens:
        clean = re.sub(r'[^\w]', '', t.lower())
        token_counts[clean] = token_counts.get(clean, 0) + 1

    for token in tokens:
        # Find token position in sentence
        idx = sentence.find(token, offset)
        if idx == -1:
            idx = offset
        token_end = idx + len(token)
        offset = token_end

        clean = re.sub(r'[^\w]', '', token.lower())
        is_repeated = token_counts.get(clean, 0) > 1 and len(clean) > 2
        flagged = is_repeated and sentence_score > 0.6

        token_highlights.append({
            "token": token,
            "start": idx,
            "end": token_end,
            "flagged": flagged,
        })

    return token_highlights
