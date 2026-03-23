"""
analyzer/scoring.py
-------------------
Compute the final AI-likelihood score from individual feature scores.

Pipeline:
  feature_scores (dict) + weights (dict) → weighted_sum → final_score (0–100)
  → classification label

Classification thresholds:
  0–30:  Human-like
  30–70: Mixed / Uncertain
  70–100: Likely AI
"""

from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Default classification labels
CLASSIFICATION_LABELS = {
    "human": "Human-like",
    "mixed": "Mixed / Uncertain",
    "ai": "Likely AI",
}


def compute_final_score(
    feature_scores: Dict[str, float],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted sum of normalized feature scores.

    Math:
        final_score = Σ (weight_i * score_i)   where Σ weights = 1.0
        result ∈ [0, 1] → multiplied by 100 for display

    Parameters
    ----------
    feature_scores : dict  feature_name → score in [0,1]
    weights        : dict  feature_name → weight in [0,1]  (should sum to 1.0)

    Returns
    -------
    (final_score_0_100, contribution_dict)
        - final_score_0_100: float in [0, 100]
        - contribution_dict: feature_name → weighted_contribution (0–100 scale)
    """
    total_weight = 0.0
    weighted_sum = 0.0
    contributions: Dict[str, float] = {}

    for feature_name, weight in weights.items():
        score = feature_scores.get(feature_name, 0.5)  # fallback to neutral
        contribution = weight * score
        contributions[feature_name] = round(contribution * 100, 2)
        weighted_sum += contribution
        total_weight += weight

    if total_weight == 0:
        logger.warning("All feature weights are zero — returning 50 (neutral).")
        return 50.0, contributions

    # Normalize in case weights don't sum exactly to 1.0
    final_score = (weighted_sum / total_weight) * 100.0
    final_score = max(0.0, min(100.0, final_score))

    logger.debug(f"Final score: {final_score:.2f}  (total_weight={total_weight:.3f})")
    return round(final_score, 2), contributions


def classify(score: float, thresholds: Dict[str, float]) -> str:
    """
    Classify text based on final AI-likelihood score.

    Parameters
    ----------
    score      : float in [0, 100]
    thresholds : dict with keys 'human_max' and 'mixed_max'

    Returns
    -------
    str: classification label
    """
    human_max = thresholds.get("human_max", 30)
    mixed_max = thresholds.get("mixed_max", 70)

    if score <= human_max:
        return CLASSIFICATION_LABELS["human"]
    elif score <= mixed_max:
        return CLASSIFICATION_LABELS["mixed"]
    else:
        return CLASSIFICATION_LABELS["ai"]


def get_classification_color(classification: str) -> str:
    """Return a CSS color class name for the classification."""
    mapping = {
        CLASSIFICATION_LABELS["human"]: "human",
        CLASSIFICATION_LABELS["mixed"]: "mixed",
        CLASSIFICATION_LABELS["ai"]: "ai",
    }
    return mapping.get(classification, "mixed")
