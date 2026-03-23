"""
analyzer/pipeline.py
--------------------
Main analysis pipeline orchestrator.

Pipeline stages:
  INPUT → PREPROCESS → FEATURE EXTRACTION → NORMALIZATION
        → SCORING → HIGHLIGHT MAPPING → EXPLANATION → OUTPUT

Each stage is modular and independently testable.
"""

import re
import time
import logging
import hashlib
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

import yaml

from analyzer.features import ALL_FEATURES, FEATURE_MAP
from analyzer.normalization import normalize_scores, normalize_sentence_scores
from analyzer.scoring import compute_final_score, classify
from analyzer.highlighting import build_highlights
from analyzer.explanation import generate_explanation

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = "config/weights.yaml"


def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate the YAML config file."""
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Config loaded from {path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config not found at {path}. Using defaults.")
        return _default_config()
    except yaml.YAMLError as e:
        logger.error(f"Config parse error: {e}. Using defaults.")
        return _default_config()


def _default_config() -> Dict[str, Any]:
    """Fallback config with equal weights if YAML not found."""
    n = len(ALL_FEATURES)
    weight = round(1.0 / n, 4)
    return {
        "features": {f.name: weight for f in ALL_FEATURES},
        "thresholds": {"human_max": 30, "mixed_max": 70, "ai_min": 70},
        "highlight_thresholds": {f.name: 0.60 for f in ALL_FEATURES},
        "settings": {
            "moving_ttr_window": 50,
            "repetition_ngram_size": 3,
            "min_sentences": 2,
            "max_input_chars": 100000,
            "cache_enabled": True,
            "debug": False,
            "log_level": "INFO",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str, max_chars: int = 100000) -> Tuple[str, List[str]]:
    """
    Clean and segment input text.

    Returns
    -------
    (cleaned_text, sentences)
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")

    # Security: truncate oversized input
    if len(text) > max_chars:
        logger.warning(f"Input truncated from {len(text)} to {max_chars} chars.")
        text = text[:max_chars]

    # Normalize whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Sentence tokenize
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    sentences = [s.strip() for s in sentences if s.strip()]
    return text, sentences


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(text: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Run all features on the text.

    Returns
    -------
    dict: feature_name → full feature result dict
    """
    results = {}
    enabled_features = config.get("features", {})

    for feature in ALL_FEATURES:
        if feature.name not in enabled_features:
            logger.debug(f"Feature '{feature.name}' not in config, skipping.")
            continue
        try:
            t0 = time.perf_counter()
            result = feature.compute(text)
            elapsed = time.perf_counter() - t0
            logger.debug(f"Feature '{feature.name}' computed in {elapsed*1000:.1f}ms")
            results[feature.name] = result
        except Exception as e:
            logger.error(f"Feature '{feature.name}' failed: {e}")
            results[feature.name] = {
                "raw": 0.0,
                "score": 0.5,
                "sentence_scores": [],
                "description": f"Error: {e}",
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(
    feature_results: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Extract and normalize scores from feature results.

    Returns
    -------
    (normalized_scores, normalized_per_sentence_scores)
    """
    raw_scores = {name: res["score"] for name, res in feature_results.items()}
    per_sentence = {name: res.get("sentence_scores", []) for name, res in feature_results.items()}

    normalized = normalize_scores(raw_scores)
    normalized_per_sentence = normalize_sentence_scores(per_sentence)
    return normalized, normalized_per_sentence


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_text(
    normalized_scores: Dict[str, float],
    config: Dict[str, Any],
) -> Tuple[float, str, Dict[str, float]]:
    """
    Compute final score and classification.

    Returns
    -------
    (final_score, classification, contributions)
    """
    weights = config.get("features", {})
    thresholds = config.get("thresholds", {"human_max": 30, "mixed_max": 70})

    final_score, contributions = compute_final_score(normalized_scores, weights)
    classification = classify(final_score, thresholds)
    return final_score, classification, contributions


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Highlight mapping
# ─────────────────────────────────────────────────────────────────────────────

def map_highlights(
    text: str,
    sentences: List[str],
    per_sentence_scores: Dict[str, List[float]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build sentence-level highlight objects."""
    weights = config.get("features", {})
    thresholds = config.get("highlight_thresholds", {})
    return build_highlights(text, sentences, per_sentence_scores, weights, thresholds)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Explanation
# ─────────────────────────────────────────────────────────────────────────────

def build_explanation(
    final_score: float,
    classification: str,
    normalized_scores: Dict[str, float],
    contributions: Dict[str, float],
) -> Dict[str, Any]:
    """Generate human-readable explanation."""
    return generate_explanation(final_score, classification, normalized_scores, contributions)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    text: str,
    config: Optional[Dict[str, Any]] = None,
    include_explanation: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Full analysis pipeline: text → structured result.

    Parameters
    ----------
    text                : input text to analyze
    config              : loaded config dict (if None, loads from default path)
    include_explanation : whether to include explanation layer
    debug               : include raw feature results in output

    Returns
    -------
    {
        "scores": {feature_name: score, ...},
        "final_score": float,
        "classification": str,
        "highlights": [...],
        "explanation": {...},
        "metadata": {...},
    }
    """
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    settings = config.get("settings", {})
    max_chars = settings.get("max_input_chars", 100000)

    # ── Stage 1: Preprocess ──────────────────────────────────────────────────
    cleaned_text, sentences = preprocess(text, max_chars=max_chars)
    logger.info(f"Preprocessed: {len(cleaned_text)} chars, {len(sentences)} sentences")

    # Minimum content check
    min_sents = settings.get("min_sentences", 2)
    if len(sentences) < min_sents:
        logger.warning(f"Only {len(sentences)} sentence(s). Results may be unreliable.")

    # ── Stage 2: Feature Extraction ──────────────────────────────────────────
    feature_results = extract_features(cleaned_text, config)

    # ── Stage 3: Normalization ───────────────────────────────────────────────
    normalized_scores, per_sentence_scores = normalize_features(feature_results)

    # ── Stage 4: Scoring ─────────────────────────────────────────────────────
    final_score, classification, contributions = score_text(normalized_scores, config)

    # ── Stage 5: Highlight Mapping ───────────────────────────────────────────
    highlights = map_highlights(cleaned_text, sentences, per_sentence_scores, config)

    # ── Stage 6: Explanation ─────────────────────────────────────────────────
    explanation = None
    if include_explanation:
        explanation = build_explanation(
            final_score, classification, normalized_scores, contributions
        )

    # ── Stage 7: Output assembly ─────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start

    result = {
        "scores": normalized_scores,
        "final_score": final_score,
        "classification": classification,
        "highlights": highlights,
        "metadata": {
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "sentence_count": len(sentences),
            "analysis_time_ms": round(elapsed * 1000, 1),
        },
    }

    if explanation:
        result["explanation"] = explanation

    if debug:
        result["debug"] = {
            name: {
                "raw": res.get("raw"),
                "score": res.get("score"),
                "description": res.get("description"),
            }
            for name, res in feature_results.items()
        }

    logger.info(
        f"Analysis complete: score={final_score:.1f}, "
        f"class='{classification}', time={elapsed*1000:.0f}ms"
    )
    return result
