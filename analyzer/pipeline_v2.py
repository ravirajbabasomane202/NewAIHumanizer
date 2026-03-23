"""
analyzer/pipeline_v2.py
-----------------------
Upgraded pipeline: v1 + v2 features, ML scoring, calibration.

Pipeline:
  INPUT → PREPROCESS → FEATURE EXTRACTION (v1+v2) → NORMALIZATION
        → ML SCORING → HIGHLIGHT MAPPING → EXPLANATION → OUTPUT
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import yaml

from analyzer.features import ALL_FEATURES
from analyzer.features_v2 import V2_FEATURES
from analyzer.normalization import normalize_scores, normalize_sentence_scores
from analyzer.scoring import classify
from analyzer.highlighting import build_highlights
from analyzer.explanation import generate_explanation
from analyzer.ml_scorer import MLScorer, FEATURE_NAMES
from analyzer.pipeline import preprocess, load_config

logger = logging.getLogger(__name__)

# ── Global ML scorer (singleton, trained once on first use) ──────────────────
_ml_scorer: Optional[MLScorer] = None


def get_ml_scorer(force_retrain: bool = False) -> MLScorer:
    """Return (and lazily train) the global ML scorer."""
    global _ml_scorer
    if _ml_scorer is None:
        _ml_scorer = MLScorer()
    if not _ml_scorer.is_trained or force_retrain:
        if not force_retrain and not _ml_scorer.load():
            logger.info("Training ML scorer (first run)...")
            _ml_scorer.train()
        elif force_retrain:
            _ml_scorer.train()
    return _ml_scorer


def extract_all_features(text: str, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Run all 25 features (v1 + v2) on text.
    """
    results = {}
    all_feats = ALL_FEATURES + V2_FEATURES

    for feature in all_feats:
        try:
            t0 = time.perf_counter()
            result = feature.compute(text)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.debug(f"Feature '{feature.name}': {result.get('score', '?'):.3f} in {elapsed:.1f}ms")
            results[feature.name] = result
        except Exception as e:
            logger.error(f"Feature '{feature.name}' failed: {e}")
            results[feature.name] = {
                "raw": 0.0, "score": 0.5, "sentence_scores": [], "description": f"Error: {e}"
            }

    return results


def analyze_v2(
    text: str,
    config: Optional[Dict[str, Any]] = None,
    include_explanation: bool = True,
    debug: bool = False,
    use_ml: bool = True,
) -> Dict[str, Any]:
    """
    Full upgraded pipeline: 25 features + ML scoring.

    Returns same structure as v1 analyze() but with richer fields:
    {
        "scores":          { feature_name: score, ... },
        "final_score":     float (0-100),
        "ml_score":        float (0-100, from ML model),
        "classification":  str,
        "confidence":      str,
        "highlights":      [...],
        "explanation":     {...},
        "feature_importances": {...},
        "ml_metrics":      {...},
        "metadata":        {...},
    }
    """
    t_start = time.perf_counter()

    if config is None:
        config = load_config()

    settings = config.get("settings", {})
    max_chars = settings.get("max_input_chars", 100000)

    # Stage 1: Preprocess
    cleaned_text, sentences = preprocess(text, max_chars=max_chars)

    # Stage 2: Extract all 25 features
    feature_results = extract_all_features(cleaned_text, config)

    # Stage 3: Normalize
    raw_scores = {name: res["score"] for name, res in feature_results.items()}
    per_sentence = {name: res.get("sentence_scores", []) for name, res in feature_results.items()}
    normalized = normalize_scores(raw_scores)
    normalized_per_sentence = normalize_sentence_scores(per_sentence)

    # Stage 4: ML scoring
    ml_result = {"score": 50.0, "label": "Mixed / Uncertain", "confidence": "Low", "method": "none"}
    ml_scorer = None

    if use_ml:
        try:
            ml_scorer = get_ml_scorer()
            ml_result = ml_scorer.predict(normalized)
        except Exception as e:
            logger.error(f"ML scorer failed: {e}. Falling back to weighted sum.")

    ml_score = ml_result["score"]
    classification = ml_result["label"]
    confidence = ml_result["confidence"]

    # Also compute v1-style weighted sum for blending / comparison
    weights = config.get("features", {})
    total_w = sum(weights.get(k, 0) for k in normalized)
    if total_w > 0:
        weighted_score = sum(weights.get(k, 0) * v for k, v in normalized.items()) / total_w * 100
    else:
        weighted_score = ml_score

    # Blend: 70% ML + 30% heuristic (graceful degradation)
    if use_ml and ml_scorer and ml_scorer.is_trained:
        final_score = 0.70 * ml_score + 0.30 * weighted_score
    else:
        final_score = weighted_score
        thresholds = config.get("thresholds", {"human_max": 30, "mixed_max": 70})
        classification = classify(final_score, thresholds)
        confidence = "Low"

    final_score = round(min(100.0, max(0.0, final_score)), 2)

    # Stage 5: Highlights
    hl_thresholds = config.get("highlight_thresholds", {})
    # Use combined weights (v1 + v2 equal weight for v2 features)
    all_weights = dict(weights)
    v2_default_w = 0.04  # give v2 features modest default weight
    for feat in V2_FEATURES:
        if feat.name not in all_weights:
            all_weights[feat.name] = v2_default_w

    highlights = build_highlights(
        cleaned_text, sentences, normalized_per_sentence, all_weights, hl_thresholds
    )

    # Stage 6: Explanation
    explanation = None
    if include_explanation:
        # Include contribution from all features
        contributions = {name: round(normalized.get(name, 0) * 100, 2) for name in normalized}
        explanation = generate_explanation(final_score, classification, normalized, contributions)

    elapsed = time.perf_counter() - t_start

    result: Dict[str, Any] = {
        "scores": normalized,
        "final_score": final_score,
        "ml_score": round(ml_score, 2),
        "weighted_score": round(weighted_score, 2),
        "classification": classification,
        "confidence": confidence,
        "highlights": highlights,
        "metadata": {
            "char_count": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
            "sentence_count": len(sentences),
            "analysis_time_ms": round(elapsed * 1000, 1),
            "features_computed": len(feature_results),
            "scoring_method": ml_result.get("method", "heuristic"),
        },
    }

    if explanation:
        result["explanation"] = explanation

    if ml_scorer and ml_scorer.is_trained:
        result["feature_importances"] = ml_scorer.feature_importances
        result["ml_train_metrics"] = ml_scorer.metrics

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
        f"v2 analysis: score={final_score:.1f} (ml={ml_score:.1f}|heur={weighted_score:.1f}), "
        f"class='{classification}', time={elapsed*1000:.0f}ms"
    )
    return result
