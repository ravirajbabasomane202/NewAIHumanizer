"""
analyzer/pipeline_modern.py
============================
Modern AI detection pipeline — replaces v2 pipeline.

Key differences from v2:
  - Uses FeatureExtractor (12 calibrated features) instead of 25 noisy ones
  - DominanceScorer replaces naive weighted average
  - Anti-contradiction logic overrides stuck "Mixed" classifications
  - Dynamic thresholds based on confidence
  - Per-sentence contributions for intelligent highlighting

Pipeline:
  INPUT → PREPROCESS → FEATURE EXTRACTION → DOMINANCE SCORING
        → ML SCORING → ENSEMBLE BLEND → ANTI-CONTRADICTION
        → CONFIDENCE → CLASSIFICATION → HIGHLIGHTING → EXPLANATION
"""

import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

from analyzer.detector import (
    FeatureExtractor, ModernMLScorer, get_modern_scorer,
    dominance_score, capped_weighted_score,
    anti_contradiction_override, estimate_confidence,
    classify_with_dynamic_threshold,
    AI_DOMINANT_FEATURES, HUMAN_COUNTER_FEATURES,
    _sentences, _alpha_words, _sigmoid, _clip01, _safe_div,
)
from analyzer.pipeline import preprocess, load_config

logger = logging.getLogger(__name__)

# Feature display names for UI
FEATURE_DISPLAY_NAMES = {
    "burstiness":           "Sentence Length Variation",
    "sentence_uniformity":  "Sentence Structure Uniformity",
    "lexical_density":      "Lexical Density",
    "phrase_repetition":    "Phrase Repetition",
    "coherence_smoothness": "Semantic Coherence Smoothness",
    "compression_signal":   "Compression Signal",
    "contraction_absence":  "Contraction Absence",
    "sentence_length_cv":   "Sentence Length Consistency",
    "transition_density":   "AI Transition Word Density",
    "punct_uniformity":     "Punctuation Uniformity",
    "hapax_signal":         "Vocabulary Uniqueness",
    "sentiment_flatness":   "Sentiment Flatness",
}

# Human-readable reason strings for highlighting
FEATURE_REASONS = {
    "burstiness":           "uniform sentence lengths",
    "sentence_uniformity":  "repetitive sentence openings",
    "lexical_density":      "high content-word density",
    "phrase_repetition":    "repeated phrase patterns",
    "coherence_smoothness": "over-smooth topic transitions",
    "compression_signal":   "high text compressibility",
    "contraction_absence":  "no contractions (AI avoids 'don't', 'I'm')",
    "sentence_length_cv":   "consistent sentence length pattern",
    "transition_density":   "AI transition words (furthermore, however, etc.)",
    "punct_uniformity":     "uniform punctuation usage",
    "hapax_signal":         "low vocabulary uniqueness",
    "sentiment_flatness":   "flat emotional tone",
}

# Singleton extractor
_extractor = FeatureExtractor()


def analyze_modern(
    text: str,
    config: Optional[Dict[str, Any]] = None,
    include_explanation: bool = True,
    debug: bool = False,
    use_ml: bool = True,
) -> Dict[str, Any]:
    """
    Full modern detection pipeline.
    
    Returns:
    {
        "scores":          {feature→score},
        "final_score":     0-100,
        "dominance_score": 0-100,
        "ml_score":        0-100,
        "classification":  str,
        "confidence":      {"level", "score", ...},
        "highlights":      [...],
        "explanation":     {...},
        "metadata":        {...},
        "debug":           {...} (if debug=True)
    }
    """
    t0 = time.perf_counter()
    
    if config is None:
        config = load_config()
    
    max_chars = config.get("settings", {}).get("max_input_chars", 100000)
    cleaned_text, sent_list = preprocess(text, max_chars=max_chars)
    
    # ── Stage 1: Feature extraction ─────────────────────────────────────────
    features = _extractor.extract(cleaned_text)
    per_sentence_features = _extractor.extract_per_sentence(cleaned_text)
    
    # ── Stage 2: Dominance score (rule-based) ────────────────────────────────
    dom_score = dominance_score(features)
    dom_score_100 = dom_score * 100
    
    # ── Stage 3: ML score ────────────────────────────────────────────────────
    ml_score_100 = dom_score_100  # default
    ml_result = {}
    
    if use_ml:
        try:
            scorer = get_modern_scorer()
            ml_result = scorer.predict(features)
            ml_prob = ml_result.get("ml_probability", dom_score)
            ml_score_100 = ml_prob * 100
        except Exception as e:
            logger.warning(f"ML scorer failed: {e}. Using dominance only.")
    
    # ── Stage 4: Ensemble blend ──────────────────────────────────────────────
    # 55% dominance (strong rule-based) + 45% ML
    if use_ml and ml_result:
        raw_score = 0.55 * dom_score_100 + 0.45 * ml_score_100
    else:
        raw_score = dom_score_100
    
    # ── Stage 5: Anti-contradiction override ─────────────────────────────────
    final_score, override_reason = anti_contradiction_override(
        raw_score, features, dom_score
    )
    
    # ── Stage 6: Confidence estimation ──────────────────────────────────────
    confidence = estimate_confidence(features, final_score, dom_score)
    
    # ── Stage 7: Dynamic classification ─────────────────────────────────────
    classification = classify_with_dynamic_threshold(final_score, confidence, dom_score)
    
    # ── Stage 8: Highlights ──────────────────────────────────────────────────
    highlights = _build_highlights(cleaned_text, sent_list, per_sentence_features, features)
    
    # ── Stage 9: Explanation ─────────────────────────────────────────────────
    explanation = None
    if include_explanation:
        explanation = _build_explanation(
            final_score, classification, features, confidence, override_reason
        )
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    result: Dict[str, Any] = {
        "scores":           features,
        "final_score":      round(final_score, 2),
        "dominance_score":  round(dom_score_100, 2),
        "ml_score":         round(ml_score_100, 2),
        "classification":   classification,
        "confidence":       confidence,
        "highlights":       highlights,
        "metadata": {
            "char_count":         len(cleaned_text),
            "word_count":         len(cleaned_text.split()),
            "sentence_count":     len(sent_list),
            "analysis_time_ms":   round(elapsed_ms, 1),
            "features_computed":  len(features),
            "override_reason":    override_reason or "none",
            "scoring_method":     "modern_ensemble",
        },
    }
    
    if explanation:
        result["explanation"] = explanation
    
    # Feature importances from ML model
    try:
        scorer = get_modern_scorer()
        result["feature_importances"] = scorer.importances
        result["ml_train_metrics"] = scorer.metrics
    except Exception:
        pass
    
    if debug:
        result["debug"] = {
            "dom_score_raw":      round(dom_score, 4),
            "raw_score_pre_override": round(raw_score, 2),
            "override_reason":    override_reason,
            "per_feature": {
                name: {
                    "score": round(score, 4),
                    "is_ai_signal": score > 0.60,
                    "is_human_signal": score < 0.35,
                }
                for name, score in features.items()
            },
        }
    
    logger.info(
        f"Modern: score={final_score:.1f} (dom={dom_score_100:.1f}|ml={ml_score_100:.1f}), "
        f"class='{classification}', conf={confidence['level']}, {elapsed_ms:.0f}ms"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Highlight builder — contribution-based (not just threshold)
# ─────────────────────────────────────────────────────────────────────────────

def _build_highlights(
    text: str,
    sentences: List[str],
    per_sent: Dict[str, List[float]],
    global_features: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Build sentence highlights based on feature contributions.
    Each sentence gets a contribution-weighted score.
    """
    if not sentences:
        return []
    
    # Find character positions
    positions = _find_positions(text, sentences)
    highlights = []
    
    for i, (sent, (start, end)) in enumerate(zip(sentences, positions)):
        # Collect per-sentence feature scores
        sent_features = {}
        for feat_name, scores in per_sent.items():
            if i < len(scores):
                sent_features[feat_name] = scores[i]
            else:
                sent_features[feat_name] = global_features.get(feat_name, 0.5)
        
        # Dominance score for this sentence
        sent_dom = dominance_score(sent_features)
        sent_score_100 = round(sent_dom * 100, 1)
        
        # Find which features fire for this sentence
        reasons = []
        for feat_name in AI_DOMINANT_FEATURES:
            feat_score = sent_features.get(feat_name, 0.5)
            if feat_score > 0.60:
                reason = FEATURE_REASONS.get(feat_name, feat_name)
                reasons.append(reason)
        
        # Label
        if sent_dom >= 0.65:
            label = "AI"
        elif sent_dom >= 0.38:
            label = "Mixed"
        else:
            label = "Human"
        
        highlights.append({
            "text":    sent,
            "start":   start,
            "end":     end,
            "label":   label,
            "score":   sent_score_100,
            "reasons": reasons[:3],  # top 3 reasons
        })
    
    return highlights


def _find_positions(text: str, sentences: List[str]) -> List[Tuple[int, int]]:
    """Find start/end char positions of each sentence in text."""
    import re
    positions = []
    search_from = 0
    for sent in sentences:
        escaped = re.escape(sent.strip())
        match = re.search(escaped, text[search_from:])
        if match:
            s = search_from + match.start()
            e = search_from + match.end()
            positions.append((s, e))
            search_from = e
        else:
            positions.append((search_from, min(search_from + len(sent), len(text))))
            search_from = min(search_from + len(sent), len(text))
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# Explanation builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_explanation(
    score: float,
    classification: str,
    features: Dict[str, float],
    confidence: Dict[str, Any],
    override_reason: str,
) -> Dict[str, Any]:
    """Build human-readable explanation."""
    
    # Top AI signals
    top_ai = sorted(
        [(k, v) for k, v in features.items() if v > 0.58],
        key=lambda x: -x[1]
    )[:5]
    
    # Top human signals
    top_human = sorted(
        [(k, v) for k, v in features.items() if v < 0.35],
        key=lambda x: x[1]
    )[:3]
    
    # Summary
    if classification == "Likely AI":
        summary = (
            f"This text is likely AI-generated (score: {score:.0f}/100, "
            f"confidence: {confidence['level']}). "
            f"Strong AI indicators: {', '.join(FEATURE_DISPLAY_NAMES.get(k,'?').lower() for k, _ in top_ai[:3])}."
        )
    elif classification == "Human-like":
        summary = (
            f"This text appears human-written (score: {score:.0f}/100, "
            f"confidence: {confidence['level']}). "
            f"Human indicators: {', '.join(FEATURE_DISPLAY_NAMES.get(k,'?').lower() for k, _ in top_human[:3])}."
        )
    else:
        summary = (
            f"This text shows mixed signals (score: {score:.0f}/100, "
            f"confidence: {confidence['level']}). "
            "Could be lightly edited AI text or formal human writing."
        )
    
    # Key findings
    findings = []
    for feat, score_val in top_ai[:4]:
        display = FEATURE_DISPLAY_NAMES.get(feat, feat)
        reason = FEATURE_REASONS.get(feat, "")
        findings.append(f"⚠ {display} ({score_val:.0%}): {reason}")
    for feat, score_val in top_human[:2]:
        display = FEATURE_DISPLAY_NAMES.get(feat, feat)
        findings.append(f"✓ {display} ({score_val:.0%}): suggests human authorship")
    
    if override_reason and override_reason != "none":
        findings.append(f"ℹ Anti-contradiction: {override_reason}")
    
    # Feature details
    feature_details = []
    for feat, feat_score in sorted(features.items(), key=lambda x: -x[1]):
        feature_details.append({
            "feature":      feat,
            "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
            "score":        round(feat_score, 3),
            "score_pct":    round(feat_score * 100, 1),
            "flagged":      feat_score > 0.60,
            "detail":       FEATURE_REASONS.get(feat, ""),
        })
    
    return {
        "summary":         summary,
        "key_findings":    findings,
        "feature_details": feature_details,
        "confidence":      confidence["level"],
    }
