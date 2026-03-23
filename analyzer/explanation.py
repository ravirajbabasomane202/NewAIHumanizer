"""
analyzer/explanation.py
-----------------------
Generates human-readable explanations for the AI-likelihood analysis.

Takes structured results and produces:
  - A top-level summary paragraph
  - Per-feature bullet explanations
  - Key evidence for the classification
"""

from typing import Dict, Any, List


# Threshold for a feature to be considered "flagged"
FLAG_THRESHOLD = 0.60

# Feature descriptions for explanation text
FEATURE_EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "perplexity": {
        "high": "The text uses predictable, expected word sequences typical of AI generation.",
        "low":  "Word choices are varied and sometimes unexpected, suggesting human authorship.",
    },
    "burstiness": {
        "high": "Sentence lengths are unusually uniform — AI text often lacks natural rhythm variation.",
        "low":  "Sentence lengths vary naturally, consistent with human writing patterns.",
    },
    "sentence_diversity": {
        "high": "Many sentences share the same grammatical structure, a hallmark of AI generation.",
        "low":  "Sentence structures are varied, indicating natural writing.",
    },
    "lexical_diversity": {
        "high": "The vocabulary is limited and repetitive, common in AI-generated text.",
        "low":  "Rich and varied vocabulary suggests human authorship.",
    },
    "repetition": {
        "high": "Repeated phrases and n-grams are detected at above-normal density.",
        "low":  "Low phrase repetition — no unusual repetition patterns found.",
    },
    "semantic_predictability": {
        "high": "Each sentence closely mirrors the previous one semantically — over-smooth transitions.",
        "low":  "Sentence meanings shift naturally, as expected in human writing.",
    },
    "syntactic_complexity": {
        "high": "Sentences tend to be simple and clause-light — AI often avoids complex structures.",
        "low":  "Sentences contain nested clauses and varied structure.",
    },
    "function_word_dist": {
        "high": "The ratio of function words (stopwords) deviates from typical English patterns.",
        "low":  "Function word distribution matches expected English baselines.",
    },
    "entropy": {
        "high": "Word frequency distribution is concentrated — limited token diversity.",
        "low":  "Word frequency distribution is broad, indicating diverse language use.",
    },
    "ngram_frequency_bias": {
        "high": "The text over-relies on common English words, avoiding rare or domain-specific terms.",
        "low":  "The text uses a mix of common and less common vocabulary.",
    },
    "over_coherence": {
        "high": "Inter-sentence similarity is unnaturally uniform — paragraphs feel 'too smooth'.",
        "low":  "Coherence varies naturally across the text.",
    },
    "emotional_variability": {
        "high": "Emotional tone is flat and consistent throughout — AI rarely modulates sentiment.",
        "low":  "Emotional tone shifts naturally, consistent with human expression.",
    },
    "error_patterns": {
        "high": "No grammatical errors or typos were detected — AI text is typically error-free.",
        "low":  "Minor errors or informal patterns were found, suggesting human authorship.",
    },
    "contextual_depth": {
        "high": "The text lacks personal pronouns and subjective language — feels impersonal.",
        "low":  "Personal voice and subjective markers are present.",
    },
    "stopword_patterning": {
        "high": "Stopwords are distributed unusually uniformly across text sections.",
        "low":  "Stopword distribution varies naturally across the text.",
    },
}

# Feature display names
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "perplexity":             "Word Predictability",
    "burstiness":             "Sentence Length Variation",
    "sentence_diversity":     "Sentence Structure Diversity",
    "lexical_diversity":      "Vocabulary Richness",
    "repetition":             "Phrase Repetition",
    "semantic_predictability":"Semantic Flow Smoothness",
    "syntactic_complexity":   "Syntactic Complexity",
    "function_word_dist":     "Function Word Distribution",
    "entropy":                "Word Entropy",
    "ngram_frequency_bias":   "Common Vocabulary Bias",
    "over_coherence":         "Coherence Uniformity",
    "emotional_variability":  "Emotional Variation",
    "error_patterns":         "Error / Typo Density",
    "contextual_depth":       "Personal Voice Depth",
    "stopword_patterning":    "Stopword Positioning",
}


def generate_explanation(
    final_score: float,
    classification: str,
    feature_scores: Dict[str, float],
    contributions: Dict[str, float],
) -> Dict[str, Any]:
    """
    Generate a structured human-readable explanation.

    Parameters
    ----------
    final_score     : float, 0–100 AI-likelihood score
    classification  : str label ("Human-like", "Mixed / Uncertain", "Likely AI")
    feature_scores  : dict  feature_name → normalized score [0,1]
    contributions   : dict  feature_name → weighted contribution (0–100 scale)

    Returns
    -------
    {
        "summary": str,
        "key_findings": List[str],
        "feature_details": List[dict],
        "confidence": str,
    }
    """
    # Identify top AI-flagged features (score >= threshold)
    flagged = [
        (name, score)
        for name, score in feature_scores.items()
        if score >= FLAG_THRESHOLD
    ]
    flagged_sorted = sorted(flagged, key=lambda x: x[1], reverse=True)

    # Identify human-like features (low scores)
    human_features = [
        (name, score)
        for name, score in feature_scores.items()
        if score < 0.35
    ]
    human_sorted = sorted(human_features, key=lambda x: x[1])

    # Build summary
    summary = _build_summary(final_score, classification, flagged_sorted, human_sorted)

    # Key findings
    key_findings = _build_key_findings(flagged_sorted, human_sorted)

    # Per-feature details
    feature_details = _build_feature_details(feature_scores, contributions)

    # Confidence assessment
    confidence = _assess_confidence(final_score, len(flagged), len(human_features))

    return {
        "summary": summary,
        "key_findings": key_findings,
        "feature_details": feature_details,
        "confidence": confidence,
    }


def _build_summary(
    final_score: float,
    classification: str,
    flagged: List[tuple],
    human: List[tuple],
) -> str:
    """Build the main summary paragraph."""

    if classification == "Likely AI":
        intro = (
            f"This text is likely AI-generated (score: {final_score:.0f}/100). "
            "Multiple linguistic markers consistent with AI authorship were detected."
        )
        if flagged:
            top_reasons = ", ".join(
                FEATURE_DISPLAY_NAMES.get(n, n).lower()
                for n, _ in flagged[:3]
            )
            intro += f" Key indicators include: {top_reasons}."

    elif classification == "Human-like":
        intro = (
            f"This text appears to be human-written (score: {final_score:.0f}/100). "
            "Most linguistic markers align with natural human writing patterns."
        )
        if human:
            top_human = ", ".join(
                FEATURE_DISPLAY_NAMES.get(n, n).lower()
                for n, _ in human[:3]
            )
            intro += f" Strong human indicators: {top_human}."

    else:  # Mixed
        intro = (
            f"This text shows mixed signals (score: {final_score:.0f}/100). "
            "Some features suggest AI generation while others indicate human authorship. "
            "This could be lightly edited AI text, or AI-assisted human writing."
        )

    return intro


def _build_key_findings(
    flagged: List[tuple],
    human: List[tuple],
) -> List[str]:
    """Build a list of key finding strings."""
    findings = []

    for name, score in flagged[:5]:
        explanation = FEATURE_EXPLANATIONS.get(name, {}).get("high", "")
        display = FEATURE_DISPLAY_NAMES.get(name, name)
        if explanation:
            findings.append(f"⚠ {display} ({score:.0%}): {explanation}")

    for name, score in human[:2]:
        explanation = FEATURE_EXPLANATIONS.get(name, {}).get("low", "")
        display = FEATURE_DISPLAY_NAMES.get(name, name)
        if explanation:
            findings.append(f"✓ {display} ({score:.0%}): {explanation}")

    return findings


def _build_feature_details(
    feature_scores: Dict[str, float],
    contributions: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build per-feature detail objects for UI consumption."""
    details = []
    for name, score in sorted(feature_scores.items(), key=lambda x: -x[1]):
        high_text = FEATURE_EXPLANATIONS.get(name, {}).get("high", "")
        low_text = FEATURE_EXPLANATIONS.get(name, {}).get("low", "")
        detail_text = high_text if score >= 0.5 else low_text
        details.append({
            "feature": name,
            "display_name": FEATURE_DISPLAY_NAMES.get(name, name),
            "score": round(score, 3),
            "score_pct": round(score * 100, 1),
            "contribution": contributions.get(name, 0.0),
            "flagged": score >= FLAG_THRESHOLD,
            "detail": detail_text,
        })
    return details


def _assess_confidence(
    final_score: float,
    flagged_count: int,
    human_count: int,
) -> str:
    """Assess confidence level of the classification."""
    # Strong signal: score far from 50 and consistent features
    distance_from_center = abs(final_score - 50)

    if distance_from_center >= 30 and (flagged_count >= 8 or human_count >= 8):
        return "High"
    elif distance_from_center >= 15:
        return "Medium"
    else:
        return "Low"


def format_explanation_text(explanation: Dict[str, Any]) -> str:
    """Format explanation as plain text for CLI output."""
    lines = ["=" * 60, "ANALYSIS EXPLANATION", "=" * 60, ""]
    lines.append(explanation["summary"])
    lines.append("")

    if explanation["key_findings"]:
        lines.append("KEY FINDINGS:")
        for finding in explanation["key_findings"]:
            lines.append(f"  {finding}")
        lines.append("")

    lines.append(f"Confidence: {explanation['confidence']}")
    lines.append("")
    lines.append("FEATURE BREAKDOWN:")
    for detail in explanation["feature_details"]:
        bar = "█" * int(detail["score_pct"] / 10) + "░" * (10 - int(detail["score_pct"] / 10))
        flag = " ⚠" if detail["flagged"] else "  "
        lines.append(
            f"  {flag} {detail['display_name']:<35} [{bar}] {detail['score_pct']:5.1f}%"
        )

    return "\n".join(lines)
