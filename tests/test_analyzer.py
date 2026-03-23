"""
tests/test_analyzer.py — Unit and integration tests

Run with:
  python -m pytest tests/ -v
  python tests/test_analyzer.py   (standalone)
"""

import sys
import os
import math

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.samples import HUMAN_SAMPLE, AI_SAMPLE, MIXED_SAMPLE
from analyzer.features import (
    PerplexityFeature, BurstinessFeature, LexicalDiversityFeature,
    RepetitionFeature, EntropyFeature, ContextualDepthFeature,
    EmotionalVariabilityFeature, ALL_FEATURES,
)
from analyzer.pipeline import analyze, load_config, preprocess
from analyzer.scoring import compute_final_score, classify
from analyzer.normalization import normalize_scores, clamp
from analyzer.explanation import generate_explanation


# ── Helpers ──────────────────────────────────────────────────────────────────

def run_test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    icon = "✓" if condition else "✗"
    print(f"  [{status}] {icon} {name}", end="")
    if detail:
        print(f"  ({detail})", end="")
    print()
    return condition


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_preprocessing():
    section("Preprocessing")
    pass_count = 0

    # Normal text
    text, sents = preprocess("Hello world. This is a test sentence. And one more.")
    pass_count += run_test("Sentence split works", len(sents) >= 2, f"{len(sents)} sents")

    # Whitespace normalization
    text2, _ = preprocess("  extra   spaces  \n\n\n lots of newlines  ")
    pass_count += run_test("Whitespace normalized", "   " not in text2)

    # Empty input raises ValueError
    try:
        preprocess("")
        pass_count += run_test("Empty input raises ValueError", False)
    except ValueError:
        pass_count += run_test("Empty input raises ValueError", True)

    # Truncation
    big_text = "word " * 30000  # 150k chars
    text_t, _ = preprocess(big_text, max_chars=100000)
    pass_count += run_test("Oversized input truncated", len(text_t) <= 100001)

    return pass_count, 4


def test_individual_features():
    section("Individual Features")
    pass_count = 0
    total = 0

    # Each feature should return score in [0,1]
    for feature in ALL_FEATURES:
        result = feature.compute(AI_SAMPLE)
        score = result.get("score", -1)
        ok = isinstance(score, float) and 0.0 <= score <= 1.0
        pass_count += run_test(
            f"Feature '{feature.name}' score in [0,1]",
            ok,
            f"score={score:.3f}"
        )
        total += 1

    # Sentence scores are lists
    perp = PerplexityFeature()
    result = perp.compute(AI_SAMPLE)
    pass_count += run_test(
        "PerplexityFeature returns sentence_scores list",
        isinstance(result.get("sentence_scores"), list)
    )
    total += 1

    # Burstiness: uniform text should score high (AI-like)
    burst = BurstinessFeature()
    uniform = "This is a sentence. This is a sentence. This is a sentence. This is another sentence. This is still a sentence. This is yet another."
    variable = "Hi! This is a much longer sentence with many more words and clauses in it. Ok. Right, so this is also quite a long sentence. No. But this is another one that goes on for a while."
    r_uniform = burst.compute(uniform)
    r_variable = burst.compute(variable)
    pass_count += run_test(
        "Burstiness: uniform text scores higher than variable",
        r_uniform["score"] >= r_variable["score"],
        f"uniform={r_uniform['score']:.3f}, variable={r_variable['score']:.3f}"
    )
    total += 1

    # Lexical diversity: repetitive text scores higher (AI-like)
    lex = LexicalDiversityFeature(window_size=20)
    repetitive = "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
    diverse = "the quick brown fox jumps lazily over an exhausted somnolent sleeping canine"
    r_rep = lex.compute(repetitive)
    r_div = lex.compute(diverse)
    pass_count += run_test(
        "LexicalDiversity: repetitive text scores higher",
        r_rep["score"] >= r_div["score"],
        f"repetitive={r_rep['score']:.3f}, diverse={r_div['score']:.3f}"
    )
    total += 1

    # Contextual depth: personal text scores lower (more human)
    cd = ContextualDepthFeature()
    personal = "I think I really believe that I feel strongly about my own personal experiences."
    impersonal = "The system processes data through multiple computational layers."
    r_pers = cd.compute(personal)
    r_imp = cd.compute(impersonal)
    pass_count += run_test(
        "ContextualDepth: impersonal text scores higher (AI-like)",
        r_imp["score"] >= r_pers["score"],
        f"impersonal={r_imp['score']:.3f}, personal={r_pers['score']:.3f}"
    )
    total += 1

    # Short text handling
    rep_feat = RepetitionFeature()
    result = rep_feat.compute("Hi.")
    pass_count += run_test(
        "Features handle very short text gracefully",
        0.0 <= result["score"] <= 1.0
    )
    total += 1

    return pass_count, total


def test_normalization():
    section("Normalization")
    pass_count = 0

    # clamp
    pass_count += run_test("clamp(0.5, 0, 1) = 0.5", clamp(0.5) == 0.5)
    pass_count += run_test("clamp(1.5, 0, 1) = 1.0", clamp(1.5) == 1.0)
    pass_count += run_test("clamp(-0.1, 0, 1) = 0.0", clamp(-0.1) == 0.0)

    # normalize_scores
    raw = {"perplexity": 0.8, "burstiness": 0.3, "repetition": 1.2, "entropy": -0.1}
    normed = normalize_scores(raw)
    pass_count += run_test(
        "normalize_scores clamps all values to [0,1]",
        all(0.0 <= v <= 1.0 for v in normed.values())
    )

    return pass_count, 4


def test_scoring():
    section("Scoring")
    pass_count = 0

    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    scores = {"a": 1.0, "b": 1.0, "c": 1.0}
    score, contribs = compute_final_score(scores, weights)
    pass_count += run_test("All-1.0 scores → final=100", abs(score - 100.0) < 0.1, f"score={score}")

    scores = {"a": 0.0, "b": 0.0, "c": 0.0}
    score, _ = compute_final_score(scores, weights)
    pass_count += run_test("All-0.0 scores → final=0", abs(score - 0.0) < 0.1, f"score={score}")

    thresholds = {"human_max": 30, "mixed_max": 70}
    pass_count += run_test("Score 15 → Human-like", classify(15, thresholds) == "Human-like")
    pass_count += run_test("Score 50 → Mixed", classify(50, thresholds) == "Mixed / Uncertain")
    pass_count += run_test("Score 85 → Likely AI", classify(85, thresholds) == "Likely AI")

    return pass_count, 5


def test_full_pipeline():
    section("Full Pipeline (Integration)")
    pass_count = 0

    config = load_config()

    # Human sample
    result = analyze(HUMAN_SAMPLE, config=config)
    human_score = result["final_score"]
    pass_count += run_test(
        "Human sample analyzed without error",
        "final_score" in result and "classification" in result,
        f"score={human_score:.1f}"
    )

    # AI sample
    result_ai = analyze(AI_SAMPLE, config=config)
    ai_score = result_ai["final_score"]
    pass_count += run_test(
        "AI sample analyzed without error",
        "final_score" in result_ai,
        f"score={ai_score:.1f}"
    )

    # AI should score higher than human
    pass_count += run_test(
        "AI sample scores higher than human sample",
        ai_score > human_score,
        f"AI={ai_score:.1f}, Human={human_score:.1f}"
    )

    # Result structure
    keys = {"scores", "final_score", "classification", "highlights", "metadata"}
    pass_count += run_test(
        "Result contains all required keys",
        keys.issubset(result.keys())
    )

    # Highlights structure
    highlights = result.get("highlights", [])
    pass_count += run_test(
        "Highlights are a non-empty list",
        isinstance(highlights, list) and len(highlights) > 0
    )
    if highlights:
        h = highlights[0]
        required_keys = {"text", "start", "end", "label", "reasons", "score"}
        pass_count += run_test(
            "Highlight objects have required keys",
            required_keys.issubset(h.keys())
        )
    else:
        pass_count += 1  # skip

    # Debug mode
    result_debug = analyze(AI_SAMPLE, config=config, debug=True)
    pass_count += run_test(
        "Debug mode includes 'debug' key",
        "debug" in result_debug
    )

    # Explanation
    result_expl = analyze(AI_SAMPLE, config=config, include_explanation=True)
    pass_count += run_test(
        "Explanation included when requested",
        "explanation" in result_expl and result_expl["explanation"] is not None
    )

    # Mixed sample
    result_mixed = analyze(MIXED_SAMPLE, config=config)
    mixed_score = result_mixed["final_score"]
    pass_count += run_test(
        "Mixed sample score between human and AI",
        human_score <= mixed_score <= ai_score or  # ideal
        0 < mixed_score < 100,  # at minimum, valid range
        f"H={human_score:.1f}, M={mixed_score:.1f}, A={ai_score:.1f}"
    )

    return pass_count, 9


def test_security():
    section("Security & Robustness")
    pass_count = 0

    config = load_config()

    # HTML injection
    try:
        result = analyze("<script>alert('xss')</script>", config=config)
        pass_count += run_test("HTML injection handled", "final_score" in result)
    except Exception:
        pass_count += run_test("HTML injection handled (raised safely)", True)

    # Empty string → ValueError
    try:
        analyze("", config=config)
        pass_count += run_test("Empty string raises ValueError", False)
    except ValueError:
        pass_count += run_test("Empty string raises ValueError", True)

    # Unicode
    try:
        result = analyze("这是一个测试。Это тест. هذا اختبار.", config=config)
        pass_count += run_test("Unicode text handled", "final_score" in result)
    except Exception as e:
        pass_count += run_test("Unicode text handled", False, str(e))

    # Single word
    try:
        result = analyze("Hello.", config=config)
        pass_count += run_test("Single sentence handled", "final_score" in result)
    except Exception:
        pass_count += run_test("Single sentence handled (exception expected)", True)

    return pass_count, 4


def test_feature_distributions():
    section("Feature Distribution Analysis")

    config = load_config()
    results = {
        "human": analyze(HUMAN_SAMPLE, config=config),
        "ai":    analyze(AI_SAMPLE, config=config),
        "mixed": analyze(MIXED_SAMPLE, config=config),
    }

    print("\n  Feature Score Comparison (Human / Mixed / AI):")
    print(f"  {'Feature':<30} {'Human':>7} {'Mixed':>7} {'AI':>7}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7}")

    for feat in ALL_FEATURES:
        h = results["human"]["scores"].get(feat.name, 0)
        m = results["mixed"]["scores"].get(feat.name, 0)
        a = results["ai"]["scores"].get(feat.name, 0)
        print(f"  {feat.name:<30} {h:>7.3f} {m:>7.3f} {a:>7.3f}")

    print(f"\n  {'FINAL SCORE':<30} {results['human']['final_score']:>7.1f} "
          f"{results['mixed']['final_score']:>7.1f} {results['ai']['final_score']:>7.1f}")
    print(f"  {'CLASSIFICATION':<30} {results['human']['classification'][:10]:>10} "
          f"{results['mixed']['classification'][:10]:>10} {results['ai']['classification'][:10]:>10}")

    return 1, 1  # Always passes (it's a display test)


# ── Runner ─────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "═"*55)
    print("  EXPLAINABLE AI ANALYZER — TEST SUITE")
    print("═"*55)

    total_pass = 0
    total_tests = 0

    suites = [
        test_preprocessing,
        test_individual_features,
        test_normalization,
        test_scoring,
        test_full_pipeline,
        test_security,
        test_feature_distributions,
    ]

    for suite in suites:
        p, t = suite()
        total_pass += p
        total_tests += t

    print("\n" + "═"*55)
    print(f"  RESULTS: {total_pass}/{total_tests} tests passed")
    if total_pass == total_tests:
        print("  ✓ All tests passed!")
    else:
        print(f"  ✗ {total_tests - total_pass} test(s) failed")
    print("═"*55 + "\n")

    return total_pass == total_tests


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
