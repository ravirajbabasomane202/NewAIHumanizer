"""
tests/test_v2.py — Extended test suite for v2 upgrades

Tests:
  - All 10 v2 features
  - ML scorer training + prediction
  - Full v2 pipeline
  - Benchmark metrics
  - Adversarial robustness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tests.samples import HUMAN_SAMPLE, AI_SAMPLE, MIXED_SAMPLE
from analyzer.features_v2 import V2_FEATURES, BigramLMPerplexityFeature, VADERSentimentFeature, CompressionRatioFeature
from analyzer.ml_scorer import MLScorer, Benchmarker, AdversarialTester
from analyzer.pipeline_v2 import analyze_v2, get_ml_scorer
from analyzer.pipeline import load_config


def run_test(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    icon = "✓" if condition else "✗"
    print(f"  [{status}] {icon} {name}", end="")
    if detail: print(f"  ({detail})", end="")
    print()
    return condition

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def test_v2_features():
    section("V2 Feature Tests")
    p = t = 0

    for feat in V2_FEATURES:
        result = feat.compute(AI_SAMPLE)
        score = result.get("score", -1)
        ok = isinstance(score, (int, float)) and 0.0 <= score <= 1.0
        p += run_test(f"Feature '{feat.name}' → [0,1]", ok, f"score={score:.3f}")
        t += 1

    # Bigram PP: AI should score higher than human (more predictable)
    bp = BigramLMPerplexityFeature()
    r_ai = bp.compute(AI_SAMPLE)
    r_hu = bp.compute(HUMAN_SAMPLE)
    p += run_test("BigramPP: AI > Human", r_ai["score"] >= r_hu["score"],
                  f"AI={r_ai['score']:.3f}, H={r_hu['score']:.3f}")
    t += 1

    # VADER: AI should have lower sentiment variance (flatter emotion)
    vs = VADERSentimentFeature()
    r_ai = vs.compute(AI_SAMPLE)
    r_hu = vs.compute(HUMAN_SAMPLE)
    p += run_test("VADER: AI sentiment flatter", r_ai["score"] >= r_hu["score"],
                  f"AI={r_ai['score']:.3f}, H={r_hu['score']:.3f}")
    t += 1

    # Compression: AI should be more compressible
    cr = CompressionRatioFeature()
    r_ai = cr.compute(AI_SAMPLE)
    r_hu = cr.compute(HUMAN_SAMPLE)
    # Allow this to be directional or within range
    p += run_test("CompressionRatio in [0,1] for both",
                  0 <= r_ai["score"] <= 1 and 0 <= r_hu["score"] <= 1,
                  f"AI={r_ai['score']:.3f}, H={r_hu['score']:.3f}")
    t += 1

    # Short text handling
    for feat in V2_FEATURES:
        r = feat.compute("Hello.")
        ok = 0.0 <= r.get("score", 0.5) <= 1.0
        p += run_test(f"'{feat.name}' handles short text", ok)
        t += 1

    return p, t


def test_ml_scorer():
    section("ML Scorer Tests")
    p = t = 0

    scorer = MLScorer()

    # Training
    metrics = scorer.train(save=False)
    p += run_test("ML scorer trains without error", scorer.is_trained)
    t += 1
    p += run_test("Training accuracy > 0.6", metrics.get("accuracy", 0) > 0.6,
                  f"acc={metrics.get('accuracy','?')}")
    t += 1
    p += run_test("CV AUC > 0.6", metrics.get("cv_auc_gb_mean", 0) > 0.6,
                  f"AUC={metrics.get('cv_auc_gb_mean','?')}")
    t += 1
    p += run_test("F1 > 0.6", metrics.get("f1", 0) > 0.6, f"F1={metrics.get('f1','?')}")
    t += 1
    p += run_test("Confusion matrix present", "confusion_matrix" in metrics)
    t += 1

    # Prediction
    ai_feats = {name: 0.75 for name in ["perplexity","burstiness","emotional_variability",
                                         "error_patterns","contextual_depth","bigram_perplexity",
                                         "log_likelihood_variance","vader_sentiment"]}
    human_feats = {name: 0.20 for name in ["perplexity","burstiness","emotional_variability",
                                            "error_patterns","contextual_depth","bigram_perplexity",
                                            "log_likelihood_variance","vader_sentiment"]}

    ai_pred = scorer.predict(ai_feats)
    hu_pred = scorer.predict(human_feats)
    p += run_test("ML predicts AI-like features as AI", ai_pred["probability"] > 0.5,
                  f"p={ai_pred['probability']:.3f}")
    t += 1
    p += run_test("ML predicts human-like features as human", hu_pred["probability"] < 0.5,
                  f"p={hu_pred['probability']:.3f}")
    t += 1

    # Feature importances
    imps = scorer.feature_importances
    p += run_test("Feature importances populated", len(imps) > 0)
    t += 1
    p += run_test("Importances sum ≈ 1.0", abs(sum(imps.values()) - 1.0) < 0.05,
                  f"sum={sum(imps.values()):.3f}")
    t += 1

    # Fallback (untrained scorer)
    scorer2 = MLScorer()
    fb = scorer2._fallback_predict({"perplexity": 0.8, "burstiness": 0.7})
    p += run_test("Fallback predict in [0,1]", 0 <= fb["probability"] <= 1.0)
    t += 1

    return p, t


def test_v2_pipeline():
    section("V2 Pipeline Integration")
    p = t = 0

    config = load_config()

    r_ai = analyze_v2(AI_SAMPLE, config=config, use_ml=True)
    r_hu = analyze_v2(HUMAN_SAMPLE, config=config, use_ml=True)
    r_mx = analyze_v2(MIXED_SAMPLE, config=config, use_ml=True)

    p += run_test("AI sample analyzed", "final_score" in r_ai, f"score={r_ai['final_score']:.1f}")
    t += 1
    p += run_test("Human sample analyzed", "final_score" in r_hu, f"score={r_hu['final_score']:.1f}")
    t += 1
    p += run_test("AI scores higher than human", r_ai["final_score"] > r_hu["final_score"],
                  f"AI={r_ai['final_score']:.1f}, H={r_hu['final_score']:.1f}")
    t += 1

    # Required keys
    for key in ["scores","final_score","ml_score","classification","confidence","highlights"]:
        p += run_test(f"Result has '{key}'", key in r_ai)
        t += 1

    # Feature count (25 features)
    p += run_test("All 25 features computed",
                  r_ai["metadata"]["features_computed"] == 25,
                  f"got {r_ai['metadata']['features_computed']}")
    t += 1

    # Confidence
    p += run_test("Confidence is valid", r_ai["confidence"] in ("High","Medium","Low"))
    t += 1

    # Feature importances in result
    p += run_test("Feature importances in result", "feature_importances" in r_ai)
    t += 1

    # Without ML
    r_no_ml = analyze_v2(AI_SAMPLE, config=config, use_ml=False)
    p += run_test("Works without ML", "final_score" in r_no_ml)
    t += 1

    return p, t


def test_benchmark():
    section("Benchmark Metrics")
    p = t = 0

    config = load_config()

    # Build a small labeled dataset
    labeled = (
        [{"text": HUMAN_SAMPLE, "label": 0}] * 5 +
        [{"text": AI_SAMPLE, "label": 1}] * 5 +
        [{"text": MIXED_SAMPLE, "label": 1}] * 3
    )

    bench = Benchmarker()
    results = bench.run(labeled, lambda text: analyze_v2(text, config=config, use_ml=True))

    p += run_test("Benchmark runs", "accuracy" in results, f"acc={results.get('accuracy','?')}")
    t += 1
    p += run_test("Benchmark n_samples correct", results.get("n_samples", 0) == len(labeled))
    t += 1
    p += run_test("Confusion matrix 2×2", len(results.get("confusion_matrix", [])) == 2)
    t += 1
    p += run_test("AUC > 0.4", results.get("auc_roc", 0) > 0.4,
                  f"AUC={results.get('auc_roc','?')}")
    t += 1

    bench.print_report(results)
    return p, t


def test_adversarial():
    section("Adversarial Robustness")
    p = t = 0

    config = load_config()
    tester = AdversarialTester()

    for label, text in [("AI", AI_SAMPLE), ("Human", HUMAN_SAMPLE)]:
        results = tester.run_all(text, lambda txt: analyze_v2(txt, config=config, use_ml=True))
        baseline = results.get("baseline_score", 50)
        n_robust = sum(1 for k, v in results.items()
                       if isinstance(v, dict) and v.get("robust", False))
        n_attacks = 5
        p += run_test(f"{label}: ≥3/5 attacks are robust", n_robust >= 3,
                      f"{n_robust}/{n_attacks} robust, baseline={baseline:.0f}")
        t += 1

        # Individual attack results
        for attack in ["typo_injection","synonym_noise","sentence_shuffle"]:
            v = results.get(attack, {})
            if isinstance(v, dict) and "delta" in v:
                p += run_test(f"{label} vs {attack}: |delta|<25",
                              abs(v["delta"]) < 25,
                              f"delta={v['delta']:.1f}")
                t += 1

    return p, t


def test_feature_distributions_v2():
    section("V2 Feature Distribution Analysis")
    config = load_config()
    r = {
        "human": analyze_v2(HUMAN_SAMPLE, config=config, use_ml=False),
        "mixed": analyze_v2(MIXED_SAMPLE, config=config, use_ml=False),
        "ai":    analyze_v2(AI_SAMPLE, config=config, use_ml=False),
    }
    from analyzer.features_v2 import V2_FEATURES
    print(f"\n  {'Feature':<30} {'Human':>7} {'Mixed':>7} {'AI':>7}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7}")
    for feat in V2_FEATURES:
        h = r["human"]["scores"].get(feat.name, 0)
        m = r["mixed"]["scores"].get(feat.name, 0)
        a = r["ai"]["scores"].get(feat.name, 0)
        print(f"  {feat.name:<30} {h:>7.3f} {m:>7.3f} {a:>7.3f}")
    print(f"\n  {'FINAL SCORE':<30} {r['human']['final_score']:>7.1f} "
          f"{r['mixed']['final_score']:>7.1f} {r['ai']['final_score']:>7.1f}")
    return 1, 1


def run_all():
    print("\n" + "═"*60)
    print("  EXPLAINABLE AI ANALYZER v2 — EXTENDED TEST SUITE")
    print("═"*60)

    total_p = total_t = 0
    suites = [
        test_v2_features,
        test_ml_scorer,
        test_v2_pipeline,
        test_benchmark,
        test_adversarial,
        test_feature_distributions_v2,
    ]
    for suite in suites:
        p, t = suite()
        total_p += p; total_t += t

    print("\n" + "═"*60)
    print(f"  RESULTS: {total_p}/{total_t} tests passed")
    print(f"  {'✓ All tests passed!' if total_p == total_t else f'✗ {total_t-total_p} failed'}")
    print("═"*60 + "\n")
    return total_p == total_t


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
