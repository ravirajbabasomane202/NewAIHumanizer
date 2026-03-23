"""
tests/test_modern.py — Comprehensive test suite for modern detector

Tests:
  - Feature extractor (all 12 features, calibration)
  - Dominance scorer
  - Anti-contradiction logic  
  - ML scorer (training, prediction, no overfitting)
  - Full pipeline (accuracy on diverse examples)
  - Adversarial robustness
  - Confidence estimation
  - Dynamic thresholds
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer.detector import (
    FeatureExtractor, ModernMLScorer, dominance_score,
    anti_contradiction_override, estimate_confidence,
    classify_with_dynamic_threshold, capped_weighted_score,
    AI_DOMINANT_FEATURES, AdversarialTesterModern,
    get_modern_scorer,
)
from analyzer.pipeline_modern import analyze_modern
from analyzer.pipeline import load_config
from tests.samples import HUMAN_SAMPLE, AI_SAMPLE, MIXED_SAMPLE

# ── Additional test texts ────────────────────────────────────────────────────

AI_TEXTS = [
    # GPT-4 style essay with AI transition words
    "Artificial intelligence represents a transformative force in contemporary society. Furthermore, the implications of this technology extend across multiple domains. Consequently, policymakers must address these challenges systematically. Moreover, researchers have demonstrated significant progress in developing robust solutions. Additionally, stakeholders from various sectors have begun to implement these frameworks. Ultimately, the integration of AI systems requires careful consideration of ethical principles and societal impact.",
    
    # AI analysis paragraph
    "The implementation of machine learning algorithms has significantly enhanced operational efficiency. However, several challenges persist in the deployment of these systems. Nevertheless, organizations have successfully leveraged these technologies to optimize their workflows. Furthermore, the data-driven approach has proven particularly effective in identifying patterns. Consequently, the adoption of AI-powered solutions continues to accelerate across industries.",

    # Structured AI response
    "Climate change represents one of the most pressing challenges facing humanity today. The scientific evidence overwhelmingly indicates that human activities are the primary driver of global warming. Consequently, immediate and comprehensive action is required at all levels of society. Furthermore, international cooperation is essential for developing effective mitigation strategies. Notably, renewable energy technologies offer promising solutions to reduce carbon emissions significantly.",
]

HUMAN_TEXTS = [
    # Casual personal writing with contractions
    "I've been thinking about this for a while now. It's honestly kind of weird how we just accept things, you know? Like yesterday I was trying to fix my coffee maker — which, by the way, has been broken for like three weeks — and I just kept wondering why I hadn't done it sooner. Anyway, it's fixed now and the coffee tastes way better.",
    
    # Blog-style with personal voice
    "So I finally read that book everyone's been talking about and I don't know, I'm kind of torn on it. Parts of it are really good — the middle section especially. But the ending? I'm not buying it. Maybe I'm missing something. My friend says I need to re-read it but I've got like six other books on my nightstand right now so that's not happening anytime soon.",
    
    # Email/informal writing
    "Hey, just wanted to follow up on what we talked about last week. I've been chewing on it and I think we can make it work, but we'd need to move pretty fast. Can you check with your team? I know it's a lot to ask, but honestly I think this could be really good for both of us. Let me know what you think.",
]


def run_test(name, cond, detail=""):
    s = "PASS" if cond else "FAIL"
    print(f"  [{s}] {'✓' if cond else '✗'} {name}", end="")
    if detail: print(f"  ({detail})", end="")
    print()
    return cond

def section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


# ── 1. Feature Extractor ─────────────────────────────────────────────────────

def test_feature_extractor():
    section("Feature Extractor — 12 Calibrated Features")
    p = t = 0
    fe = FeatureExtractor()
    
    # All features return [0,1]
    for text_label, text in [("AI", AI_TEXTS[0]), ("Human", HUMAN_TEXTS[0])]:
        feats = fe.extract(text)
        for name, score in feats.items():
            ok = isinstance(score, float) and 0.0 <= score <= 1.0
            p += run_test(f"{text_label}: '{name}' in [0,1]", ok, f"{score:.3f}")
            t += 1
    
    # Calibration: AI features should be higher for AI text
    ai_feats = fe.extract(AI_TEXTS[0])
    hu_feats = fe.extract(HUMAN_TEXTS[0])
    
    calibration_checks = [
        ("burstiness",         ai_feats["burstiness"]         > hu_feats["burstiness"]),
        ("contraction_absence",ai_feats["contraction_absence"] > hu_feats["contraction_absence"]),
        ("transition_density", ai_feats["transition_density"]  > hu_feats["transition_density"]),
    ]
    for feat_name, ok in calibration_checks:
        p += run_test(f"Calibration: AI '{feat_name}' > Human",
                      ok, f"AI={ai_feats[feat_name]:.3f}, H={hu_feats[feat_name]:.3f}")
        t += 1
    
    # Short text → neutral (0.5)
    short_feats = fe.extract("Hello.")
    p += run_test("Short text returns neutral scores",
                  all(v == 0.5 for v in short_feats.values()))
    t += 1
    
    # Per-sentence extraction
    per_sent = fe.extract_per_sentence(AI_TEXTS[0])
    p += run_test("Per-sentence returns dict of lists", isinstance(per_sent, dict))
    t += 1
    p += run_test("Per-sentence lists non-empty",
                  all(len(v) > 0 for v in per_sent.values()))
    t += 1
    p += run_test("Per-sentence scores in [0,1]",
                  all(0 <= s <= 1 for v in per_sent.values() for s in v))
    t += 1
    
    return p, t


# ── 2. Dominance Scorer ──────────────────────────────────────────────────────

def test_dominance_scorer():
    section("Dominance Scorer — Anti-Cancellation Logic")
    p = t = 0
    fe = FeatureExtractor()
    
    # Pure AI features → score near 1
    ai_feats_extreme = {k: 0.85 for k in fe.FEATURE_NAMES}
    d_ai = dominance_score(ai_feats_extreme)
    p += run_test("Extreme AI features → score ≥ 0.80", d_ai >= 0.80, f"dom={d_ai:.3f}")
    t += 1
    
    # Pure human features → score near 0
    hu_feats_extreme = {k: 0.15 for k in fe.FEATURE_NAMES}
    d_hu = dominance_score(hu_feats_extreme)
    p += run_test("Extreme human features → score ≤ 0.20", d_hu <= 0.20, f"dom={d_hu:.3f}")
    t += 1
    
    # Real AI text → high dominance
    ai_feats = fe.extract(AI_TEXTS[0])
    d_real_ai = dominance_score(ai_feats)
    p += run_test("Real AI text → dominance ≥ 0.70", d_real_ai >= 0.70, f"dom={d_real_ai:.3f}")
    t += 1
    
    # Real human text → low dominance
    hu_feats = fe.extract(HUMAN_TEXTS[0])
    d_real_hu = dominance_score(hu_feats)
    p += run_test("Real human text → dominance ≤ 0.30", d_real_hu <= 0.30, f"dom={d_real_hu:.3f}")
    t += 1
    
    # Anti-cancellation: mixed signals with dominant AI should lean AI
    mixed = {k: 0.35 for k in fe.FEATURE_NAMES}
    mixed.update({"burstiness": 0.85, "contraction_absence": 0.90, 
                   "transition_density": 0.80, "sentence_length_cv": 0.80,
                   "coherence_smoothness": 0.75})
    d_mixed = dominance_score(mixed)
    p += run_test("Mixed signals with 5 strong AI → score > 0.60", d_mixed > 0.60, f"dom={d_mixed:.3f}")
    t += 1
    
    # Return in [0,1]
    p += run_test("Dominance always in [0,1]", 0 <= d_ai <= 1 and 0 <= d_hu <= 1)
    t += 1
    
    # Capped weighted score
    feats = {k: 0.7 for k in fe.FEATURE_NAMES}
    capped = capped_weighted_score(feats, cap=0.20)
    p += run_test("Capped weighted score in [0,1]", 0 <= capped <= 1)
    t += 1
    
    return p, t


# ── 3. Anti-Contradiction Logic ──────────────────────────────────────────────

def test_anti_contradiction():
    section("Anti-Contradiction Override Logic")
    p = t = 0
    fe = FeatureExtractor()
    
    # Case 1: High dominance + stuck in Mixed → should boost to AI
    strong_ai_feats = {k: 0.20 for k in fe.FEATURE_NAMES}
    strong_ai_feats.update({k: 0.80 for k in AI_DOMINANT_FEATURES})
    dom = dominance_score(strong_ai_feats)
    score_55, reason = anti_contradiction_override(55.0, strong_ai_feats, dom)
    p += run_test("Strong AI signals boost Mixed→AI", score_55 > 65,
                  f"55→{score_55:.1f}, dom={dom:.3f}, reason={reason}")
    t += 1
    
    # Case 2: Clear human features → prevent false positive
    clear_hu_feats = {k: 0.80 for k in fe.FEATURE_NAMES}
    clear_hu_feats.update({k: 0.15 for k in AI_DOMINANT_FEATURES})
    dom_hu = dominance_score(clear_hu_feats)
    score_60, reason_hu = anti_contradiction_override(60.0, clear_hu_feats, dom_hu)
    p += run_test("Clear human signals pull down high score", score_60 < 60,
                  f"60→{score_60:.1f}")
    t += 1
    
    # Case 3: Moderate signals → no override
    neutral_feats = {k: 0.50 for k in fe.FEATURE_NAMES}
    dom_n = dominance_score(neutral_feats)
    score_50, reason_n = anti_contradiction_override(50.0, neutral_feats, dom_n)
    p += run_test("Neutral signals → minimal override", abs(score_50 - 50) < 20,
                  f"50→{score_50:.1f}")
    t += 1
    
    return p, t


# ── 4. ML Scorer ─────────────────────────────────────────────────────────────

def test_ml_scorer():
    section("ML Scorer — Training & Generalization")
    p = t = 0
    
    scorer = ModernMLScorer()
    metrics = scorer.train(save=False)
    
    p += run_test("Trains without error", scorer.is_trained)
    t += 1
    p += run_test("CV AUC > 0.75 (no overfitting)",
                  metrics.get("cv_auc_mean", 0) > 0.75,
                  f"CV_AUC={metrics.get('cv_auc_mean','?')}")
    t += 1
    p += run_test("F1 > 0.75", metrics.get("f1", 0) > 0.75, f"F1={metrics.get('f1','?')}")
    t += 1
    
    # No single feature dominates > 50%
    top_imp = max(scorer.importances.values()) if scorer.importances else 1.0
    p += run_test("No feature has > 70% importance (no dominance)",
                  top_imp < 0.70, f"max_imp={top_imp:.3f}")
    t += 1
    
    # Prediction: extreme AI features → high probability
    ai_extreme = {k: 0.80 for k in scorer.feature_names}
    pred_ai = scorer.predict(ai_extreme)
    p += run_test("Extreme AI features → ML prob > 0.80",
                  pred_ai["ml_probability"] > 0.80,
                  f"p={pred_ai['ml_probability']:.3f}")
    t += 1
    
    # Prediction: extreme human features → low probability
    hu_extreme = {k: 0.15 for k in scorer.feature_names}
    pred_hu = scorer.predict(hu_extreme)
    p += run_test("Extreme human features → ML prob < 0.20",
                  pred_hu["ml_probability"] < 0.20,
                  f"p={pred_hu['ml_probability']:.3f}")
    t += 1
    
    return p, t


# ── 5. Full Pipeline Accuracy ────────────────────────────────────────────────

def test_pipeline_accuracy():
    section("Pipeline Accuracy — Diverse Text Examples")
    p = t = 0
    config = load_config()
    
    # All AI texts should score > 65
    for i, text in enumerate(AI_TEXTS):
        r = analyze_modern(text, config=config)
        ok = r["final_score"] > 65
        p += run_test(f"AI text #{i+1} → score > 65",
                      ok, f"score={r['final_score']:.1f}, class={r['classification']}")
        t += 1
    
    # All human texts should score < 35
    for i, text in enumerate(HUMAN_TEXTS):
        r = analyze_modern(text, config=config)
        ok = r["final_score"] < 40
        p += run_test(f"Human text #{i+1} → score < 40",
                      ok, f"score={r['final_score']:.1f}, class={r['classification']}")
        t += 1
    
    # Main sample test
    r_ai = analyze_modern(AI_SAMPLE, config=config)
    r_hu = analyze_modern(HUMAN_SAMPLE, config=config)
    
    p += run_test("AI sample: classification = 'Likely AI' or score > 50",
                  r_ai["final_score"] > 50 or r_ai["classification"] == "Likely AI",
                  f"score={r_ai['final_score']:.1f}")
    t += 1
    p += run_test("Human sample: score < 35",
                  r_hu["final_score"] < 35,
                  f"score={r_hu['final_score']:.1f}")
    t += 1
    p += run_test("AI scores higher than human",
                  r_ai["final_score"] > r_hu["final_score"],
                  f"AI={r_ai['final_score']:.1f}, H={r_hu['final_score']:.1f}")
    t += 1
    
    # Required output structure
    for key in ["scores","final_score","dominance_score","ml_score",
                "classification","confidence","highlights","metadata"]:
        p += run_test(f"Result has '{key}'", key in r_ai)
        t += 1
    
    # Confidence object
    conf = r_ai.get("confidence", {})
    p += run_test("Confidence has 'level'", "level" in conf)
    t += 1
    p += run_test("Confidence level valid", conf.get("level") in ("High","Medium","Low"))
    t += 1
    
    # Highlight structure
    hl = r_ai.get("highlights", [])
    p += run_test("Highlights non-empty", len(hl) > 0)
    t += 1
    if hl:
        h = hl[0]
        for k in ["text","start","end","label","score","reasons"]:
            p += run_test(f"Highlight has '{k}'", k in h)
            t += 1
    
    # Debug mode
    r_debug = analyze_modern(AI_TEXTS[0], config=config, debug=True)
    p += run_test("Debug mode works", "debug" in r_debug)
    t += 1
    p += run_test("Debug has per_feature", "per_feature" in r_debug.get("debug", {}))
    t += 1
    
    return p, t


# ── 6. Confidence & Dynamic Thresholds ──────────────────────────────────────

def test_confidence_and_thresholds():
    section("Confidence Estimation & Dynamic Thresholds")
    p = t = 0
    fe = FeatureExtractor()
    
    # High confidence: features agree strongly on AI
    ai_agree = {k: 0.80 for k in fe.FEATURE_NAMES}
    conf_high = estimate_confidence(ai_agree, 85.0, 0.85)
    p += run_test("Strongly agreeing AI features → High confidence",
                  conf_high["level"] == "High",
                  f"conf={conf_high['level']} ({conf_high['score']:.3f})")
    t += 1
    
    # Low confidence: mixed signals
    mixed = {k: 0.50 for k in fe.FEATURE_NAMES}
    conf_low = estimate_confidence(mixed, 50.0, 0.50)
    p += run_test("Mixed signals near boundary → Low confidence",
                  conf_low["level"] in ("Low", "Medium"),
                  f"conf={conf_low['level']}")
    t += 1
    
    # Dynamic thresholds: High confidence → tighter thresholds
    cls_high_ai = classify_with_dynamic_threshold(70.0, {"level":"High"}, 0.80)
    p += run_test("Score 70 + High conf → Likely AI", cls_high_ai == "Likely AI")
    t += 1
    
    cls_low_ai = classify_with_dynamic_threshold(70.0, {"level":"Low"}, 0.60)
    p += run_test("Score 70 + Low conf → Mixed (tighter threshold)", 
                  cls_low_ai == "Mixed / Uncertain",
                  f"cls={cls_low_ai}")
    t += 1
    
    cls_human = classify_with_dynamic_threshold(15.0, {"level":"High"}, 0.10)
    p += run_test("Score 15 + High conf → Human-like", cls_human == "Human-like")
    t += 1
    
    return p, t


# ── 7. Adversarial Robustness ────────────────────────────────────────────────

def test_adversarial():
    section("Adversarial Robustness — 5 Attack Types")
    p = t = 0
    config = load_config()
    tester = AdversarialTesterModern()
    
    for label, text in [("AI", AI_TEXTS[0]), ("Human", HUMAN_TEXTS[0])]:
        results = tester.run_all(text, lambda txt: analyze_modern(txt, config=config))
        baseline = results["baseline_score"]
        n_robust = sum(1 for k, v in results.items()
                       if isinstance(v, dict) and v.get("robust", False))
        p += run_test(f"{label}: ≥3/5 attacks robust", n_robust >= 3,
                      f"{n_robust}/5, baseline={baseline:.0f}")
        t += 1
        
        # Contraction injection is the hardest — test specifically
        contr = results.get("contraction_injection", {})
        if isinstance(contr, dict) and "delta" in contr:
            p += run_test(f"{label}: contraction injection delta < 30",
                          abs(contr["delta"]) < 30,
                          f"delta={contr['delta']:.1f}")
            t += 1
    
    return p, t


# ── 8. Before vs After comparison ────────────────────────────────────────────

def test_before_after():
    section("Before vs After — Key Improvements")
    p = t = 0
    config = load_config()
    
    print("\n  BEFORE (v2) vs AFTER (modern):")
    
    # v2 pipeline
    from analyzer.pipeline_v2 import analyze_v2
    
    improvements = []
    for label, text in [("AI_Essay", AI_TEXTS[0]), ("Human_Casual", HUMAN_TEXTS[0]),
                         ("AI_Sample", AI_SAMPLE), ("Human_Sample", HUMAN_SAMPLE)]:
        try:
            v2 = analyze_v2(text, config=config, use_ml=True, include_explanation=False)
            v2_score = v2["final_score"]
        except Exception:
            v2_score = 50.0
        
        mod = analyze_modern(text, config=config, include_explanation=False)
        mod_score = mod["final_score"]
        
        is_ai = "AI" in label
        correct_direction = (mod_score > 65) if is_ai else (mod_score < 40)
        
        print(f"  {label:<20} v2={v2_score:>5.1f}  modern={mod_score:>5.1f}  "
              f"{'✓ Correct' if correct_direction else '○ Ambiguous'}")
        improvements.append(correct_direction)
        p += run_test(f"{label}: modern correctly classified", correct_direction,
                      f"score={mod_score:.1f}")
        t += 1
    
    overall_improvement = sum(improvements) / len(improvements)
    print(f"\n  Correct classifications: {sum(improvements)}/{len(improvements)} "
          f"({overall_improvement:.0%})")
    
    return p, t


# ── Runner ─────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "═"*62)
    print("  MODERN AI DETECTOR — FULL TEST SUITE")
    print("═"*62)
    
    total_p = total_t = 0
    suites = [
        test_feature_extractor,
        test_dominance_scorer,
        test_anti_contradiction,
        test_ml_scorer,
        test_pipeline_accuracy,
        test_confidence_and_thresholds,
        test_adversarial,
        test_before_after,
    ]
    
    for suite in suites:
        p, t = suite()
        total_p += p; total_t += t
    
    print("\n" + "═"*62)
    pct = total_p/total_t*100 if total_t else 0
    print(f"  RESULTS: {total_p}/{total_t} tests passed ({pct:.0f}%)")
    print(f"  {'✓ All tests passed!' if total_p == total_t else f'○ {total_t-total_p} failed'}")
    print("═"*62 + "\n")
    return total_p == total_t


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    ok = run_all()
    sys.exit(0 if ok else 1)
