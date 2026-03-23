"""
humanizer/pipeline.py
=====================
Integration between Humanizer and the modern AI detector.

Provides:
  HumanizerPipeline  — full end-to-end: analyze → humanize → re-analyze → report
  run_comparison     — quick before/after CLI report
  batch_humanize     — process multiple texts
"""

import sys
import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class HumanizerPipeline:
    """
    Full pipeline: Input Text → Detect → Humanize → Re-detect → Report.
    
    Usage:
        pipeline = HumanizerPipeline()
        report = pipeline.run("Your AI text here...", mode="balanced")
        print(report["summary"])
    """
    
    def __init__(self):
        # Lazy imports to avoid circular deps
        from analyzer.pipeline_modern import analyze_modern
        from analyzer.pipeline import load_config
        from humanizer.humanizer import HumanizerEngine, HumanizerConfig
        
        self.config = load_config()
        self._analyze = lambda t: analyze_modern(t, config=self.config, include_explanation=False)
        self._engine = HumanizerEngine()
        self._HumanizerConfig = HumanizerConfig
    
    def run(
        self,
        text: str,
        mode: str = "balanced",
        max_passes: int = 3,
        seed: int = 42,
        target_score: float = 50.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline run.
        
        Returns dict with:
          original_text, humanized_text,
          original_score, humanized_score, score_change, improvement_pct,
          original_class, humanized_class,
          transformations_applied, per_feature_changes,
          passes_used, time_ms
        """
        t0 = time.perf_counter()
        
        # Step 1: Original detection
        orig_result = self._analyze(text)
        orig_score = orig_result["final_score"]
        orig_class = orig_result["classification"]
        orig_feats = orig_result["scores"]
        
        if verbose:
            print(f"[Pipeline] Original score: {orig_score:.1f} ({orig_class})")
        
        # Step 2: Humanize with feedback loop
        config = self._HumanizerConfig(
            mode=mode,
            max_passes=max_passes,
            seed=seed,
            target_score=target_score,
        )
        
        hum_result = self._engine.humanize(text, config, analyze_fn=self._analyze)
        humanized_text = hum_result.text
        
        # Step 3: Final detection
        final_result = self._analyze(humanized_text)
        final_score = final_result["final_score"]
        final_class = final_result["classification"]
        final_feats = final_result["scores"]
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if verbose:
            print(f"[Pipeline] Final score: {final_score:.1f} ({final_class})")
            print(f"[Pipeline] Improvement: {orig_score - final_score:.1f} pts "
                  f"({(orig_score - final_score)/max(orig_score,1)*100:.0f}%)")
        
        # Per-feature changes
        feat_changes = {}
        for k in set(orig_feats) | set(final_feats):
            before = orig_feats.get(k, 0)
            after  = final_feats.get(k, 0)
            feat_changes[k] = {
                "before":    round(before, 3),
                "after":     round(after, 3),
                "delta":     round(after - before, 3),
                "improved":  (after - before) < -0.05,
            }
        
        change = orig_score - final_score
        pct = (change / orig_score * 100) if orig_score else 0
        
        # Human-readable summary
        summary = _build_summary(
            text, humanized_text,
            orig_score, final_score, orig_class, final_class,
            hum_result.transformations_applied, feat_changes
        )
        
        return {
            "original_text":         text,
            "humanized_text":        humanized_text,
            "original_score":        round(orig_score, 2),
            "humanized_score":       round(final_score, 2),
            "score_change":          round(change, 2),
            "improvement_pct":       round(pct, 1),
            "original_class":        orig_class,
            "humanized_class":       final_class,
            "original_confidence":   orig_result.get("confidence", {}).get("level", "?"),
            "humanized_confidence":  final_result.get("confidence", {}).get("level", "?"),
            "transformations_applied": hum_result.transformations_applied,
            "per_feature_changes":   feat_changes,
            "features_improved":     sum(1 for v in feat_changes.values() if v["improved"]),
            "passes_used":           hum_result.passes_applied,
            "mode":                  mode,
            "time_ms":               round(elapsed_ms, 1),
            "summary":               summary,
        }
    
    def run_comparison_report(self, text: str, modes: List[str] = None) -> str:
        """Run all three modes and return a comparison table."""
        if modes is None:
            modes = ["subtle", "balanced", "aggressive"]
        
        lines = ["", "═" * 65, "  HUMANIZER MODE COMPARISON", "═" * 65]
        lines.append(f"  Original text: {text[:80]}{'...' if len(text) > 80 else ''}")
        lines.append("")
        
        orig = self._analyze(text)
        orig_score = orig["final_score"]
        lines.append(f"  {'Mode':<12} {'Score':>8} {'Change':>8} {'Class':<20} {'Transforms'}")
        lines.append(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*20} {'─'*20}")
        lines.append(f"  {'Original':<12} {orig_score:>8.1f} {'—':>8} {orig['classification']:<20}")
        
        for mode in modes:
            result = self.run(text, mode=mode)
            delta = result["score_change"]
            lines.append(
                f"  {mode:<12} {result['humanized_score']:>8.1f} "
                f"{'-'+str(round(delta,1)):>8} {result['humanized_class']:<20} "
                f"{', '.join(result['transformations_applied'][:3])}"
            )
        
        lines.append("═" * 65)
        return "\n".join(lines)


def run_comparison(text: str, mode: str = "balanced") -> str:
    """Quick comparison report — CLI-friendly."""
    pipeline = HumanizerPipeline()
    result = pipeline.run(text, mode=mode, verbose=False)
    return _format_cli_report(result)


def batch_humanize(
    texts: List[str],
    mode: str = "balanced",
    analyze_fn=None,
) -> List[Dict[str, Any]]:
    """
    Humanize a list of texts.
    Returns list of result dicts.
    """
    from humanizer.humanizer import humanize_text
    
    results = []
    for i, text in enumerate(texts):
        logger.info(f"Humanizing text {i+1}/{len(texts)}")
        result = humanize_text(text, mode=mode, analyze_fn=analyze_fn)
        results.append({
            "index":           i,
            "original_text":   text[:100] + "..." if len(text) > 100 else text,
            "humanized_text":  result.text,
            "original_score":  result.original_score,
            "humanized_score": result.humanized_score,
            "score_change":    result.score_change,
            "improvement_pct": result.improvement_pct,
            "transforms":      result.transformations_applied,
        })
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary(
    orig_text: str,
    hum_text: str,
    orig_score: float,
    final_score: float,
    orig_class: str,
    final_class: str,
    transforms: List[str],
    feat_changes: Dict,
) -> str:
    change = orig_score - final_score
    pct = (change / orig_score * 100) if orig_score else 0
    improved_feats = [k for k, v in feat_changes.items() if v["improved"]]
    
    lines = [
        f"Score dropped from {orig_score:.0f} → {final_score:.0f} "
        f"({change:+.0f} pts, {pct:.0f}% reduction).",
        f"Classification: {orig_class} → {final_class}.",
        f"Transformations applied: {', '.join(transforms[:5])}.",
    ]
    if improved_feats:
        lines.append(f"Features improved: {', '.join(improved_feats[:4])}.")
    return " ".join(lines)


def _format_cli_report(result: Dict[str, Any]) -> str:
    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
    BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"; CYAN = "\033[96m"
    
    orig = result["original_score"]
    final = result["humanized_score"]
    change = result["score_change"]
    pct = result["improvement_pct"]
    
    score_color = GREEN if change > 20 else (YELLOW if change > 5 else RED)
    
    lines = [
        f"\n{BOLD}{'═'*65}{RESET}",
        f"{BOLD}  HUMANIZER RESULT  [{result['mode'].upper()} mode]{RESET}",
        f"{'═'*65}",
        f"\n  {BOLD}Original Score:{RESET}   {RED}{orig:.1f}/100{RESET}  ({result['original_class']})",
        f"  {BOLD}Humanized Score:{RESET}  {score_color}{final:.1f}/100{RESET}  ({result['humanized_class']})",
        f"  {BOLD}Improvement:{RESET}      {score_color}{change:+.1f} pts  ({pct:.0f}% reduction){RESET}",
        f"  {BOLD}Passes Used:{RESET}      {result['passes_used']}",
        f"  {BOLD}Time:{RESET}             {result['time_ms']:.0f}ms",
        "",
        f"  {BOLD}Transformations:{RESET}",
    ]
    for t in result["transformations_applied"]:
        lines.append(f"    {CYAN}→{RESET} {t}")
    
    lines.append(f"\n  {BOLD}Feature Changes:{RESET}")
    feat_changes = result["per_feature_changes"]
    for feat, vals in sorted(feat_changes.items(), key=lambda x: x[1]["delta"]):
        delta = vals["delta"]
        if abs(delta) < 0.02:
            continue
        color = GREEN if delta < 0 else RED
        bar = "▼" if delta < 0 else "▲"
        lines.append(f"    {color}{bar} {feat:<28}{RESET} {vals['before']:.3f} → {vals['after']:.3f}  ({delta:+.3f})")
    
    lines.append(f"\n  {BOLD}Humanized Text Preview:{RESET}")
    preview = result["humanized_text"][:300]
    lines.append(f"  {DIM}{preview}{'...' if len(result['humanized_text']) > 300 else ''}{RESET}")
    
    lines.append(f"\n{'═'*65}\n")
    return "\n".join(lines)
