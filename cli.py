"""
cli.py — Command-Line Interface for AI Text Analyzer

Usage:
  python cli.py "Some text to analyze"
  python cli.py --file input.txt
  python cli.py --file input.txt --debug
  echo "Some text" | python cli.py
  python cli.py --demo
"""

import sys
import json
import argparse
import logging

from analyzer.pipeline import analyze, load_config
from analyzer.explanation import format_explanation_text


def main():
    parser = argparse.ArgumentParser(
        description="Explainable AI vs Human Text Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "Large language models generate text by..."
  python cli.py --file essay.txt
  python cli.py --file essay.txt --output json
  python cli.py --demo
  echo "Some text here" | python cli.py
        """,
    )

    parser.add_argument(
        "text",
        nargs="?",
        help="Text to analyze (or pipe via stdin)",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        help="Read input from a file",
    )
    parser.add_argument(
        "--output", "-o",
        choices=["pretty", "json", "minimal"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include raw feature scores in output",
    )
    parser.add_argument(
        "--no-explanation",
        action="store_true",
        help="Skip explanation layer (faster)",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="config/weights.yaml",
        help="Path to weights config YAML",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with built-in AI and human text samples",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity",
    )
    parser.add_argument(
        "--export-html",
        metavar="PATH",
        help="Export analysis report as HTML file",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config = load_config(args.config)

    if args.demo:
        _run_demo(config)
        return

    # Get input text
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(0)

    text = text.strip()
    if not text:
        print("Error: No text provided.", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    try:
        result = analyze(
            text,
            config=config,
            include_explanation=not args.no_explanation,
            debug=args.debug,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.output == "json":
        print(json.dumps(result, indent=2))
    elif args.output == "minimal":
        print(f"Score: {result['final_score']:.1f}/100  |  {result['classification']}")
    else:
        _pretty_print(result)

    # Export HTML if requested
    if args.export_html:
        _export_html(text, result, args.export_html)


def _pretty_print(result: dict):
    """Pretty-print analysis result to terminal."""
    score = result["final_score"]
    classification = result["classification"]
    meta = result.get("metadata", {})

    # Color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[96m"

    color = RED if score >= 70 else (YELLOW if score >= 30 else GREEN)

    print(f"\n{BOLD}{'─' * 65}{RESET}")
    print(f"{BOLD}  EXPLAINABLE AI TEXT ANALYZER  {RESET}{DIM}v1.0{RESET}")
    print(f"{'─' * 65}")

    # Score meter
    filled = int(score / 5)
    meter = "█" * filled + "░" * (20 - filled)
    print(f"\n  Score:  {color}{BOLD}{score:.1f}/100{RESET}  [{color}{meter}{RESET}]")
    print(f"  Result: {color}{BOLD}{classification}{RESET}")
    print(f"  Words:  {meta.get('word_count','?')}  │  "
          f"Sentences: {meta.get('sentence_count','?')}  │  "
          f"Time: {meta.get('analysis_time_ms','?')}ms")

    # Feature breakdown
    print(f"\n  {BOLD}Feature Scores:{RESET}")
    scores = result.get("scores", {})
    from analyzer.explanation import FEATURE_DISPLAY_NAMES
    for name, score_val in sorted(scores.items(), key=lambda x: -x[1]):
        pct = score_val * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        flag_col = RED if pct >= 60 else (YELLOW if pct >= 35 else GREEN)
        display = FEATURE_DISPLAY_NAMES.get(name, name)[:30]
        print(f"  {flag_col}{bar}{RESET} {pct:5.1f}%  {DIM}{display}{RESET}")

    # Highlights
    highlights = result.get("highlights", [])
    ai_sents = [h for h in highlights if h["label"] == "AI"]
    if ai_sents:
        print(f"\n  {BOLD}AI-flagged sentences:{RESET}")
        for h in ai_sents[:5]:
            short = h["text"][:80] + ("..." if len(h["text"]) > 80 else "")
            reasons = ", ".join(h["reasons"][:2]) if h["reasons"] else "—"
            print(f"  {RED}▶{RESET} \"{DIM}{short}{RESET}\"")
            print(f"    {DIM}Reasons: {reasons}{RESET}")

    # Explanation
    explanation = result.get("explanation")
    if explanation:
        print(f"\n{BOLD}{'─' * 65}{RESET}")
        print(format_explanation_text(explanation))

    print(f"{'─' * 65}\n")


def _run_demo(config: dict):
    """Run analysis on built-in demo texts."""
    from tests.samples import HUMAN_SAMPLE, AI_SAMPLE, MIXED_SAMPLE

    demos = [
        ("HUMAN SAMPLE", HUMAN_SAMPLE),
        ("AI SAMPLE", AI_SAMPLE),
        ("MIXED SAMPLE", MIXED_SAMPLE),
    ]

    for label, text in demos:
        print(f"\n{'='*65}")
        print(f"  DEMO: {label}")
        print(f"{'='*65}")
        print(f"  Text preview: {text[:100]}...")
        result = analyze(text, config=config, include_explanation=False)
        _pretty_print(result)


def _export_html(text: str, result: dict, path: str):
    """Export HTML report to file."""
    try:
        from app import _render_html_report, _build_highlighted_html
        html = _render_html_report(text, result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nHTML report saved to: {path}")
    except Exception as e:
        print(f"Export failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
