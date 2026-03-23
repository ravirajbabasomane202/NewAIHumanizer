"""
app.py — Flask Web UI  (v3: unified automatic humanizer, no mode buttons)
"""
import os, io, logging
from flask import Flask, render_template, request, jsonify, send_file
from analyzer.pipeline_modern import analyze_modern, FEATURE_DISPLAY_NAMES
from analyzer.detector import get_modern_scorer, AdversarialTesterModern
from analyzer.pipeline import load_config

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-v3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
CONFIG = load_config()

try:
    _scorer = get_modern_scorer()
    logger.info(f"Scorer ready (trained={_scorer.is_trained})")
except Exception as e:
    logger.warning(f"Scorer warm-up: {e}")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(silent=True) if request.is_json else {}
    text = str((data or {}).get("text", request.form.get("text", ""))).strip()
    debug = bool((data or {}).get("debug", False))

    if not text:
        return jsonify({"error": "No text provided."}), 400
    max_chars = CONFIG.get("settings", {}).get("max_input_chars", 100000)
    if len(text) > max_chars:
        return jsonify({"error": f"Text too long. Max {max_chars} chars."}), 413

    try:
        result = analyze_modern(text, config=CONFIG, include_explanation=True, debug=debug)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("Analysis error")
        return jsonify({"error": "Analysis error. Please try again."}), 500


@app.route("/humanize", methods=["POST"])
def humanize_route():
    """
    Automatic maximum humanization — no mode selection needed.
    Always runs aggressive mode with 3 passes and feedback loop.
    """
    data = request.get_json(silent=True) if request.is_json else {}
    text = str((data or {}).get("text", request.form.get("text", ""))).strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    max_chars = CONFIG.get("settings", {}).get("max_input_chars", 100000)
    if len(text) > max_chars:
        return jsonify({"error": "Text too long."}), 413

    try:
        from humanizer.pipeline import HumanizerPipeline
        pipeline = HumanizerPipeline()
        result = pipeline.run(text, mode="aggressive", max_passes=3, target_score=45.0)
        return jsonify(result)
    except Exception as e:
        logger.exception("Humanizer error")
        return jsonify({"error": str(e)}), 500


@app.route("/adversarial", methods=["POST"])
def adversarial_route():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "No text."}), 400
    try:
        tester = AdversarialTesterModern()
        results = tester.run_all(text, lambda t: analyze_modern(t, config=CONFIG, include_explanation=False))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export/html", methods=["POST"])
def export_html():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", request.form.get("text", ""))).strip()
    if not text:
        return jsonify({"error": "No text."}), 400
    try:
        result = analyze_modern(text, config=CONFIG)
        html = _render_report(text, result)
        buf = io.BytesIO(html.encode("utf-8")); buf.seek(0)
        return send_file(buf, mimetype="text/html", as_attachment=True,
                         download_name="ai_analysis.html")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    scorer = get_modern_scorer()
    return jsonify({"status": "ok", "version": "3.0.0", "ml_trained": scorer.is_trained})


def _render_report(text, result):
    score = result.get("final_score", 0)
    cls = result.get("classification", "Unknown")
    conf = result.get("confidence", {})
    expl = result.get("explanation", {})
    meta = result.get("metadata", {})
    highlights = result.get("highlights", [])
    color = {"Likely AI":"#ef4444","Human-like":"#22c55e","Mixed / Uncertain":"#f59e0b"}.get(cls,"#888")
    hl_html = _hl_html(text, highlights)
    summary = expl.get("summary","")
    findings_html = "".join(f"<li>{f}</li>" for f in expl.get("key_findings",[]))
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>AI Analysis</title>
<style>body{{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:1rem 2rem;background:#080c14;color:#e2e8f0}}
h1{{color:#60a5fa;border-bottom:2px solid #1e3051;padding-bottom:.5rem}}
.card{{background:#0d1526;border:1px solid #1e3051;border-radius:10px;padding:1.25rem;margin:1rem 0}}
.score{{font-size:3.5rem;font-weight:900;color:{color};font-family:monospace}}
.badge{{display:inline-block;padding:3px 12px;border-radius:4px;font-size:.85rem;font-weight:700;background:{color}22;color:{color};border:1px solid {color}44}}
.hl-ai{{background:rgba(239,68,68,.2);border-bottom:2px solid #ef4444;border-radius:2px;padding:0 2px}}
.hl-mixed{{background:rgba(245,158,11,.15);border-bottom:2px solid #f59e0b;border-radius:2px;padding:0 2px}}
.hl-human{{background:rgba(34,197,94,.08)}}
.ht{{line-height:2;font-size:.95rem;white-space:pre-wrap}}
</style></head><body>
<h1>AI Text Analysis</h1>
<div class="card" style="display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start">
  <div><div class="score">{score:.0f}<span style="font-size:1.5rem">/100</span></div><div class="badge">{cls}</div></div>
  <div style="flex:1;color:#94a3b8;font-size:.85rem;line-height:1.8">
    Words: {meta.get('word_count','?')} · Sentences: {meta.get('sentence_count','?')} · Confidence: {conf.get('level','?')}<br>
    {summary}
  </div>
</div>
<div class="card"><div class="ht">{hl_html}</div></div>
</body></html>"""


def _hl_html(text, highlights):
    if not highlights: return text.replace("&","&amp;").replace("<","&lt;")
    r = []; last = 0
    lc = {"AI":"hl-ai","Mixed":"hl-mixed","Human":"hl-human"}
    for h in sorted(highlights, key=lambda x: x["start"]):
        s, e = h["start"], h["end"]
        if s > last: r.append(text[last:s].replace("&","&amp;").replace("<","&lt;"))
        tip = "; ".join(h.get("reasons",[])[:2]).replace('"','&quot;')
        r.append(f'<span class="{lc.get(h["label"],"")}" title="{tip}">{text[s:e].replace("&","&amp;").replace("<","&lt;")}</span>')
        last = e
    if last < len(text): r.append(text[last:].replace("&","&amp;").replace("<","&lt;"))
    return "".join(r)



@app.route("/humanize/until-zero", methods=["POST"])
def humanize_until_zero():
    """
    Iterative humanization — repeats rounds until score < threshold or no progress.
    Returns JSON with all rounds + final best result.
    
    Body: {"text": "...", "target": 15, "max_rounds": 10}
    """
    data = request.get_json(silent=True) if request.is_json else {}
    text    = str((data or {}).get("text", "")).strip()
    target  = float((data or {}).get("target", 15.0))   # stop below this score
    max_rounds = int((data or {}).get("max_rounds", 10))

    if not text:
        return jsonify({"error": "No text provided."}), 400
    max_chars = CONFIG.get("settings", {}).get("max_input_chars", 100000)
    if len(text) > max_chars:
        return jsonify({"error": "Text too long."}), 413

    try:
        from humanizer.pipeline import HumanizerPipeline
        from analyzer.pipeline_modern import analyze_modern

        pipeline = HumanizerPipeline()
        analyze_fn = lambda t: analyze_modern(t, config=CONFIG, include_explanation=False)

        # Get starting score
        start_result = analyze_fn(text)
        start_score  = start_result["final_score"]

        rounds = []
        best_score = start_score
        best_text  = text
        current    = text
        no_improve_streak = 0

        for round_num in range(max_rounds):
            # Try 4 seeds, keep the best result this round
            round_best_score = float("inf")
            round_best_text  = current

            for seed in [42, 17, 99, 7]:
                r = pipeline.run(
                    current, mode="aggressive",
                    max_passes=3, seed=seed, target_score=target
                )
                if r["humanized_score"] < round_best_score:
                    round_best_score = r["humanized_score"]
                    round_best_text  = r["humanized_text"]
                    round_best_data  = r

            # Track best ever
            improved = round_best_score < best_score
            if improved:
                best_score = round_best_score
                best_text  = round_best_text
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            rounds.append({
                "round":          round_num + 1,
                "score_before":   round(analyze_fn(current)["final_score"], 1),
                "score_after":    round(round_best_score, 1),
                "best_so_far":    round(best_score, 1),
                "improved":       improved,
                "classification": round_best_data.get("humanized_class", ""),
                "transforms":     round_best_data.get("transformations_applied", [])[:4],
            })

            current = round_best_text

            # Stop conditions
            if best_score <= target:
                break
            if no_improve_streak >= 2:
                break

        # Final analysis of best text
        final_result = analyze_fn(best_text)

        return jsonify({
            "original_text":    text,
            "best_text":        best_text,
            "original_score":   round(start_score, 2),
            "best_score":       round(best_score, 2),
            "total_drop":       round(start_score - best_score, 2),
            "improvement_pct":  round((start_score - best_score) / max(start_score, 1) * 100, 1),
            "rounds_used":      len(rounds),
            "target_reached":   best_score <= target,
            "final_class":      final_result.get("classification", ""),
            "final_confidence": final_result.get("confidence", {}).get("level", ""),
            "rounds":           rounds,
            "per_feature_changes": {
                k: {
                    "before": round(start_result["scores"].get(k, 0), 3),
                    "after":  round(final_result["scores"].get(k, 0), 3),
                    "delta":  round(final_result["scores"].get(k, 0) - start_result["scores"].get(k, 0), 3),
                    "improved": (final_result["scores"].get(k, 0) - start_result["scores"].get(k, 0)) < -0.05,
                }
                for k in set(start_result.get("scores", {})) | set(final_result.get("scores", {}))
            },
        })

    except Exception as e:
        logger.exception("Until-zero error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
