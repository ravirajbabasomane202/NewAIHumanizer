"""
api.py — Standalone REST API for AI Text Analyzer

Endpoints:
  POST /analyze            — Full analysis
  POST /analyze/batch      — Analyze multiple texts
  GET  /features           — List available features + metadata
  GET  /health             — Health check

Run standalone: python api.py
Or mount on existing Flask: from api import api_bp; app.register_blueprint(api_bp, url_prefix='/api')
"""

import json
import logging
from flask import Flask, Blueprint, request, jsonify
from analyzer.pipeline import analyze, load_config
from analyzer.features import ALL_FEATURES, FEATURE_MAP
from analyzer.explanation import FEATURE_DISPLAY_NAMES, NORMALIZATION_STRATEGIES as _NS

# Import normalization strategies for feature listing
try:
    from analyzer.normalization import NORMALIZATION_STRATEGIES
except ImportError:
    NORMALIZATION_STRATEGIES = {}

logger = logging.getLogger(__name__)

# ── Blueprint (can be mounted on app.py) ─────────────────────────────────────
api_bp = Blueprint("api", __name__)
CONFIG = load_config()


@api_bp.route("/analyze", methods=["POST"])
def api_analyze():
    """
    POST /analyze
    
    Request body (JSON):
    {
        "text": "Text to analyze...",
        "debug": false,       // optional: include raw feature scores
        "explanation": true   // optional: include explanation layer
    }
    
    Response:
    {
        "scores": { feature_name: score_0_to_1, ... },
        "final_score": 68.4,
        "classification": "Likely AI",
        "highlights": [...],
        "explanation": {...},
        "metadata": {...}
    }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Field 'text' is required and must be non-empty"}), 400

    debug = bool(data.get("debug", False))
    include_explanation = bool(data.get("explanation", True))

    max_chars = CONFIG.get("settings", {}).get("max_input_chars", 100000)
    if len(text) > max_chars:
        return jsonify({
            "error": f"Text exceeds maximum length of {max_chars} characters.",
            "provided_length": len(text),
        }), 413

    try:
        result = analyze(
            text,
            config=CONFIG,
            include_explanation=include_explanation,
            debug=debug,
        )
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.exception("API analysis error")
        return jsonify({"error": "Internal error during analysis"}), 500


@api_bp.route("/analyze/batch", methods=["POST"])
def api_analyze_batch():
    """
    POST /analyze/batch

    Request body:
    {
        "texts": ["text 1...", "text 2...", ...],
        "explanation": false
    }

    Response:
    {
        "results": [
            { "index": 0, "final_score": ..., "classification": ..., "scores": {...} },
            ...
        ]
    }

    Maximum 10 texts per batch.
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    include_explanation = bool(data.get("explanation", False))

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "'texts' must be a non-empty array"}), 400

    if len(texts) > 10:
        return jsonify({"error": "Batch limit is 10 texts per request"}), 400

    results = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            results.append({"index": i, "error": "Empty text"})
            continue
        try:
            r = analyze(text, config=CONFIG, include_explanation=include_explanation)
            results.append({
                "index": i,
                "final_score": r["final_score"],
                "classification": r["classification"],
                "scores": r["scores"],
                "metadata": r["metadata"],
            })
            if include_explanation:
                results[-1]["explanation"] = r.get("explanation")
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify({"results": results}), 200


@api_bp.route("/features", methods=["GET"])
def list_features():
    """
    GET /features
    
    Returns metadata about all available features.
    """
    features_info = []
    weights = CONFIG.get("features", {})
    hl_thresholds = CONFIG.get("highlight_thresholds", {})

    for feature in ALL_FEATURES:
        features_info.append({
            "name": feature.name,
            "display_name": FEATURE_DISPLAY_NAMES.get(feature.name, feature.name),
            "description": feature.description,
            "weight": weights.get(feature.name, 0.0),
            "highlight_threshold": hl_thresholds.get(feature.name, 0.6),
            "normalization": NORMALIZATION_STRATEGIES.get(feature.name, ""),
        })

    return jsonify({
        "feature_count": len(features_info),
        "features": features_info,
    }), 200


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "1.0.0"}), 200


# ── Standalone runner ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix="/api")

# Also mount at root for direct access
app.register_blueprint(api_bp, url_prefix="")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, port=5001)
