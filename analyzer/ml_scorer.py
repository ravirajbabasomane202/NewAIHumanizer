"""
analyzer/ml_scorer.py
---------------------
ML-based scoring layer that replaces the static weighted-sum.

Architecture:
  1. Feature vector (25 features × normalized scores)
  2. StandardScaler (z-score normalization)
  3. GradientBoostingClassifier  (primary model — handles non-linear feature interactions)
  4. LogisticRegression           (secondary model — for explainability / calibration)
  5. CalibratedClassifierCV       (Platt scaling → well-calibrated probabilities)
  6. VotingEnsemble               (blend both models)

Training:
  - Uses a synthetic corpus of human + AI text features (generated at init).
  - In production you would replace _generate_training_data() with real labeled data.
  - Model trains in <1 second (lightweight, no GPU).

Evaluation metrics exposed:
  - accuracy, precision, recall, F1, AUC-ROC
  - Confusion matrix
  - Cross-validation scores
"""

import math
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report,
)
import joblib

logger = logging.getLogger(__name__)

# Feature names in fixed order (matches feature extraction order)
FEATURE_NAMES = [
    # v1 features
    "perplexity", "burstiness", "sentence_diversity", "lexical_diversity",
    "repetition", "semantic_predictability", "syntactic_complexity",
    "function_word_dist", "entropy", "ngram_frequency_bias",
    "over_coherence", "emotional_variability", "error_patterns",
    "contextual_depth", "stopword_patterning",
    # v2 features
    "bigram_perplexity", "log_likelihood_variance", "vader_sentiment",
    "compression_ratio", "stylometric_fingerprint", "vocab_richness_curve",
    "punctuation_diversity", "hapax_ratio", "readability_score",
    "sentence_position_bias",
]

MODEL_PATH = Path(__file__).parent.parent / "models" / "ml_scorer.joblib"


class MLScorer:
    """
    Trained ML classifier that predicts AI-likelihood from feature vectors.

    Usage:
        scorer = MLScorer()
        scorer.train()  # or scorer.load()
        result = scorer.predict(feature_dict)
        # → {"probability": 0.72, "score": 72.0, "confidence": "High"}
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        )
        self._lr = LogisticRegression(
            C=0.5,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
        self.model: Optional[Any] = None
        self.is_trained = False
        self._train_metrics: Dict[str, Any] = {}
        self._feature_importances: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the ML scorer.

        If X and y are not provided, uses the synthetic training corpus.
        Returns evaluation metrics.
        """
        if X is None or y is None:
            logger.info("Generating synthetic training corpus...")
            X, y = self._generate_training_data(n_samples=2000)

        logger.info(f"Training on {len(X)} samples ({y.sum()} AI, {len(y)-y.sum()} human)...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        gb_cv = cross_val_score(self._gb, X_scaled, y, cv=cv, scoring="roc_auc")
        lr_cv = cross_val_score(self._lr, X_scaled, y, cv=cv, scoring="roc_auc")
        logger.info(f"CV AUC — GB: {gb_cv.mean():.3f}±{gb_cv.std():.3f}, LR: {lr_cv.mean():.3f}±{lr_cv.std():.3f}")

        # Fit both models
        self._gb.fit(X_scaled, y)
        self._lr.fit(X_scaled, y)

        # Calibrate GB with Platt scaling
        calibrated_gb = CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                subsample=0.8, min_samples_leaf=3, random_state=42,
            ),
            method="sigmoid", cv=3,
        )
        calibrated_gb.fit(X_scaled, y)

        # Soft voting ensemble: calibrated GB + LR
        self.model = calibrated_gb  # primary
        self._lr_fitted = self._lr

        # Full-set metrics
        y_pred = (calibrated_gb.predict_proba(X_scaled)[:, 1] >= 0.5).astype(int)
        y_prob = calibrated_gb.predict_proba(X_scaled)[:, 1]
        acc = accuracy_score(y, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average="binary")
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred).tolist()

        self._train_metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
            "confusion_matrix": cm,
            "cv_auc_gb_mean": round(float(gb_cv.mean()), 4),
            "cv_auc_gb_std": round(float(gb_cv.std()), 4),
            "cv_auc_lr_mean": round(float(lr_cv.mean()), 4),
            "n_train": len(X),
        }

        # Feature importances from GB
        importances = self._gb.feature_importances_
        self._feature_importances = {
            name: round(float(imp), 4)
            for name, imp in sorted(
                zip(FEATURE_NAMES[:len(importances)], importances),
                key=lambda x: -x[1]
            )
        }

        logger.info(
            f"Training complete — Accuracy={acc:.3f}, AUC={auc:.3f}, F1={f1:.3f}"
        )

        self.is_trained = True

        if save:
            self.save()

        return self._train_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, feature_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict AI probability from feature dict.

        Parameters
        ----------
        feature_scores : dict  feature_name → score in [0,1]

        Returns
        -------
        {
            "probability": float,   # [0,1] calibrated probability
            "score": float,         # 0–100
            "label": str,
            "confidence": str,
            "method": str,
        }
        """
        if not self.is_trained:
            # Fallback: weighted sum (same as v1)
            return self._fallback_predict(feature_scores)

        vec = self._feature_dict_to_vector(feature_scores)
        vec_scaled = self.scaler.transform([vec])

        prob_gb = float(self.model.predict_proba(vec_scaled)[0, 1])
        prob_lr = float(self._lr_fitted.predict_proba(vec_scaled)[0, 1])

        # Ensemble: 70% GB + 30% LR
        prob = 0.7 * prob_gb + 0.3 * prob_lr
        score = prob * 100.0

        label = self._classify(prob)
        confidence = self._confidence(prob)

        return {
            "probability": round(prob, 4),
            "score": round(score, 2),
            "label": label,
            "confidence": confidence,
            "method": "ml_ensemble",
        }

    def predict_sentence(
        self,
        sentence_feature_scores: Dict[str, List[float]],
        sentence_idx: int,
    ) -> Dict[str, Any]:
        """Predict for a single sentence's feature score slice."""
        sent_scores = {
            name: scores[sentence_idx]
            for name, scores in sentence_feature_scores.items()
            if sentence_idx < len(scores)
        }
        return self.predict(sent_scores)

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "model": self.model,
            "lr": self._lr_fitted if hasattr(self, "_lr_fitted") else self._lr,
            "metrics": self._train_metrics,
            "importances": self._feature_importances,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Optional[Path] = None) -> bool:
        path = path or MODEL_PATH
        if not path.exists():
            logger.warning(f"No saved model at {path}. Will use fallback scoring.")
            return False
        try:
            bundle = joblib.load(path)
            self.scaler = bundle["scaler"]
            self.model = bundle["model"]
            self._lr_fitted = bundle["lr"]
            self._train_metrics = bundle.get("metrics", {})
            self._feature_importances = bundle.get("importances", {})
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────
    # Synthetic training data generation
    # ──────────────────────────────────────────────────────────────────────

    def _generate_training_data(
        self, n_samples: int = 2000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic feature vectors representing human (0) and AI (1) text.

        Each feature is drawn from realistic distributions calibrated against
        known properties of AI vs human writing.

        Human text characteristics (low AI scores):
          - High burstiness, high lexical diversity, high contextual depth
          - Low error patterns score (more errors), high hapax ratio
          - High emotional variability, low compression ratio

        AI text characteristics (high AI scores):
          - Low burstiness, high perplexity score, low contextual depth
          - High error patterns score (near-perfect), low hapax ratio
          - Low emotional variability, high compression ratio
        """
        rng = np.random.RandomState(42)
        n_features = len(FEATURE_NAMES)

        # Base distributions for each class per feature (mean, std)
        # (human_mean, human_std, ai_mean, ai_std)
        _distributions = {
            "perplexity":              (0.30, 0.12, 0.62, 0.12),
            "burstiness":              (0.20, 0.12, 0.65, 0.15),
            "sentence_diversity":      (0.20, 0.15, 0.55, 0.18),
            "lexical_diversity":       (0.22, 0.12, 0.55, 0.15),
            "repetition":              (0.05, 0.06, 0.18, 0.10),
            "semantic_predictability": (0.12, 0.10, 0.45, 0.18),
            "syntactic_complexity":    (0.28, 0.18, 0.58, 0.15),
            "function_word_dist":      (0.30, 0.18, 0.62, 0.15),
            "entropy":                 (0.15, 0.10, 0.45, 0.15),
            "ngram_frequency_bias":    (0.25, 0.15, 0.50, 0.18),
            "over_coherence":          (0.42, 0.20, 0.72, 0.15),
            "emotional_variability":   (0.25, 0.18, 0.68, 0.15),
            "error_patterns":          (0.45, 0.20, 0.78, 0.12),
            "contextual_depth":        (0.22, 0.15, 0.72, 0.15),
            "stopword_patterning":     (0.35, 0.18, 0.62, 0.15),
            "bigram_perplexity":       (0.28, 0.12, 0.68, 0.12),
            "log_likelihood_variance": (0.28, 0.15, 0.70, 0.12),
            "vader_sentiment":         (0.22, 0.15, 0.68, 0.15),
            "compression_ratio":       (0.28, 0.15, 0.62, 0.15),
            "stylometric_fingerprint": (0.30, 0.15, 0.65, 0.12),
            "vocab_richness_curve":    (0.25, 0.15, 0.62, 0.15),
            "punctuation_diversity":   (0.28, 0.18, 0.62, 0.15),
            "hapax_ratio":             (0.25, 0.15, 0.60, 0.15),
            "readability_score":       (0.28, 0.18, 0.60, 0.15),
            "sentence_position_bias":  (0.28, 0.18, 0.62, 0.15),
        }

        half = n_samples // 2
        X_list = []
        y_list = []

        for label, is_ai in [(0, False), (1, True)]:
            n = half
            for _ in range(n):
                vec = []
                for feat in FEATURE_NAMES:
                    dist = _distributions.get(feat, (0.35, 0.15, 0.60, 0.15))
                    if is_ai:
                        mu, sigma = dist[2], dist[3]
                    else:
                        mu, sigma = dist[0], dist[1]
                    val = rng.normal(mu, sigma)
                    # Clip to [0,1] and add slight noise
                    val = float(np.clip(val + rng.normal(0, 0.02), 0.0, 1.0))
                    vec.append(val)

                # Inject realistic class-conditional correlations
                if is_ai:
                    # AI: high features are correlated (uniform style)
                    noise = rng.normal(0, 0.05)
                    vec = [min(1.0, max(0.0, v + noise * 0.3)) for v in vec]
                else:
                    # Human: more feature independence and variability
                    for i in range(len(vec)):
                        if rng.random() < 0.2:
                            vec[i] = float(np.clip(rng.uniform(0, 1), 0, 1))

                X_list.append(vec)
                y_list.append(label)

        # Shuffle
        combined = list(zip(X_list, y_list))
        rng.shuffle(combined)
        X_arr, y_arr = zip(*combined)

        return np.array(X_arr), np.array(y_arr)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _feature_dict_to_vector(self, feature_scores: Dict[str, float]) -> List[float]:
        return [feature_scores.get(name, 0.5) for name in FEATURE_NAMES]

    def _classify(self, prob: float) -> str:
        if prob >= 0.70:
            return "Likely AI"
        elif prob >= 0.30:
            return "Mixed / Uncertain"
        else:
            return "Human-like"

    def _confidence(self, prob: float) -> str:
        dist = abs(prob - 0.5)
        if dist >= 0.35:
            return "High"
        elif dist >= 0.20:
            return "Medium"
        else:
            return "Low"

    def _fallback_predict(self, feature_scores: Dict[str, float]) -> Dict[str, Any]:
        """Static weighted sum fallback when model is not trained."""
        weights = {
            "perplexity": 0.08, "burstiness": 0.07, "bigram_perplexity": 0.09,
            "log_likelihood_variance": 0.08, "lexical_diversity": 0.06,
            "repetition": 0.06, "semantic_predictability": 0.06,
            "syntactic_complexity": 0.05, "function_word_dist": 0.05,
            "entropy": 0.05, "over_coherence": 0.06, "vader_sentiment": 0.05,
            "error_patterns": 0.04, "contextual_depth": 0.05,
            "stylometric_fingerprint": 0.05, "compression_ratio": 0.05,
            "hapax_ratio": 0.05,
        }
        total_w = sum(weights.values())
        score = sum(weights.get(k, 0) * v for k, v in feature_scores.items()) / total_w
        score = min(1.0, max(0.0, score))
        return {
            "probability": round(score, 4),
            "score": round(score * 100, 2),
            "label": self._classify(score),
            "confidence": self._confidence(score),
            "method": "weighted_fallback",
        }

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._train_metrics

    @property
    def feature_importances(self) -> Dict[str, float]:
        return self._feature_importances


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark utilities
# ─────────────────────────────────────────────────────────────────────────────

class Benchmarker:
    """
    Evaluate the full pipeline against labeled examples.

    Usage:
        bench = Benchmarker()
        results = bench.run(labeled_examples)
        bench.print_report(results)
    """

    def run(
        self,
        labeled_examples: List[Dict[str, Any]],
        analyze_fn,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        labeled_examples : list of {"text": str, "label": 0|1}  (1 = AI)
        analyze_fn       : callable(text) → {"final_score": float, ...}
        """
        y_true = []
        y_pred = []
        y_prob = []
        errors = []

        for ex in labeled_examples:
            try:
                result = analyze_fn(ex["text"])
                prob = result["final_score"] / 100.0
                pred = 1 if prob >= 0.5 else 0
                y_true.append(ex["label"])
                y_pred.append(pred)
                y_prob.append(prob)
            except Exception as e:
                errors.append(str(e))

        if not y_true:
            return {"error": "No successful predictions"}

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_prob_arr = np.array(y_prob)

        acc = accuracy_score(y_true_arr, y_pred_arr)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_true_arr, y_prob_arr)
        except Exception:
            auc = 0.5
        cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

        return {
            "n_samples": len(y_true),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
            "confusion_matrix": cm,
            "errors": errors,
        }

    def print_report(self, results: Dict[str, Any]) -> None:
        print("\n" + "═"*50)
        print("  BENCHMARK RESULTS")
        print("═"*50)
        for k, v in results.items():
            if k not in ("confusion_matrix", "errors"):
                print(f"  {k:<25} {v}")
        if "confusion_matrix" in results:
            cm = results["confusion_matrix"]
            print(f"\n  Confusion Matrix:")
            print(f"            Pred Human  Pred AI")
            if len(cm) >= 2:
                print(f"  True Human    {cm[0][0]:>5}      {cm[0][1]:>5}")
                print(f"  True AI       {cm[1][0]:>5}      {cm[1][1]:>5}")
        print("═"*50)


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial robustness tests
# ─────────────────────────────────────────────────────────────────────────────

class AdversarialTester:
    """
    Tests pipeline robustness against common adversarial attacks.

    Attacks:
      1. Typo injection   — randomly swap/insert chars
      2. Synonym noise    — replace common words with near-synonyms
      3. Sentence shuffle — randomize sentence order
      4. Whitespace noise — add/remove extra spaces
      5. Punctuation strip — remove most punctuation
    """

    _SYNONYMS = {
        "important": "crucial", "however": "nevertheless", "therefore": "consequently",
        "shows": "demonstrates", "uses": "employs", "large": "substantial",
        "small": "minimal", "good": "beneficial", "bad": "detrimental",
        "many": "numerous", "often": "frequently", "very": "considerably",
        "the": "the", "a": "a",  # no-ops to avoid over-substitution
    }

    def inject_typos(self, text: str, rate: float = 0.03) -> str:
        words = text.split()
        rng = random.Random(42)
        result = []
        for word in words:
            if rng.random() < rate and len(word) > 3:
                op = rng.choice(["swap", "double", "drop"])
                if op == "swap" and len(word) > 2:
                    i = rng.randint(0, len(word)-2)
                    word = word[:i] + word[i+1] + word[i] + word[i+2:]
                elif op == "double":
                    i = rng.randint(0, len(word)-1)
                    word = word[:i] + word[i] + word[i:]
                elif op == "drop" and len(word) > 2:
                    i = rng.randint(0, len(word)-1)
                    word = word[:i] + word[i+1:]
            result.append(word)
        return " ".join(result)

    def inject_synonyms(self, text: str) -> str:
        for orig, syn in self._SYNONYMS.items():
            text = re.sub(r'\b' + orig + r'\b', syn, text, flags=re.IGNORECASE)
        return text

    def shuffle_sentences(self, text: str) -> str:
        from analyzer.features import _tokenize_sentences
        sents = _tokenize_sentences(text)
        if len(sents) < 3:
            return text
        rng = random.Random(42)
        rng.shuffle(sents)
        return " ".join(sents)

    def strip_punctuation(self, text: str) -> str:
        return re.sub(r'[.,;:!?—–]', '', text)

    def add_whitespace_noise(self, text: str) -> str:
        rng = random.Random(42)
        words = text.split()
        result = []
        for w in words:
            result.append(w)
            if rng.random() < 0.05:
                result.append("")  # double space
        return " ".join(result)

    def run_all(self, text: str, analyze_fn) -> Dict[str, Any]:
        """Run all attacks and report score stability."""
        baseline = analyze_fn(text)["final_score"]

        attacks = {
            "typo_injection":    self.inject_typos(text),
            "synonym_noise":     self.inject_synonyms(text),
            "sentence_shuffle":  self.shuffle_sentences(text),
            "punct_strip":       self.strip_punctuation(text),
            "whitespace_noise":  self.add_whitespace_noise(text),
        }

        results = {"baseline_score": baseline}
        for name, attacked_text in attacks.items():
            try:
                score = analyze_fn(attacked_text)["final_score"]
                delta = score - baseline
                results[name] = {
                    "score": round(score, 1),
                    "delta": round(delta, 1),
                    "robust": abs(delta) < 15.0,  # <15pt shift = robust
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        robust_count = sum(1 for k, v in results.items()
                           if isinstance(v, dict) and v.get("robust", False))
        results["robustness_score"] = f"{robust_count}/{len(attacks)}"
        return results

import re
