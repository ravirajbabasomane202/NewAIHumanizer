"""
analyzer/features_v2.py
-----------------------
Enhanced feature set — upgraded from v1.

New / improved features:
  16. BigramLMPerplexity     — real bigram LM with interpolation (replaces fake unigram PP)
  17. LogLikelihoodVariance  — variance of token log-probs (GPTZero-style signal)
  18. VADERSentiment         — rule-based VADER-style sentiment (no transformers needed)
  19. CompressionRatio       — zlib compression as complexity proxy
  20. StyloFingerprint       — stylometric feature bundle (avg word len, punct density, etc.)
  21. VocabRichnessCurve     — TTR curve slope (how fast vocab saturates)
  22. SentencePositionBias   — do sentence lengths follow a suspicious pattern?
  23. PunctuationDiversity   — entropy of punctuation marks used
  24. HapaxRatio             — ratio of words appearing exactly once
  25. ReadabilityScore       — Flesch-Kincaid approximation

All return score ∈ [0,1] where 1 = most AI-like.
"""

import re
import math
import zlib
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

from analyzer.features import (
    Feature, _tokenize_sentences, _tokenize_words,
    _ngrams, _safe_div, _sigmoid, _clip01, _STOPWORDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 16 · Bigram Language Model Perplexity
# ─────────────────────────────────────────────────────────────────────────────

class BigramLMPerplexityFeature(Feature):
    """
    True bigram LM perplexity with Kneser-Ney-style interpolation.

    Math:
        P_interp(w_i | w_{i-1}) = λ · P_bigram + (1-λ) · P_unigram
        PP = exp( -1/N Σ log P_interp(w_i | w_{i-1}) )

    Normalization: sigmoid-inverse centred at log(30).
    Lower PP → more predictable → AI-like → higher score.

    Advantage over v1: bigram context captures phrase-level predictability,
    which is a stronger signal than unigram alone.
    """
    name = "bigram_perplexity"
    description = "Bigram LM perplexity — captures phrase-level predictability"

    LAMBDA = 0.7   # interpolation weight for bigram vs unigram

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        tokens = _tokenize_words(text)
        if len(tokens) < 10:
            return self._empty("Too few words for bigram LM")

        # Unigram counts
        uni_counts = Counter(tokens)
        total_uni = len(tokens)
        vocab_size = len(uni_counts)

        # Bigram counts
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        bi_counts: Dict[Tuple, int] = Counter(bigrams)
        # Context counts (how many times w_{i-1} appeared before any word)
        ctx_counts: Counter = Counter(t for t, _ in bigrams)

        def log_prob_interp(w_prev: str, w_cur: str) -> float:
            # Bigram probability (Laplace smoothed)
            bi_p = (bi_counts.get((w_prev, w_cur), 0) + 1) / (ctx_counts.get(w_prev, 0) + vocab_size)
            # Unigram probability
            uni_p = (uni_counts.get(w_cur, 0) + 1) / (total_uni + vocab_size)
            # Interpolated
            p = self.LAMBDA * bi_p + (1 - self.LAMBDA) * uni_p
            return math.log(max(p, 1e-12))

        def sent_pp(sent: str) -> float:
            words = _tokenize_words(sent)
            if len(words) < 2:
                return 30.0
            log_sum = sum(
                log_prob_interp(words[i], words[i+1])
                for i in range(len(words) - 1)
            )
            return math.exp(-log_sum / max(len(words) - 1, 1))

        pps = [sent_pp(s) for s in sentences]
        avg_pp = sum(pps) / max(len(pps), 1)
        score = 1.0 - _sigmoid(math.log(max(avg_pp, 1e-9)), k=0.9, x0=math.log(30))

        ss = [round(1.0 - _sigmoid(math.log(max(p, 1e-9)), k=0.9, x0=math.log(30)), 4) for p in pps]

        return {
            "raw": round(avg_pp, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"Bigram PP≈{avg_pp:.1f}. {'Predictable (AI-like)' if score > 0.6 else 'Variable (human-like)'}"
        }


# ─────────────────────────────────────────────────────────────────────────────
# 17 · Log-Likelihood Variance
# ─────────────────────────────────────────────────────────────────────────────

class LogLikelihoodVarianceFeature(Feature):
    """
    Variance of per-sentence log-likelihood scores (GPTZero-inspired signal).

    Math:
        LL(s) = mean log P_bigram(w_i | w_{i-1}) for sentence s
        signal = std(LL across sentences)

    Low variance → flat, uniformly predictable text → AI-like.
    Normalization: 1 - sigmoid(std, k=10, x0=0.5).
    """
    name = "log_likelihood_variance"
    description = "Variance of per-sentence log-likelihood (GPTZero-style signal)"

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        if len(sentences) < 3:
            return self._empty("Need ≥3 sentences")

        tokens = _tokenize_words(text)
        if len(tokens) < 10:
            return self._empty()

        uni = Counter(tokens); total = len(tokens); vocab = len(uni)
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        bi_cnt = Counter(bigrams)
        ctx_cnt = Counter(t for t, _ in bigrams)

        def sent_ll(sent: str) -> float:
            words = _tokenize_words(sent)
            if len(words) < 2:
                return -3.0
            lp = []
            for i in range(len(words) - 1):
                bi_p = (bi_cnt.get((words[i], words[i+1]), 0) + 1) / (ctx_cnt.get(words[i], 0) + vocab)
                uni_p = (uni.get(words[i+1], 0) + 1) / (total + vocab)
                p = 0.7 * bi_p + 0.3 * uni_p
                lp.append(math.log(max(p, 1e-12)))
            return sum(lp) / max(len(lp), 1)

        lls = [sent_ll(s) for s in sentences]
        mean_ll = sum(lls) / len(lls)
        std_ll = math.sqrt(sum((x - mean_ll) ** 2 for x in lls) / len(lls))

        # Low variance → AI-like
        score = 1.0 - _sigmoid(std_ll, k=8.0, x0=0.4)

        ss = []
        for ll in lls:
            dev = abs(ll - mean_ll)
            ss.append(round(1.0 - _clip01(dev, 0, 2.0), 4))

        return {
            "raw": round(std_ll, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"LL std={std_ll:.3f}. {'Uniform likelihood (AI-like)' if score > 0.6 else 'Variable likelihood (human)'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 18 · VADER-style Sentiment (rule-based, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

class VADERSentimentFeature(Feature):
    """
    Rule-based compound sentiment score per sentence using a curated lexicon
    + valence modifiers (negation, intensifiers, punctuation boosts).

    Replaces the keyword-counting v1 approach with a proper valence model.

    Score: std(sentiment_per_sentence) — low std = flat = AI-like.
    Normalization: 1 - sigmoid(std, k=15, x0=0.04).
    """
    name = "vader_sentiment"
    description = "VADER-style sentiment variance across sentences"

    # Compact sentiment lexicon: word → valence [-1, 1]
    _LEXICON: Dict[str, float] = {
        # Strongly positive
        "excellent": 1.0, "outstanding": 1.0, "superb": 1.0, "wonderful": 0.9,
        "amazing": 0.9, "fantastic": 0.9, "brilliant": 0.85, "great": 0.8,
        "love": 0.8, "perfect": 0.85, "best": 0.8, "incredible": 0.9,
        "happy": 0.75, "joy": 0.8, "delightful": 0.8, "beautiful": 0.75,
        "nice": 0.65, "good": 0.65, "helpful": 0.6, "positive": 0.65,
        "success": 0.7, "win": 0.7, "glad": 0.7, "pleased": 0.65,
        "fun": 0.65, "enjoy": 0.7, "safe": 0.6, "strong": 0.55,
        "proud": 0.7, "grateful": 0.75, "excited": 0.75, "hope": 0.6,
        # Mildly positive
        "fine": 0.3, "okay": 0.25, "decent": 0.35, "useful": 0.4,
        "interesting": 0.4, "clear": 0.3, "easy": 0.35, "better": 0.5,
        # Strongly negative
        "terrible": -1.0, "awful": -0.95, "horrible": -0.95, "dreadful": -0.9,
        "hate": -0.9, "worst": -0.9, "disgusting": -0.9, "miserable": -0.85,
        "tragic": -0.8, "disaster": -0.8, "fail": -0.75, "failure": -0.8,
        "bad": -0.75, "sad": -0.7, "angry": -0.7, "fear": -0.65,
        "ugly": -0.65, "stupid": -0.7, "broken": -0.6, "wrong": -0.55,
        "danger": -0.65, "pain": -0.7, "hurt": -0.65, "weak": -0.5,
        "poor": -0.55, "sick": -0.6, "dark": -0.4, "cold": -0.3,
        # Mildly negative
        "problem": -0.45, "difficult": -0.4, "hard": -0.3, "concern": -0.35,
        "issue": -0.3, "risk": -0.4, "mistake": -0.5, "wrong": -0.55,
    }

    _NEGATORS = {"not","no","never","neither","nor","nothing","nobody","nowhere",
                 "n't","cannot","without","lack","lacks","lacking","absent"}
    _INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "incredibly": 1.4,
                     "absolutely": 1.4, "quite": 1.1, "rather": 1.1,
                     "really": 1.2, "truly": 1.2, "highly": 1.2,
                     "somewhat": 0.8, "slightly": 0.7, "barely": 0.5}

    def _sentence_valence(self, sent: str) -> float:
        words = _tokenize_words(sent)
        if not words:
            return 0.0

        valences = []
        for i, word in enumerate(words):
            if word not in self._LEXICON:
                continue
            val = self._LEXICON[word]

            # Check preceding word for negation (window=3)
            window = words[max(0, i-3):i]
            negated = any(w in self._NEGATORS for w in window)
            if negated:
                val *= -0.74  # VADER-style negation dampening

            # Intensifiers (preceding 1-2 words)
            for j in range(max(0, i-2), i):
                mult = self._INTENSIFIERS.get(words[j], None)
                if mult is not None:
                    val *= mult
                    break

            valences.append(val)

        if not valences:
            return 0.0

        # VADER compound-style aggregation
        raw_sum = sum(valences)
        # Normalize to [-1, 1]
        alpha = 15.0
        compound = raw_sum / math.sqrt(raw_sum ** 2 + alpha)
        return compound

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        if len(sentences) < 2:
            return self._empty()

        sentiments = [self._sentence_valence(s) for s in sentences]
        mean_s = sum(sentiments) / len(sentiments)
        std_s = math.sqrt(sum((s - mean_s) ** 2 for s in sentiments) / len(sentiments))

        # Low variance → AI-like
        score = 1.0 - _sigmoid(std_s, k=15.0, x0=0.05)

        ss = [round(1.0 - _clip01(abs(s - mean_s), 0, 0.5), 4) for s in sentiments]

        return {
            "raw": round(std_s, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"Sentiment std={std_s:.3f} (VADER-style). {'Flat emotion (AI-like)' if score > 0.6 else 'Emotional range (human)'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 19 · Compression Ratio
# ─────────────────────────────────────────────────────────────────────────────

class CompressionRatioFeature(Feature):
    """
    Compression-based complexity detection.

    Math:
        ratio = len(zlib.compress(text_bytes)) / len(text_bytes)

    High compression (low ratio) → high redundancy / repetition → AI-like.
    Normalization: score = 1 - ratio (clipped to [0.3, 0.9] range first).

    Intuition: AI text tends to be more compressible because it uses
    predictable phrase patterns; human text is informationally denser.
    """
    name = "compression_ratio"
    description = "zlib compression ratio — high compressibility = AI-like redundancy"

    def compute(self, text: str) -> Dict[str, Any]:
        if len(text) < 50:
            return self._empty("Text too short for compression analysis")

        encoded = text.encode("utf-8")
        compressed = zlib.compress(encoded, level=9)
        ratio = len(compressed) / len(encoded)

        # Map: typical human text ≈ 0.55-0.75, AI text ≈ 0.35-0.55
        score = 1.0 - _clip01(ratio, 0.30, 0.80)

        sentences = _tokenize_sentences(text)
        ss = []
        for sent in sentences:
            if len(sent) < 20:
                ss.append(round(score, 4))
                continue
            enc = sent.encode("utf-8")
            comp = zlib.compress(enc, level=9)
            r = len(comp) / len(enc)
            ss.append(round(1.0 - _clip01(r, 0.3, 0.9), 4))

        return {
            "raw": round(ratio, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"Compression ratio={ratio:.3f}. {'High redundancy (AI-like)' if score > 0.5 else 'Dense (human-like)'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 20 · Stylometric Fingerprint
# ─────────────────────────────────────────────────────────────────────────────

class StyloFingerprintFeature(Feature):
    """
    Bundle of stylometric micro-features aggregated into one score.

    Sub-features (each normalized to [0,1]):
        (a) avg_word_length — AI tends toward moderate lengths (4-7 chars)
        (b) punct_density  — AI uses punctuation at very predictable rates
        (c) digit_ratio    — ratio of token-digit presence
        (d) all_caps_ratio — shouty words more human
        (e) contraction_ratio — "don't", "I've" → more human

    Final = weighted average of sub-feature AI-signals.
    """
    name = "stylometric_fingerprint"
    description = "Stylometric micro-features bundle (word length, punctuation, contractions)"

    _CONTRACTIONS = {
        "don't","doesn't","didn't","won't","wouldn't","can't","couldn't",
        "i'm","i've","i'll","i'd","you're","you've","you'll","they're",
        "we're","we've","it's","that's","there's","what's","who's",
        "isn't","aren't","wasn't","weren't","haven't","hadn't","shouldn't",
    }

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        raw_tokens = text.split()
        alpha_tokens = _tokenize_words(text)

        if not raw_tokens or not alpha_tokens:
            return self._empty()

        # (a) Avg word length — AI: typically 4.5-5.5 chars (moderate, clean)
        avg_wl = sum(len(w) for w in alpha_tokens) / len(alpha_tokens)
        # Humans skew either shorter (informal) or longer (academic)
        # AI clusters in the middle → signal via distance from extremes
        wl_signal = 1.0 - abs(avg_wl - 5.0) / 4.0
        wl_signal = max(0.0, min(1.0, wl_signal))

        # (b) Punctuation density per word
        punct_chars = sum(1 for c in text if c in string.punctuation)
        punct_density = _safe_div(punct_chars, max(len(raw_tokens), 1))
        # AI tends to use exactly the "right" amount (0.3-0.6 punct/word)
        punct_signal = 1.0 - abs(punct_density - 0.45) / 0.45
        punct_signal = max(0.0, min(1.0, punct_signal))

        # (c) Digit ratio — humans use more numbers in natural writing
        digit_count = sum(1 for t in raw_tokens if any(c.isdigit() for c in t))
        digit_ratio = _safe_div(digit_count, len(raw_tokens))
        digit_signal = 1.0 - _sigmoid(digit_ratio, k=20, x0=0.05)

        # (d) All-caps ratio — informal emphasis, more human
        caps_count = sum(1 for t in raw_tokens if t.isupper() and len(t) > 1)
        caps_ratio = _safe_div(caps_count, len(raw_tokens))
        caps_signal = 1.0 - _sigmoid(caps_ratio, k=30, x0=0.02)

        # (e) Contractions — strongly human
        text_lower = text.lower()
        contr_count = sum(1 for c in self._CONTRACTIONS if c in text_lower)
        contr_ratio = _safe_div(contr_count, len(alpha_tokens))
        contr_signal = 1.0 - _sigmoid(contr_ratio, k=30, x0=0.03)

        # Weighted aggregate
        weights = [0.25, 0.2, 0.15, 0.15, 0.25]
        signals = [wl_signal, punct_signal, digit_signal, caps_signal, contr_signal]
        score = sum(w * s for w, s in zip(weights, signals))

        # Per-sentence
        ss = []
        for sent in sentences:
            words = _tokenize_words(sent)
            raw = sent.split()
            if not words:
                ss.append(0.5); continue
            awl = sum(len(w) for w in words) / len(words)
            wls = max(0.0, min(1.0, 1.0 - abs(awl - 5.0) / 4.0))
            contr_s = sum(1 for c in self._CONTRACTIONS if c in sent.lower())
            cs = 1.0 - _sigmoid(_safe_div(contr_s, max(len(words), 1)), k=30, x0=0.03)
            ss.append(round((wls + cs) / 2, 4))

        return {
            "raw": round(score, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": (
                f"Stylometric bundle: avg_wlen={avg_wl:.1f}, "
                f"punct_dens={punct_density:.2f}, contr_ratio={contr_ratio:.3f}. "
                f"{'AI-pattern' if score > 0.6 else 'Human-pattern'}."
            )
        }


# ─────────────────────────────────────────────────────────────────────────────
# 21 · Vocabulary Richness Curve (TTR slope)
# ─────────────────────────────────────────────────────────────────────────────

class VocabRichnessCurveFeature(Feature):
    """
    Measures how quickly vocabulary saturates as text grows.

    Math:
        Sample TTR at positions [10%, 20%, ..., 100%] of text.
        Fit slope of TTR vs position.
        Steep negative slope → vocab saturates fast → AI-like.

    Normalization: score = sigmoid(-slope, k=5, x0=0.5).
    """
    name = "vocab_richness_curve"
    description = "Vocabulary saturation rate (TTR slope over text progression)"

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if len(tokens) < 20:
            return self._empty()

        n = len(tokens)
        steps = 10
        ttrs = []
        xs = []
        for i in range(1, steps + 1):
            end = max(1, i * n // steps)
            window = tokens[:end]
            ttrs.append(len(set(window)) / len(window))
            xs.append(i)

        # Linear regression slope
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ttrs) / len(ttrs)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ttrs))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = _safe_div(num, den)  # negative = vocab saturates → AI-like

        # More negative slope = faster saturation = AI-like = higher score
        score = _sigmoid(-slope, k=4.0, x0=0.3)
        score = max(0.0, min(1.0, score))

        sentences = _tokenize_sentences(text)
        ss = [round(score, 4)] * len(sentences)

        return {
            "raw": round(slope, 5),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"TTR slope={slope:.4f}. {'Fast saturation (AI-like)' if score > 0.6 else 'Sustained vocab growth (human)'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 22 · Punctuation Diversity
# ─────────────────────────────────────────────────────────────────────────────

class PunctuationDiversityFeature(Feature):
    """
    Shannon entropy of punctuation marks used.

    Math:
        H_punct = -Σ p(mark) * log2(p(mark))

    Low entropy → monotonous punctuation → AI-like.
    Normalization: 1 - H_norm (normalized by log2(unique_marks)).
    """
    name = "punctuation_diversity"
    description = "Entropy of punctuation mark distribution"

    _PUNCT_SET = set(".,;:!?—–-()[]{}\"'…")

    def compute(self, text: str) -> Dict[str, Any]:
        punct = [c for c in text if c in self._PUNCT_SET]
        if len(punct) < 5:
            return self._empty("Too little punctuation")

        counts = Counter(punct)
        total = len(punct)
        H = -sum((c / total) * math.log2(c / total) for c in counts.values())
        max_H = math.log2(len(counts)) if len(counts) > 1 else 1.0
        norm = _safe_div(H, max_H)
        score = 1.0 - norm  # Low diversity → AI-like

        sentences = _tokenize_sentences(text)
        ss = []
        for sent in sentences:
            sp = [c for c in sent if c in self._PUNCT_SET]
            if len(sp) < 2:
                ss.append(round(score, 4)); continue
            cnt = Counter(sp)
            tot = len(sp)
            h = -sum((c/tot)*math.log2(c/tot) for c in cnt.values())
            mh = math.log2(len(cnt)) if len(cnt) > 1 else 1.0
            ss.append(round(1.0 - _safe_div(h, mh), 4))

        return {
            "raw": round(norm, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"Punct entropy={norm:.3f}. {'Monotonous (AI-like)' if score > 0.6 else 'Diverse punctuation'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 23 · Hapax Ratio
# ─────────────────────────────────────────────────────────────────────────────

class HapaxRatioFeature(Feature):
    """
    Ratio of hapax legomena (words appearing exactly once).

    Math:
        hapax_ratio = |{w : count(w) == 1}| / |vocabulary|

    Low hapax ratio → repetitive vocabulary → AI-like.
    Normalization: score = 1 - hapax_ratio (inverted, clipped).
    """
    name = "hapax_ratio"
    description = "Ratio of words appearing exactly once (vocabulary uniqueness)"

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if len(tokens) < 20:
            return self._empty()

        counts = Counter(tokens)
        hapax_count = sum(1 for c in counts.values() if c == 1)
        hapax_ratio = _safe_div(hapax_count, len(counts))

        # High hapax → unique vocabulary → human-like → low score
        score = 1.0 - hapax_ratio

        sentences = _tokenize_sentences(text)
        ss = []
        for sent in sentences:
            words = _tokenize_words(sent)
            if not words:
                ss.append(0.5); continue
            cnt = Counter(words)
            hr = _safe_div(sum(1 for c in cnt.values() if c == 1), max(len(cnt), 1))
            ss.append(round(1.0 - hr, 4))

        return {
            "raw": round(hapax_ratio, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"Hapax ratio={hapax_ratio:.3f}. {'Low uniqueness (AI-like)' if score > 0.5 else 'High uniqueness (human)'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 24 · Readability Score (Flesch-Kincaid Grade approximation)
# ─────────────────────────────────────────────────────────────────────────────

class ReadabilityScoreFeature(Feature):
    """
    Flesch-Kincaid Grade Level approximation.

    Math:
        FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    AI text tends to target mid-range readability (grade 8-12).
    Very high or very low FK → more human-like (extremes are human).
    Signal: distance from the "AI sweet spot" (grade 9-11).
    score = 1 - sigmoid(|FK - 10|, k=0.3, x0=3.0)
    """
    name = "readability_score"
    description = "Flesch-Kincaid grade — AI clusters in mid-range readability"

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Rough syllable count via vowel-group detection."""
        word = word.lower().strip(".,;:!?\"'")
        if not word:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for c in word:
            is_v = c in vowels
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        # Silent-e rule
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        tokens = _tokenize_words(text)
        if not sentences or not tokens:
            return self._empty()

        n_sents = max(len(sentences), 1)
        n_words = max(len(tokens), 1)
        n_syllables = sum(self._count_syllables(w) for w in tokens)

        fk = 0.39 * (n_words / n_sents) + 11.8 * (n_syllables / n_words) - 15.59
        # AI sweet spot ~9-11; humans are more spread out
        dist = abs(fk - 10.0)
        score = 1.0 - _sigmoid(dist, k=0.35, x0=3.0)
        score = max(0.0, min(1.0, score))

        ss = []
        for sent in sentences:
            words = _tokenize_words(sent)
            if not words:
                ss.append(0.5); continue
            syll = sum(self._count_syllables(w) for w in words)
            fks = 0.39 * len(words) + 11.8 * (syll / len(words)) - 15.59
            d = abs(fks - 10.0)
            ss.append(round(1.0 - _sigmoid(d, k=0.35, x0=3.0), 4))

        return {
            "raw": round(fk, 2),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"FK grade≈{fk:.1f}. {'Mid-range (AI-typical)' if score > 0.5 else 'Extreme grade (human-typical)'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# 25 · Sentence Position Bias
# ─────────────────────────────────────────────────────────────────────────────

class SentencePositionBiasFeature(Feature):
    """
    Detects suspicious length patterns across sentence positions.

    AI models often produce:
      - First sentences that are markedly longer (topic-sentence pattern)
      - Consistent "intro-body-conclusion" structure that is length-detectable

    Math:
        First-third mean vs last-third mean vs middle-third mean.
        Low variance among thirds AND high first-sent dominance → AI signal.

    score = blend of (low variance signal) + (first-sentence length ratio).
    """
    name = "sentence_position_bias"
    description = "Suspicious sentence-length patterns across text position"

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        if len(sentences) < 4:
            return self._empty()

        lengths = [len(_tokenize_words(s)) for s in sentences]
        n = len(lengths)
        t = max(n // 3, 1)

        thirds = [
            sum(lengths[:t]) / max(t, 1),
            sum(lengths[t:2*t]) / max(t, 1),
            sum(lengths[2*t:]) / max(n - 2*t, 1),
        ]

        mean_t = sum(thirds) / 3
        std_t = math.sqrt(sum((x - mean_t) ** 2 for x in thirds) / 3)

        # Low variance across thirds → structured, AI-like
        low_var_signal = 1.0 - _sigmoid(std_t, k=0.5, x0=3.0)

        # First sentence dominance (AI often starts with long "topic" sentence)
        first_len = lengths[0]
        overall_mean = sum(lengths) / n
        first_dom = _clip01(first_len / (overall_mean + 1e-9), 0.8, 2.5)

        score = 0.6 * low_var_signal + 0.4 * first_dom
        score = max(0.0, min(1.0, score))

        ss = []
        for i, l in enumerate(lengths):
            dev_from_mean = abs(l - overall_mean) / (overall_mean + 1e-9)
            ss.append(round(1.0 - _clip01(dev_from_mean, 0, 1.5), 4))

        return {
            "raw": round(std_t, 4),
            "score": round(score, 4),
            "sentence_scores": ss,
            "description": f"Positional length std={std_t:.2f}. {'Structured pattern (AI-like)' if score > 0.5 else 'Natural structure'}."
        }


# ─────────────────────────────────────────────────────────────────────────────
# V2 Feature registry
# ─────────────────────────────────────────────────────────────────────────────

V2_FEATURES = [
    BigramLMPerplexityFeature(),
    LogLikelihoodVarianceFeature(),
    VADERSentimentFeature(),
    CompressionRatioFeature(),
    StyloFingerprintFeature(),
    VocabRichnessCurveFeature(),
    PunctuationDiversityFeature(),
    HapaxRatioFeature(),
    ReadabilityScoreFeature(),
    SentencePositionBiasFeature(),
]

V2_FEATURE_MAP = {f.name: f for f in V2_FEATURES}
