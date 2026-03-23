"""
analyzer/features.py
--------------------
15 linguistic/statistical features for AI text detection.

Uses NLTK when available, falls back to pure-stdlib implementations.
All features output scores normalized to [0, 1].
"""

import re
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Optional NLTK with graceful fallback
# ---------------------------------------------------------------------------

_NLTK_AVAILABLE = False
try:
    import nltk
    _RESOURCES = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt",     "punkt"),
        ("corpora/stopwords",    "stopwords"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger",     "averaged_perceptron_tagger"),
    ]
    for _path, _name in _RESOURCES:
        try:
            nltk.data.find(_path)
        except LookupError:
            try:
                nltk.download(_name, quiet=True)
            except Exception:
                pass
    from nltk.tokenize import sent_tokenize as _nltk_sent
    from nltk.tokenize import word_tokenize as _nltk_word
    from nltk.corpus import stopwords as _nltk_sw
    from nltk.tag import pos_tag as _nltk_pos
    _NLTK_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Stopwords (hardcoded fallback so NLTK is not required)
# ---------------------------------------------------------------------------

_SW_FALLBACK = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","can","will","just","should","now","don","ll","re","ve",
}

def _get_stopwords():
    if _NLTK_AVAILABLE:
        try:
            return set(_nltk_sw.words("english"))
        except Exception:
            pass
    return _SW_FALLBACK

_STOPWORDS = _get_stopwords()

_COMMON_UNIGRAMS = {
    "the","be","to","of","and","a","in","that","have","it","for","not","on",
    "with","he","as","you","do","at","this","but","his","by","from","they",
    "we","say","her","she","or","an","will","my","one","all","would","there",
    "their","what","so","up","out","if","about","who","get","which","go","me",
    "when","make","can","like","time","no","just","him","know","take","people",
    "into","year","your","good","some","could","them","see","other","than",
    "then","now","look","only","come","its","over","think","also","back",
    "after","use","two","how","our","work","first","well","way","even","new",
    "want","because","any","these","give","day","most","us",
}

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _tokenize_sentences(text: str) -> List[str]:
    if _NLTK_AVAILABLE:
        try:
            return [s.strip() for s in _nltk_sent(text.strip()) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]


def _tokenize_words(text: str) -> List[str]:
    if _NLTK_AVAILABLE:
        try:
            return [t for t in _nltk_word(text.lower()) if t.isalpha()]
        except Exception:
            pass
    return re.findall(r"[a-zA-Z]+", text.lower())


def _pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    if _NLTK_AVAILABLE:
        try:
            return _nltk_pos(tokens)
        except Exception:
            pass
    # Simple heuristic fallback
    _jj_sfx = ("ful","less","ous","ive","al","ible","able","ic")
    _vb_sfx = ("ate","ize","ify","ise","ed","ing")
    _nn_sfx = ("tion","ment","ness","ity","ism","ance","ence","ship")
    result = []
    for t in tokens:
        tl = t.lower()
        if tl in _STOPWORDS:
            tag = "DT"
        elif tl.endswith(_jj_sfx):
            tag = "JJ"
        elif tl.endswith(_vb_sfx):
            tag = "VB"
        elif tl.endswith(_nn_sfx):
            tag = "NN"
        else:
            tag = "NN"
        result.append((t, tag))
    return result

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def _sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if x < x0 else 1.0

def _clip01(v: float, lo: float, hi: float) -> float:
    if hi == lo: return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Feature(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def compute(self, text: str) -> Dict[str, Any]:
        pass

    def _empty(self, reason: str = "Insufficient text") -> Dict[str, Any]:
        return {"raw": 0.0, "score": 0.5, "sentence_scores": [], "description": reason}

# ---------------------------------------------------------------------------
# 1 · Perplexity
# ---------------------------------------------------------------------------

class PerplexityFeature(Feature):
    """
    Laplace-smoothed unigram perplexity.
    PP = exp(-1/N Σ log P(w_i))
    Low PP → predictable → AI-like → score near 1.
    Normalization: sigmoid-inverse centred at log(50).
    """
    name = "perplexity"
    description = "Approximate perplexity — low = predictable (AI-like)"

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        tokens = _tokenize_words(text)
        if len(tokens) < 5:
            return self._empty("Too few words")

        total = len(tokens); vocab = len(set(tokens))
        counts = Counter(tokens)

        def pp(sent):
            words = _tokenize_words(sent)
            if not words: return 50.0
            lp = sum(math.log((counts.get(w,0)+1)/(total+vocab)) for w in words)
            return math.exp(-lp / len(words))

        pps = [pp(s) for s in sentences]
        avg = sum(pps) / max(len(pps), 1)
        score = 1.0 - _sigmoid(math.log(max(avg,1e-9)), k=0.8, x0=math.log(50))
        ss = [round(1.0 - _sigmoid(math.log(max(p,1e-9)), k=0.8, x0=math.log(50)), 4) for p in pps]

        return {"raw": round(avg,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Avg PP≈{avg:.1f}. {'Predictable (AI)' if score>0.6 else 'Variable (human)'}."}


# ---------------------------------------------------------------------------
# 2 · Burstiness
# ---------------------------------------------------------------------------

class BurstinessFeature(Feature):
    """
    CoV = std(lengths) / mean(lengths).
    Low CoV → uniform sentences → AI-like.
    """
    name = "burstiness"
    description = "Sentence length variation — low = uniform (AI-like)"

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        if len(sentences) < 3:
            return self._empty("Need ≥3 sentences")
        lengths = [len(_tokenize_words(s)) for s in sentences]
        mean = sum(lengths) / len(lengths)
        if mean == 0: return self._empty()
        std = math.sqrt(sum((l-mean)**2 for l in lengths) / len(lengths))
        bursty = _safe_div(std, mean)
        score = 1.0 - _sigmoid(bursty, k=5.0, x0=0.4)
        ss = [round(1.0 - _clip01(abs(l-mean)/(mean+1e-9), 0, 1.5), 4) for l in lengths]
        return {"raw": round(bursty,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"CoV={bursty:.3f}. {'Uniform (AI)' if score>0.6 else 'Variable (human)'}."}


# ---------------------------------------------------------------------------
# 3 · Sentence Structure Diversity
# ---------------------------------------------------------------------------

class SentenceDiversityFeature(Feature):
    """
    D = unique_POS_sequences / total_sentences.
    Low D → repetitive structure → AI-like → score = 1-D.
    """
    name = "sentence_diversity"
    description = "POS-tag sequence uniqueness ratio"

    def compute(self, text: str) -> Dict[str, Any]:
        sentences = _tokenize_sentences(text)
        if len(sentences) < 2: return self._empty()
        seqs = []
        for s in sentences:
            words = _tokenize_words(s)
            seqs.append(tuple(t for _,t in _pos_tag(words)) if words else ())
        diversity = _safe_div(len(set(seqs)), len(seqs))
        score = 1.0 - diversity
        counts = Counter(seqs)
        ss = [round(_clip01(counts[q]-1, 0, 5)*0.8, 4) for q in seqs]
        return {"raw": round(diversity,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"POS diversity={diversity:.3f}. {'Repetitive' if score>0.5 else 'Varied'}."}


# ---------------------------------------------------------------------------
# 4 · Lexical Diversity
# ---------------------------------------------------------------------------

class LexicalDiversityFeature(Feature):
    """
    MTTR via sliding window.
    Low MTTR → limited vocab → AI-like → score = 1 - MTTR.
    """
    name = "lexical_diversity"
    description = "Vocabulary richness (Moving TTR)"

    def __init__(self, window_size: int = 50):
        self.w = window_size

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if not tokens: return self._empty()
        if len(tokens) < self.w // 2:
            ttr = len(set(tokens)) / len(tokens)
            return {"raw": round(ttr,4), "score": round(1-ttr,4), "sentence_scores": [],
                    "description": f"TTR={ttr:.3f} (short text)."}
        w = self.w
        ttrs = [len(set(tokens[i:i+w]))/w for i in range(0, len(tokens)-w+1, max(1,w//2))]
        mttr = sum(ttrs) / max(len(ttrs),1)
        score = 1.0 - mttr
        sents = _tokenize_sentences(text)
        ss = []
        for s in sents:
            words = _tokenize_words(s)
            if not words: ss.append(0.5); continue
            ss.append(round(1.0 - len(set(words))/len(words), 4))
        return {"raw": round(mttr,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"MTTR={mttr:.3f}. {'Limited vocab (AI)' if score>0.5 else 'Rich vocab'}."}


# ---------------------------------------------------------------------------
# 5 · Repetition
# ---------------------------------------------------------------------------

class RepetitionFeature(Feature):
    """
    R = (total-unique_ngrams) / total_ngrams. Already in [0,1].
    """
    name = "repetition"
    description = "Repeated n-gram density"

    def __init__(self, n: int = 3):
        self.n = n

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if len(tokens) < self.n + 2: return self._empty()
        ng = _ngrams(tokens, self.n)
        counts = Counter(ng)
        repeated = sum(c-1 for c in counts.values() if c > 1)
        ratio = _safe_div(repeated, len(ng))
        sents = _tokenize_sentences(text)
        ss = []
        for s in sents:
            words = _tokenize_words(s)
            if len(words) < self.n: ss.append(0.0); continue
            sngs = _ngrams(words, self.n)
            cnt = Counter(sngs)
            rep = sum(c-1 for c in cnt.values() if c > 1)
            ss.append(round(_safe_div(rep, max(len(sngs),1)), 4))
        return {"raw": round(ratio,4), "score": round(ratio,4), "sentence_scores": ss,
                "description": f"Repeated {self.n}-gram ratio={ratio:.3f}."}


# ---------------------------------------------------------------------------
# 6 · Semantic Predictability
# ---------------------------------------------------------------------------

class SemanticPredictabilityFeature(Feature):
    """
    SP = mean cosine_sim(TF-IDF(s_i), TF-IDF(s_{i+1})).
    High SP → over-smooth → AI-like → score = SP.
    """
    name = "semantic_predictability"
    description = "Average adjacent-sentence TF-IDF similarity"

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        if len(sents) < 2: return self._empty()
        N = len(sents)
        doc_freq: Counter = Counter()
        for s in sents:
            doc_freq.update(set(_tokenize_words(s)))
        idf = {w: math.log((N+1)/(df+1))+1 for w,df in doc_freq.items()}

        def vec(s):
            words = _tokenize_words(s)
            if not words: return {}
            tf = Counter(words); tot = len(words)
            return {w: (c/tot)*idf.get(w,1.0) for w,c in tf.items()}

        def cos(v1, v2):
            common = set(v1) & set(v2)
            if not common: return 0.0
            dot = sum(v1[w]*v2[w] for w in common)
            n1 = math.sqrt(sum(x**2 for x in v1.values()))
            n2 = math.sqrt(sum(x**2 for x in v2.values()))
            return _safe_div(dot, n1*n2)

        vecs = [vec(s) for s in sents]
        sims = [cos(vecs[i], vecs[i+1]) for i in range(N-1)]
        avg = sum(sims) / max(len(sims),1)
        ss = [round(sims[i] if i < len(sims) else (sims[-1] if sims else 0.0), 4) for i in range(N)]
        return {"raw": round(avg,4), "score": round(avg,4), "sentence_scores": ss,
                "description": f"Avg similarity={avg:.3f}. {'High (AI)' if avg>0.4 else 'Natural flow'}."}


# ---------------------------------------------------------------------------
# 7 · Syntactic Complexity
# ---------------------------------------------------------------------------

class SyntacticComplexityFeature(Feature):
    """
    clause_density = (1 + subordinators + commas//2) / word_count.
    Low → simple → AI-like.
    """
    name = "syntactic_complexity"
    description = "Clause density per sentence"
    _SUBS = {"although","because","since","while","if","unless","until","when",
             "where","which","who","whom","whose","that","whether","though",
             "after","before","as","once","whenever"}

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        if not sents: return self._empty()

        def cx(s):
            words = _tokenize_words(s)
            if not words: return 0.0
            subs = sum(1 for w in words if w in self._SUBS)
            clauses = 1 + subs + s.count(",")//2
            return _safe_div(clauses, len(words))

        comps = [cx(s) for s in sents]
        avg = sum(comps) / max(len(comps),1)
        score = 1.0 - _sigmoid(avg, k=20, x0=0.12)
        ss = [round(1.0 - _sigmoid(c, k=20, x0=0.12), 4) for c in comps]
        return {"raw": round(avg,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Clause density={avg:.3f}. {'Simple (AI)' if score>0.6 else 'Complex'}."}


# ---------------------------------------------------------------------------
# 8 · Function Word Distribution
# ---------------------------------------------------------------------------

class FunctionWordDistributionFeature(Feature):
    """
    Deviation from expected stopword ratio (≈0.45).
    High deviation → unusual pattern → AI signal.
    """
    name = "function_word_dist"
    description = "Stopword ratio deviation from baseline"
    _BASE = 0.45

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        words = _tokenize_words(text)
        if not words: return self._empty()
        sw = sum(1 for w in words if w in _STOPWORDS)
        ratio = _safe_div(sw, len(words))
        dev = abs(ratio - self._BASE)
        score = _sigmoid(dev, k=15, x0=0.10)
        ss = []
        for s in sents:
            w = _tokenize_words(s)
            if not w: ss.append(0.5); continue
            sc = sum(1 for x in w if x in _STOPWORDS)
            ss.append(round(_sigmoid(abs(_safe_div(sc,len(w)) - self._BASE), k=15, x0=0.10), 4))
        return {"raw": round(ratio,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"SW ratio={ratio:.3f} (base≈{self._BASE}). Dev={dev:.3f}."}


# ---------------------------------------------------------------------------
# 9 · Entropy
# ---------------------------------------------------------------------------

class EntropyFeature(Feature):
    """
    H_norm = H / log2(vocab). Low → concentrated → AI-like → score = 1-H_norm.
    """
    name = "entropy"
    description = "Normalized Shannon entropy of word distribution"

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if len(tokens) < 5: return self._empty()
        counts = Counter(tokens); total = len(tokens); vocab = len(counts)
        H = -sum((c/total)*math.log2(c/total) for c in counts.values())
        maxH = math.log2(vocab) if vocab > 1 else 1.0
        norm = _safe_div(H, maxH)
        score = 1.0 - norm
        sents = _tokenize_sentences(text)
        ss = []
        for s in sents:
            w = _tokenize_words(s)
            if not w: ss.append(0.5); continue
            cnt = Counter(w); tot = len(w)
            h = -sum((c/tot)*math.log2(c/tot) for c in cnt.values())
            mh = math.log2(len(cnt)) if len(cnt)>1 else 1.0
            ss.append(round(1.0 - _safe_div(h, mh), 4))
        return {"raw": round(norm,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Norm entropy={norm:.3f}. {'Low (AI)' if score>0.5 else 'High (human)'}."}


# ---------------------------------------------------------------------------
# 10 · N-gram Frequency Bias
# ---------------------------------------------------------------------------

class NgramFrequencyBiasFeature(Feature):
    """
    Common-word ratio. High → safe AI vocabulary.
    score = sigmoid(ratio, k=12, x0=0.55).
    """
    name = "ngram_frequency_bias"
    description = "Over-reliance on common English words"

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if not tokens: return self._empty()
        cc = sum(1 for t in tokens if t in _COMMON_UNIGRAMS)
        bias = _safe_div(cc, len(tokens))
        score = _sigmoid(bias, k=12, x0=0.55)
        sents = _tokenize_sentences(text)
        ss = []
        for s in sents:
            w = _tokenize_words(s)
            if not w: ss.append(0.5); continue
            c = sum(1 for x in w if x in _COMMON_UNIGRAMS)
            ss.append(round(_sigmoid(_safe_div(c,len(w)), k=12, x0=0.55), 4))
        return {"raw": round(bias,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Common ratio={bias:.3f}. {'High (AI)' if bias>0.55 else 'Varied'}."}


# ---------------------------------------------------------------------------
# 11 · Over-Coherence
# ---------------------------------------------------------------------------

class OverCoherenceFeature(Feature):
    """
    Low std(inter-sentence similarities) → over-coherent → AI-like.
    """
    name = "over_coherence"
    description = "Uniformly high inter-sentence coherence"

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        if len(sents) < 3: return self._empty()
        sp_result = SemanticPredictabilityFeature().compute(text)
        sims = sp_result.get("sentence_scores", [])
        if len(sims) < 2: return self._empty()
        mean_s = sum(sims) / len(sims)
        std_s = math.sqrt(sum((s-mean_s)**2 for s in sims) / len(sims))
        score = 1.0 - _sigmoid(std_s, k=15, x0=0.15)
        ss = [round(1.0 - _clip01(abs(s-mean_s), 0, 0.5), 4) for s in sims]
        return {"raw": round(std_s,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Sim std={std_s:.3f}. {'Over-coherent (AI)' if score>0.6 else 'Normal'}."}


# ---------------------------------------------------------------------------
# 12 · Emotional Variability
# ---------------------------------------------------------------------------

class EmotionalVariabilityFeature(Feature):
    """
    Low std(per-sentence sentiment) → flat emotion → AI-like.
    """
    name = "emotional_variability"
    description = "Variance of sentence-level sentiment"

    _POS = {"good","great","excellent","happy","joy","love","wonderful","amazing",
            "best","better","brilliant","beautiful","fantastic","positive","success",
            "win","gain","helpful","kind","perfect","nice","glad","proud","laugh",
            "smile","fun","hope","strong","safe","free","clear","bright","easy",
            "fast","powerful","warm","sweet","fresh"}
    _NEG = {"bad","terrible","awful","sad","hate","horrible","worst","worse","ugly",
            "negative","fail","loss","harm","problem","wrong","weak","danger","fear",
            "pain","cry","dark","cold","slow","poor","sick","dead","hard","difficult",
            "broken","error","stupid","angry"}

    def _s(self, sent):
        w = _tokenize_words(sent)
        if not w: return 0.0
        return _safe_div(sum(1 for x in w if x in self._POS) - sum(1 for x in w if x in self._NEG), len(w))

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        if len(sents) < 2: return self._empty()
        sentiments = [self._s(s) for s in sents]
        mean_s = sum(sentiments) / len(sentiments)
        std_s = math.sqrt(sum((x-mean_s)**2 for x in sentiments) / len(sentiments))
        score = 1.0 - _sigmoid(std_s, k=20, x0=0.05)
        ss = [round(1.0 - _clip01(abs(x-mean_s), 0, 0.3), 4) for x in sentiments]
        return {"raw": round(std_s,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Sentiment std={std_s:.3f}. {'Flat emotion (AI)' if score>0.6 else 'Emotional range'}."}


# ---------------------------------------------------------------------------
# 13 · Error Patterns
# ---------------------------------------------------------------------------

class ErrorPatternsFeature(Feature):
    """
    Heuristic error density. Low → AI perfection → high score.
    """
    name = "error_patterns"
    description = "Low error density = AI-like grammatical perfection"

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        words = text.split()
        if not words: return self._empty()
        errors = 0
        for i in range(len(words)-1):
            w1 = re.sub(r'[^\w]','',words[i].lower())
            w2 = re.sub(r'[^\w]','',words[i+1].lower())
            if w1 and w1 == w2: errors += 1
        errors += sum(1 for w in words if len(w) > 18)
        errors += len(re.findall(r'[a-z]\.[A-Z]', text))
        errors += len(re.findall(r'[?!]{3,}', text))
        density = _safe_div(errors, len(words))
        score = 1.0 - _sigmoid(density, k=50, x0=0.03)
        ss = []
        for s in sents:
            w = s.split()
            if not w: ss.append(0.5); continue
            e = 0
            for i in range(len(w)-1):
                x1 = re.sub(r'[^\w]','',w[i].lower())
                x2 = re.sub(r'[^\w]','',w[i+1].lower())
                if x1 and x1 == x2: e += 1
            e += sum(1 for x in w if len(x)>18)
            ss.append(round(1.0 - _sigmoid(_safe_div(e,len(w)), k=50, x0=0.03), 4))
        return {"raw": round(density,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Error density≈{density:.4f}. {'Near-perfect (AI)' if score>0.6 else 'Errors found (human)'}."}


# ---------------------------------------------------------------------------
# 14 · Contextual Depth
# ---------------------------------------------------------------------------

class ContextualDepthFeature(Feature):
    """
    CD = (first-person + subjective markers) / word_count.
    Low → impersonal → AI-like.
    """
    name = "contextual_depth"
    description = "Personal voice and subjective language density"
    _FP   = {"i","me","my","mine","myself","we","us","our","ours","ourselves"}
    _SUBJ = {"think","feel","believe","suppose","guess","reckon","suspect",
             "imagine","wonder","hope","wish","seem","appear","perhaps",
             "maybe","probably","honestly","personally","frankly","actually","really"}

    def compute(self, text: str) -> Dict[str, Any]:
        sents = _tokenize_sentences(text)
        tokens = _tokenize_words(text)
        if not tokens: return self._empty()
        fp   = sum(1 for t in tokens if t in self._FP)
        subj = sum(1 for t in tokens if t in self._SUBJ)
        cd = _safe_div(fp+subj, len(tokens))
        score = 1.0 - _sigmoid(cd, k=20, x0=0.06)
        ss = []
        for s in sents:
            w = _tokenize_words(s)
            if not w: ss.append(0.5); continue
            p = sum(1 for x in w if x in self._FP)
            sj = sum(1 for x in w if x in self._SUBJ)
            ss.append(round(1.0 - _sigmoid(_safe_div(p+sj,len(w)), k=20, x0=0.06), 4))
        return {"raw": round(cd,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"Personal ratio={cd:.3f}. {'Impersonal (AI)' if score>0.6 else 'Personal voice'}."}


# ---------------------------------------------------------------------------
# 15 · Stopword Patterning
# ---------------------------------------------------------------------------

class StopwordPatterningFeature(Feature):
    """
    Divide text into thirds; std(SW_ratio per third).
    Low std → uniform → AI-like.
    """
    name = "stopword_patterning"
    description = "Positional stopword density variation"

    def compute(self, text: str) -> Dict[str, Any]:
        tokens = _tokenize_words(text)
        if len(tokens) < 9: return self._empty()
        t = len(tokens) // 3
        parts = [tokens[:t], tokens[t:2*t], tokens[2*t:]]
        ratios = [_safe_div(sum(1 for w in p if w in _STOPWORDS), len(p)) if p else 0.0 for p in parts]
        mean_r = sum(ratios) / 3
        std_r = math.sqrt(sum((r-mean_r)**2 for r in ratios) / 3)
        score = 1.0 - _sigmoid(std_r, k=30, x0=0.04)
        sents = _tokenize_sentences(text)
        ss = [round(score, 4)] * len(sents)
        return {"raw": round(std_r,4), "score": round(score,4), "sentence_scores": ss,
                "description": f"SW positional std={std_r:.3f}. {'Uniform (AI)' if score>0.6 else 'Varied'}."}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_FEATURES: List[Feature] = [
    PerplexityFeature(),
    BurstinessFeature(),
    SentenceDiversityFeature(),
    LexicalDiversityFeature(),
    RepetitionFeature(),
    SemanticPredictabilityFeature(),
    SyntacticComplexityFeature(),
    FunctionWordDistributionFeature(),
    EntropyFeature(),
    NgramFrequencyBiasFeature(),
    OverCoherenceFeature(),
    EmotionalVariabilityFeature(),
    ErrorPatternsFeature(),
    ContextualDepthFeature(),
    StopwordPatterningFeature(),
]

FEATURE_MAP: Dict[str, Feature] = {f.name: f for f in ALL_FEATURES}
