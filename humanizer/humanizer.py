"""
humanizer/humanizer.py
======================
Humanizer Engine — transforms AI-generated text into human-like writing.

Strategy: Break AI patterns rather than mask them.
Each transformation targets a specific detector feature.

Targeted features (in priority order):
  1. transition_density    → Replace/remove AI transition words
  2. contraction_absence   → Inject natural contractions  
  3. burstiness            → Vary sentence lengths (split/merge)
  4. sentence_length_cv    → Create natural length rhythm
  5. sentiment_flatness    → Inject emotional variation
  6. lexical_density       → Replace formal words with casual equivalents
  7. punct_uniformity      → Vary punctuation patterns
  8. compression_signal    → Add natural filler/redundancy

Modes:
  subtle     — light touch, minimal changes
  balanced   — recommended, meaningful transformation
  aggressive — maximum humanization

IMPORTANT: All transformations are linguistically meaningful.
           No random word shuffling. Meaning is preserved.
"""

import re
import math
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

# English utilities — all 29 modules
try:
    from humanizer.english_utils import (
        NumberToWords, NumberToOrdinalWords, Ordinalize,
        Vocabulary, Inflector, DEFAULT_INFLECTOR,
        ArticleHandler, CollectionHumanizer,
        StringHumanizer, StringDehumanizer,
        CasingTransformer, Truncator, QuantityFormatter,
        MetricNumerals, DateHumanizer, TimeSpanHumanizer,
        TimeToClockNotation, AgeFormatter, TimeUnitSymbols,
        RomanNumerals, ByteSize, WordsToNumber,
        EnglishGrammarFixer, Validator,
        HeadingConverter, NumberScalingHelpers,
        NumberToTimeSpan, ByteRate, FluentDateBuilder,
        TransformerPipeline,
        # New classes
        DateToOrdinalWords, ByteSizeExtensions,
        DateHumanizeAlgorithm, DefaultHumanizer,
        CollectionFormatter, EnglishOrdinalizer,
        EnglishFormattingRules,
        EnumHumanizer, EnumDehumanizer, StringConcat,
        TupleFormatter, PrepositionHandler,
        PrecisionHumanizer, ResourceKeys, ResourceRetrieval,
        LocalizationRegistry, PolyfillShims, DefaultFormatter,
        Configurator, EnglishGrammarDetector,
        _apply_numbers_to_words, _apply_words_to_numbers, _apply_ordinalize,
    )
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HumanizerConfig:
    """Configuration for a humanization run."""
    mode: str = "balanced"           # subtle | balanced | aggressive
    max_passes: int = 3              # iterative refinement passes
    target_score: float = 50.0      # stop when below this
    seed: Optional[int] = 42        # reproducibility (None = random)
    
    # Per-transformation enable/disable
    enable_transition_replace: bool = True
    enable_contraction_inject: bool = False  # Academic writing uses full forms, not contractions
    enable_sentence_variation: bool = True
    enable_personal_voice: bool = True
    enable_sentiment_variation: bool = True
    enable_lexical_casual: bool = True
    enable_punct_variation: bool = True
    enable_filler_phrases: bool = True
    enable_structural_rewrite: bool = True
    enable_coherence_disruption: bool = True
    # New transformers
    enable_semantic_phrase_rewrite: bool = True
    enable_sentence_reorder: bool = True
    enable_paragraph_flow: bool = True
    enable_memory_simulation: bool = True
    
    @property
    def intensity(self) -> float:
        """Transformation intensity 0-1."""
        return {"subtle": 0.25, "balanced": 0.55, "aggressive": 0.85}.get(self.mode, 0.55)
    
    @property
    def rates(self) -> Dict[str, float]:
        """Per-transformation application rates."""
        i = self.intensity
        return {
            "transition_replace": min(1.0, i * 1.6),
            "contraction":        0.0,                  # disabled — academic writing
            "sentence_split":     min(0.5, i * 0.8),
            "sentence_merge":     min(0.3, i * 0.55),
            "personal_voice":     min(0.12, i * 0.18),
            "sentiment":          min(0.15, i * 0.22),
            "lexical_casual":     min(0.55, i * 0.9),
            "punct_vary":         min(0.35, i * 0.65),
            "filler":             min(0.10, i * 0.16),
            "structural":         min(0.25, i * 0.45),
            "coherence_break":    min(0.20, i * 0.35),
            # New transformers
            "semantic_phrase":    min(0.70, i * 1.1),
            "sentence_reorder":   min(0.25, i * 0.40),
            "paragraph_flow":     min(0.12, i * 0.20),
            "memory_sim":         min(0.30, i * 0.50),
            # English utils stage A
            "eu_number":          min(0.45, i * 0.75),
            "eu_metric":          min(0.35, i * 0.60),
            "eu_collection":      min(0.40, i * 0.65),
        }


@dataclass
class TransformationResult:
    """Result of a humanization pass."""
    text: str
    original_score: float
    humanized_score: float
    score_change: float
    passes_applied: int
    transformations_applied: List[str] = field(default_factory=list)
    per_feature_change: Dict[str, float] = field(default_factory=dict)
    
    @property
    def improvement_pct(self) -> float:
        if self.original_score == 0:
            return 0.0
        return (self.original_score - self.humanized_score) / self.original_score * 100


# ─────────────────────────────────────────────────────────────────────────────
# Core vocabulary tables
# ─────────────────────────────────────────────────────────────────────────────

# AI transition words → human alternatives
# Format: original → list of casual replacements
AI_TRANSITIONS = {
    # Additive
    "furthermore":     ["in addition", "additionally", "it should also be noted that"],
    "moreover":        ["in addition", "it is further noted that", "beyond this"],
    "additionally":    ["furthermore", "in addition", "it is also worth noting that"],
    "in addition":     ["furthermore", "additionally", "it is also the case that"],
    "not only that":   ["beyond this", "furthermore"],

    # Causal / result
    "consequently":    ["as a result", "accordingly", "it follows that"],
    "therefore":       ["thus", "accordingly", "as a result", "it follows that"],
    "thus":            ["accordingly", "as a consequence", "therefore"],
    "hence":           ["consequently", "therefore", "as a result"],
    "as a result":     ["consequently", "accordingly", "therefore"],
    "for this reason": ["consequently", "accordingly", "as a result"],

    # Contrast
    "nevertheless":    ["notwithstanding", "despite this", "however"],
    "nonetheless":     ["notwithstanding", "despite this", "however"],
    "however":         ["nonetheless", "notwithstanding", "in contrast"],
    "notwithstanding": ["nevertheless", "despite this", "however"],
    "despite this":    ["nevertheless", "notwithstanding", "even so"],
    "on the contrary": ["in contrast", "conversely", "to the contrary"],

    # Emphasis
    "significantly":   ["considerably", "markedly", "substantially", "notably"],
    "substantially":   ["considerably", "markedly", "significantly"],
    "considerably":    ["substantially", "markedly", "significantly"],
    "notably":         ["particularly", "especially", "significantly"],
    "importantly":     ["crucially", "significantly", "it is notable that"],
    "fundamentally":   ["at its core", "in essence", "inherently"],
    "essentially":     ["in essence", "fundamentally", "at its core"],
    "ultimately":      ["in conclusion", "upon reflection", "in the final analysis"],
    "particularly":    ["especially", "specifically", "notably"],
    "specifically":    ["in particular", "especially", "precisely"],
    "generally":       ["in general", "broadly speaking", "as a rule"],
    "typically":       ["generally", "ordinarily", "as a rule"],
    "overall":         ["in summary", "broadly speaking", "on the whole"],
    "comprehensively": ["thoroughly", "systematically", "in its entirety"],
    "systematically":  ["methodically", "rigorously", "in a structured manner"],
    "strategically":   ["deliberately", "purposefully", "with careful consideration"],

    # Action verbs (casual → academic)
    "utilize":         ["employ", "apply", "make use of"],
    "utilizes":        ["employs", "applies"],
    "utilized":        ["employed", "applied"],
    "facilitate":      ["enable", "support", "promote"],
    "facilitates":     ["enables", "supports"],
    "implement":       ["apply", "employ", "institute"],
    "implements":      ["applies", "employs"],
    "implemented":     ["applied", "employed", "instituted"],
    "demonstrate":     ["illustrate", "establish", "indicate"],
    "demonstrates":    ["illustrates", "establishes"],
    "demonstrated":    ["illustrated", "established"],
    "indicate":        ["suggest", "demonstrate", "reveal"],
    "indicates":       ["suggests", "demonstrates"],
    "encompass":       ["encompass", "subsume", "incorporate"],
    "encompasses":     ["encompasses", "subsumes"],
    "incorporate":     ["integrate", "subsume", "include"],
    "incorporates":    ["integrates", "subsumes"],
    "integrate":       ["synthesise", "incorporate", "consolidate"],
    "integrates":      ["synthesises", "incorporates"],
    "optimize":        ["enhance", "refine", "improve"],
    "optimizes":       ["enhances", "refines"],
    "streamline":      ["rationalise", "simplify", "systematise"],
    "leverage":        ["employ", "utilise", "harness"],
    "robust":          ["rigorous", "comprehensive", "well-founded"],
    "comprehensive":   ["thorough", "exhaustive", "extensive"],
    "significant":     ["substantial", "notable", "considerable"],
    "substantial":     ["considerable", "significant", "notable"],
    "approach":        ["methodology", "framework", "paradigm"],
    "framework":       ["theoretical framework", "conceptual model", "structure"],
    "paradigm":        ["theoretical model", "conceptual framework", "approach"],
    "endeavor":        ["undertaking", "effort", "pursuit"],
    "commence":        ["initiate", "begin", "inaugurate"],
    "terminate":       ["conclude", "cease", "discontinue"],
    "ascertain":       ["determine", "establish", "identify"],
    "ameliorate":      ["improve", "enhance", "rectify"],
    "disseminate":     ["disseminate", "promulgate", "distribute"],
    "subsequently":    ["thereafter", "subsequently", "following this"],
    "concurrently":    ["simultaneously", "in parallel", "concomitantly"],
    "in order to":     ["in order to", "so as to", "with the aim of"],
    "prior to":        ["prior to", "preceding", "before"],
    "subsequent to":   ["following", "subsequent to", "after"],
    "in the event that": ["in the event that", "should", "if"],
    "in terms of":     ["with regard to", "concerning", "in relation to"],
}

# Contractions to inject: expanded → contracted
CONTRACTION_MAP = {
    r"\bI am\b":          "I'm",
    r"\bI have\b":        "I've",
    r"\bI will\b":        "I'll",
    r"\bI would\b":       "I'd",
    r"\bI had\b":         "I'd",
    r"\byou are\b":       "you're",
    r"\byou have\b":      "you've",
    r"\byou will\b":      "you'll",
    r"\bthey are\b":      "they're",
    r"\bthey have\b":     "they've",
    r"\bwe are\b":        "we're",
    r"\bwe have\b":       "we've",
    r"\bit is\b":         "it's",
    r"\bthat is\b":       "that's",
    r"\bthere is\b":      "there's",
    r"\bwhat is\b":       "what's",
    r"\bwho is\b":        "who's",
    r"\bdo not\b":        "don't",
    r"\bdoes not\b":      "doesn't",
    r"\bdid not\b":       "didn't",
    r"\bwill not\b":      "won't",
    r"\bwould not\b":     "wouldn't",
    r"\bcannot\b":        "can't",
    r"\bcould not\b":     "couldn't",
    r"\bshould not\b":    "shouldn't",
    r"\bhave not\b":      "haven't",
    r"\bhas not\b":       "hasn't",
    r"\bhad not\b":       "hadn't",
    r"\bis not\b":        "isn't",
    r"\bare not\b":       "aren't",
    r"\bwas not\b":       "wasn't",
    r"\bwere not\b":      "weren't",
}

# Personal voice insertions (sentence-level)
PERSONAL_VOICE_OPENERS = [
    "It is worth noting that", "Evidence suggests that", "It is arguable that",
    "This analysis indicates that", "It may be observed that", "As the literature suggests,",
    "Critically,", "From a theoretical standpoint,", "Upon closer examination,",
    "It is significant that", "Scholarly consensus holds that", "It can be argued that",
    "The evidence indicates that", "As this paper contends,", "Notably,",
]

PERSONAL_VOICE_CLOSERS = [
    ", as the evidence suggests", ", as this analysis demonstrates",
    ", a finding of considerable significance", ", as scholars have noted",
    ", which warrants further investigation", ", according to the available evidence",
    ", a conclusion supported by the literature",
]

# Master set of all opener prefixes — used to prevent re-injection on already-transformed sentences
_ALL_KNOWN_OPENERS = frozenset([
    p.lower().rstrip(',').rstrip(' ').rstrip('that').strip()
    for p in (
        PERSONAL_VOICE_OPENERS
        + ["In this context", "As previously noted", "With respect to this point",
           "It is therefore evident that", "Upon examination", "In light of this",
           "With regard to the foregoing", "As the analysis reveals",
           "Furthermore", "Moreover", "In addition", "Consequently",
           "Nevertheless", "Notwithstanding this", "Accordingly",
           "In contrast", "By extension", "In particular",
           "On a related note", "Worth mentioning here", "A quick aside",
           "That brings up another point", "Which raises an interesting question",
           "It is of considerable significance", "It bears emphasising",
           "A critical consideration here", "The evidence demonstrates",
           "It is therefore", "It follows that", "Upon closer examination",
           "It can be argued", "It is arguable", "Evidence suggests",
           "As the literature", "As this paper", "Scholarly consensus",
           "From a theoretical", "The evidence indicates", "This analysis"]
    )
])

def _already_has_injected_opener(sent: str) -> bool:
    """Return True if sentence already starts with an injected academic/filler opener."""
    s = sent.strip().lower()
    for opener in _ALL_KNOWN_OPENERS:
        if s.startswith(opener):
            return True
    return False

def _already_has_injected_closer(sent: str) -> bool:
    """Return True if sentence already ends with an injected academic closer."""
    s = sent.strip().lower().rstrip('.')
    closers = [c.lower().strip(', ') for c in PERSONAL_VOICE_CLOSERS]
    for c in closers:
        if s.endswith(c):
            return True
    return False

# Emotional/expressive phrases to add variety
EMOTIONAL_INJECTIONS = [
    # Academic emphasis/hedging
    ("what is particularly noteworthy is", "notably,"),
    ("it is of considerable significance that", "significantly,"),
    ("it bears emphasising that", "it is worth emphasising that"),
    ("a critical consideration here is", "crucially,"),
    ("the evidence demonstrates that", "empirical evidence indicates that"),
    # Academic hedging
    ("to some extent,", "in certain respects,"),
    ("it is arguable that", "one may contend that"),
    ("from a conceptual standpoint,", "from a theoretical perspective,"),
    ("broadly speaking,", "in general terms,"),
]

# Academic transition phrases that vary sentence rhythm
NATURAL_FILLERS = [
    "In this context, ",
    "As previously noted, ",
    "With respect to this point, ",
    "It is therefore evident that ",
    "Upon examination, ",
    "In light of this, ",
    "With regard to the foregoing, ",
    "As the analysis reveals, ",
]

# Punctuation variations
PUNCT_VARIATIONS = {
    ", ":  [", ", "; ", ", "],
    ". ":  [". ", ".\n", ". "],
}

# Academic sentence connectors
NATURAL_CONNECTORS = [
    "Furthermore, ", "Moreover, ", "In addition, ", "Consequently, ",
    "Nevertheless, ", "Notwithstanding this, ", "Accordingly, ",
    "In contrast, ", "By extension, ", "In particular, ",
]



# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, preserving trailing punctuation."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]


def _count_words(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))


def _capitalize_first(s: str) -> str:
    """Capitalize first letter, lowercase rest of first word."""
    if not s:
        return s
    return s[0].upper() + s[1:]


def _ends_with_punct(s: str) -> bool:
    return bool(s) and s[-1] in '.!?'


def _ensure_punct(s: str) -> str:
    """Ensure sentence ends with punctuation."""
    s = s.strip()
    if s and s[-1] not in '.!?,;:':
        s += '.'
    return s


def _word_boundary_replace(text: str, pattern: str, replacement: str, rng: random.Random) -> str:
    """Replace pattern (case-insensitive, word boundary) with one of the replacements."""
    if isinstance(replacement, list):
        replacement = rng.choice(replacement)
    
    def replace_match(m):
        orig = m.group(0)
        # Preserve sentence-initial capitalization
        if orig[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement
    
    return re.sub(r'\b' + re.escape(pattern) + r'\b', replace_match, text, flags=re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 1: AI Transition Word Replacement
# ─────────────────────────────────────────────────────────────────────────────

class TransitionReplacer:
    """
    Replaces AI-style transition words with casual alternatives.
    Target feature: transition_density → reduce to < 0.03
    """
    
    def apply(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        """Returns (transformed_text, replacements_made)."""
        count = 0
        
        # Sort by length (longest first) to avoid partial matches
        sorted_transitions = sorted(AI_TRANSITIONS.items(), key=lambda x: -len(x[0]))
        
        for ai_phrase, alternatives in sorted_transitions:
            if rng.random() > rate:
                continue
            
            # Check if phrase exists (case-insensitive)
            pattern = r'\b' + re.escape(ai_phrase) + r'\b'
            if not re.search(pattern, text, re.IGNORECASE):
                continue
            
            replacement = rng.choice(alternatives)
            
            def replace_match(m):
                orig = m.group(0)
                rep = replacement
                # Preserve capitalization
                if orig[0].isupper():
                    rep = rep[0].upper() + rep[1:]
                return rep
            
            new_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                count += 1
        
        return text, count


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 2: Contraction Injection
# ─────────────────────────────────────────────────────────────────────────────

class ContractionInjector:
    """
    Injects natural contractions into the text.
    Target feature: contraction_absence → reduce score to < 0.30
    Strategy: Replace expanded forms (I am → I'm, do not → don't)
    """
    
    def apply(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        count = 0
        for pattern, contraction in CONTRACTION_MAP.items():
            if rng.random() > rate:
                continue
            new_text = re.sub(pattern, contraction, text, flags=re.IGNORECASE, count=2)
            if new_text != text:
                count += 1
                text = new_text
        return text, count


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 3: Sentence Variation Engine
# ─────────────────────────────────────────────────────────────────────────────

class SentenceVariationEngine:
    """
    Creates burstiness by varying sentence lengths.
    Target features: burstiness, sentence_length_cv
    
    Operations:
    - Split: long sentences (>20 words) → two shorter ones
    - Merge: adjacent short sentences (<8 words) → one medium
    - Fragment: occasionally drop trailing subordinate clause to make it brief
    """
    
    SPLIT_THRESHOLD = 18   # words — sentences above this can be split
    MERGE_THRESHOLD = 8    # words — sentences below this can be merged
    
    # Conjunctions where we can split (before these words)
    SPLIT_POINTS = [
        r',?\s+and\s+',
        r',?\s+but\s+',
        r',?\s+which\s+',
        r',?\s+where\s+',
        r',?\s+while\s+',
        r',?\s+although\s+',
        r',?\s+because\s+',
        r',?\s+since\s+',
        r';\s+',
    ]
    
    def apply(self, sentences: List[str], split_rate: float, merge_rate: float,
              rng: random.Random) -> List[str]:
        result = []
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            wc = _count_words(sent)
            
            # Try to split long sentence
            if wc > self.SPLIT_THRESHOLD and rng.random() < split_rate:
                split = self._try_split(sent, rng)
                if split:
                    result.extend(split)
                    i += 1
                    continue
            
            # Try to merge with next short sentence
            if (wc < self.MERGE_THRESHOLD and i + 1 < len(sentences)
                    and _count_words(sentences[i+1]) < self.MERGE_THRESHOLD
                    and rng.random() < merge_rate):
                merged = self._merge(sent, sentences[i+1], rng)
                result.append(merged)
                i += 2
                continue
            
            result.append(sent)
            i += 1
        
        return result
    
    def _try_split(self, sent: str, rng: random.Random) -> Optional[List[str]]:
        """Try to split a sentence at a natural conjunction point."""
        rng.shuffle(self.SPLIT_POINTS)
        
        for pattern in self.SPLIT_POINTS:
            matches = list(re.finditer(pattern, sent, re.IGNORECASE))
            if not matches:
                continue
            
            # Prefer splitting near the middle
            mid = len(sent) // 2
            best = min(matches, key=lambda m: abs(m.start() - mid))
            
            # Only split if both parts are substantial
            part1 = sent[:best.start()].strip()
            # Get what comes after the conjunction
            after_conj = sent[best.end():]
            
            if _count_words(part1) < 4 or _count_words(after_conj) < 4:
                continue
            
            # Determine connector type for part2
            matched_text = best.group(0).strip().lower().strip(',').strip()
            
            if matched_text in ('and', 'but', 'while', 'although', 'because', 'since'):
                # Convert to independent sentence
                part2 = _capitalize_first(after_conj.strip())
                # Sometimes keep a natural connector
                if matched_text == 'but' and rng.random() < 0.6:
                    part2 = "But " + after_conj.strip()
                elif matched_text == 'and' and rng.random() < 0.4:
                    part2 = rng.choice(["And ", "Plus, "]) + after_conj.strip()
            else:
                part2 = _capitalize_first(after_conj.strip())
            
            part1 = _ensure_punct(part1)
            part2 = _ensure_punct(part2)
            
            if _count_words(part1) >= 4 and _count_words(part2) >= 4:
                return [part1, part2]
        
        return None
    
    def _merge(self, s1: str, s2: str, rng: random.Random) -> str:
        """Merge two short sentences with a natural connector."""
        s1 = s1.rstrip('.!?').strip()
        s2_lower = s2[0].lower() + s2[1:] if s2 else s2
        connectors =  [", and ", ", and also ", ", so ", ", but "]
        return s1 + rng.choice(connectors) + s2_lower


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 4: Personal Voice Injection
# ─────────────────────────────────────────────────────────────────────────────

class PersonalVoiceInjector:
    """
    Adds subjective tone: hedges, personal pronouns, opinions.
    Target features: sentence_uniformity (varies openings), sentiment_flatness
    """
    
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        # Build a set of known opener prefixes to avoid double-injection
        _OPENER_PREFIXES = tuple(p.lower()[:18] for p in PERSONAL_VOICE_OPENERS)
        _CLOSER_SUFFIXES = tuple(c.lower().strip(' ,.')[-18:] for c in PERSONAL_VOICE_CLOSERS)
        # Also guard against all filler/connector prefixes
        _FILLER_PREFIXES = tuple(f.lower().strip()[:16] for f in NATURAL_FILLERS + NATURAL_CONNECTORS)

        result = []
        for sent in sentences:
            # Never inject on short sentences or fragments
            if rng.random() > rate or _count_words(sent) < 8:
                result.append(sent)
                continue

            sent_low = sent.lower()

            # Skip if sentence already begins with any known injected phrase
            if any(sent_low.startswith(p) for p in _OPENER_PREFIXES + _FILLER_PREFIXES):
                result.append(sent)
                continue

            op = rng.choice(["opener", "closer", "hedge"])

            if op == "opener":
                opener = rng.choice(PERSONAL_VOICE_OPENERS)
                sent = opener[0].upper() + opener[1:] + ", " + sent[0].lower() + sent[1:]

            elif op == "closer":
                # Skip if sentence already ends with a known closer
                if any(sent_low.rstrip('.').endswith(s) for s in _CLOSER_SUFFIXES):
                    result.append(sent)
                    continue
                if (sent.endswith('.') and not sent.endswith('...')
                        and '—' not in sent[-5:] and len(sent) > 30):
                    closer = rng.choice(PERSONAL_VOICE_CLOSERS)
                    sent = sent[:-1] + closer + '.'

            elif op == "hedge":
                hedges = [("arguably", 0.3), ("considerably", 0.3), ("notably", 0.4)]
                hedge, hedge_rate = rng.choice(hedges)
                if rng.random() < hedge_rate:
                    sent = re.sub(
                        r'\b(very|quite|extremely|highly|particularly|especially)\b',
                        hedge, sent, count=1, flags=re.IGNORECASE
                    )

            result.append(sent)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 5: Sentiment Variation
# ─────────────────────────────────────────────────────────────────────────────

class SentimentVariator:
    """
    Adds emotional variation across sentences.
    Target feature: sentiment_flatness → std(sentiment) increases
    
    Strategy: mark some sentences with mild enthusiasm or mild skepticism
    """
    
    _ENTHUSIASM = [
        "Of particular significance is the finding that ",
        "It is especially noteworthy that ",
        "A critical observation here is that ",
    ]
    
    _SKEPTICISM = [
        "It is worth questioning whether ",
        "That said, it remains uncertain whether ",
        "The empirical basis for whether ",
    ]
    
    _POSITIVE_INJECTIONS = [
        ", a finding of considerable scholarly significance",
        ", a result with meaningful theoretical implications",
        ", a conclusion that merits further investigation",
    ]
    
    _CAUTIOUS_INJECTIONS = [
        ", though the matter is not without complexity",
        ", subject to the limitations of the available evidence",
        " — notwithstanding certain important qualifications",
    ]
    
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        if len(sentences) < 3:
            return sentences

        # Build suffix guards
        _POS_SUFFIXES = tuple(p.lower().strip(' ,.')[-22:] for p in self._POSITIVE_INJECTIONS)
        _CAUT_SUFFIXES = tuple(c.lower().strip(' ,.')[-22:] for c in self._CAUTIOUS_INJECTIONS)

        result = list(sentences)
        n = len(result)

        # Apply to at most ONE sentence per call to avoid piling up
        if rng.random() < rate and n > 2:
            idx = rng.randint(1, min(3, n-1))
            sent_low = result[idx].lower().rstrip('.')
            if not any(sent_low.endswith(s) for s in _POS_SUFFIXES + _CAUT_SUFFIXES):
                inj = rng.choice(self._POSITIVE_INJECTIONS + self._CAUTIOUS_INJECTIONS)
                if result[idx].endswith('.'):
                    result[idx] = result[idx][:-1] + inj + '.'

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 6: Lexical Casualization
# ─────────────────────────────────────────────────────────────────────────────

class LexicalCasualizer:
    """
    Replaces overly formal/academic vocabulary with natural alternatives.
    Target feature: lexical_density (reduces content-word formality)
    """
    
    # Academic register — replace overly simple or vague words with precise academic equivalents
    FORMAL_TO_CASUAL = {
        r"\brequires\b":         ["necessitates", "demands", "calls for"],
        r"\brequire\b":          ["necessitate", "demand", "call for"],
        r"\bprovides\b":         ["furnishes", "offers", "presents"],
        r"\bprovide\b":          ["furnish", "offer", "present"],
        r"\baddresses\b":        ["examines", "investigates", "analyses"],
        r"\baddress\b":          ["examine", "investigate", "analyse"],
        r"\bpresents\b":         ["articulates", "elucidates", "sets forth"],
        r"\bpresent\b":          ["articulate", "elucidate", "set forth"],
        r"\bexamines\b":         ["scrutinises", "analyses", "investigates"],
        r"\bexamine\b":          ["scrutinise", "analyse", "investigate"],
        r"\bpotential\b":        ["prospective", "plausible", "latent"],
        r"\bnumerous\b":         ["a considerable number of", "a multiplicity of", "manifold"],
        r"\bmultiple\b":         ["several", "a number of", "various"],
        r"\bvarious\b":          ["diverse", "a range of", "disparate"],
        r"\bindividuals\b":      ["persons", "subjects", "participants"],
        r"\bpeople\b":           ["individuals", "persons", "populations"],
        r"\bobtain\b":           ["acquire", "procure", "derive"],
        r"\bpurchase\b":         ["procure", "acquire"],
        r"\bsufficient\b":       ["adequate", "commensurate", "satisfactory"],
        r"\badditional\b":       ["supplementary", "further", "ancillary"],
        r"\bemploy\b":           ["utilise", "apply", "deploy"],
        r"\bmodify\b":           ["alter", "adapt", "revise"],
        r"\bextensive\b":        ["comprehensive", "wide-ranging", "exhaustive"],
        r"\bcritical\b":         ["pivotal", "essential", "fundamental"],
        r"\bessential\b":        ["indispensable", "fundamental", "integral"],
        r"\bprimary\b":          ["principal", "predominant", "foremost"],
        r"\bfundamental\b":      ["foundational", "axiomatic", "constitutive"],
        r"\bchallenges\b":       ["constraints", "impediments", "limitations"],
        r"\bchallenge\b":        ["constraint", "impediment", "limitation"],
        r"\bcontemporary\b":     ["present-day", "current", "prevailing"],
        r"\brapid\b":            ["accelerated", "expeditious", "swift"],
        r"\benables\b":          ["permits", "facilitates", "allows"],
        r"\benable\b":           ["permit", "facilitate", "allow"],
        r"\butilizes\b":         ["employs", "applies", "deploys"],
        r"\butilize\b":          ["employ", "apply", "deploy"],
        r"\butilized\b":         ["employed", "applied", "deployed"],
        r"\bfacilitate\b":       ["enable", "support", "advance"],
        r"\bfacilitates\b":      ["enables", "supports", "advances"],
        r"\brobust\b":           ["rigorous", "methodologically sound", "well-grounded"],
        r"\bimplementation\b":   ["operationalisation", "application", "execution"],
        r"\bmechanism\b":        ["mechanism", "process", "modality"],
        r"\bmechanisms\b":       ["mechanisms", "processes", "modalities"],
        r"\bapproach\b":         ["methodology", "paradigm", "theoretical framework"],
        r"\bshow\b":             ["demonstrate", "illustrate", "indicate"],
        r"\bshows\b":            ["demonstrates", "illustrates", "indicates"],
        r"\bshowed\b":           ["demonstrated", "illustrated", "indicated"],
        r"\buse\b":              ["employ", "utilise", "apply"],
        r"\buses\b":             ["employs", "utilises", "applies"],
        r"\bused\b":             ["employed", "utilised", "applied"],
        r"\bget\b":              ["obtain", "acquire", "derive"],
        r"\bgets\b":             ["obtains", "acquires"],
        r"\bgot\b":              ["obtained", "acquired"],
        r"\bbig\b":              ["substantial", "considerable", "significant"],
        r"\bsmall\b":            ["limited", "negligible", "minimal"],
        r"\bhelps\b":            ["facilitates", "supports", "promotes"],
        r"\bhelp\b":             ["facilitate", "support", "promote"],
        r"\btried\b":            ["endeavoured", "sought to", "attempted to"],
        r"\btry\b":              ["endeavour", "seek to", "attempt to"],
        r"\bstart\b":            ["initiate", "commence", "inaugurate"],
        r"\bstarts\b":           ["initiates", "commences"],
        r"\bstarted\b":          ["initiated", "commenced"],
        r"\bend\b":              ["conclude", "terminate", "cease"],
        r"\bends\b":             ["concludes", "terminates"],
        r"\bended\b":            ["concluded", "terminated"],
        r"\blook at\b":          ["examine", "investigate", "scrutinise"],
        r"\blooks at\b":         ["examines", "investigates"],
        r"\bfind\b":             ["identify", "establish", "ascertain"],
        r"\bfinds\b":            ["identifies", "establishes"],
        r"\bfound\b":            ["identified", "established", "demonstrated"],
        r"\bthink\b":            ["contend", "posit", "argue"],
        r"\bthinks\b":           ["contends", "posits"],
        r"\bthought\b":          ["contended", "posited"],
        r"\bsay\b":              ["assert", "contend", "maintain"],
        r"\bsays\b":             ["asserts", "contends"],
        r"\bsaid\b":             ["asserted", "contended", "maintained"],
        r"\bmake\b":             ["constitute", "render", "produce"],
        r"\bmakes\b":            ["constitutes", "renders"],
        r"\bmade\b":             ["constituted", "rendered"],
        r"\bkids\b":             ["children", "young people"],
        r"\bfolks\b":            ["individuals", "persons"],
        r"\bbuy\b":              ["purchase", "procure", "acquire"],
        r"\bbuys\b":             ["purchases", "procures"],
        r"\bneed\b":             ["require", "necessitate"],
        r"\bneeds\b":            ["requires", "necessitates"],
        r"\bgood\b":             ["effective", "advantageous", "beneficial"],
        r"\bbad\b":              ["detrimental", "adverse", "deleterious"],
        r"\bsee\b":              ["observe", "discern", "identify"],
        r"\bsees\b":             ["observes", "discerns"],
        r"\bsaw\b":              ["observed", "discerned", "identified"],
    }
    
    def apply(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        count = 0
        items = list(self.FORMAL_TO_CASUAL.items())
        rng.shuffle(items)
        
        for pattern, alternatives in items:
            if rng.random() > rate:
                continue
            if not re.search(pattern, text, re.IGNORECASE):
                continue
            
            replacement = rng.choice(alternatives)
            
            def replace_fn(m, rep=replacement):
                orig = m.group(0)
                if orig[0].isupper():
                    return rep[0].upper() + rep[1:]
                return rep
            
            new_text = re.sub(pattern, replace_fn, text, count=2, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                count += 1
        
        return text, count


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 7: Punctuation Variation
# ─────────────────────────────────────────────────────────────────────────────

class PunctuationVariator:
    """
    Introduces punctuation diversity: em dashes, ellipses, parentheses.
    Target feature: punct_uniformity → entropy increases
    """
    
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        result = []
        for sent in sentences:
            if rng.random() > rate:
                result.append(sent)
                continue
            
            op = rng.choice(["parenthetical", "ellipsis"])
            
            if op == "em_dash":
                # Replace a comma-clause with em dash
                # e.g. "X, which Y," → "X — Y —"
                match = re.search(r',\s+(\w)', sent)
                if match and sent.count(',') >= 2:
                    pos = match.start()
                    sent = sent[:pos] + ' —' + sent[pos+1:]
            
            elif op == "parenthetical":
                # Wrap a short qualifying phrase in parentheses
                # "X, Y," → "X (Y)"
                match = re.search(r',\s+((?:at least|for instance|for example|in theory|in practice|in some cases|particularly|especially|notably)[^,]+),', sent)
                if match:
                    phrase = match.group(1)
                    sent = sent[:match.start()] + f' ({phrase})' + sent[match.end():]
            
            elif op == "ellipsis":
                # Add ellipsis at end for trailing thought (aggressive mode)
                if sent.endswith('.') and rng.random() < 0.2:
                    sent = sent[:-1] + '...'
            
            elif op == "comma_to_dash":
                # Replace one interior comma with em dash for rhythm
                commas = [(m.start(), m.end()) for m in re.finditer(r',', sent)]
                if len(commas) >= 2:
                    idx = rng.randint(0, len(commas)-1)
                    pos, end = commas[idx]
                    sent = sent[:pos] + ' —' + sent[end:]
            
            result.append(sent)
        
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 8: Natural Filler & Redundancy
# ─────────────────────────────────────────────────────────────────────────────

class FillerInjector:
    """
    Adds natural filler phrases and minor redundancy.
    Target feature: compression_signal (less compressible text)
    
    Strategy: add sentence-opening fillers that humans naturally use.
    """
    
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        result = []
        n = len(sentences)
        
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        result = []
        n = len(sentences)

        # Build prefix guards from all known fillers and connectors
        _ALL_PREFIXES = tuple(
            p.lower().strip()[:16]
            for p in NATURAL_FILLERS + NATURAL_CONNECTORS
        )
        # Also guard common academic/personal openers already in use
        _EXTRA_GUARDS = tuple(p.lower()[:16] for p in PERSONAL_VOICE_OPENERS)

        for i, sent in enumerate(sentences):
            # Never inject on first sentence or short fragments
            if rng.random() > rate or i == 0 or _count_words(sent) < 8:
                result.append(sent)
                continue

            sent_low = sent.lower()

            # Skip if sentence already starts with a known filler/connector/opener
            if any(sent_low.startswith(p) for p in _ALL_PREFIXES + _EXTRA_GUARDS):
                result.append(sent)
                continue

            op = rng.choice(["filler_opener", "natural_connector", "minor_redundancy"])

            if op == "filler_opener":
                filler = rng.choice(NATURAL_FILLERS)
                sent = filler + sent[0].lower() + sent[1:]

            elif op == "natural_connector":
                connector = rng.choice(NATURAL_CONNECTORS)
                first_word = sent.split()[0].lower() if sent.split() else ""
                if first_word not in ("but", "and", "so", "yet", "or"):
                    sent = connector + sent[0].lower() + sent[1:]

            elif op == "minor_redundancy":
                # Only add if no trailing aside already present
                _ASIDE_MARKERS = (" at least in principle", " broadly speaking",
                                  " roughly", " in most cases", " or something close to it")
                already_has = any(sent.lower().rstrip('.').endswith(a.strip()) for a in _ASIDE_MARKERS)
                if not already_has and _count_words(sent) > 10 and rng.random() < 0.25:
                    asides = [
                        ", at least in principle",
                        ", broadly speaking",
                        ", in most cases",
                    ]
                    if sent.endswith('.'):
                        sent = sent[:-1] + rng.choice(asides) + '.'

            result.append(sent)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 9: Structural Rewriting
# ─────────────────────────────────────────────────────────────────────────────

class StructuralRewriter:
    """
    Changes sentence structure: clause reordering, fronting, inversion.
    Target feature: sentence_uniformity (varies sentence patterns)
    """
    
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        result = []
        for sent in sentences:
            if rng.random() > rate:
                result.append(sent)
                continue
            
            op = rng.choice(["front_clause", "passive_to_active_hint", "add_fragment"])
            
            if op == "front_clause":
                # Move prepositional/temporal clause to front
                # "X happens because Y" → "Because Y, X happens"
                match = re.search(
                    r'^(.{15,}?)\s+(because|since|although|if|when|while)\s+(.+)$',
                    sent, re.IGNORECASE
                )
                if match:
                    main, conj, clause = match.group(1), match.group(2), match.group(3)
                    # Only reorder if main is long enough
                    if _count_words(main) > 5 and _count_words(clause) > 3:
                        conj_cap = conj[0].upper() + conj[1:]
                        clause_end = _ensure_punct(clause)
                        if clause_end.endswith('.'):
                            clause_end = clause_end[:-1]
                        new_sent = f"{conj_cap} {clause_end}, {main[0].lower() + main[1:]}."
                        if _count_words(new_sent) == _count_words(sent):  # no words lost
                            sent = new_sent
            
            elif op == "passive_to_active_hint":
                # Add active-voice phrasing as alternative
                # Simple pattern: "X is done by Y" → hint toward "Y does X"
                # This is complex NLP; we do a safer version: reword "is being" constructs
                sent = re.sub(r'\bis being\b', 'is', sent, flags=re.IGNORECASE)
                sent = re.sub(r'\bare being\b', 'are', sent, flags=re.IGNORECASE)
            
            elif op == "add_fragment":
                # Turn a qualifier clause into a crisp standalone sentence,
                # then optionally follow with a short punchy echo fragment.
                # e.g. "X, which is significant." → "X. Significant, indeed."
                if rng.random() < 0.35:
                    match = re.search(
                        r',\s+which\s+(is|are|was|were)\s+([\w\s]{3,25}?)\.?$',
                        sent, re.IGNORECASE
                    )
                    if match:
                        qualifier = match.group(2).strip().rstrip('.')
                        sent = sent[:match.start()] + '.'
                        if qualifier and len(qualifier.split()) <= 5:
                            # Academic fragment echoes
                            echoes = [
                                f"{_capitalize_first(qualifier)}. Notably so.",
                                f"{_capitalize_first(qualifier)}, in point of fact.",
                                f"Indeed — {qualifier}.",
                                f"{_capitalize_first(qualifier)}. The literature concurs.",
                            ]
                            sent = sent + " " + rng.choice(echoes)
            
            result.append(sent)
        
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 10a: Phrase-Level Semantic Rewriter
# ─────────────────────────────────────────────────────────────────────────────

class SemanticPhraseRewriter:
    """
    Rewrites common AI phrase patterns using academic paraphrase templates.
    Goes beyond word substitution — replaces full predicate phrases.
    Target: lexical_density, compression_signal
    """

    # (regex pattern, list of academic paraphrase templates)
    PHRASE_TEMPLATES = [
        # Performs/functions
        (r'\bperforms\s+(efficiently|effectively|well)\b',
         ["operates with considerable efficacy",
          "demonstrates a high degree of effectiveness",
          "functions with notable precision"]),
        (r'\bworks\s+(well|effectively|efficiently)\b',
         ["operates effectively",
          "proves demonstrably effective",
          "yields satisfactory results"]),
        (r'\bgets the job done\b',
         ["fulfils the required function",
          "achieves the designated objective",
          "meets the established criteria"]),
        # Plays a role
        (r'\bplays\s+a\s+(key|major|critical|significant|important)\s+role\s+in\b',
         ["constitutes a central determinant of",
          "exerts a considerable influence upon",
          "functions as a primary factor in"]),
        (r'\bplays\s+a\s+role\s+in\b',
         ["contributes to",
          "serves as a factor in",
          "exerts influence upon"]),
        # Sheds light
        (r'\bsheds?\s+light\s+on\b',
         ["illuminates",
          "elucidates",
          "contributes to an understanding of"]),
        # Paves the way
        (r'\bpaves?\s+the\s+way\s+for\b',
         ["establishes the conditions for",
          "creates the prerequisite conditions for",
          "lays the groundwork for"]),
        # Takes into account
        (r'\btakes?\s+into\s+account\b',
         ["accounts for",
          "incorporates consideration of",
          "subsumes within its purview"]),
        # In light of the fact that
        (r'\bin\s+light\s+of\s+the\s+fact\s+that\b',
         ["given that",
          "considering that",
          "in view of the fact that"]),
        # Due to the fact that
        (r'\bdue\s+to\s+the\s+fact\s+that\b',
         ["because",
          "given that",
          "on account of the fact that"]),
        # It is important to note
        (r'\bit\s+is\s+important\s+to\s+note\s+(that)?\b',
         ["it is noteworthy that",
          "it warrants emphasis that",
          "attention must be drawn to the fact that"]),
        # It is worth noting
        (r'\bit\s+is\s+worth\s+noting\s+(that)?\b',
         ["it is notable that",
          "it merits acknowledgement that",
          "it bears noting that"]),
        # There is a need for
        (r'\bthere\s+is\s+a\s+need\s+for\b',
         ["there exists a requirement for",
          "a necessity for X has been identified",
          "the literature identifies a demand for"]),
        # Carry out
        (r'\bcarr(?:y|ies|ied)\s+out\b',
         ["conduct",
          "execute",
          "undertake"]),
        # A wide range of
        (r'\ba\s+wide\s+range\s+of\b',
         ["a diverse array of",
          "a broad spectrum of",
          "an extensive variety of"]),
        # A large number of
        (r'\ba\s+large\s+number\s+of\b',
         ["a considerable proportion of",
          "a substantial number of",
          "a significant quantity of"]),
        # In today's world / in today's society
        (r'\bin\s+today\'?s?\s+(world|society|era|age)\b',
         ["in the contemporary context",
          "in the present era",
          "within the current scholarly landscape"]),
        # Over the past / in recent years
        (r'\b(?:over\s+the\s+past|in\s+recent)\s+years\b',
         ["in the preceding years",
          "over recent decades",
          "throughout the recent period under examination"]),
        # In conclusion / in summary (at start of sentence)
        (r'(?i)^in\s+conclusion[,\s]',
         ["To summarise the foregoing analysis, ",
          "In concluding this discussion, ",
          "Drawing together the threads of the argument, "]),
        # First and foremost
        (r'\bfirst\s+and\s+foremost\b',
         ["principally",
          "most significantly",
          "of primary importance"]),
        # Last but not least
        (r'\blast\s+but\s+not\s+least\b',
         ["finally, and of equal importance",
          "of no lesser significance",
          "concluding this enumeration"]),
        # AI-favourite phrase: "delve into"
        (r'\bdelves?\s+into\b',
         ["examines",
          "investigates",
          "analyses in depth"]),
        # AI-favourite: "it is evident that"
        (r'\bit\s+is\s+(?:clear|evident|apparent)\s+that\b',
         ["the evidence demonstrates that",
          "the foregoing analysis establishes that",
          "it may be concluded that"]),
        # Boasts / boast
        (r'\bboasts?\b',
         ["possesses",
          "is characterised by",
          "demonstrates"]),
    ]

    def apply(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        count = 0
        templates = list(self.PHRASE_TEMPLATES)
        rng.shuffle(templates)

        for pattern, replacements in templates:
            if rng.random() > rate:
                continue
            if not re.search(pattern, text, re.IGNORECASE):
                continue
            replacement = rng.choice(replacements)

            def replace_fn(m, rep=replacement):
                orig = m.group(0)
                # Preserve sentence-initial capitalisation
                if orig and orig[0].isupper():
                    return rep[0].upper() + rep[1:]
                return rep

            new_text = re.sub(pattern, replace_fn, text, count=1, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                count += 1

        return text, count


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 10b: Cross-Sentence Reorderer
# ─────────────────────────────────────────────────────────────────────────────

class SentenceReorderer:
    """
    Swaps adjacent non-opening, non-closing sentence pairs in the middle of
    a paragraph to break the perfectly linear A→B→C AI pattern.
    Target: sentence_uniformity, coherence_smoothness
    """

    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        n = len(sentences)
        if n < 5:
            return sentences  # too short to safely reorder

        result = list(sentences)
        # Only reorder in the middle third — never touch opening/closing sentences
        lo = max(1, n // 4)
        hi = min(n - 2, 3 * n // 4)

        if hi <= lo:
            return sentences

        # Attempt at most one swap per call to avoid chaos
        if rng.random() < rate:
            idx = rng.randint(lo, hi - 1)
            # Swap sentences[idx] and sentences[idx+1]
            result[idx], result[idx + 1] = result[idx + 1], result[idx]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 10c: Paragraph Flow Variator
# ─────────────────────────────────────────────────────────────────────────────

class ParagraphFlowVariator:
    """
    Breaks the AI pattern of uniformly-sized, perfectly structured paragraphs.
    Inserts occasional short standalone emphasis sentences between longer ones,
    and forces paragraph breaks at natural points.
    Target: burstiness, sentence_uniformity, compression_signal
    """

    # Short standalone sentences injected for rhythm
    _STANDALONE_LINES = [
        "This distinction is critical.",
        "The implications merit careful consideration.",
        "The evidence is unambiguous on this point.",
        "This finding warrants particular emphasis.",
        "The significance of this cannot be overstated.",
        "Such nuance is frequently overlooked.",
        "This remains an open question in the literature.",
        "The practical ramifications are substantial.",
    ]

    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        n = len(sentences)
        if n < 4:
            return sentences

        result = []
        for i, sent in enumerate(sentences):
            result.append(sent)
            # After a long sentence in the middle, occasionally insert a
            # short standalone line for rhythm variation
            if (i > 0 and i < n - 1
                    and rng.random() < rate
                    and _count_words(sent) > 20):
                standalone = rng.choice(self._STANDALONE_LINES)
                # Guard: don't insert the same standalone twice
                if standalone not in result:
                    result.append(standalone)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 10d: Human Memory Simulator
# ─────────────────────────────────────────────────────────────────────────────

class MemorySimulator:
    """
    Adds human-like back-references and forward-pointers — patterns that
    imply the writer is thinking across the whole text, not sentence by sentence.
    Target: coherence_smoothness, sentence_uniformity
    Only fires once per text to avoid stacking.
    """

    _BACK_REFERENCES = [
        "As established earlier, ",
        "As previously noted, ",
        "Returning to the earlier point, ",
        "As discussed above, ",
        "Consistent with the foregoing analysis, ",
        "Building on the point introduced earlier, ",
    ]

    _FORWARD_POINTERS = [
        "This will be examined further below.",
        "This point will be returned to in the subsequent analysis.",
        "A fuller treatment of this issue follows.",
        "This question merits further elaboration.",
    ]

    _EMPHASIS_BRIDGES = [
        "This is, in essence, the central argument.",
        "The preceding point is fundamental to the broader analysis.",
        "This tension lies at the heart of the issue.",
        "The significance of this observation extends beyond the immediate context.",
    ]

    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        n = len(sentences)
        if n < 6:
            return sentences

        result = list(sentences)

        # Guard: if any back-reference already present, skip
        _back_prefixes = tuple(b.lower()[:18] for b in self._BACK_REFERENCES)
        already_has = any(
            s.lower().startswith(p) for s in result for p in _back_prefixes
        )
        if already_has:
            return result

        # Insert a back-reference at a mid-sentence (not first or last 2)
        if rng.random() < rate:
            idx = rng.randint(max(2, n // 3), min(n - 2, 2 * n // 3))
            ref = rng.choice(self._BACK_REFERENCES)
            sent = result[idx]
            # Only prepend if sentence doesn't already open with a connector
            first = sent.split()[0].lower() if sent.split() else ""
            if first not in ("furthermore", "moreover", "additionally", "however",
                             "nevertheless", "notwithstanding", "consequently"):
                result[idx] = ref + sent[0].lower() + sent[1:]

        # Occasionally insert a forward pointer or emphasis bridge after a long sentence
        if rng.random() < rate * 0.6 and n > 8:
            idx = rng.randint(n // 4, n // 2)
            if _count_words(result[idx]) > 18:
                bridge = rng.choice(self._FORWARD_POINTERS + self._EMPHASIS_BRIDGES)
                if bridge not in result:
                    result.insert(idx + 1, bridge)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Transformation 10: Coherence Disruption
# ─────────────────────────────────────────────────────────────────────────────

class CoherenceDisruptor:
    """
    Slightly reduces over-smooth sentence-to-sentence transitions.
    Target feature: coherence_smoothness (increases natural variation)
    
    Strategy: occasionally insert a micro-tangent or pivot sentence.
    """
    
    _PIVOT_PHRASES = [
        "On a related note, ",
        "That brings up another point: ",
        "Which raises an interesting question: ",
        "Worth mentioning here: ",
        "A quick aside: ",
    ]
    
    def apply(self, sentences: List[str], rate: float, rng: random.Random) -> List[str]:
        if len(sentences) < 4:
            return sentences
        
        result = list(sentences)
        n = len(result)
        
        # Insert one pivot phrase at a random mid-point
        if rng.random() < rate:
            idx = rng.randint(n//3, 2*n//3)
            if idx < n:
                sent = result[idx]
                pivot = rng.choice(self._PIVOT_PHRASES)
                first = sent.split()[0].lower() if sent.split() else ""
                if first not in ("but", "and", "so", "yet", "although"):
                    result[idx] = pivot + sent[0].lower() + sent[1:]
        
        return result


# ─────────────────────────────────────────────────────────────────────────────
# English Utilities Integration Layer
# Applies all 29 english_utils modules at appropriate pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

class EnglishUtilsTransformer:
    """
    Integrates all 29 english_utils modules into the humanization pipeline.

    Stage A  — pre-structural (text-level):
        number_variation, article_fix, quantity_fix, metric_notation,
        ordinal_variation, collection_humanize, string_humanize

    Stage B  — post-structural (sentence-level):
        casing_fix, grammar_fix, truncation_safety

    Stage C  — final validation:
        validator check + double-space / capitalisation cleanup
    """

    def __init__(self):
        self._available = _UTILS_AVAILABLE

    # ── Stage A: text-level ──────────────────────────────────────────────────

    def apply_number_variation(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        """
        Vary how numbers are expressed:
          - Small integers (2-12) → written words ("three", "twelve")
          - Ordinal context ("2 step") → "2nd step"
          - Large round numbers → metric notation ("1000000" → "1M")
          - Written numbers → digits in formal contexts
        """
        if not self._available:
            return text, 0
        count = 0

        # 1. Small integers to words (selective — only 2-12, avoid dates/years)
        if rng.random() < rate:
            def _num_to_word(m):
                nonlocal count
                n = int(m.group(1))
                # Skip years (1900-2100), percentages, IDs
                before = text[:m.start()].rstrip()
                after  = text[m.end():]
                if (1900 <= n <= 2100 or
                        (before and before[-1] == '%') or
                        (after and after.lstrip().startswith('%'))):
                    return m.group(0)
                word = NumberToWords.convert(n)
                count += 1
                return word
            text = re.sub(r'\b([2-9]|1[0-2])\b', _num_to_word, text)

        # 2. Large round numbers → compact metric
        if rng.random() < rate * 0.6:
            def _to_metric(m):
                nonlocal count
                n = int(m.group(0).replace(',', ''))
                if n >= 10_000:
                    compact = MetricNumerals.to_metric(n)
                    count += 1
                    return compact
                return m.group(0)
            text = re.sub(r'\b\d{5,}(?:,\d{3})*\b', _to_metric, text)

        # 3. Ordinal context — "1st", "2nd" etc.
        if rng.random() < rate * 0.5:
            new_text = _apply_ordinalize(text)
            if new_text != text:
                text = new_text
                count += 1

        return text, count

    def apply_article_fix(self, text: str) -> str:
        """Fix 'a/an' agreement after word substitutions."""
        if not self._available:
            return text
        return ArticleHandler.fix_articles_in_text(text)

    def apply_quantity_fix(self, text: str) -> str:
        """Fix number-noun agreement using EnglishGrammarFixer."""
        if not self._available:
            return text
        # Use the fixer with only spacing fixes (number agreement side effect)
        fixer = EnglishGrammarFixer()
        return fixer.fix(text, fix_articles=False, fix_spacing=True, fix_casing=False)

    def apply_collection_humanize(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        """
        Detect patterns like "X, Y and Z" and normalise to Oxford comma style.
        Also humanizes bulleted/numbered list fragments if detected.
        """
        if not self._available:
            return text, 0
        count = 0
        if rng.random() < rate:
            # Replace " and " (without comma before it) in list-like contexts
            # "A, B and C" → "A, B, and C"
            new_text = re.sub(
                r'(\w[\w\s]*),\s+(\w[\w\s]*)\s+and\s+(\w)',
                lambda m: f"{m.group(1)}, {m.group(2)}, and {m.group(3)}",
                text
            )
            if new_text != text:
                text = new_text
                count += 1
        return text, count

    def apply_metric_notation(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        """Parse metric abbreviations in text and normalise them."""
        if not self._available:
            return text, 0
        count = 0
        if rng.random() < rate:
            # Ensure metric values are consistent: "10k" → "10K"
            new_text = re.sub(
                r'\b(\d+(?:\.\d+)?)\s*([kmgKMG])\b',
                lambda m: m.group(1) + m.group(2).upper(),
                text
            )
            if new_text != text:
                text = new_text
                count += 1
        return text, count

    def apply_inflection_fix(self, text: str, rate: float, rng: random.Random) -> Tuple[str, int]:
        """
        Fix inflection errors that arise from word substitutions.
        Uses DEFAULT_INFLECTOR singleton for correct plural/singular forms.
        """
        if not self._available:
            return text, 0
        count = 0
        if rng.random() < rate:
            # Fix obvious plural mismatches: "these constraint" → "these constraints"
            def _fix_plural(m):
                nonlocal count
                det, word = m.group(1), m.group(2)
                plural_dets = {"these", "those", "many", "several", "multiple",
                               "various", "numerous", "few", "all", "both"}
                if det.lower() in plural_dets:
                    plural = DEFAULT_INFLECTOR.pluralize(word)
                    if plural != word:
                        count += 1
                        return f"{det} {plural}"
                return m.group(0)
            text = re.sub(r'\b(these|those|many|several|multiple|various|numerous|few)\s+([a-z]+)\b',
                          _fix_plural, text, flags=re.IGNORECASE)

            # Fix "a constraints" → "a constraint"
            def _fix_singular(m):
                nonlocal count
                art, word = m.group(1), m.group(2)
                singular = DEFAULT_INFLECTOR.singularize(word)
                if singular != word:
                    count += 1
                    return f"{art} {singular}"
                return m.group(0)
            text = re.sub(r'\b(a|an|one)\s+([a-z]{4,}s)\b',
                          _fix_singular, text, flags=re.IGNORECASE)

        return text, count

    # ── Stage B: sentence-level ──────────────────────────────────────────────

    def apply_casing_fix(self, sentences: List[str]) -> List[str]:
        """Ensure every sentence is properly sentence-cased after transforms."""
        if not self._available:
            return sentences
        result = []
        for sent in sentences:
            # Fix sentences that lost their capital letter during transforms
            fixed = CasingTransformer.to_sentence_case(sent)
            result.append(fixed)
        return result

    def apply_truncation_safety(self, sentences: List[str], max_words: int = 80) -> List[str]:
        """
        Split any sentence that ballooned over max_words (from bad merges)
        using the Truncator as a safety net.
        """
        if not self._available:
            return sentences
        result = []
        for sent in sentences:
            wc = len(sent.split())
            if wc > max_words:
                truncated = Truncator.truncate(sent, max_words, by_words=True)
                result.append(truncated)
            else:
                result.append(sent)
        return result

    # ── Stage C: validation + cleanup ───────────────────────────────────────

    def apply_final_cleanup(self, text: str) -> str:
        """Final pass: fix articles, spacing, capitalisation, duplicates."""
        if not self._available:
            return text

        # Use EnglishFormattingRules for a comprehensive cleanup
        rules = EnglishFormattingRules()
        text = rules.apply(text, rules=[
            "articles",
            "double_spaces",
            "duplicate_words",
            "sentence_case",
        ])

        return text.strip()

    def get_validation_warnings(self, text: str) -> List[str]:
        """Return readability warnings from Validator."""
        if not self._available:
            return []
        return Validator.validate(text)


# ─────────────────────────────────────────────────────────────────────────────
# Main Humanizer Engine
# ─────────────────────────────────────────────────────────────────────────────

class HumanizerEngine:
    """
    Orchestrates all 10 transformations with configurable intensity.
    Implements iterative feedback loop: re-analyze and re-apply if score still high.
    """
    
    def __init__(self):
        self.transition_replacer  = TransitionReplacer()
        self.contraction_injector = ContractionInjector()
        self.sentence_variator    = SentenceVariationEngine()
        self.personal_voice       = PersonalVoiceInjector()
        self.sentiment_variator   = SentimentVariator()
        self.lexical_casualizer   = LexicalCasualizer()
        self.punct_variator       = PunctuationVariator()
        self.filler_injector      = FillerInjector()
        self.structural_rewriter  = StructuralRewriter()
        self.coherence_disruptor  = CoherenceDisruptor()
        # New transformers
        self.semantic_phrase_rewriter = SemanticPhraseRewriter()
        self.sentence_reorderer       = SentenceReorderer()
        self.paragraph_flow_variator  = ParagraphFlowVariator()
        self.memory_simulator         = MemorySimulator()
        # English utils integration (all 29 modules)
        self.english_utils            = EnglishUtilsTransformer()
    
    def humanize(
        self,
        text: str,
        config: Optional[HumanizerConfig] = None,
        analyze_fn = None,
    ) -> TransformationResult:
        """
        Full humanization pipeline with optional iterative refinement.
        
        Parameters
        ----------
        text       : input text (AI-generated)
        config     : HumanizerConfig (defaults to balanced)
        analyze_fn : callable(text) → {"final_score": float, ...}
                     If provided, enables feedback loop.
        """
        if config is None:
            config = HumanizerConfig()
        
        rng = random.Random(config.seed)
        
        # Get original score
        original_score = 50.0
        original_features = {}
        if analyze_fn:
            try:
                orig_result = analyze_fn(text)
                original_score = orig_result.get("final_score", 50.0)
                original_features = orig_result.get("scores", {})
            except Exception as e:
                logger.warning(f"Analysis failed: {e}")
        
        # ── Iterative refinement loop ─────────────────────────────────────
        current_text = text
        applied = []
        
        for pass_num in range(config.max_passes):
            logger.debug(f"Humanization pass {pass_num+1}/{config.max_passes}")
            
            # On later passes, be more aggressive on structural transforms
            # but NEVER increase phrase-injection rates (they must stay capped)
            pass_rates = deepcopy(config.rates)
            if pass_num > 0:
                _structural_keys = {"transition_replace", "sentence_split",
                                    "sentence_merge", "lexical_casual",
                                    "punct_vary", "structural", "coherence_break",
                                    "semantic_phrase", "sentence_reorder",
                                    "eu_number", "eu_metric", "eu_collection"}
                for k in _structural_keys:
                    pass_rates[k] = min(1.0, pass_rates[k] * (1.0 + pass_num * 0.3))
                # Phrase-injection keys: cut rates on repeat passes to near zero
                for k in ("personal_voice", "sentiment", "filler"):
                    pass_rates[k] = pass_rates[k] * max(0.0, 1.0 - pass_num * 0.7)
            
            current_text, pass_applied = self._apply_one_pass(
                current_text, config, pass_rates, rng, pass_num=pass_num
            )
            applied.extend(pass_applied)
            
            # Check score after this pass
            if analyze_fn and pass_num < config.max_passes - 1:
                try:
                    interim = analyze_fn(current_text)
                    interim_score = interim.get("final_score", original_score)
                    logger.debug(f"Pass {pass_num+1} score: {interim_score:.1f}")
                    
                    # Stop early if target reached
                    if interim_score <= config.target_score:
                        logger.debug(f"Target score {config.target_score} reached early.")
                        break
                except Exception:
                    pass
        
        # ── Post-loop cleanup: trim if text grew too much ────────────────
        current_text = self._trim_to_length(current_text, text, max_ratio=1.9)
        
        # ── Final scoring ─────────────────────────────────────────────────
        final_score = original_score
        final_features = {}
        if analyze_fn:
            try:
                final_result = analyze_fn(current_text)
                final_score = final_result.get("final_score", original_score)
                final_features = final_result.get("scores", {})
            except Exception:
                pass
        
        # Per-feature changes
        feat_changes = {
            k: round(final_features.get(k, 0) - original_features.get(k, 0), 3)
            for k in set(original_features) | set(final_features)
        }
        
        return TransformationResult(
            text=current_text,
            original_score=original_score,
            humanized_score=final_score,
            score_change=original_score - final_score,
            passes_applied=min(pass_num + 1, config.max_passes),
            transformations_applied=list(dict.fromkeys(applied)),  # deduplicate, preserve order
            per_feature_change=feat_changes,
        )
    
    def _apply_one_pass(
        self,
        text: str,
        config: HumanizerConfig,
        rates: Dict[str, float],
        rng: random.Random,
        pass_num: int = 0,
    ) -> Tuple[str, List[str]]:
        """Apply all enabled transformations in one pass."""
        applied = []

        # ── Stage A: English utils — text-level pre-processing ───────────────

        # A1. Number variation (words ↔ digits, metric, ordinals)
        new_text, count = self.english_utils.apply_number_variation(
            text, rates.get("eu_number", 0.35), rng
        )
        if count:
            text = new_text
            applied.append(f"number_variation({count})")

        # A2. Metric notation normalisation
        new_text, count = self.english_utils.apply_metric_notation(
            text, rates.get("eu_metric", 0.25), rng
        )
        if count:
            text = new_text
            applied.append(f"metric_notation({count})")

        # A3. Oxford-comma collection humanisation
        new_text, count = self.english_utils.apply_collection_humanize(
            text, rates.get("eu_collection", 0.30), rng
        )
        if count:
            text = new_text
            applied.append(f"collection_humanize({count})")

        # ── Text-level transformations (work on full text string) ─────────
        
        # 1. Transition word replacement (highest priority — strongest signal)
        if config.enable_transition_replace:
            new_text, count = self.transition_replacer.apply(
                text, rates["transition_replace"], rng
            )
            if count:
                text = new_text
                applied.append(f"transition_replace({count})")
        
        # 2. Lexical casualization
        if config.enable_lexical_casual:
            new_text, count = self.lexical_casualizer.apply(
                text, rates["lexical_casual"], rng
            )
            if count:
                text = new_text
                applied.append(f"lexical_casual({count})")
        
        # 3. Contraction injection — apply with boosted rate on pass 2+
        if config.enable_contraction_inject:
            contr_rate = min(1.0, rates["contraction"] * (1.0 + pass_num * 0.5))
            new_text, count = self.contraction_injector.apply(text, contr_rate, rng)
            if count:
                text = new_text
                applied.append(f"contraction_inject({count})")
        
        # ── Sentence-level transformations ────────────────────────────────
        sentences = _split_sentences(text)
        
        # 4. Sentence variation (split/merge for burstiness)
        if config.enable_sentence_variation:
            orig_count = len(sentences)
            # Boost split rate on later passes to break uniform length
            split_rate = min(0.9, rates["sentence_split"] * (1.0 + pass_num * 0.4))
            sentences = self.sentence_variator.apply(
                sentences, split_rate, rates["sentence_merge"], rng
            )
            delta = len(sentences) - orig_count
            if delta:
                applied.append(f"sentence_variation({delta:+d})")
        
        # 5. Personal voice injection
        if config.enable_personal_voice:
            new_sents = self.personal_voice.apply(sentences, rates["personal_voice"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("personal_voice")
        
        # 6. Sentiment variation
        # 6. Sentiment variation — targeted injection on pass 2+
        if config.enable_sentiment_variation:
            sent_rate = min(0.8, rates["sentiment"] * (1.0 + pass_num * 0.3))
            new_sents = self.sentiment_variator.apply(sentences, sent_rate, rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("sentiment_variation")
        
        # 7. Punctuation variation
        if config.enable_punct_variation:
            new_sents = self.punct_variator.apply(sentences, rates["punct_vary"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("punct_variation")
        
        # 8. Filler phrases — only on first pass to avoid stacking
        if config.enable_filler_phrases and pass_num == 0:
            new_sents = self.filler_injector.apply(sentences, rates["filler"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("filler_inject")
        
        # 9. Structural rewriting
        if config.enable_structural_rewrite:
            new_sents = self.structural_rewriter.apply(sentences, rates["structural"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("structural_rewrite")
        
        # 10. Coherence disruption
        if config.enable_coherence_disruption:
            new_sents = self.coherence_disruptor.apply(sentences, rates["coherence_break"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("coherence_disrupt")

        # 11. Phrase-level semantic rewriting (phrase templates)
        if config.enable_semantic_phrase_rewrite:
            text = ' '.join(sentences)
            new_text, count = self.semantic_phrase_rewriter.apply(text, rates["semantic_phrase"], rng)
            if count:
                text = new_text
                sentences = _split_sentences(text)
                applied.append(f"semantic_phrase({count})")

        # 12. Cross-sentence reordering — only on pass 0 to avoid chaos
        if config.enable_sentence_reorder and pass_num == 0:
            new_sents = self.sentence_reorderer.apply(sentences, rates["sentence_reorder"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("sentence_reorder")

        # 13. Paragraph flow variation (standalone short lines) — pass 0 only
        if config.enable_paragraph_flow and pass_num == 0:
            new_sents = self.paragraph_flow_variator.apply(sentences, rates["paragraph_flow"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("paragraph_flow")

        # 14. Memory simulation (back-references, forward pointers) — pass 0 only
        if config.enable_memory_simulation and pass_num == 0:
            new_sents = self.memory_simulator.apply(sentences, rates["memory_sim"], rng)
            if new_sents != sentences:
                sentences = new_sents
                applied.append("memory_sim")
        
        # ── Stage B: English utils — sentence-level post-processing ─────────

        # B1. Fix inflection errors from word substitutions
        new_text, count = self.english_utils.apply_inflection_fix(
            ' '.join(sentences), rates.get("lexical_casual", 0.4) * 0.7, rng
        )
        if count:
            sentences = _split_sentences(new_text)
            applied.append(f"inflection_fix({count})")

        # B2. Fix a/an article agreement
        text_pre_article = ' '.join(sentences)
        text_post_article = self.english_utils.apply_article_fix(text_pre_article)
        if text_post_article != text_pre_article:
            sentences = _split_sentences(text_post_article)
            applied.append("article_fix")

        # B3. Fix number-noun agreement ("1 items" → "1 item")
        text_pre_qty = ' '.join(sentences)
        text_post_qty = self.english_utils.apply_quantity_fix(text_pre_qty)
        if text_post_qty != text_pre_qty:
            sentences = _split_sentences(text_post_qty)
            applied.append("quantity_fix")

        # B4. Sentence casing fix (after all transforms)
        sentences = self.english_utils.apply_casing_fix(sentences)

        # B5. Truncation safety net (sentences > 80 words)
        sentences = self.english_utils.apply_truncation_safety(sentences, max_words=80)

        # Reassemble text
        text = ' '.join(sentences)

        # ── Stage C: English utils — final cleanup + validation ─────────────
        text = self.english_utils.apply_final_cleanup(text)

        # Warn on validation issues (debug only)
        if logger.isEnabledFor(logging.DEBUG):
            warnings = self.english_utils.get_validation_warnings(text)
            for w in warnings:
                logger.debug(f"[Validator] {w}")

        # Legacy punctuation cleanup
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'\.{2,}(?!\.)', '.', text)
        text = re.sub(r'\.{2,},', '.', text)
        text = re.sub(r'\.,', '.', text)
        text = re.sub(r'\.\s*\.(?!\.)', '. ', text)
        text = re.sub(r',\s*,', ',', text)
        text = re.sub(r'\s*—\s*', ', ', text)
        text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)

        return text, applied
    
    def _trim_to_length(self, text: str, original_text: str, max_ratio: float = 1.8) -> str:
        """Trim humanized text if it grew too long relative to original."""
        orig_words = len(original_text.split())
        curr_words = len(text.split())
        if orig_words > 0 and curr_words / orig_words > max_ratio:
            # Keep sentences until we approach the limit
            sents = _split_sentences(text)
            result = []
            total = 0
            limit = int(orig_words * max_ratio)
            for s in sents:
                wc = len(s.split())
                if total + wc > limit and result:
                    break
                result.append(s)
                total += wc
            text = ' '.join(result)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_engine = HumanizerEngine()


def humanize_text(
    text: str,
    mode: str = "balanced",
    max_passes: int = 3,
    seed: int = 42,
    analyze_fn=None,
    target_score: float = 50.0,
) -> TransformationResult:
    """
    Humanize AI-generated text.
    
    Parameters
    ----------
    text         : AI-generated text to transform
    mode         : 'subtle' | 'balanced' | 'aggressive'
    max_passes   : max refinement iterations (1-3)
    seed         : random seed for reproducibility
    analyze_fn   : detection function(text) → {"final_score": float, ...}
                   If provided, enables score-aware feedback loop
    target_score : stop refining when score drops below this
    
    Returns
    -------
    TransformationResult with:
        .text              — humanized text
        .original_score    — AI detection score before
        .humanized_score   — AI detection score after
        .score_change      — points dropped
        .improvement_pct   — percentage reduction
        .transformations_applied — list of applied transforms
    """
    config = HumanizerConfig(
        mode=mode,
        max_passes=max_passes,
        seed=seed,
        target_score=target_score,
    )
    return _engine.humanize(text, config, analyze_fn)


def apply_transformations(
    text: str,
    transformations: List[str],
    mode: str = "balanced",
    seed: int = 42,
) -> str:
    """
    Apply specific transformations by name.
    
    Parameters
    ----------
    transformations : list of names, e.g. ["transition_replace", "contraction_inject"]
    
    Available names:
        transition_replace, contraction_inject, sentence_variation,
        personal_voice, sentiment_variation, lexical_casual,
        punct_variation, filler_inject, structural_rewrite, coherence_disrupt
    """
    config = HumanizerConfig(mode=mode, seed=seed, max_passes=1,
        enable_transition_replace   = "transition_replace"  in transformations,
        enable_contraction_inject   = "contraction_inject"  in transformations,
        enable_sentence_variation   = "sentence_variation"  in transformations,
        enable_personal_voice       = "personal_voice"      in transformations,
        enable_sentiment_variation  = "sentiment_variation" in transformations,
        enable_lexical_casual       = "lexical_casual"      in transformations,
        enable_punct_variation      = "punct_variation"     in transformations,
        enable_filler_phrases       = "filler_inject"       in transformations,
        enable_structural_rewrite   = "structural_rewrite"  in transformations,
        enable_coherence_disruption = "coherence_disrupt"   in transformations,
    )
    result = _engine.humanize(text, config, analyze_fn=None)
    return result.text


def evaluate_score_change(
    original_text: str,
    humanized_text: str,
    analyze_fn,
) -> Dict[str, Any]:
    """
    Compare detection scores before and after humanization.
    
    Parameters
    ----------
    original_text  : original AI text
    humanized_text : text after humanization
    analyze_fn     : detection function
    
    Returns
    -------
    dict with scores, change, improvement, and per-feature breakdown
    """
    orig_result = analyze_fn(original_text)
    hum_result  = analyze_fn(humanized_text)
    
    orig_score = orig_result.get("final_score", 0)
    hum_score  = hum_result.get("final_score", 0)
    orig_feats = orig_result.get("scores", {})
    hum_feats  = hum_result.get("scores", {})
    
    change = orig_score - hum_score
    pct    = (change / orig_score * 100) if orig_score else 0
    
    feat_changes = {}
    for k in set(orig_feats) | set(hum_feats):
        before = orig_feats.get(k, 0)
        after  = hum_feats.get(k, 0)
        feat_changes[k] = {
            "before": round(before, 3),
            "after":  round(after, 3),
            "delta":  round(after - before, 3),
            "improved": (after - before) < -0.05,
        }
    
    return {
        "original_score":       round(orig_score, 2),
        "humanized_score":      round(hum_score, 2),
        "score_change":         round(change, 2),
        "improvement_pct":      round(pct, 1),
        "original_class":       orig_result.get("classification", "?"),
        "humanized_class":      hum_result.get("classification", "?"),
        "original_confidence":  orig_result.get("confidence", {}).get("level", "?"),
        "humanized_confidence": hum_result.get("confidence", {}).get("level", "?"),
        "feature_changes":      feat_changes,
        "features_improved":    sum(1 for v in feat_changes.values() if v["improved"]),
    }
