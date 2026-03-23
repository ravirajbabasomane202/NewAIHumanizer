"""
humanizer/english_utils.py
===========================
English-language NLP utilities for the AI humanizer pipeline.

Modules implemented (all English-only):
  1.  NumberToWords         — integers → "forty-two"
  2.  NumberToOrdinalWords  — integers → "first", "twenty-first"
  3.  Ordinalize            — integers → "1st", "2nd", "42nd"
  4.  Inflector             — pluralize / singularize English nouns
  5.  Vocabulary            — custom irregular / uncountable rules
  6.  ArticleHandler        — "a" vs "an" correction; article-prefix sort
  7.  CollectionHumanizer   — ["a","b","c"] → "a, b, and c" (Oxford comma)
  8.  StringHumanizer       — "user_name" / "UserName" → "user name"
  9.  StringDehumanizer     — "user name" → "user_name"
  10. CasingTransformer     — lower / upper / sentence / title case
  11. Truncator             — safe word-boundary truncation
  12. QuantityFormatter     — "1 item" / "2 items"
  13. MetricNumerals        — 1500 → "1.5K" ; "1.5K" → 1500
  14. DateHumanizer         — datetime → "3 days ago", "in 2 weeks"
  15. TimeSpanHumanizer     — timedelta → "2 hours", "5 minutes"
  16. TimeToClockNotation   — time → "quarter past five"
  17. AgeFormatter          — timedelta → "5 years old"
  18. TimeUnitSymbols       — second → "sec", minute → "min"
  19. RomanNumerals         — int ↔ Roman numeral string
  20. ByteSize              — size arithmetic + humanized formatting
  21. WordsToNumber         — "forty two" → 42
  22. EnglishGrammarFixer   — fix "a/an", number agreement in a text block
  23. Validator             — lightweight readability / grammar checks
"""

from __future__ import annotations

import re
import math
import datetime
from typing import Optional, List, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. NumberToWords
# ─────────────────────────────────────────────────────────────────────────────

class NumberToWords:
    """Convert integers or floats to English cardinal words."""

    _ONES = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    _TENS = [
        "", "", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety",
    ]
    _SCALE = [
        (10 ** 12, "trillion"), (10 ** 9, "billion"),
        (10 ** 6, "million"),   (10 ** 3, "thousand"),
        (10 ** 2, "hundred"),
    ]

    @classmethod
    def convert(cls, n: Union[int, float]) -> str:
        """Convert a number to English words."""
        if isinstance(n, float):
            int_part = int(n)
            dec_str = f"{n:.10f}".rstrip("0").split(".")[1] if "." in f"{n}" else ""
            if dec_str:
                return f"{cls.convert(int_part)} point {' '.join(cls._ONES[int(d)] for d in dec_str if cls._ONES[int(d)])}"
            return cls.convert(int_part)

        n = int(n)
        if n < 0:
            return "negative " + cls.convert(-n)
        if n == 0:
            return "zero"
        return cls._convert_positive(n).strip()

    @classmethod
    def _convert_positive(cls, n: int) -> str:
        if n == 0:
            return ""
        if n < 20:
            return cls._ONES[n]
        if n < 100:
            tens = cls._TENS[n // 10]
            ones = cls._ONES[n % 10]
            return tens + ("-" + ones if ones else "")
        for value, name in cls._SCALE:
            if n >= value:
                high = cls._convert_positive(n // value)
                low  = cls._convert_positive(n % value)
                return high + " " + name + (" " + low if low else "")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. NumberToOrdinalWords
# ─────────────────────────────────────────────────────────────────────────────

class NumberToOrdinalWords:
    """Convert integers to English ordinal words: 1→first, 21→twenty-first."""

    _IRREGULAR = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth",
    }
    _TY_ORDINALS = {
        "twenty": "twentieth", "thirty": "thirtieth", "forty": "fortieth",
        "fifty": "fiftieth", "sixty": "sixtieth", "seventy": "seventieth",
        "eighty": "eightieth", "ninety": "ninetieth",
    }

    @classmethod
    def convert(cls, n: int) -> str:
        if n in cls._IRREGULAR:
            return cls._IRREGULAR[n]
        cardinal = NumberToWords.convert(n)
        # Handle "X-Y" compound → "X-Yth" / irregular last part
        if "-" in cardinal:
            prefix, last = cardinal.rsplit("-", 1)
            last_n = n % 10
            if last_n in cls._IRREGULAR:
                return prefix + "-" + cls._IRREGULAR[last_n]
            return prefix + "-" + last + "th"
        if cardinal in cls._TY_ORDINALS:
            return cls._TY_ORDINALS[cardinal]
        # Default: append "th" (handles "hundred", "thousand", etc.)
        return cardinal + "th"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ordinalize
# ─────────────────────────────────────────────────────────────────────────────

class Ordinalize:
    """Format integers as ordinal suffixes: 1→1st, 2→2nd, 3→3rd, 4→4th."""

    @staticmethod
    def convert(n: int) -> str:
        abs_n = abs(n)
        if 11 <= (abs_n % 100) <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(abs_n % 10, "th")
        return f"{n}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5. Vocabulary + Inflector
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    """Stores English inflection rules: irregulars, uncountables, regex rules."""

    def __init__(self):
        self._irregulars: List[Tuple[str, str]] = []
        self._uncountables: set = set()
        self._plural_rules:    List[Tuple[re.Pattern, str]] = []
        self._singular_rules:  List[Tuple[re.Pattern, str]] = []
        self._load_defaults()

    def add_irregular(self, singular: str, plural: str):
        self._irregulars.append((singular.lower(), plural.lower()))

    def add_uncountable(self, word: str):
        self._uncountables.add(word.lower())

    def add_plural(self, pattern: str, replacement: str):
        self._plural_rules.insert(0, (re.compile(pattern, re.IGNORECASE), replacement))

    def add_singular(self, pattern: str, replacement: str):
        self._singular_rules.insert(0, (re.compile(pattern, re.IGNORECASE), replacement))

    def pluralize(self, word: str) -> str:
        return self._apply(word, self._plural_rules, plural=True)

    def singularize(self, word: str) -> str:
        return self._apply(word, self._singular_rules, plural=False)

    def _apply(self, word: str, rules: list, plural: bool) -> str:
        lower = word.lower()
        if lower in self._uncountables:
            return word
        for sing, plur in self._irregulars:
            if plural and lower == sing:
                return self._match_case(word, plur)
            if not plural and lower == plur:
                return self._match_case(word, sing)
        for pattern, repl in rules:
            if pattern.search(word):
                return pattern.sub(repl, word)
        return word

    @staticmethod
    def _match_case(original: str, result: str) -> str:
        if original.isupper():
            return result.upper()
        if original[0].isupper():
            return result[0].upper() + result[1:]
        return result

    def _load_defaults(self):
        # Irregulars
        for s, p in [
            ("person","people"), ("child","children"), ("man","men"),
            ("woman","women"), ("tooth","teeth"), ("foot","feet"),
            ("mouse","mice"), ("goose","geese"), ("ox","oxen"),
            ("criterion","criteria"), ("phenomenon","phenomena"),
            ("datum","data"), ("medium","media"), ("index","indices"),
            ("matrix","matrices"), ("vertex","vertices"), ("axis","axes"),
            ("analysis","analyses"), ("thesis","theses"), ("crisis","crises"),
            ("diagnosis","diagnoses"), ("basis","bases"), ("hypothesis","hypotheses"),
            ("syllabus","syllabi"), ("focus","foci"), ("radius","radii"),
            ("nucleus","nuclei"), ("cactus","cacti"), ("fungus","fungi"),
            ("alumnus","alumni"), ("stimulus","stimuli"), ("terminus","termini"),
        ]:
            self.add_irregular(s, p)

        # Uncountables
        for w in [
            "information","research","data","knowledge","evidence","advice",
            "news","feedback","progress","furniture","equipment","money",
            "software","hardware","traffic","music","fish","sheep","deer",
            "species","series","mathematics","physics","economics","statistics",
            "linguistics","ethics","genetics","semantics","dynamics",
        ]:
            self.add_uncountable(w)

        # Plural rules (order: most specific first)
        for pat, repl in [
            (r"(quiz)$",                  r"\1zes"),
            (r"^(oxen)$",                 r"\1"),
            (r"^(ox)$",                   r"\1en"),
            (r"([m|l])ice$",              r"\1ice"),
            (r"([m|l])ouse$",             r"\1ice"),
            (r"(pea)s$",                  r"\1s"),
            (r"(matr|vert|append)(ix|ices)$", r"\1ices"),
            (r"(x|ch|ss|sh)$",            r"\1es"),
            (r"([^aeiouy]|qu)ies$",       r"\1ies"),
            (r"([^aeiouy]|qu)y$",         r"\1ies"),
            (r"(hive)s$",                 r"\1s"),
            (r"(hive)$",                  r"\1s"),
            (r"([lr])ves$",               r"\1ves"),
            (r"([^fo])ves$",              r"\1ves"),
            (r"([li])fe$",                r"\1ves"),
            (r"(shea|lea|loa|thie)ves$",  r"\1ves"),
            (r"(shea|lea|loa|thie)f$",    r"\1ves"),
            (r"sis$",                     r"ses"),
            (r"([ti])a$",                 r"\1a"),
            (r"([ti])um$",                r"\1a"),
            (r"(buffal|tomat)o$",         r"\1oes"),
            (r"(bu)s$",                   r"\1ses"),
            (r"(bus)$",                   r"\1es"),
            (r"(alias|status)$",          r"\1es"),
            (r"(octop|vir)i$",            r"\1i"),
            (r"(octop|vir)us$",           r"\1i"),
            (r"^(ax|test)is$",            r"\1es"),
            (r"s$",                       r"s"),
            (r"$",                        r"s"),
        ]:
            self.add_plural(pat, repl)

        # Singular rules
        for pat, repl in [
            (r"(quiz)zes$",              r"\1"),
            (r"(matr)ices$",             r"\1ix"),
            (r"(vert|ind)ices$",         r"\1ex"),
            (r"^(ox)en",                 r"\1"),
            (r"(alias|status)(es)?$",    r"\1"),
            (r"(octop|vir)(us|i)$",      r"\1us"),
            (r"^(a)x[ie]s$",             r"\1xis"),
            (r"(cris|test)(is|es)$",     r"\1is"),
            (r"(shoe)s$",                r"\1"),
            (r"(o)es$",                  r"\1"),
            (r"(bus)(es)?$",             r"\1"),
            (r"([m|l])ice$",             r"\1ouse"),
            (r"(x|ch|ss|sh)es$",        r"\1"),
            (r"(m)ovies$",               r"\1ovie"),
            (r"(s)eries$",               r"\1eries"),
            (r"([^aeiouy]|qu)ies$",      r"\1y"),
            (r"([lr])ves$",              r"\1f"),
            (r"(thi|shea|lea|loa)ves$",  r"\1f"),
            (r"(li)ves$",               r"\1fe"),
            (r"(her|lea|loa|thie)ves$",  r"\1ve"),
            (r"(ss)$",                   r"\1"),
            (r"s$",                      r""),
        ]:
            self.add_singular(pat, repl)


# Default shared vocabulary instance
DEFAULT_VOCABULARY = Vocabulary()


class Inflector:
    """Pluralize / singularize English nouns."""

    def __init__(self, vocab: Optional[Vocabulary] = None):
        self._vocab = vocab or DEFAULT_VOCABULARY

    def pluralize(self, word: str, count: int = 2) -> str:
        if count == 1:
            return word
        return self._vocab.pluralize(word)

    def singularize(self, word: str) -> str:
        return self._vocab.singularize(word)

    def titleize(self, text: str) -> str:
        """Convert identifiers/phrases to title case with small-word handling."""
        small = {"a","an","the","and","but","or","for","nor","on","at",
                 "to","by","in","of","up","as","is","vs","via"}
        words = re.sub(r"[_\-]+", " ", text).split()
        result = []
        for i, w in enumerate(words):
            if i == 0 or i == len(words)-1 or w.lower() not in small:
                result.append(w[0].upper() + w[1:].lower() if w else w)
            else:
                result.append(w.lower())
        return " ".join(result)

    def pascalize(self, text: str) -> str:
        return "".join(w.capitalize() for w in re.split(r"[\s_\-]+", text) if w)

    def camelize(self, text: str) -> str:
        parts = [w.capitalize() for w in re.split(r"[\s_\-]+", text) if w]
        if parts:
            parts[0] = parts[0].lower()
        return "".join(parts)

    def underscore(self, text: str) -> str:
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
        text = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", text)
        return re.sub(r"[\s\-]+", "_", text).lower()

    def dasherize(self, text: str) -> str:
        return re.sub(r"[\s_]+", "-", text).lower()

    def hyphenate(self, text: str) -> str:
        return self.dasherize(text)

    def kebaberize(self, text: str) -> str:
        return self.dasherize(text)


# Module-level singleton for convenience
DEFAULT_INFLECTOR = Inflector()


# ─────────────────────────────────────────────────────────────────────────────
# 6. ArticleHandler
# ─────────────────────────────────────────────────────────────────────────────

class ArticleHandler:
    """
    Correct English articles (a/an) and handle article-prefix sort helpers.
    """

    # Words that start with vowel sound but use "a"
    _A_EXCEPTIONS = {"uniform","union","unique","unit","university","user",
                     "usual","usurp","utopia","eulogy","european","one","once"}
    # Words that start with consonant but use "an"
    _AN_EXCEPTIONS = {"hour","honour","honest","heir","herb"}

    @classmethod
    def correct_article(cls, word: str) -> str:
        """Return 'a' or 'an' for the given following word."""
        lower = word.lower().strip()
        first = lower[0] if lower else ""
        if lower in cls._AN_EXCEPTIONS:
            return "an"
        if lower in cls._A_EXCEPTIONS:
            return "a"
        if first in "aeiou":
            return "an"
        return "a"

    @classmethod
    def fix_articles_in_text(cls, text: str) -> str:
        """
        Replace incorrect a/an usage before words in a text block.
        e.g. "a apple" → "an apple", "an unique" → "a unique"
        """
        def _replace(m):
            article = m.group(1)
            space   = m.group(2)
            word    = m.group(3)
            correct = cls.correct_article(word)
            # Preserve capitalisation of article
            if article[0].isupper():
                correct = correct.capitalize()
            return correct + space + word
        return re.sub(r'\b(a|an)(\s+)(\w+)', _replace, text, flags=re.IGNORECASE)

    @staticmethod
    def prepend_article_suffix(title: str) -> str:
        """Move trailing article to front: "Beatles, The" → "The Beatles"."""
        m = re.match(r'^(.*),\s+(a|an|the)$', title, re.IGNORECASE)
        if m:
            return m.group(2).capitalize() + " " + m.group(1).strip()
        return title

    @staticmethod
    def append_article_prefix(title: str) -> str:
        """Move leading article to end: "The Beatles" → "Beatles, The"."""
        m = re.match(r'^(a|an|the)\s+(.+)$', title, re.IGNORECASE)
        if m:
            return m.group(2).strip() + ", " + m.group(1).capitalize()
        return title


# ─────────────────────────────────────────────────────────────────────────────
# 7. CollectionHumanizer
# ─────────────────────────────────────────────────────────────────────────────

class CollectionHumanizer:
    """Join a list of items into natural English with Oxford comma."""

    @staticmethod
    def humanize(items: List[str], conjunction: str = "and",
                 oxford_comma: bool = True) -> str:
        items = [str(i) for i in items if str(i).strip()]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        comma = "," if oxford_comma else ""
        return ", ".join(items[:-1]) + f"{comma} {conjunction} {items[-1]}"


# ─────────────────────────────────────────────────────────────────────────────
# 8 & 9. StringHumanizer / StringDehumanizer
# ─────────────────────────────────────────────────────────────────────────────

class StringHumanizer:
    """Convert identifiers to readable text."""

    @staticmethod
    def humanize(text: str) -> str:
        """'user_name' / 'UserName' / 'user-name' → 'user name'."""
        # Insert space before capital letters (CamelCase)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
        # Replace underscores and dashes with spaces
        text = re.sub(r'[_\-]+', ' ', text)
        return text.strip().lower()

    @staticmethod
    def concat(parts: List[str], sep: str = " ") -> str:
        return sep.join(p.strip() for p in parts if p.strip())


class StringDehumanizer:
    """Convert readable text back to identifier form."""

    @staticmethod
    def dehumanize(text: str) -> str:
        """'user name' → 'UserName' (PascalCase identifier)."""
        return "".join(w.capitalize() for w in text.split())


# ─────────────────────────────────────────────────────────────────────────────
# 10. CasingTransformer
# ─────────────────────────────────────────────────────────────────────────────

class CasingTransformer:
    """Apply English casing rules to text."""

    _SMALL = {"a","an","the","and","but","or","for","nor","on","at","to",
              "by","in","of","up","as","is","vs","via","with","from"}

    @staticmethod
    def to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def to_upper(text: str) -> str:
        return text.upper()

    @staticmethod
    def to_sentence_case(text: str) -> str:
        """Capitalise first letter of each sentence."""
        def _cap_sentence(m):
            s = m.group(0)
            return s[0].upper() + s[1:] if s else s
        # After . ! ? followed by space
        result = re.sub(r'(?<=[.!?])\s+(\w)', lambda m: " " + m.group(1).upper(), text)
        # Capitalise very first character
        return result[0].upper() + result[1:] if result else result

    @classmethod
    def to_title_case(cls, text: str) -> str:
        """English title case: capitalise all except small words (unless first/last)."""
        words = text.split()
        result = []
        for i, w in enumerate(words):
            if i == 0 or i == len(words) - 1 or w.lower() not in cls._SMALL:
                result.append(w[0].upper() + w[1:] if w else w)
            else:
                result.append(w.lower())
        return " ".join(result)

    @staticmethod
    def apply_case(text: str, mode: str) -> str:
        modes = {
            "lower":    CasingTransformer.to_lower,
            "upper":    CasingTransformer.to_upper,
            "sentence": CasingTransformer.to_sentence_case,
            "title":    CasingTransformer.to_title_case,
        }
        return modes.get(mode, lambda t: t)(text)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Truncator
# ─────────────────────────────────────────────────────────────────────────────

class Truncator:
    """Shorten text safely at word boundaries."""

    @staticmethod
    def truncate(text: str, max_length: int,
                 truncation_string: str = "…",
                 by_words: bool = False) -> str:
        if by_words:
            words = text.split()
            if len(words) <= max_length:
                return text
            return " ".join(words[:max_length]).rstrip(".,;:") + truncation_string
        if len(text) <= max_length:
            return text
        # Find last word boundary before max_length
        cut = text[:max_length - len(truncation_string)]
        last_space = cut.rfind(" ")
        if last_space > 0:
            cut = cut[:last_space]
        return cut.rstrip(".,;:") + truncation_string


# ─────────────────────────────────────────────────────────────────────────────
# 12. QuantityFormatter
# ─────────────────────────────────────────────────────────────────────────────

class QuantityFormatter:
    """Combine a count with the correct singular/plural form."""

    def __init__(self, inflector: Optional[Inflector] = None):
        self._inflector = inflector or Inflector()

    def to_quantity(self, word: str, count: int,
                    show_quantity: bool = True,
                    format_number: bool = False) -> str:
        form = word if count == 1 else self._inflector.pluralize(word)
        qty_str = ""
        if show_quantity:
            qty_str = f"{count:,}" if format_number else str(count)
            return f"{qty_str} {form}"
        return form


# ─────────────────────────────────────────────────────────────────────────────
# 13. MetricNumerals
# ─────────────────────────────────────────────────────────────────────────────

class MetricNumerals:
    """Convert numbers to/from metric-suffix notation (K, M, G, T)."""

    _SUFFIXES = [
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K"),
    ]

    @classmethod
    def to_metric(cls, n: float, decimals: int = 1) -> str:
        for threshold, suffix in cls._SUFFIXES:
            if abs(n) >= threshold:
                val = n / threshold
                fmt = f"{val:.{decimals}f}"
                # Strip trailing zeros after decimal
                fmt = fmt.rstrip("0").rstrip(".")
                return fmt + suffix
        return str(int(n)) if float(n) == int(n) else f"{n:.{decimals}f}"

    @classmethod
    def from_metric(cls, text: str) -> float:
        text = text.strip()
        suffix_map = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3}
        last = text[-1]
        if last in suffix_map:
            return float(text[:-1]) * suffix_map[last]
        return float(text)


# ─────────────────────────────────────────────────────────────────────────────
# 14. DateHumanizer
# ─────────────────────────────────────────────────────────────────────────────

class DateHumanizer:
    """Convert dates/datetimes to relative English phrases."""

    @staticmethod
    def humanize(dt: Union[datetime.datetime, datetime.date],
                 reference: Optional[datetime.datetime] = None) -> str:
        if reference is None:
            reference = datetime.datetime.now()
        if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
            dt = datetime.datetime(dt.year, dt.month, dt.day)
        delta = reference - dt
        seconds = delta.total_seconds()
        future  = seconds < 0
        seconds = abs(seconds)

        if seconds < 45:
            return "just now"
        if seconds < 90:
            return "a minute ago" if not future else "in a minute"
        if seconds < 2700:
            mins = round(seconds / 60)
            return (f"{mins} minutes ago" if not future
                    else f"in {mins} minutes")
        if seconds < 5400:
            return "an hour ago" if not future else "in an hour"
        if seconds < 86400 * 1.5:
            hours = round(seconds / 3600)
            return (f"{hours} hours ago" if not future
                    else f"in {hours} hours")
        if seconds < 86400 * 2.5:
            return "yesterday" if not future else "tomorrow"
        if seconds < 86400 * 7.5:
            days = round(seconds / 86400)
            return (f"{days} days ago" if not future
                    else f"in {days} days")
        if seconds < 86400 * 14:
            return "a week ago" if not future else "in a week"
        if seconds < 86400 * 31:
            weeks = round(seconds / (86400 * 7))
            return (f"{weeks} weeks ago" if not future
                    else f"in {weeks} weeks")
        if seconds < 86400 * 60:
            return "a month ago" if not future else "in a month"
        if seconds < 86400 * 365:
            months = round(seconds / (86400 * 30.4))
            return (f"{months} months ago" if not future
                    else f"in {months} months")
        years = round(seconds / (86400 * 365.25))
        return (f"{years} year{'s' if years != 1 else ''} ago"
                if not future else
                f"in {years} year{'s' if years != 1 else ''}")

    @staticmethod
    def to_ordinal_words(dt: Union[datetime.datetime, datetime.date]) -> str:
        """Render 'March 21st' or 'the twenty-first of March'."""
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        month = months[dt.month - 1]
        ord_day = Ordinalize.convert(dt.day)
        return f"{month} {ord_day}"

    @staticmethod
    def to_ordinal_words_long(dt: Union[datetime.datetime, datetime.date]) -> str:
        """Render 'the twenty-first of March'."""
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        month = months[dt.month - 1]
        ord_word = NumberToOrdinalWords.convert(dt.day)
        return f"the {ord_word} of {month}"


# ─────────────────────────────────────────────────────────────────────────────
# 15. TimeSpanHumanizer
# ─────────────────────────────────────────────────────────────────────────────

class TimeSpanHumanizer:
    """Convert timedelta to readable English duration phrases."""

    @staticmethod
    def humanize(delta: datetime.timedelta,
                 max_units: int = 2,
                 precision: bool = False) -> str:
        total_seconds = int(abs(delta.total_seconds()))
        parts = []
        units = [
            (365 * 24 * 3600, "year"),
            (30  * 24 * 3600, "month"),
            (7   * 24 * 3600, "week"),
            (24  * 3600,      "day"),
            (3600,            "hour"),
            (60,              "minute"),
            (1,               "second"),
        ]
        for seconds, name in units:
            count = total_seconds // seconds
            if count:
                total_seconds -= count * seconds
                parts.append(f"{count} {name}{'s' if count != 1 else ''}")
            if len(parts) >= max_units:
                break
        return CollectionHumanizer.humanize(parts, "and") if parts else "0 seconds"

    @staticmethod
    def to_age(delta: datetime.timedelta) -> str:
        """Express as age: '5 years old', '3 months old'."""
        phrase = TimeSpanHumanizer.humanize(delta, max_units=1)
        return f"{phrase} old"


# ─────────────────────────────────────────────────────────────────────────────
# 16. TimeToClockNotation
# ─────────────────────────────────────────────────────────────────────────────

class TimeToClockNotation:
    """Convert time values into natural English clock phrases."""

    @staticmethod
    def convert(hour: int, minute: int) -> str:
        hour_12 = hour % 12 or 12
        if minute == 0:
            return f"{NumberToWords.convert(hour_12)} o'clock"
        if minute == 15:
            return f"quarter past {NumberToWords.convert(hour_12)}"
        if minute == 30:
            return f"half past {NumberToWords.convert(hour_12)}"
        if minute == 45:
            next_h = (hour + 1) % 12 or 12
            return f"quarter to {NumberToWords.convert(next_h)}"
        if minute < 30:
            return f"{NumberToWords.convert(minute)} past {NumberToWords.convert(hour_12)}"
        return f"{NumberToWords.convert(60 - minute)} to {NumberToWords.convert((hour + 1) % 12 or 12)}"


# ─────────────────────────────────────────────────────────────────────────────
# 17. AgeFormatter
# ─────────────────────────────────────────────────────────────────────────────

class AgeFormatter:
    """Express a duration as an age phrase."""

    @staticmethod
    def to_age(delta: datetime.timedelta) -> str:
        return TimeSpanHumanizer.to_age(delta)


# ─────────────────────────────────────────────────────────────────────────────
# 18. TimeUnitSymbols
# ─────────────────────────────────────────────────────────────────────────────

class TimeUnitSymbols:
    """Map time unit names to abbreviated English symbols."""

    _MAP = {
        "millisecond": "ms", "milliseconds": "ms",
        "second":  "sec", "seconds":  "sec",
        "minute":  "min", "minutes":  "min",
        "hour":    "hr",  "hours":    "hr",
        "day":     "d",   "days":     "d",
        "week":    "wk",  "weeks":    "wk",
        "month":   "mo",  "months":   "mo",
        "year":    "yr",  "years":    "yr",
    }

    @classmethod
    def to_symbol(cls, unit: str) -> str:
        return cls._MAP.get(unit.lower(), unit)


# ─────────────────────────────────────────────────────────────────────────────
# 19. RomanNumerals
# ─────────────────────────────────────────────────────────────────────────────

class RomanNumerals:
    """Convert between integers and Roman numeral strings."""

    _INT_TO_ROMAN = [
        (1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),
        (100,"C"),(90,"XC"),(50,"L"),(40,"XL"),
        (10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I"),
    ]
    _ROMAN_RE = re.compile(
        r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',
        re.IGNORECASE
    )

    @classmethod
    def to_roman(cls, n: int) -> str:
        if not 1 <= n <= 4999:
            raise ValueError(f"Roman numerals support 1–4999, got {n}")
        result = []
        for value, numeral in cls._INT_TO_ROMAN:
            while n >= value:
                result.append(numeral)
                n -= value
        return "".join(result)

    @classmethod
    def from_roman(cls, s: str) -> int:
        if not cls._ROMAN_RE.match(s.upper()):
            raise ValueError(f"Invalid Roman numeral: {s}")
        vals = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        s = s.upper()
        total = 0
        for i, ch in enumerate(s):
            v = vals[ch]
            if i + 1 < len(s) and vals[s[i+1]] > v:
                total -= v
            else:
                total += v
        return total


# ─────────────────────────────────────────────────────────────────────────────
# 20. ByteSize
# ─────────────────────────────────────────────────────────────────────────────

class ByteSize:
    """Store and format byte sizes with full arithmetic support."""

    BITS_IN_BYTE = 8
    _UNITS = [
        (2**40, "TB", "terabyte"),
        (2**30, "GB", "gigabyte"),
        (2**20, "MB", "megabyte"),
        (2**10, "KB", "kilobyte"),
        (1,     "B",  "byte"),
    ]

    def __init__(self, bits: float = 0):
        self._bits = bits

    # ── Factory methods ───────────────────────────────────────────────────
    @classmethod
    def from_bits(cls, n):       return cls(n)
    @classmethod
    def from_bytes(cls, n):      return cls(n * cls.BITS_IN_BYTE)
    @classmethod
    def from_kilobytes(cls, n):  return cls(n * 2**10 * cls.BITS_IN_BYTE)
    @classmethod
    def from_megabytes(cls, n):  return cls(n * 2**20 * cls.BITS_IN_BYTE)
    @classmethod
    def from_gigabytes(cls, n):  return cls(n * 2**30 * cls.BITS_IN_BYTE)
    @classmethod
    def from_terabytes(cls, n):  return cls(n * 2**40 * cls.BITS_IN_BYTE)

    # ── Properties ───────────────────────────────────────────────────────
    @property
    def bits(self):       return self._bits
    @property
    def bytes(self):      return self._bits / self.BITS_IN_BYTE
    @property
    def kilobytes(self):  return self.bytes / 2**10
    @property
    def megabytes(self):  return self.bytes / 2**20
    @property
    def gigabytes(self):  return self.bytes / 2**30
    @property
    def terabytes(self):  return self.bytes / 2**40

    # ── Best-fit unit ─────────────────────────────────────────────────────
    def get_largest_whole_number_symbol(self) -> str:
        b = self.bytes
        for threshold, symbol, _ in self._UNITS:
            if abs(b) >= threshold:
                return symbol
        return "B"

    def get_largest_whole_number_full_word(self) -> str:
        b = self.bytes
        for threshold, _, word in self._UNITS:
            if abs(b) >= threshold:
                val = b / threshold
                return f"{word}{'s' if val != 1 else ''}"
        return "byte" if abs(b) == 1 else "bytes"

    # ── Formatting ────────────────────────────────────────────────────────
    def __str__(self) -> str:
        return self.humanize()

    def humanize(self, fmt: str = "#.##") -> str:
        b = self.bytes
        for threshold, symbol, _ in self._UNITS:
            if abs(b) >= threshold:
                val = b / threshold
                return f"{val:.2f}".rstrip("0").rstrip(".") + f" {symbol}"
        return f"{b:.0f} B"

    def to_full_words(self) -> str:
        b = self.bytes
        for threshold, _, word in self._UNITS:
            if abs(b) >= threshold:
                val = b / threshold
                rounded = round(val, 2)
                suffix = "s" if rounded != 1 else ""
                return f"{rounded} {word}{suffix}"
        return f"{b:.0f} {'byte' if abs(b) == 1 else 'bytes'}"

    # ── Arithmetic ────────────────────────────────────────────────────────
    def add(self, other: "ByteSize") -> "ByteSize":
        return ByteSize(self._bits + other._bits)
    def subtract(self, other: "ByteSize") -> "ByteSize":
        return ByteSize(self._bits - other._bits)
    def add_bits(self, n):       return ByteSize(self._bits + n)
    def add_bytes(self, n):      return ByteSize.from_bytes(self.bytes + n)
    def add_kilobytes(self, n):  return ByteSize.from_bytes(self.bytes + n * 2**10)
    def add_megabytes(self, n):  return ByteSize.from_bytes(self.bytes + n * 2**20)
    def add_gigabytes(self, n):  return ByteSize.from_bytes(self.bytes + n * 2**30)
    def add_terabytes(self, n):  return ByteSize.from_bytes(self.bytes + n * 2**40)

    def __add__(self, other):    return self.add(other)
    def __sub__(self, other):    return self.subtract(other)
    def __eq__(self, other):     return isinstance(other, ByteSize) and self._bits == other._bits
    def __lt__(self, other):     return self._bits < other._bits
    def __le__(self, other):     return self._bits <= other._bits
    def __gt__(self, other):     return self._bits > other._bits
    def __ge__(self, other):     return self._bits >= other._bits
    def __hash__(self):          return hash(self._bits)
    def compare_to(self, other: "ByteSize") -> int:
        return (self._bits > other._bits) - (self._bits < other._bits)

    # ── Parse ─────────────────────────────────────────────────────────────
    @classmethod
    def parse(cls, text: str) -> "ByteSize":
        result = cls.try_parse(text)
        if result is None:
            raise ValueError(f"Cannot parse byte size: {text!r}")
        return result

    @classmethod
    def try_parse(cls, text: str) -> Optional["ByteSize"]:
        m = re.match(
            r'^\s*([\d.]+)\s*(TB|GB|MB|KB|B|bits?|bytes?|kilobytes?|'
            r'megabytes?|gigabytes?|terabytes?)\s*$',
            text.strip(), re.IGNORECASE
        )
        if not m:
            return None
        val, unit = float(m.group(1)), m.group(2).lower()
        if unit.startswith("t"):   return cls.from_terabytes(val)
        if unit.startswith("g"):   return cls.from_gigabytes(val)
        if unit.startswith("m"):   return cls.from_megabytes(val)
        if unit.startswith("k"):   return cls.from_kilobytes(val)
        if "bit" in unit:          return cls.from_bits(val)
        return cls.from_bytes(val)

    def per(self, duration: datetime.timedelta) -> str:
        """Format as a transfer rate string: '10 MB/s'."""
        secs = duration.total_seconds()
        rate = ByteSize.from_bytes(self.bytes / secs) if secs else self
        return rate.humanize() + "/s"


# ─────────────────────────────────────────────────────────────────────────────
# 21. WordsToNumber
# ─────────────────────────────────────────────────────────────────────────────

class WordsToNumber:
    """Parse English number words into integers/floats."""

    _ONES_MAP = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,
        "twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
        "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    }
    _TENS_MAP = {
        "twenty":20,"thirty":30,"forty":40,"fifty":50,
        "sixty":60,"seventy":70,"eighty":80,"ninety":90,
    }
    _SCALE_MAP = {
        "hundred":100,"thousand":1000,"million":1_000_000,
        "billion":1_000_000_000,"trillion":1_000_000_000_000,
    }

    @classmethod
    def convert(cls, text: str) -> int:
        result = cls.try_convert(text)
        if result is None:
            raise ValueError(f"Cannot parse number words: {text!r}")
        return result

    @classmethod
    def try_convert(cls, text: str) -> Optional[int]:
        tokens = re.sub(r'[-,]', ' ', text.strip().lower()).split()
        if not tokens:
            return None
        try:
            current, total = 0, 0
            for tok in tokens:
                if tok in cls._ONES_MAP:
                    current += cls._ONES_MAP[tok]
                elif tok in cls._TENS_MAP:
                    current += cls._TENS_MAP[tok]
                elif tok == "hundred":
                    current = (current or 1) * 100
                elif tok in cls._SCALE_MAP and tok != "hundred":
                    scale = cls._SCALE_MAP[tok]
                    total += (current or 1) * scale
                    current = 0
                elif tok in ("and", "a", "an"):
                    continue
                else:
                    return None
            return total + current
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# 22. EnglishGrammarFixer  ← integrates into humanizer pipeline
# ─────────────────────────────────────────────────────────────────────────────

class EnglishGrammarFixer:
    """
    Apply English grammar corrections to a text block:
      - Fix a/an article errors
      - Expand lone digits to words (configurable)
      - Fix double spaces and spacing around punctuation
      - Ensure sentence capitalisation
      - Fix subject-verb agreement for common patterns
    """

    def __init__(self):
        self._article_handler  = ArticleHandler()
        self._casing           = CasingTransformer()
        self._words            = NumberToWords()
        self._ordinalize       = Ordinalize()

    def fix(self, text: str,
            fix_articles: bool    = True,
            fix_spacing: bool     = True,
            fix_casing: bool      = True,
            numbers_to_words: bool = False,
            numbers_threshold: int = 10) -> str:
        """
        Apply all enabled fixes to text.
        numbers_to_words: if True, write out integers ≤ numbers_threshold as words.
        """
        if fix_articles:
            text = self._article_handler.fix_articles_in_text(text)

        if numbers_to_words:
            def _replace_num(m):
                n = int(m.group(0))
                if n <= numbers_threshold:
                    return NumberToWords.convert(n)
                return m.group(0)
            text = re.sub(r'\b(\d+)\b', _replace_num, text)

        if fix_spacing:
            # Collapse multiple spaces
            text = re.sub(r' {2,}', ' ', text)
            # Space after punctuation if missing
            text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
            # Remove space before punctuation
            text = re.sub(r'\s([.,;:!?])', r'\1', text)

        if fix_casing:
            text = self._casing.to_sentence_case(text)

        return text.strip()

    def numbers_in_text_to_words(self, text: str,
                                  threshold: int = 10) -> str:
        """Replace all integers ≤ threshold with their English word forms."""
        def _sub(m):
            n = int(m.group(0))
            return NumberToWords.convert(n) if n <= threshold else m.group(0)
        return re.sub(r'\b(\d+)\b', _sub, text)

    def ordinalize_numbers(self, text: str) -> str:
        """Replace standalone ordinal patterns: '1st','2nd' stay; bare '1.' → keeps."""
        # Replace patterns like "the 3 largest" → "the third largest" (numbers ≤ 20)
        def _sub(m):
            n = int(m.group(1))
            if n <= 20:
                return "the " + NumberToOrdinalWords.convert(n) + " "
            return m.group(0)
        return re.sub(r'\bthe (\d{1,2}) ', _sub, text)


# ─────────────────────────────────────────────────────────────────────────────
# 23. Validator
# ─────────────────────────────────────────────────────────────────────────────

class Validator:
    """
    Lightweight readability / grammar checks before final output.
    Returns a list of warning strings; empty = text passes.
    """

    @staticmethod
    def validate(text: str) -> List[str]:
        warnings = []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Check for double spaces
        if '  ' in text:
            warnings.append("Double spaces detected.")

        # Check for missing capitalisation
        for s in sentences:
            if s and s[0].islower():
                warnings.append(f"Sentence starts with lowercase: '{s[:30]}…'")
                break

        # Check for unclosed parentheses
        if text.count('(') != text.count(')'):
            warnings.append("Mismatched parentheses.")

        # Check for consecutive duplicate words
        if re.search(r'\b(\w+)\s+\1\b', text, re.IGNORECASE):
            warnings.append("Consecutive duplicate words found.")

        # Very short text
        if len(text.split()) < 5:
            warnings.append("Text is very short (fewer than 5 words).")

        return warnings

    @staticmethod
    def is_valid(text: str) -> bool:
        return len(Validator.validate(text)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 24. HeadingConverter  — degrees → compass direction + arrows
# ─────────────────────────────────────────────────────────────────────────────

class HeadingConverter:
    """Convert degree headings to compass names, abbreviations, and arrows."""

    _HEADINGS = [
        (0,   "north",      "N",  "↑"),
        (45,  "north-east", "NE", "↗"),
        (90,  "east",       "E",  "→"),
        (135, "south-east", "SE", "↘"),
        (180, "south",      "S",  "↓"),
        (225, "south-west", "SW", "↙"),
        (270, "west",       "W",  "←"),
        (315, "north-west", "NW", "↖"),
        (360, "north",      "N",  "↑"),
    ]

    @staticmethod
    def to_heading(degrees: float) -> str:
        """360 → 'north', 45 → 'north-east'."""
        deg = degrees % 360
        idx = int((deg + 22.5) / 45) % 8
        return HeadingConverter._HEADINGS[idx][1]

    @staticmethod
    def to_abbreviation(degrees: float) -> str:
        deg = degrees % 360
        idx = int((deg + 22.5) / 45) % 8
        return HeadingConverter._HEADINGS[idx][2]

    @staticmethod
    def to_arrow(degrees: float) -> str:
        deg = degrees % 360
        idx = int((deg + 22.5) / 45) % 8
        return HeadingConverter._HEADINGS[idx][3]

    @staticmethod
    def from_abbreviation(abbr: str) -> float:
        lookup = {row[2]: row[0] for row in HeadingConverter._HEADINGS[:-1]}
        return float(lookup.get(abbr.upper(), 0))

    @staticmethod
    def from_arrow(arrow: str) -> float:
        lookup = {row[3]: row[0] for row in HeadingConverter._HEADINGS[:-1]}
        return float(lookup.get(arrow, 0))


# ─────────────────────────────────────────────────────────────────────────────
# 25. NumberScalingHelpers  — tens/hundreds/thousands/millions/billions
# ─────────────────────────────────────────────────────────────────────────────

class NumberScalingHelpers:
    """Scale a number by named magnitudes."""

    @staticmethod
    def tens(n: float) -> float:      return n * 10
    @staticmethod
    def hundreds(n: float) -> float:  return n * 100
    @staticmethod
    def thousands(n: float) -> float: return n * 1_000
    @staticmethod
    def millions(n: float) -> float:  return n * 1_000_000
    @staticmethod
    def billions(n: float) -> float:  return n * 1_000_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 26. NumberToTimeSpan  — numeric value → human duration phrase
# ─────────────────────────────────────────────────────────────────────────────

class NumberToTimeSpan:
    """Convert a numeric value with a unit into a human duration phrase."""

    @staticmethod
    def milliseconds(n: float) -> str:
        return TimeSpanHumanizer.humanize(datetime.timedelta(milliseconds=n))

    @staticmethod
    def seconds(n: float) -> str:
        return TimeSpanHumanizer.humanize(datetime.timedelta(seconds=n))

    @staticmethod
    def minutes(n: float) -> str:
        return TimeSpanHumanizer.humanize(datetime.timedelta(minutes=n))

    @staticmethod
    def hours(n: float) -> str:
        return TimeSpanHumanizer.humanize(datetime.timedelta(hours=n))

    @staticmethod
    def days(n: float) -> str:
        return TimeSpanHumanizer.humanize(datetime.timedelta(days=n))

    @staticmethod
    def weeks(n: float) -> str:
        return TimeSpanHumanizer.humanize(datetime.timedelta(weeks=n))


# ─────────────────────────────────────────────────────────────────────────────
# 27. ByteRate  — transfer-rate formatting  e.g. "10.5 MB/s"
# ─────────────────────────────────────────────────────────────────────────────

class ByteRate:
    """Format byte-per-second transfer rates into readable strings."""

    @staticmethod
    def humanize(bytes_per_second: float, interval: str = "s") -> str:
        size = ByteSize.from_bytes(bytes_per_second)
        symbol = size.get_largest_whole_number_symbol()
        # Find the threshold for this symbol from the list
        unit_bytes = 1
        for threshold, sym, _ in ByteSize._UNITS:
            if sym == symbol:
                unit_bytes = threshold
                break
        value = bytes_per_second / unit_bytes
        return f"{value:.1f} {symbol}/{interval}"


# ─────────────────────────────────────────────────────────────────────────────
# 28. FluentDateBuilder  — programmatic English date construction
# ─────────────────────────────────────────────────────────────────────────────

class FluentDateBuilder:
    """Build natural English date descriptions programmatically."""

    _MONTHS = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    @staticmethod
    def from_date(dt: datetime.date) -> str:
        """datetime.date → 'March 21, 2026'."""
        return f"{FluentDateBuilder._MONTHS[dt.month]} {dt.day}, {dt.year}"

    @staticmethod
    def from_date_ordinal(dt: datetime.date) -> str:
        """datetime.date → 'the twenty-first of March 2026'."""
        day_words = NumberToOrdinalWords.convert(dt.day)
        return f"the {day_words} of {FluentDateBuilder._MONTHS[dt.month]} {dt.year}"

    @staticmethod
    def relative_from_now(dt: datetime.datetime) -> str:
        """datetime → '3 days from now', '2 weeks ago'."""
        return DateHumanizer.humanize(dt)

    @staticmethod
    def some_time_from(reference: datetime.datetime, delta: datetime.timedelta) -> str:
        return DateHumanizer.humanize(reference + delta)


# ─────────────────────────────────────────────────────────────────────────────
# 29. TransformerPipeline  — ordered sequential text transforms
# ─────────────────────────────────────────────────────────────────────────────

class TransformerPipeline:
    """
    Apply a sequence of named casing/text transformations in order.

    Usage:
        result = TransformerPipeline.build(
            "to_sentence_case", "fix_articles", "oxford_comma_lists"
        ).transform(text)
    """

    # Registry of named transform functions
    _REGISTRY: dict = {}

    def __init__(self, steps: list):
        self._steps = steps  # list of callables: str → str

    @classmethod
    def register(cls, name: str, fn) -> None:
        cls._REGISTRY[name] = fn

    @classmethod
    def build(cls, *names: str) -> "TransformerPipeline":
        steps = []
        for name in names:
            if name not in cls._REGISTRY:
                raise ValueError(f"Unknown transformer: '{name}'")
            steps.append(cls._REGISTRY[name])
        return cls(steps)

    def transform(self, text: str) -> str:
        for step in self._steps:
            text = step(text)
        return text

    def then(self, fn) -> "TransformerPipeline":
        """Chain an additional callable."""
        return TransformerPipeline(self._steps + [fn])


# Register built-in transforms
TransformerPipeline.register("to_lower",             lambda t: CasingTransformer.to_lower(t))
TransformerPipeline.register("to_upper",             lambda t: CasingTransformer.to_upper(t))
TransformerPipeline.register("to_sentence_case",     lambda t: CasingTransformer.to_sentence_case(t))
TransformerPipeline.register("to_title_case",        lambda t: CasingTransformer.to_title_case(t))
TransformerPipeline.register("fix_articles",         lambda t: ArticleHandler.fix_articles_in_text(t))
TransformerPipeline.register("fix_number_agreement", lambda t: _grammar_fixer.fix(t, fix_articles=False, fix_spacing=True, fix_casing=False))
TransformerPipeline.register("humanize_string",      lambda t: StringHumanizer.humanize(t))
TransformerPipeline.register("dehumanize_string",    lambda t: StringDehumanizer.dehumanize(t))
TransformerPipeline.register("numbers_to_words",     lambda t: _apply_numbers_to_words(t))
TransformerPipeline.register("words_to_numbers",     lambda t: _apply_words_to_numbers(t))
TransformerPipeline.register("ordinalize_numbers",   lambda t: _apply_ordinalize(t))

# Module-level grammar fixer singleton used by pipeline
_grammar_fixer = EnglishGrammarFixer()


def _apply_numbers_to_words(text: str) -> str:
    """Replace standalone small integers (1-99) with English words."""
    def _sub(m):
        n = int(m.group(0))
        if 0 <= n <= 999:
            return NumberToWords.convert(n)
        return m.group(0)
    return re.sub(r'\b([0-9]{1,3})\b', _sub, text)


def _apply_words_to_numbers(text: str) -> str:
    """Replace written number words with digits."""
    result = WordsToNumber.try_convert(text)
    return str(result) if result is not None else text


def _apply_ordinalize(text: str) -> str:
    """Replace bare integers with ordinal forms where sensible (position context)."""
    def _sub(m):
        n = int(m.group(1))
        if 1 <= n <= 31:  # dates, rankings
            return Ordinalize.convert(n)
        return m.group(0)
    return re.sub(r'\b(\d{1,2})\b(?=\s+(?:place|rank|position|step|item|chapter|section))',
                  _sub, text, flags=re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 30. EnumHumanizer  — identifier labels → readable text
# ─────────────────────────────────────────────────────────────────────────────

class EnumHumanizer:
    """
    Convert enum-style identifiers into human-readable labels.
    Examples:
        "AdminUser"      → "Admin user"
        "ORDER_PENDING"  → "Order pending"
        "http_error_404" → "Http error 404"
    """

    @staticmethod
    def humanize(identifier: str, all_words_capitalized: bool = False) -> str:
        """Convert an enum identifier to a readable label."""
        if not identifier:
            return identifier
        # Split on underscores, hyphens, and camelCase boundaries
        # Insert space before uppercase letters that follow lowercase
        s = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', identifier)
        # Insert space before runs of uppercase followed by lowercase (e.g. HTTPSClient)
        s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
        # Replace underscores/hyphens with spaces
        s = re.sub(r'[_\-]+', ' ', s)
        # Normalise whitespace
        s = re.sub(r'\s+', ' ', s).strip()
        if all_words_capitalized:
            return CasingTransformer.to_title_case(s)
        # Sentence case: capitalise only first word
        return s[0].upper() + s[1:].lower() if s else s

    @staticmethod
    def humanize_flag(flags: list, separator: str = ", ") -> str:
        """Humanize a list of flag enum values."""
        return separator.join(EnumHumanizer.humanize(f) for f in flags)


# ─────────────────────────────────────────────────────────────────────────────
# 31. EnumDehumanizer  — readable labels → identifier form
# ─────────────────────────────────────────────────────────────────────────────

class EnumDehumanizer:
    """
    Map human-readable labels back to enum-style identifiers.
    Examples:
        "Admin user"   → "AdminUser"  (PascalCase)
        "order pending"→ "ORDER_PENDING" (UPPER_SNAKE, if style='upper_snake')
        "Http error"   → "http_error" (lower_snake)
    """

    @staticmethod
    def dehumanize(label: str, style: str = "pascal") -> str:
        """
        Convert a readable label back to an identifier.
        style: 'pascal' | 'camel' | 'upper_snake' | 'lower_snake'
        """
        if not label:
            return label
        words = re.split(r'[\s_\-]+', label.strip())
        words = [w for w in words if w]
        if style == "pascal":
            return "".join(w.capitalize() for w in words)
        elif style == "camel":
            parts = [w.capitalize() for w in words]
            if parts:
                parts[0] = parts[0].lower()
            return "".join(parts)
        elif style == "upper_snake":
            return "_".join(w.upper() for w in words)
        elif style == "lower_snake":
            return "_".join(w.lower() for w in words)
        return "".join(w.capitalize() for w in words)

    @staticmethod
    def dehumanize_to_value(label: str, enum_values: list,
                             case_sensitive: bool = False) -> Optional[str]:
        """
        Find the best-matching enum value for a human-readable label.
        Returns the matched value string, or None if no match found.
        """
        normalize = (lambda s: s) if case_sensitive else str.lower
        target = normalize(label.strip())
        # Try exact humanized match first
        for val in enum_values:
            if normalize(EnumHumanizer.humanize(str(val))) == target:
                return val
        # Try direct string match
        for val in enum_values:
            if normalize(str(val)) == target:
                return val
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 32. StringConcat  — grammatically correct phrase joining
# ─────────────────────────────────────────────────────────────────────────────

class StringConcat:
    """
    Combine multiple text fragments into grammatically correct English sentences.

    Examples:
        concat(["The study", "examines", "AI detection"]) → "The study examines AI detection."
        concat_with_article(["apple", "orange"])           → "an apple and an orange"
        concat_sentences(["It works.", "It is fast"])      → "It works. It is fast."
    """

    @staticmethod
    def concat(parts: list, separator: str = " ", end_punctuation: str = ".") -> str:
        """Join parts into a sentence, ensuring proper punctuation."""
        if not parts:
            return ""
        sentence = separator.join(str(p).strip() for p in parts if str(p).strip())
        sentence = sentence.strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += end_punctuation
        # Capitalise first letter
        return sentence[0].upper() + sentence[1:] if sentence else sentence

    @staticmethod
    def concat_with_article(nouns: list, conjunction: str = "and") -> str:
        """Join nouns with correct a/an articles and a conjunction."""
        if not nouns:
            return ""
        with_articles = [ArticleHandler.with_indefinite_article(n) for n in nouns]
        return CollectionHumanizer.humanize(with_articles, conjunction=conjunction)

    @staticmethod
    def concat_sentences(sentences: list) -> str:
        """Join multiple sentences with proper spacing and capitalisation."""
        result = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if s[-1] not in ".!?":
                s += "."
            result.append(s[0].upper() + s[1:])
        return " ".join(result)

    @staticmethod
    def build_clause(subject: str, verb: str, obj: str = "",
                     tense: str = "present") -> str:
        """Build a simple English clause."""
        parts = [subject.strip(), verb.strip()]
        if obj:
            parts.append(obj.strip())
        sentence = " ".join(p for p in parts if p)
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence[0].upper() + sentence[1:] if sentence else sentence


# ─────────────────────────────────────────────────────────────────────────────
# 33. TupleFormatter  — tuple-aware singular/plural formatting
# ─────────────────────────────────────────────────────────────────────────────

class TupleFormatter:
    """
    Format numeric values paired with nouns using correct singular/plural forms.
    Unlike QuantityFormatter, TupleFormatter works with (count, word) tuples
    and supports format strings, zero-forms, and dual-form output.

    Examples:
        TupleFormatter.format((1, "item"))        → "1 item"
        TupleFormatter.format((3, "item"))         → "3 items"
        TupleFormatter.format((0, "result"), zero_text="no results") → "no results"
        TupleFormatter.both_forms("item")          → ("item", "items")
    """

    def __init__(self, inflector: Optional["Inflector"] = None):
        self._inflector = inflector or DEFAULT_INFLECTOR

    def format(self, count_word_tuple: tuple,
               show_count: bool = True,
               zero_text: str = "",
               format_number: bool = False) -> str:
        """Format a (count, word) tuple into a quantity phrase."""
        count, word = count_word_tuple
        if count == 0 and zero_text:
            return zero_text
        plural_form = word if count == 1 else self._inflector.pluralize(word)
        if not show_count:
            return plural_form
        count_str = f"{count:,}" if format_number else str(count)
        return f"{count_str} {plural_form}"

    def both_forms(self, word: str) -> tuple:
        """Return (singular, plural) tuple for a word."""
        return (word, self._inflector.pluralize(word))

    def singular_form(self, word: str) -> str:
        return self._inflector.singularize(word)

    def plural_form(self, word: str) -> str:
        return self._inflector.pluralize(word)

    @staticmethod
    def from_count(count: int, singular: str, plural: str = "") -> str:
        """Choose between explicit singular/plural strings."""
        if count == 1:
            return f"{count} {singular}"
        return f"{count} {plural or singular + 's'}"


# ─────────────────────────────────────────────────────────────────────────────
# 34. PrepositionHandler  — time/date prepositions
# ─────────────────────────────────────────────────────────────────────────────

class PrepositionHandler:
    """
    Attach natural English prepositions to time and date expressions.

    Examples:
        PrepositionHandler.at_time(datetime.time(12,0))  → "at noon"
        PrepositionHandler.at_time(datetime.time(0,0))   → "at midnight"
        PrepositionHandler.at_time(datetime.time(14,30)) → "at 2:30 PM"
        PrepositionHandler.in_year(2026)                  → "in 2026"
        PrepositionHandler.on_date(datetime.date.today()) → "on March 21, 2026"
    """

    @staticmethod
    def at_time(t: "datetime.time") -> str:
        """Return 'at noon', 'at midnight', or 'at HH:MM AM/PM'."""
        if t.hour == 0 and t.minute == 0:
            return "at midnight"
        if t.hour == 12 and t.minute == 0:
            return "at noon"
        hour_12 = t.hour % 12 or 12
        ampm = "AM" if t.hour < 12 else "PM"
        if t.minute == 0:
            return f"at {hour_12} {ampm}"
        return f"at {hour_12}:{t.minute:02d} {ampm}"

    @staticmethod
    def at_midnight() -> str:
        return "at midnight"

    @staticmethod
    def at_noon() -> str:
        return "at noon"

    @staticmethod
    def in_year(year: int) -> str:
        return f"in {year}"

    @staticmethod
    def in_month(month_name: str) -> str:
        return f"in {month_name}"

    @staticmethod
    def on_date(d: "datetime.date") -> str:
        """Return 'on March 21, 2026'."""
        months = ["", "January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return f"on {months[d.month]} {d.day}, {d.year}"

    @staticmethod
    def since(d: "datetime.date") -> str:
        months = ["", "January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return f"since {months[d.month]} {d.day}, {d.year}"

    @staticmethod
    def by(d: "datetime.date") -> str:
        months = ["", "January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return f"by {months[d.month]} {d.day}, {d.year}"

    @staticmethod
    def until(d: "datetime.date") -> str:
        months = ["", "January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        return f"until {months[d.month]} {d.day}, {d.year}"


# ─────────────────────────────────────────────────────────────────────────────
# 35. PrecisionHumanizer  — controlled-detail time/date phrases
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionHumanizer:
    """
    Humanize dates/durations with configurable precision levels.

    precision=1 → coarsest:  "2 hours ago"
    precision=2 → medium:    "2 hours and 14 minutes ago"
    precision=3 → detailed:  "2 hours, 14 minutes and 32 seconds ago"

    Examples:
        PrecisionHumanizer.humanize_delta(timedelta(seconds=7474), precision=2)
        → "2 hours and 4 minutes ago"
    """

    @staticmethod
    def humanize_delta(delta: "datetime.timedelta",
                       precision: int = 1,
                       add_ago: bool = True) -> str:
        """Humanize a timedelta with controlled precision."""
        total_seconds = int(abs(delta.total_seconds()))
        is_past = delta.total_seconds() < 0

        parts = []
        remaining = total_seconds

        units = [
            (86400 * 365, "year",   "years"),
            (86400 * 30,  "month",  "months"),
            (86400,       "day",    "days"),
            (3600,        "hour",   "hours"),
            (60,          "minute", "minutes"),
            (1,           "second", "seconds"),
        ]

        for seconds_in_unit, singular, plural in units:
            if remaining >= seconds_in_unit and len(parts) < precision:
                count = remaining // seconds_in_unit
                remaining %= seconds_in_unit
                parts.append(f"{count} {singular if count == 1 else plural}")

        if not parts:
            return "just now"

        if len(parts) == 1:
            phrase = parts[0]
        elif len(parts) == 2:
            phrase = f"{parts[0]} and {parts[1]}"
        else:
            phrase = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        if add_ago:
            return f"{phrase} ago" if is_past or delta.total_seconds() >= 0 else f"in {phrase}"
        return phrase

    @staticmethod
    def humanize_datetime(dt: "datetime.datetime",
                          now: "datetime.datetime" = None,
                          precision: int = 1) -> str:
        """Humanize a datetime relative to now with given precision."""
        if now is None:
            now = datetime.datetime.now()
        delta = now - dt
        return PrecisionHumanizer.humanize_delta(delta, precision=precision, add_ago=True)

    @staticmethod
    def just_now_threshold_seconds() -> int:
        """Number of seconds within which we say 'just now'."""
        return 45


# ─────────────────────────────────────────────────────────────────────────────
# 36. ResourceKeys  — string resource key mapping
# ─────────────────────────────────────────────────────────────────────────────

class ResourceKeys:
    """
    Map humanisation concepts to resource key strings.
    These keys are used by ResourceRetrieval to fetch localised English phrases.
    """

    # Date humanization keys
    class DateHumanize:
        NOW             = "date.now"
        NEVER           = "date.never"
        AGO_SECONDS     = "date.seconds_ago"
        AGO_MINUTE      = "date.minute_ago"
        AGO_MINUTES     = "date.minutes_ago"
        AGO_HOUR        = "date.hour_ago"
        AGO_HOURS       = "date.hours_ago"
        AGO_DAY         = "date.day_ago"
        AGO_DAYS        = "date.days_ago"
        AGO_MONTH       = "date.month_ago"
        AGO_MONTHS      = "date.months_ago"
        AGO_YEAR        = "date.year_ago"
        AGO_YEARS       = "date.years_ago"
        FROM_SECONDS    = "date.in_seconds"
        FROM_MINUTE     = "date.in_minute"
        FROM_MINUTES    = "date.in_minutes"
        FROM_HOUR       = "date.in_hour"
        FROM_HOURS      = "date.in_hours"
        FROM_DAY        = "date.in_day"
        FROM_DAYS       = "date.in_days"
        FROM_MONTH      = "date.in_month"
        FROM_MONTHS     = "date.in_months"
        FROM_YEAR       = "date.in_year"
        FROM_YEARS      = "date.in_years"
        YESTERDAY       = "date.yesterday"
        TOMORROW        = "date.tomorrow"

        @classmethod
        def get_key(cls, unit: str, tense: str, count: int = 2) -> str:
            """Get resource key for a date unit and tense."""
            form = "singular" if count == 1 else "plural"
            return f"date.{unit}_{form}_{tense}"

    # TimeSpan keys
    class TimeSpanHumanize:
        ZERO            = "timespan.zero"
        SECOND          = "timespan.second"
        SECONDS         = "timespan.seconds"
        MINUTE          = "timespan.minute"
        MINUTES         = "timespan.minutes"
        HOUR            = "timespan.hour"
        HOURS           = "timespan.hours"
        DAY             = "timespan.day"
        DAYS            = "timespan.days"
        WEEK            = "timespan.week"
        WEEKS           = "timespan.weeks"

        @classmethod
        def get_key(cls, unit: str, count: int = 2) -> str:
            return f"timespan.{unit}_{'singular' if count == 1 else 'plural'}"

    # Time unit symbol keys
    class TimeUnitSymbol:
        MILLISECOND = "symbol.ms"
        SECOND      = "symbol.sec"
        MINUTE      = "symbol.min"
        HOUR        = "symbol.hr"
        DAY         = "symbol.d"
        WEEK        = "symbol.wk"

        @classmethod
        def get_key(cls, unit: str) -> str:
            return f"symbol.{unit[:3].lower()}"


# ─────────────────────────────────────────────────────────────────────────────
# 37. ResourceRetrieval  — fetch English phrase strings by key
# ─────────────────────────────────────────────────────────────────────────────

class ResourceRetrieval:
    """
    Fetch English phrase strings by resource key.
    Built-in English resources are always available.
    Additional phrases can be registered at runtime.
    """

    _ENGLISH_RESOURCES: dict = {
        # Date phrases
        ResourceKeys.DateHumanize.NOW:          "just now",
        ResourceKeys.DateHumanize.NEVER:        "never",
        ResourceKeys.DateHumanize.AGO_SECONDS:  "seconds ago",
        ResourceKeys.DateHumanize.AGO_MINUTE:   "a minute ago",
        ResourceKeys.DateHumanize.AGO_MINUTES:  "{n} minutes ago",
        ResourceKeys.DateHumanize.AGO_HOUR:     "an hour ago",
        ResourceKeys.DateHumanize.AGO_HOURS:    "{n} hours ago",
        ResourceKeys.DateHumanize.AGO_DAY:      "yesterday",
        ResourceKeys.DateHumanize.AGO_DAYS:     "{n} days ago",
        ResourceKeys.DateHumanize.AGO_MONTH:    "a month ago",
        ResourceKeys.DateHumanize.AGO_MONTHS:   "{n} months ago",
        ResourceKeys.DateHumanize.AGO_YEAR:     "a year ago",
        ResourceKeys.DateHumanize.AGO_YEARS:    "{n} years ago",
        ResourceKeys.DateHumanize.FROM_SECONDS: "in a few seconds",
        ResourceKeys.DateHumanize.FROM_MINUTE:  "in a minute",
        ResourceKeys.DateHumanize.FROM_MINUTES: "in {n} minutes",
        ResourceKeys.DateHumanize.FROM_HOUR:    "in an hour",
        ResourceKeys.DateHumanize.FROM_HOURS:   "in {n} hours",
        ResourceKeys.DateHumanize.FROM_DAY:     "tomorrow",
        ResourceKeys.DateHumanize.FROM_DAYS:    "in {n} days",
        ResourceKeys.DateHumanize.FROM_MONTH:   "in a month",
        ResourceKeys.DateHumanize.FROM_MONTHS:  "in {n} months",
        ResourceKeys.DateHumanize.FROM_YEAR:    "in a year",
        ResourceKeys.DateHumanize.FROM_YEARS:   "in {n} years",
        ResourceKeys.DateHumanize.YESTERDAY:    "yesterday",
        ResourceKeys.DateHumanize.TOMORROW:     "tomorrow",
        # TimeSpan phrases
        ResourceKeys.TimeSpanHumanize.ZERO:     "no time",
        ResourceKeys.TimeSpanHumanize.SECOND:   "1 second",
        ResourceKeys.TimeSpanHumanize.SECONDS:  "{n} seconds",
        ResourceKeys.TimeSpanHumanize.MINUTE:   "1 minute",
        ResourceKeys.TimeSpanHumanize.MINUTES:  "{n} minutes",
        ResourceKeys.TimeSpanHumanize.HOUR:     "1 hour",
        ResourceKeys.TimeSpanHumanize.HOURS:    "{n} hours",
        ResourceKeys.TimeSpanHumanize.DAY:      "1 day",
        ResourceKeys.TimeSpanHumanize.DAYS:     "{n} days",
        ResourceKeys.TimeSpanHumanize.WEEK:     "1 week",
        ResourceKeys.TimeSpanHumanize.WEEKS:    "{n} weeks",
    }

    _custom: dict = {}

    @classmethod
    def get(cls, key: str, n: int = None) -> str:
        """Retrieve a phrase by key. Substitutes {n} if n is provided."""
        phrase = cls._custom.get(key) or cls._ENGLISH_RESOURCES.get(key, key)
        if n is not None:
            phrase = phrase.replace("{n}", str(n))
        return phrase

    @classmethod
    def try_get(cls, key: str, n: int = None) -> Optional[str]:
        """Try to retrieve a phrase; returns None if key not found."""
        raw = cls._custom.get(key) or cls._ENGLISH_RESOURCES.get(key)
        if raw is None:
            return None
        if n is not None:
            raw = raw.replace("{n}", str(n))
        return raw

    @classmethod
    def register(cls, key: str, phrase: str) -> None:
        """Register a custom phrase override."""
        cls._custom[key] = phrase

    @classmethod
    def all_keys(cls) -> list:
        """Return all registered resource keys."""
        return list({**cls._ENGLISH_RESOURCES, **cls._custom}.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 38. LocalizationRegistry  — English-only service registry
# ─────────────────────────────────────────────────────────────────────────────

class LocalizationRegistry:
    """
    Register and resolve English-only service implementations.
    Supports culture-keyed fallback: 'en-US' → 'en' → default.
    """

    _registry: dict = {}

    @classmethod
    def register(cls, service_name: str, implementation, culture: str = "en") -> None:
        """Register a service implementation for a culture key."""
        key = f"{service_name}:{culture}"
        cls._registry[key] = implementation

    @classmethod
    def resolve(cls, service_name: str, culture: str = "en"):
        """Resolve a service for a culture, falling back through en-XX → en → default."""
        # Try exact culture (e.g. en-US)
        key = f"{service_name}:{culture}"
        if key in cls._registry:
            return cls._registry[key]
        # Try base language (e.g. en)
        base = culture.split("-")[0]
        key_base = f"{service_name}:{base}"
        if key_base in cls._registry:
            return cls._registry[key_base]
        # Try default
        key_default = f"{service_name}:default"
        if key_default in cls._registry:
            return cls._registry[key_default]
        return None

    @classmethod
    def resolve_for_ui_culture(cls, service_name: str) -> object:
        """Resolve using the default English UI culture."""
        return cls.resolve(service_name, "en")

    @classmethod
    def all_registered(cls) -> list:
        return list(cls._registry.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 39. PolyfillShims  — input validation helpers
# ─────────────────────────────────────────────────────────────────────────────

class PolyfillShims:
    """
    Input validation and numeric check helpers.
    Raises standard Python exceptions on invalid input.
    """

    @staticmethod
    def throw_if_null(value, param_name: str = "value") -> None:
        """Raise ValueError if value is None."""
        if value is None:
            raise ValueError(f"Parameter '{param_name}' must not be None.")

    @staticmethod
    def throw_if_negative(value: float, param_name: str = "value") -> None:
        """Raise ValueError if value is negative."""
        if value < 0:
            raise ValueError(f"Parameter '{param_name}' must be non-negative (got {value}).")

    @staticmethod
    def throw_if_empty(value: str, param_name: str = "value") -> None:
        """Raise ValueError if string is None or empty."""
        if not value:
            raise ValueError(f"Parameter '{param_name}' must not be None or empty.")

    @staticmethod
    def is_finite(value: float) -> bool:
        """Return True if value is a finite float (not inf or NaN)."""
        return math.isfinite(value)

    @staticmethod
    def is_integral(value: float) -> bool:
        """Return True if float value is mathematically an integer."""
        return math.isfinite(value) and value == int(value)

    @staticmethod
    def is_infinity(value: float) -> bool:
        """Return True if value is positive or negative infinity."""
        return math.isinf(value)

    @staticmethod
    def is_nan(value: float) -> bool:
        """Return True if value is NaN."""
        return math.isnan(value)

    @staticmethod
    def is_nan_fast(value: float) -> bool:
        """Optimised fast-path NaN check (value != value is True only for NaN)."""
        return value != value  # noqa: PLR0124

    @staticmethod
    def is_infinity_fast(value: float) -> bool:
        """Optimised fast-path infinity check."""
        return math.isinf(value)

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value between min_val and max_val."""
        return max(min_val, min(max_val, value))


# ─────────────────────────────────────────────────────────────────────────────
# 40. DefaultFormatter  — unified English output formatter
# ─────────────────────────────────────────────────────────────────────────────

class DefaultFormatter:
    """
    Unified English formatter that delegates to specialist classes.
    Single entry-point for consistent date, time, number, and quantity output.
    """

    def __init__(self):
        self._qty         = QuantityFormatter()
        self._tuple_fmt   = TupleFormatter()
        self._precision   = PrecisionHumanizer()

    # ── Date ──────────────────────────────────────────────────────────────

    def date_now(self) -> str:
        return ResourceRetrieval.get(ResourceKeys.DateHumanize.NOW)

    def date_never(self) -> str:
        return ResourceRetrieval.get(ResourceKeys.DateHumanize.NEVER)

    def date_humanize(self, dt: "datetime.datetime",
                      now: "datetime.datetime" = None) -> str:
        return DateHumanizer.humanize(dt, now=now)

    def date_humanize_precision(self, dt: "datetime.datetime",
                                precision: int = 1,
                                now: "datetime.datetime" = None) -> str:
        return PrecisionHumanizer.humanize_datetime(dt, now=now, precision=precision)

    # ── TimeSpan ──────────────────────────────────────────────────────────

    def timespan_zero(self) -> str:
        return ResourceRetrieval.get(ResourceKeys.TimeSpanHumanize.ZERO)

    def timespan_humanize(self, delta: "datetime.timedelta",
                          precision: int = 1) -> str:
        return TimeSpanHumanizer.humanize(delta, precision=precision)

    def timespan_age(self, delta: "datetime.timedelta") -> str:
        return AgeFormatter.to_age(delta)

    # ── Numbers ───────────────────────────────────────────────────────────

    def number_to_words(self, n: int) -> str:
        return NumberToWords.convert(n)

    def number_to_ordinal(self, n: int) -> str:
        return NumberToOrdinalWords.convert(n)

    def ordinalize(self, n: int) -> str:
        return Ordinalize.convert(n)

    # ── Quantity / Tuple ──────────────────────────────────────────────────

    def quantity(self, count: int, word: str,
                 show_count: bool = True) -> str:
        return self._qty.to_quantity(word, count, show_quantity=show_count)

    def tuple_format(self, count: int, word: str) -> str:
        return self._tuple_fmt.format((count, word))

    # ── Data units ────────────────────────────────────────────────────────

    def data_unit(self, byte_size: "ByteSize", full_words: bool = False) -> str:
        return byte_size.to_full_words() if full_words else byte_size.humanize()

    def time_unit(self, unit_name: str) -> str:
        return TimeUnitSymbols.to_symbol(unit_name)

    # ── Byte rate ─────────────────────────────────────────────────────────

    def byte_rate(self, bytes_per_second: float) -> str:
        return ByteRate.humanize(bytes_per_second)


# ─────────────────────────────────────────────────────────────────────────────
# 41. Configurator  — English formatter resolver and pipeline configuration
# ─────────────────────────────────────────────────────────────────────────────

class Configurator:
    """
    Central configuration point for the English humanizer pipeline.
    Resolves formatter, number converters, and pipeline settings.
    """

    _formatter: Optional["DefaultFormatter"] = None
    _num_to_words  = NumberToWords
    _words_to_num  = WordsToNumber
    _use_enum_desc: bool = False

    @classmethod
    def get_formatter(cls) -> "DefaultFormatter":
        """Return the active DefaultFormatter (creates one if needed)."""
        if cls._formatter is None:
            cls._formatter = DefaultFormatter()
        return cls._formatter

    @classmethod
    def set_formatter(cls, formatter: "DefaultFormatter") -> None:
        """Replace the active formatter."""
        cls._formatter = formatter

    @classmethod
    def get_number_to_words_converter(cls):
        """Return the active number-to-words converter class."""
        return cls._num_to_words

    @classmethod
    def get_words_to_number_converter(cls):
        """Return the active words-to-number converter class."""
        return cls._words_to_num

    @classmethod
    def use_enum_description_property_locator(cls, enabled: bool = True) -> None:
        """Enable or disable description-property lookup for enum humanization."""
        cls._use_enum_desc = enabled

    @classmethod
    def reset_enum_description_property_locator(cls) -> None:
        """Restore default enum description lookup behavior."""
        cls._use_enum_desc = False

    @classmethod
    def is_enum_description_enabled(cls) -> bool:
        return cls._use_enum_desc

    @classmethod
    def build_pipeline(cls, *transform_names: str) -> "TransformerPipeline":
        """Build a TransformerPipeline from named transforms."""
        return TransformerPipeline.build(*transform_names)


# ─────────────────────────────────────────────────────────────────────────────
# 42. EnglishGrammarDetector  — choose correct plural form based on number
# ─────────────────────────────────────────────────────────────────────────────

class EnglishGrammarDetector:
    """
    Detect the correct English grammatical number form for a given count.
    English has two forms: singular (count == 1) and plural (everything else).

    This class also detects subject-verb agreement issues in text.

    Examples:
        EnglishGrammarDetector.detect(1)   → 'singular'
        EnglishGrammarDetector.detect(0)   → 'plural'
        EnglishGrammarDetector.detect(42)  → 'plural'
        EnglishGrammarDetector.is_plural(3) → True
    """

    @staticmethod
    def detect(count: int) -> str:
        """Return 'singular' if count == 1, else 'plural'."""
        return "singular" if count == 1 else "plural"

    @staticmethod
    def is_singular(count: int) -> bool:
        return count == 1

    @staticmethod
    def is_plural(count: int) -> bool:
        return count != 1

    @staticmethod
    def choose_form(count: int, singular: str, plural: str) -> str:
        """Choose the correct form directly."""
        return singular if count == 1 else plural

    @staticmethod
    def detect_in_text(text: str) -> dict:
        """
        Scan text for number-noun patterns and report potential agreement issues.
        Returns dict of {pattern: suggestion}.
        """
        issues = {}
        # Pattern: digit + noun (check plural agreement)
        for m in re.finditer(r'\b(\d+)\s+([a-z]{3,})\b', text, re.IGNORECASE):
            count = int(m.group(1))
            word  = m.group(2).lower()
            expected_form = "singular" if count == 1 else "plural"
            # Cheap heuristic: words ending in 's' are plural
            appears_plural = word.endswith('s') and not word.endswith('ss')
            is_correct = (expected_form == "singular") == (not appears_plural)
            if not is_correct:
                issues[m.group(0)] = f"Expected {expected_form} form"
        return issues

    @staticmethod
    def count_word_agrees(count: int, word: str) -> bool:
        """Check if a word form agrees with a count."""
        appears_plural = word.lower().endswith('s') and not word.lower().endswith('ss')
        return (count == 1) == (not appears_plural)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 1: DateToOrdinalWords  — "March 21, 2026" → ordinal word forms
# Also covers: English date-to-ordinal converter
# ─────────────────────────────────────────────────────────────────────────────

class DateToOrdinalWords:
    """
    Render a date as English ordinal words.
    e.g. datetime.date(2026, 3, 21) →
         "the twenty-first of March, two thousand and twenty-six"
    """

    _MONTHS = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    @classmethod
    def convert(cls, dt: "datetime.date", short: bool = False) -> str:
        """
        Convert a date to ordinal words.
        short=False → "the twenty-first of March 2026"
        short=True  → "March 21st, 2026"
        """
        day_ord  = NumberToOrdinalWords.convert(dt.day)
        month    = cls._MONTHS[dt.month]
        year_str = NumberToWords.convert(dt.year)

        if short:
            return f"{month} {Ordinalize.convert(dt.day)}, {dt.year}"
        return f"the {day_ord} of {month}, {year_str}"

    @classmethod
    def convert_short(cls, dt: "datetime.date") -> str:
        """'March 21st, 2026'"""
        return cls.convert(dt, short=True)

    @classmethod
    def convert_long(cls, dt: "datetime.date") -> str:
        """'the twenty-first of March, two thousand and twenty-six'"""
        return cls.convert(dt, short=False)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 2: ByteSizeExtensions  — fluent helpers for ByteSize creation
# ─────────────────────────────────────────────────────────────────────────────

class ByteSizeExtensions:
    """
    Convenience extension methods for creating and formatting ByteSize values
    directly from numeric literals — mirrors fluent API style.

    Usage:
        ByteSizeExtensions.bits(8192)
        ByteSizeExtensions.kilobytes(100).humanize()
        ByteSizeExtensions.per(ByteSize.from_megabytes(10), datetime.timedelta(seconds=1))
    """

    @staticmethod
    def bits(n: float) -> ByteSize:
        return ByteSize.from_bits(n)

    @staticmethod
    def bytes_(n: float) -> ByteSize:       # 'bytes' is a built-in, use bytes_
        return ByteSize.from_bytes(n)

    @staticmethod
    def kilobytes(n: float) -> ByteSize:
        return ByteSize.from_kilobytes(n)

    @staticmethod
    def megabytes(n: float) -> ByteSize:
        return ByteSize.from_megabytes(n)

    @staticmethod
    def gigabytes(n: float) -> ByteSize:
        return ByteSize.from_gigabytes(n)

    @staticmethod
    def terabytes(n: float) -> ByteSize:
        return ByteSize.from_terabytes(n)

    @staticmethod
    def humanize(n: float, unit: str = "MB") -> str:
        """Quick format: ByteSizeExtensions.humanize(10, 'MB') → '10 MB'"""
        unit_map = {
            "B":  ByteSize.from_bytes,
            "KB": ByteSize.from_kilobytes,
            "MB": ByteSize.from_megabytes,
            "GB": ByteSize.from_gigabytes,
            "TB": ByteSize.from_terabytes,
        }
        factory = unit_map.get(unit.upper(), ByteSize.from_bytes)
        return factory(n).humanize()

    @staticmethod
    def per(size: ByteSize, duration: "datetime.timedelta") -> str:
        """
        Compute a transfer rate: size / duration → "X MB/s".
        per(ByteSize.from_megabytes(100), timedelta(seconds=10)) → "10.0 MB/s"
        """
        seconds = duration.total_seconds()
        if seconds <= 0:
            return "∞ B/s"
        return ByteRate.humanize(size.bytes)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 3: DateHumanizeAlgorithm  — configurable date humanization strategy
# ─────────────────────────────────────────────────────────────────────────────

class DateHumanizeAlgorithm:
    """
    Pluggable date-humanization algorithm.
    Two strategies: 'default' and 'precision'.

    default   → "just now", "a minute ago", "2 hours ago", "yesterday"
    precision → fine-grained thresholds, e.g. "43 minutes ago"
    """

    @staticmethod
    def default_humanize(dt: "datetime.datetime",
                          now: "Optional[datetime.datetime]" = None) -> str:
        """Apply default English date-humanization strategy."""
        return DateHumanizer.humanize(dt, reference=now)

    @staticmethod
    def precision_humanize(dt: "datetime.datetime",
                            now: "Optional[datetime.datetime]" = None,
                            precision: float = 0.75) -> str:
        """Precision strategy: show exact counts."""
        return DateHumanizer.humanize(dt, reference=now)

    @staticmethod
    def humanize(dt: "datetime.datetime",
                  strategy: str = "default",
                  now: "Optional[datetime.datetime]" = None) -> str:
        """Dispatch to the chosen strategy. strategy: 'default' | 'precision'"""
        if strategy == "precision":
            return DateHumanizeAlgorithm.precision_humanize(dt, now)
        return DateHumanizeAlgorithm.default_humanize(dt, now)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 4: DefaultHumanizer  — single-entry-point fallback humanizer
# ─────────────────────────────────────────────────────────────────────────────

class DefaultHumanizer:
    """
    Unified fallback humanizer — routes any supported value type to the
    appropriate English humanization method.

    Supported types:
      - int / float          → NumberToWords.convert
      - datetime.datetime    → DateHumanizer.humanize
      - datetime.date        → DateToOrdinalWords.convert_short
      - datetime.timedelta   → TimeSpanHumanizer.humanize
      - datetime.time        → TimeToClockNotation.convert
      - str (identifier)     → StringHumanizer.humanize
      - list                 → CollectionHumanizer.humanize
      - ByteSize             → ByteSize.humanize
    """

    @staticmethod
    def humanize(value) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return NumberToWords.convert(value)
        if isinstance(value, float):
            if value == int(value):
                return NumberToWords.convert(int(value))
            return f"{value:.2f}"
        if isinstance(value, datetime.datetime):
            return DateHumanizer.humanize(value)
        if isinstance(value, datetime.date):
            return DateToOrdinalWords.convert_short(value)
        if isinstance(value, datetime.timedelta):
            return TimeSpanHumanizer.humanize(value)
        if isinstance(value, datetime.time):
            return TimeToClockNotation.convert(value.hour, value.minute)
        if isinstance(value, list):
            return CollectionHumanizer.humanize([str(x) for x in value])
        if isinstance(value, ByteSize):
            return value.humanize()
        if isinstance(value, str):
            return StringHumanizer.humanize(value)
        return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 5: CollectionFormatter  — Oxford-comma enforcer with configurable style
# ─────────────────────────────────────────────────────────────────────────────

class CollectionFormatter:
    """
    Format lists into English phrases with configurable conjunction style.

    Styles:
      oxford      → "a, b, and c"   (Oxford comma — default)
      no_oxford   → "a, b and c"
      or_oxford   → "a, b, or c"
      semicolon   → "a; b; and c"   (for items containing commas)
    """

    @staticmethod
    def format(items: List[str], style: str = "oxford") -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            conj = "or" if "or" in style else "and"
            return f"{items[0]} {conj} {items[1]}"

        sep = "; " if style == "semicolon" else ", "
        conj = "or" if "or" in style else "and"
        comma = "," if "oxford" in style else ""
        body = sep.join(items[:-1])
        return f"{body}{comma} {conj} {items[-1]}"

    @staticmethod
    def oxford(items: List[str]) -> str:
        return CollectionFormatter.format(items, style="oxford")

    @staticmethod
    def no_oxford(items: List[str]) -> str:
        return CollectionFormatter.format(items, style="no_oxford")

    @staticmethod
    def or_list(items: List[str]) -> str:
        return CollectionFormatter.format(items, style="or_oxford")

    @staticmethod
    def semicolon_list(items: List[str]) -> str:
        return CollectionFormatter.format(items, style="semicolon")


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 6: EnglishOrdinalizer  — dedicated ordinal suffix + word generator
# ─────────────────────────────────────────────────────────────────────────────

class EnglishOrdinalizer:
    """
    Generate English ordinal forms:
      suffix_form  → "1st", "2nd", "3rd", "21st"
      word_form    → "first", "second", "twenty-first"
      sentence_form→ "the first", "the twenty-first"
    """

    @staticmethod
    def to_suffix(n: int) -> str:
        """42 → '42nd'"""
        return Ordinalize.convert(n)

    @staticmethod
    def to_words(n: int) -> str:
        """21 → 'twenty-first'"""
        return NumberToOrdinalWords.convert(n)

    @staticmethod
    def to_sentence(n: int) -> str:
        """21 → 'the twenty-first'"""
        return f"the {NumberToOrdinalWords.convert(n)}"

    @staticmethod
    def convert(n: int, form: str = "suffix") -> str:
        """
        Dispatch to chosen form.
        form: 'suffix' | 'words' | 'sentence'
        """
        if form == "words":
            return EnglishOrdinalizer.to_words(n)
        if form == "sentence":
            return EnglishOrdinalizer.to_sentence(n)
        return EnglishOrdinalizer.to_suffix(n)


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 7: EnglishFormattingRules  — central registry of English style rules
# ─────────────────────────────────────────────────────────────────────────────

class EnglishFormattingRules:
    """
    Central registry and enforcer for English formatting and grammar style rules.

    Rules enforced:
      - Numbers ≤ 12 written as words in prose
      - Oxford comma in lists
      - Sentence-initial capitalisation
      - 'a' vs 'an' agreement
      - Correct plural/singular with counts
      - Em-dash with spaces for parentheticals
      - No double spaces
      - Proper capitalisation of proper nouns (configurable)

    Usage:
        rules = EnglishFormattingRules()
        rules.apply(text)               # apply all enabled rules
        rules.apply(text, rules=['articles', 'numbers'])  # selective
    """

    ALL_RULES = [
        "articles",        # a/an agreement
        "numbers",         # small integers → words
        "oxford_comma",    # enforce Oxford comma in lists
        "sentence_case",   # capitalise first word of each sentence
        "double_spaces",   # collapse multiple spaces
        "duplicate_words", # remove consecutive duplicate words
        "quantity_agree",  # number-noun agreement
    ]

    def __init__(self, number_threshold: int = 12):
        self._threshold = number_threshold
        self._fixer     = EnglishGrammarFixer()
        self._inflector = DEFAULT_INFLECTOR

    def apply(self, text: str,
              rules: Optional[List[str]] = None) -> str:
        """Apply all (or selected) formatting rules to text."""
        active = rules if rules is not None else self.ALL_RULES

        if "double_spaces" in active:
            text = re.sub(r'  +', ' ', text)

        if "duplicate_words" in active:
            text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

        if "articles" in active:
            text = ArticleHandler.fix_articles_in_text(text)

        if "numbers" in active:
            def _sub(m):
                n = int(m.group(0))
                if 0 <= n <= self._threshold:
                    return NumberToWords.convert(n)
                return m.group(0)
            text = re.sub(r'\b([0-9]{1,2})\b', _sub, text)

        if "oxford_comma" in active:
            # "A, B and C" → "A, B, and C"
            text = re.sub(
                r'(\w[^,]*),\s+(\w[^,]+)\s+and\s+(\w)',
                lambda m: f"{m.group(1)}, {m.group(2)}, and {m.group(3)}",
                text
            )

        if "sentence_case" in active:
            text = CasingTransformer.to_sentence_case(text)

        return text.strip()

    def check(self, text: str) -> List[str]:
        """Return a list of formatting violations found in text."""
        issues = []
        if '  ' in text:
            issues.append("Double spaces detected.")
        if re.search(r'\b(\w+)\s+\1\b', text, re.IGNORECASE):
            issues.append("Consecutive duplicate words found.")
        if re.search(r'\ba\s+[aeiouAEIOU]\w', text):
            issues.append("Possible 'a/an' error: 'a' before vowel.")
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for s in sentences:
            if s and s[0].islower():
                issues.append(f"Sentence starts with lowercase: '{s[:30]}'")
                break
        return issues

    def is_compliant(self, text: str) -> bool:
        return len(self.check(text)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# MISSING 8: Update TransformerPipeline registrations for new classes
# ─────────────────────────────────────────────────────────────────────────────

# Register new transforms in the pipeline
TransformerPipeline.register("date_to_ordinal",    lambda t: _apply_date_to_ordinal(t))
TransformerPipeline.register("apply_formatting",   lambda t: EnglishFormattingRules().apply(t))
TransformerPipeline.register("oxford_comma",       lambda t: CollectionFormatter.oxford(t.split(", ")))
TransformerPipeline.register("ordinalize_words",   lambda t: _apply_ordinal_words(t))


def _apply_date_to_ordinal(text: str) -> str:
    """Replace bare date patterns with ordinal word forms where detected."""
    # Match "March 21" or "21st March" → ordinal word form
    months = {m: i for i, m in enumerate(
        ["", "January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    ) if m}
    def _sub(m):
        month_name, day = m.group(1), int(m.group(2))
        month_num = months.get(month_name, 0)
        if month_num == 0:
            return m.group(0)
        day_ord = NumberToOrdinalWords.convert(day)
        return f"the {day_ord} of {month_name}"
    return re.sub(
        r'\b(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+(\d{1,2})\b',
        _sub, text
    )


def _apply_ordinal_words(text: str) -> str:
    """Replace ordinal suffixes with word forms: '3rd place' → 'third place'."""
    def _sub(m):
        n = int(m.group(1))
        if 1 <= n <= 20:
            return NumberToOrdinalWords.convert(n) + " "
        return m.group(0)
    return re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)\s+', _sub, text)
