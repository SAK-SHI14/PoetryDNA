"""
NLP Feature Extraction for PoetryDNA Linguistic Brain.
Ported from the NLP-Sakshi notebook — 32 hand-crafted stylometric features.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

import numpy as np

# ── NLTK setup ────────────────────────────────────────────────────────────────
import nltk

for _pkg in ["cmudict", "punkt", "punkt_tab", "stopwords",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(_pkg, quiet=True)

from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

import pronouncing
import textstat

CMU_DICT = cmudict.dict()
STOP_WORDS = set(stopwords.words("english"))

# ── Thematic vocabulary lexicons ──────────────────────────────────────────────

ARCHAIC_WORDS = {
    "thee", "thou", "thy", "thine", "doth", "hath", "hast", "wherefore",
    "whence", "whilst", "oft", "ere", "yonder", "nay", "aye", "forsooth",
    "methinks", "perchance", "prithee", "wilt", "shalt", "wouldst",
    "couldst", "shouldst", "art", "tis", "twas", "twere", "twill",
    "twixt", "betwixt",
}
NATURE_WORDS = {
    "sun", "moon", "star", "sky", "cloud", "wind", "storm", "river",
    "sea", "ocean", "wave", "shore", "mountain", "hill", "forest", "tree",
    "flower", "grass", "leaf", "rain", "snow", "spring", "summer",
    "autumn", "winter", "night", "dawn", "dusk", "earth", "field",
    "meadow", "brook", "stream", "lake", "wood",
}
DARK_WORDS = {
    "death", "dead", "grave", "tomb", "darkness", "shadow", "doom",
    "despair", "grief", "sorrow", "pale", "cold", "ghost", "soul", "sin",
    "curse", "blood", "pain", "fear", "dread", "terror", "gloom", "decay",
    "ruin", "lost", "weep", "mourn", "sigh",
}
DIVINE_WORDS = {
    "god", "heaven", "angel", "divine", "holy", "sacred", "eternal",
    "paradise", "glory", "grace", "spirit", "soul", "prayer", "blessed",
    "almighty", "lord", "immortal", "celestial", "sublime", "worship",
}
SENSORY_WORDS = {
    "sweet", "soft", "warm", "bright", "rich", "deep", "pure", "tender",
    "gentle", "smooth", "fragrant", "golden", "silver", "crimson",
    "perfume", "music", "taste", "touch", "sight", "sound",
}
WORDSWORTH_WORDS = {
    "child", "childhood", "memory", "grow", "mountain", "lake", "cottage",
    "shepherd", "peasant", "solitude", "tranquil", "spots", "growth",
    "mind", "recollect", "district", "plain", "wandered", "lonely",
    "repose", "humble", "dwelling", "rustic",
}
TENNYSON_WORDS = {
    "king", "knight", "sword", "battle", "hero", "noble", "quest",
    "lancelot", "arthur", "camelot", "grail", "glory", "honor", "grief",
    "loss", "mourn", "elegy", "ring", "toll", "vale",
}
KEATS_WORDS = {
    "beauty", "urn", "grecian", "apollo", "psyche", "nightingale",
    "autumn", "ripeness", "indolence", "melancholy", "luxurious",
    "unheard", "melodies", "sweeter", "pipe", "ditties", "boughs",
}

# ── Prosody helpers ───────────────────────────────────────────────────────────

def count_syllables(word: str) -> int:
    word = word.lower().strip(string.punctuation)
    if word in CMU_DICT:
        return max(
            len([s for s in pron if s[-1].isdigit()])
            for pron in CMU_DICT[word]
        )
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def get_syllable_features(lines: list[str]) -> dict[str, float]:
    counts = []
    for line in lines:
        words = [w for w in word_tokenize(line.lower()) if w.isalpha()]
        counts.append(sum(count_syllables(w) for w in words))
    return {
        "avg_syllables_per_line": float(np.mean(counts)) if counts else 0.0,
        "std_syllables_per_line": float(np.std(counts)) if counts else 0.0,
    }


def get_iambic_score(lines: list[str]) -> float:
    iambic_count = 0
    for line in lines:
        phones: list[str] = []
        for word in word_tokenize(line.lower()):
            word = word.strip(string.punctuation)
            if word in CMU_DICT:
                phones.extend(CMU_DICT[word][0])
        stresses = [p[-1] for p in phones if p[-1].isdigit()]
        if len(stresses) >= 4:
            pairs = [
                (stresses[i], stresses[i + 1])
                for i in range(0, len(stresses) - 1, 2)
            ]
            iambic_pairs = sum(1 for a, b in pairs if a == "0" and b == "1")
            if pairs and iambic_pairs / len(pairs) > 0.5:
                iambic_count += 1
    return iambic_count / len(lines) if lines else 0.0

# ── Rhyme helpers ─────────────────────────────────────────────────────────────

def _get_last_word(line: str) -> str:
    words = re.findall(r"\b[a-zA-Z]+\b", line)
    return words[-1].lower() if words else ""


def _words_rhyme(w1: str, w2: str) -> bool:
    if not w1 or not w2 or w1 == w2:
        return False
    phones1 = pronouncing.phones_for_word(w1)
    phones2 = pronouncing.phones_for_word(w2)
    if not phones1 or not phones2:
        return w1[-2:] == w2[-2:] and len(w1) > 2
    return w2 in set(pronouncing.rhymes(w1))


def get_rhyme_features(lines: list[str]) -> dict[str, float]:
    if len(lines) < 2:
        return {"rhyme_density": 0, "is_abab": 0, "is_aabb": 0, "is_abba": 0}
    last_words = [_get_last_word(l) for l in lines]
    n = len(last_words)
    rhyme_count = total_pairs = 0
    for i in range(n):
        for j in range(i + 1, min(i + 5, n)):
            total_pairs += 1
            if _words_rhyme(last_words[i], last_words[j]):
                rhyme_count += 1
    rhyme_density = rhyme_count / total_pairs if total_pairs else 0
    is_abab = int(
        n >= 4
        and _words_rhyme(last_words[0], last_words[2])
        and _words_rhyme(last_words[1], last_words[3])
    )
    is_aabb = int(
        n >= 4
        and _words_rhyme(last_words[0], last_words[1])
        and _words_rhyme(last_words[2], last_words[3])
    )
    is_abba = int(
        n >= 4
        and _words_rhyme(last_words[0], last_words[3])
        and _words_rhyme(last_words[1], last_words[2])
    )
    return {
        "rhyme_density": rhyme_density,
        "is_abab": is_abab,
        "is_aabb": is_aabb,
        "is_abba": is_abba,
    }

# ── Vocabulary / structural / POS features ────────────────────────────────────

def get_vocabulary_features(text: str, lines: list[str]) -> dict[str, float]:
    words_raw = word_tokenize(text.lower())
    words_only = [w for w in words_raw if w.isalpha()]
    total = len(words_only) if words_only else 1
    return {
        "archaic_ratio": sum(1 for w in words_only if w in ARCHAIC_WORDS) / total,
        "nature_ratio": sum(1 for w in words_only if w in NATURE_WORDS) / total,
        "dark_ratio": sum(1 for w in words_only if w in DARK_WORDS) / total,
        "divine_ratio": sum(1 for w in words_only if w in DIVINE_WORDS) / total,
        "sensory_ratio": sum(1 for w in words_only if w in SENSORY_WORDS) / total,
        "wordsworth_ratio": sum(1 for w in words_only if w in WORDSWORTH_WORDS) / total,
        "tennyson_ratio": sum(1 for w in words_only if w in TENNYSON_WORDS) / total,
        "keats_ratio": sum(1 for w in words_only if w in KEATS_WORDS) / total,
        "type_token_ratio": len(set(words_only)) / total,
        "avg_word_length": float(np.mean([len(w) for w in words_only])) if words_only else 0.0,
        "punct_density": sum(1 for c in text if c in string.punctuation) / len(text) if text else 0.0,
        "exclaim_ratio": text.count("!") / len(lines) if lines else 0.0,
        "question_ratio": text.count("?") / len(lines) if lines else 0.0,
    }


def get_structural_features(text: str, lines: list[str]) -> dict[str, float]:
    word_counts = [len(l.split()) for l in lines]
    word_freq = Counter(text.lower().split())
    repeated = sum(
        1 for w, c in word_freq.items()
        if c >= 2 and w not in STOP_WORDS and len(w) > 3
    )
    end_punct = set(".,;:!?")
    enjambed = sum(
        1 for l in lines if l.strip() and l.strip()[-1] not in end_punct
    )
    return {
        "avg_line_length": float(np.mean(word_counts)) if word_counts else 0.0,
        "std_line_length": float(np.std(word_counts)) if word_counts else 0.0,
        "repetition_score": repeated / len(word_freq) if word_freq else 0.0,
        "enjambment_ratio": enjambed / len(lines) if lines else 0.0,
        "cap_ratio": (
            sum(1 for l in lines if l.strip() and l.strip()[0].isupper()) / len(lines)
            if lines else 0.0
        ),
        "line_count": float(len(lines)),
        "flesch_score": float(textstat.flesch_reading_ease(text)),
    }


def get_pos_features(text: str) -> dict[str, float]:
    words = word_tokenize(text)
    tagged = pos_tag(words)
    total = len(tagged) if tagged else 1
    return {
        "noun_ratio": sum(1 for _, t in tagged if t.startswith("NN")) / total,
        "verb_ratio": sum(1 for _, t in tagged if t.startswith("VB")) / total,
        "adj_ratio": sum(1 for _, t in tagged if t.startswith("JJ")) / total,
        "adv_ratio": sum(1 for _, t in tagged if t.startswith("RB")) / total,
        "prop_ratio": sum(1 for _, t in tagged if t == "NNP") / total,
    }

# ── Master extractor (32 features) ───────────────────────────────────────────

def extract_all_features(text: str) -> dict[str, float]:
    """Extract 32 hand-crafted stylometric features from a text."""
    if "/" in text:
        lines = [l.strip() for l in text.split("/") if l.strip()]
    else:
        lines = re.split(r"(?<=[.!?,;])\s+", text)
        lines = [l.strip() for l in lines if l.strip()]
    if not lines or len(lines) == 1:
        words = text.split()
        lines = [" ".join(words[i:i + 8]) for i in range(0, len(words), 8)]

    feats: dict[str, float] = {}
    feats.update(get_syllable_features(lines))
    feats["iambic_score"] = get_iambic_score(lines)
    feats.update(get_rhyme_features(lines))
    feats.update(get_vocabulary_features(text, lines))
    feats.update(get_structural_features(text, lines))
    feats.update(get_pos_features(text))
    return feats
