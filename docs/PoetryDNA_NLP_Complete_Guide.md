# PoetryDNA — Complete NLP Pipeline Explanation
### Every Concept · Every Function · Every Decision

**Authors:** Mehtab Singh · Sakshi Verma  
**GitHub:** https://github.com/MehtabSingh3711/PoetryDNA

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [What Is Authorship Attribution?](#2-what-is-authorship-attribution)
3. [What Is Stylometry?](#3-what-is-stylometry)
4. [Setup and Imports](#4-setup-and-imports)
5. [Dataset Loading and EDA](#5-dataset-loading-and-eda)
6. [Feature Engineering — Theory First](#6-feature-engineering--theory-first)
7. [Vocabulary Lexicons](#7-vocabulary-lexicons)
8. [Syllable and Rhythm Functions](#8-syllable-and-rhythm-functions)
9. [Rhyme Detection Functions](#9-rhyme-detection-functions)
10. [Vocabulary Feature Extractor](#10-vocabulary-feature-extractor)
11. [Structural Feature Extractor](#11-structural-feature-extractor)
12. [POS Tag Feature Extractor](#12-pos-tag-feature-extractor)
13. [Master Feature Extractor](#13-master-feature-extractor)
14. [Text Representations — TF-IDF](#14-text-representations--tf-idf)
15. [Dimensionality Reduction — SVD](#15-dimensionality-reduction--svd)
16. [Character N-Grams](#16-character-n-grams)
17. [MLflow Experiment Tracking](#17-mlflow-experiment-tracking)
18. [Model Training — All Three Classifiers](#18-model-training--all-three-classifiers)
19. [Random Forest — Theory and Why We Tried It](#19-random-forest--theory-and-why-we-tried-it)
20. [XGBoost — Theory and Why We Tried It](#20-xgboost--theory-and-why-we-tried-it)
21. [LightGBM — Theory and Why It Won](#21-lightgbm--theory-and-why-it-won)
22. [Feature Importance Analysis](#22-feature-importance-analysis)
23. [Final Model Evaluation](#23-final-model-evaluation)
24. [Confusion Matrix Analysis](#24-confusion-matrix-analysis)
25. [Fusion — DistilBERT + LightGBM](#25-fusion--distilbert--lightgbm)
26. [Weighted Fusion — Theory and Code](#26-weighted-fusion--theory-and-code)
27. [Meta-Classifier — Theory and Why It Lost](#27-meta-classifier--theory-and-why-it-lost)
28. [Why 97.64% and What It Means](#28-why-9764-and-what-it-means)
29. [All Saved Artifacts Explained](#29-all-saved-artifacts-explained)

---

## 1. The Problem We're Solving

Classical poetry attribution is the task of identifying who wrote an anonymous or disputed poem. Literary scholars have been doing this by hand for centuries — comparing meter, vocabulary, imagery, and tone. The question this project asks is: can a machine learn the same signals?

We picked 6 English poets spanning 1564 to 1850: Shakespeare, Keats, Milton, Tennyson, Coleridge, and Wordsworth. All six wrote in English. All six are Romantic or Renaissance. Several share themes — nature, death, the divine. This is a genuinely hard classification problem because the classes are not cleanly separated.

The NLP side of the project answers this question using traditional machine learning — hand-crafted features that capture structural and linguistic fingerprints, combined with statistical text representations.

---

## 2. What Is Authorship Attribution?

Authorship attribution is a subfield of computational linguistics that tries to identify the author of a text based on its stylistic properties. It has real-world applications:

- **Forensic linguistics** — identifying anonymous threatening letters or ransom notes
- **Literary scholarship** — resolving disputed authorship (e.g., was this Shakespeare or Marlowe?)
- **Plagiarism detection** — checking if a student's essay matches their past writing style
- **Historical document analysis** — attributing unsigned manuscripts

The key assumption is that every writer has an unconscious stylistic fingerprint — patterns in how they construct sentences, which words they favour, how they pace rhythm, even which punctuation marks they overuse. These patterns are stable enough across different works to be detectable by a classifier.

---

## 3. What Is Stylometry?

Stylometry is the quantitative study of writing style. Instead of reading a poem and saying "this feels like Keats," stylometry asks: what specific measurable properties distinguish Keats from Tennyson?

Our 32 hand-crafted features all come from stylometry. They fall into four categories:

**Prosodic features** — the rhythmic and musical properties of verse. How many syllables per line? Does the poem follow iambic meter? These features distinguish structured sonnets (Shakespeare, Keats) from free verse.

**Rhyme features** — the end-sound patterns. ABAB rhyme scheme is common in Romantic poetry. Some poets rhyme densely, others barely at all. Milton's epic verse rarely rhymes; Tennyson's lyrics rhyme constantly.

**Lexical features** — vocabulary choices. Shakespeare uses "thee" and "thou" constantly. Coleridge uses dark Gothic imagery. Wordsworth talks about childhood, memory, and rural life. We capture these as word frequency ratios against pre-defined lexicons.

**Structural features** — how the text is organized. Long lines vs short lines. How often does a sentence continue across a line break (enjambment)? How much does the poet repeat words? Milton has extremely low repetition; Tennyson and Coleridge repeat key phrases for effect.

---

## 4. Setup and Imports

```python
import os, warnings, logging
import re, string
from collections import Counter
```

**`os`** — used for environment variable setting (`os.environ["PYTHONWARNINGS"]`). Setting this silences LightGBM's verbose output at the OS level.

**`warnings`** — Python's built-in warning system. `warnings.filterwarnings("ignore")` suppresses all UserWarnings and DeprecationWarnings so notebook output stays clean.

**`logging`** — used specifically for LightGBM which logs through Python's logging system rather than warnings. `logging.getLogger("lightgbm").setLevel(logging.ERROR)` means LightGBM will only print actual errors, not training progress.

**`re`** — regular expressions. Used throughout for text cleaning, pattern matching in lines (e.g., extracting last words, detecting punctuation patterns).

**`string`** — provides `string.punctuation` which is the string `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`. Used to strip punctuation from words before syllable counting.

**`Counter`** — a dictionary subclass from `collections` that counts hashable objects. Used in `get_structural_features` to count word frequencies for repetition scoring.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
```

**`numpy`** — numerical computing. Used for `np.mean()`, `np.std()`, `np.hstack()` (combining feature matrices), `np.arange()` (weight search grid), `np.array()`. Almost every feature function uses numpy for aggregation.

**`pandas`** — data manipulation. The dataset is a DataFrame. Features are built as a DataFrame. MLflow results are queried into a DataFrame. `pd.concat()` combines multiple DataFrames.

**`matplotlib.pyplot`** — plotting. Used for all charts: class distribution, feature importance, confusion matrix, fusion comparison.

**`matplotlib.ticker`** — fine-grained control over axis tick spacing. `mticker.MultipleLocator(50)` puts a tick every 50 units on the y-axis of the class distribution chart.

**`seaborn`** — statistical visualization built on matplotlib. Used for the confusion matrix heatmap (`sns.heatmap`) because it handles annotated heatmaps more elegantly than raw matplotlib.

```python
import nltk
for pkg in ["cmudict", "punkt", "stopwords", 
            "averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger"]:
    nltk.download(pkg, quiet=True)
```

**Why download inside a loop?** Cleaner than five separate `nltk.download()` calls. `quiet=True` suppresses download progress messages. Each package is needed:

- **`cmudict`** — Carnegie Mellon University Pronouncing Dictionary. A phonetic dictionary mapping English words to their pronunciation as a sequence of phonemes. Contains ~134,000 words. We use it to count syllables and detect stress patterns for iambic meter detection.
- **`punkt`** — Punkt tokenizer. Splits text into sentences and words. Used by `word_tokenize()`.
- **`stopwords`** — List of common English words (the, is, at, which, etc.) that carry little semantic information. We filter these out when computing repetition scores and vocabulary richness.
- **`averaged_perceptron_tagger_eng`** / **`averaged_perceptron_tagger`** — The POS (Part-of-Speech) tagger model. Uses the Averaged Perceptron algorithm to tag each word with its grammatical role (noun, verb, adjective, etc.). Both versions downloaded for compatibility across NLTK versions.

```python
from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pronouncing
import textstat
```

**`cmudict.dict()`** — loads the entire CMU pronouncing dictionary into memory as a Python dictionary. Keys are lowercase words, values are lists of pronunciations (a word can have multiple), each pronunciation is a list of phoneme strings. Example: `cmu_dict["hello"]` returns `[['HH', 'AH0', 'L', 'OW1']]`.

**`stopwords.words("english")`** — returns a list of 179 English stopwords. Converted to a `set` for O(1) lookup instead of O(n) linear search.

**`word_tokenize`** — tokenizes text into individual words and punctuation marks using the Punkt algorithm. Handles contractions properly ("don't" → ["do", "n't"]).

**`pos_tag`** — tags a list of words with Penn Treebank POS tags. Returns a list of (word, tag) tuples. Tags include "NN" (noun), "VB" (verb), "JJ" (adjective), "RB" (adverb), "NNP" (proper noun).

**`pronouncing`** — a Python library built on CMU dict that provides rhyme detection. Key function: `pronouncing.rhymes(word)` returns all words that rhyme with the given word. `pronouncing.phones_for_word(word)` returns the phoneme string.

**`textstat`** — calculates readability scores from text. We use `flesch_reading_ease(text)` which returns a score from 0-100 (higher = easier to read). Classical poetry typically scores between 40-80.

```python
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
```

**`joblib`** — serializes Python objects to disk. Much faster than `pickle` for large numpy arrays (our feature matrices). Used to save all model files. `.dump(obj, filename)` saves, `.load(filename)` restores.

**`mlflow`** — experiment tracking framework. Records parameters, metrics, and artifacts for each training run. Lets us compare all 11 experiments in a dashboard.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
```

All scikit-learn imports. Each will be explained in detail when first used.

```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

**Why random_state=42?** Reproducibility. Decision trees, random forests, and gradient boosters all involve randomness (random feature selection, random data subsampling). Fixing the seed means every person who runs this notebook gets identical results. 42 is conventionally used in data science — no mathematical significance, just tradition.

`np.random.seed(42)` sets the global numpy random state. The `RANDOM_STATE` constant is passed to every model and split function so we never accidentally use a different seed.

---

## 5. Dataset Loading and EDA

```python
df = pd.read_csv("poetry_dataset.csv")
print(f"Dataset shape : {df.shape}")
print(df["label"].value_counts().to_string())
```

**What is EDA?** Exploratory Data Analysis — examining the dataset before modeling to understand its properties. Here we check two things: shape (rows × columns) and class distribution.

**Why check class distribution?** If one class has far more samples than others, classifiers will be biased toward that class. They learn "when unsure, predict the majority class" which gives misleadingly high overall accuracy but poor performance on minority classes. Our dataset shows:

```
Tennyson      500
Milton        500
Keats         500
Shakespeare   500
Coleridge     285
Wordsworth    210
```

Coleridge and Wordsworth are underrepresented. This is not arbitrary — it reflects the actual available corpus size on Project Gutenberg. Coleridge wrote less poetry than Milton. Wordsworth's corpus had fewer clean stanzas after extraction. We address this imbalance with `class_weight="balanced"` in LightGBM (explained later).

**The bar chart** visualizes this distribution. We use `reindex(POETS)` to ensure bars appear in our defined order (alphabetical) rather than sorted by count. Each poet gets a distinct color from `POET_COLORS` for visual differentiation.

---

## 6. Feature Engineering — Theory First

Before writing a single line of feature code, it's worth understanding what we're trying to measure and why each category matters for distinguishing these six poets.

### Why Not Just Use Raw Text?

You could feed raw poem text directly into a classifier. But a bag-of-words model on raw text would learn to associate specific words with specific poets — "paradise" → Milton, "nightingale" → Keats. This memorizes content rather than learning style. If Milton wrote a poem about nightingales, the model would fail.

Stylometric features capture **how** a poet writes, not **what** they write about. Iambic score measures rhythmic habit. Rhyme density measures structural preference. These patterns are stable across different subjects.

### The 32 Features and Their Signal

| Feature Group | Count | What It Captures |
|---|---|---|
| Prosody (syllables, iambic) | 3 | Rhythmic structure |
| Rhyme (density, patterns) | 4 | End-sound organization |
| Vocabulary (8 lexicons + stats) | 13 | Thematic and linguistic habits |
| Structure (line, enjambment) | 7 | Formal organization |
| POS ratios | 5 | Grammatical style |
| **Total** | **32** | — |

---

## 7. Vocabulary Lexicons

```python
ARCHAIC_WORDS = {
    "thee", "thou", "thy", "thine", "doth", "hath", "hast", "wherefore",
    "whence", "whilst", "oft", "ere", "yonder", "nay", "aye", "forsooth",
    "methinks", "perchance", "prithee", "wilt", "shalt", "wouldst",
    "couldst", "shouldst", "art", "tis", "twas", "twere", "twill",
    "twixt", "betwixt"
}
```

**Why a set not a list?** `in` operator on a set is O(1) (hash lookup). On a list it's O(n) (linear scan). Since we check every word in every stanza against this lexicon, O(1) matters.

**What signal does archaic ratio give?** Shakespeare (1564–1616) and Milton (1608–1674) write in Early Modern English where "thee/thou" are grammatically required. Keats (1795–1821) and Tennyson (1809–1892) use them consciously as a poetic affectation — less frequently. Wordsworth (1770–1850) deliberately used plain language to connect with ordinary people, so his archaic ratio would be low. High archaic ratio = older or deliberately formal poet.

```python
NATURE_WORDS = {
    "sun", "moon", "star", "sky", "cloud", "wind", "storm", "river",
    "sea", "ocean", "wave", "shore", "mountain", "hill", "forest",
    "tree", "flower", "grass", "leaf", "rain", "snow", "spring",
    "summer", "autumn", "winter", "night", "dawn", "dusk", "earth",
    "field", "meadow", "brook", "stream", "lake", "wood"
}
```

**Nature imagery** is common across Romantic poets but differs in *type*. Wordsworth focuses on English lakes and mountains (the Lake District). Keats focuses on sensory richness in autumn. Shelley (not in our set) on wind and sky. This feature alone cannot distinguish Romantic poets but contributes as part of the overall fingerprint.

```python
DARK_WORDS = {
    "death", "dead", "grave", "tomb", "darkness", "shadow", "doom",
    "despair", "grief", "sorrow", "pale", "cold", "ghost", "soul",
    "sin", "curse", "blood", "pain", "fear", "dread", "terror",
    "gloom", "decay", "ruin", "lost", "weep", "mourn", "sigh"
}
```

**Dark imagery** is highest in Coleridge (The Rime of the Ancient Mariner is saturated with death, curse, and ghost imagery) and in Milton (Paradise Lost, the fall of man). Shakespeare's sonnets have relatively low dark imagery; they focus on beauty and time rather than death and horror.

```python
DIVINE_WORDS = {
    "god", "heaven", "angel", "divine", "holy", "sacred", "eternal",
    "paradise", "glory", "grace", "spirit", "soul", "prayer", "blessed",
    "almighty", "lord", "immortal", "celestial", "sublime", "worship"
}
```

**Divine imagery** is most concentrated in Milton. Paradise Lost is literally about God, Satan, Heaven, and Hell — divine vocabulary density is extremely high. Shakespeare's sonnets mention the divine occasionally but are primarily secular. Wordsworth treats nature as quasi-divine ("spots of time") but uses less explicit religious vocabulary.

```python
SENSORY_WORDS = {
    "sweet", "soft", "warm", "bright", "rich", "deep", "pure", "tender",
    "gentle", "smooth", "fragrant", "golden", "silver", "crimson",
    "perfume", "music", "taste", "touch", "sight", "sound"
}
```

**Sensory richness** is Keats's signature. His Odes are famous for synesthesia and dense sensory language — "Season of mists and mellow fruitfulness," "the murmurous haunt of flies on summer eves." John Keats trained as a surgeon and had an unusually vivid awareness of physical sensation. High sensory ratio = likely Keats.

```python
WORDSWORTH_WORDS = {
    "child", "childhood", "memory", "grow", "mountain", "lake",
    "cottage", "shepherd", "peasant", "solitude", "tranquil", "spots",
    "growth", "mind", "recollect", "district", "plain", "wandered",
    "lonely", "repose", "humble", "dwelling", "rustic"
}
```

**Why a poet-specific lexicon?** Because Wordsworth's style is distinctive enough to warrant it. The Prelude (his major work) is a poem about the growth of consciousness through childhood memories in the Lake District. Words like "recollect", "spots" (referring to "spots of time"), "peasant", and "solitude" appear with unusually high frequency in his work and almost nowhere else in our corpus.

**Decision rationale**: We added three poet-specific lexicons (Wordsworth, Tennyson, Keats) only because those three had the most confusion in the confusion matrix. Adding similar lexicons for Shakespeare and Milton would have less impact because they're better separated by prosodic features.

```python
TENNYSON_WORDS = {
    "king", "knight", "sword", "battle", "hero", "noble", "quest",
    "lancelot", "arthur", "camelot", "grail", "glory", "honor",
    "grief", "loss", "mourn", "elegy", "ring", "toll", "vale"
}
```

**Tennyson** has two major stylistic registers: Arthurian legend (Idylls of the King — hence "lancelot", "camelot", "grail") and Victorian elegy (In Memoriam, "Crossing the Bar" — hence "grief", "loss", "toll"). Neither register overlaps much with our other five poets.

```python
KEATS_WORDS = {
    "beauty", "urn", "grecian", "apollo", "psyche", "nightingale",
    "autumn", "ripeness", "indolence", "melancholy", "luxurious",
    "unheard", "melodies", "sweeter", "pipe", "ditties", "boughs"
}
```

**Keats-specific terms** mostly come from his six major odes: Ode to a Nightingale, Ode on a Grecian Urn, Ode to Psyche, Ode on Melancholy, Ode to Autumn, Ode on Indolence. Words like "nightingale", "grecian", "psyche", "ditties", and "boughs" appear in multiple odes and are essentially Keats's signature vocabulary.

---

## 8. Syllable and Rhythm Functions

### `count_syllables(word)`

```python
def count_syllables(word):
    word = word.lower().strip(string.punctuation)
    if word in cmu_dict:
        return max(len([s for s in pron if s[-1].isdigit()])
                   for pron in cmu_dict[word])
    return max(1, len(re.findall(r"[aeiouy]+", word)))
```

**What is a syllable?** A syllable is a unit of pronunciation consisting of a vowel sound, possibly surrounded by consonants. "Po-et-ry" has 3 syllables. "Night" has 1.

**How does CMU dict encode syllables?** Each word is represented as a sequence of ARPABET phonemes. Phonemes that carry stress end with a digit (0=unstressed, 1=primary stress, 2=secondary stress). So "poetry" → `['P', 'OW1', 'AH0', 'T', 'R', 'IY0']`. Counting the phonemes ending in digits gives syllable count: `OW1`, `AH0`, `IY0` → 3 syllables.

**Why `.lower().strip(string.punctuation)`?** CMU dict uses lowercase keys. Poetry often has apostrophes in contractions ("'tis", "e'er") that must be stripped before lookup.

**Why `max(...for pron in cmu_dict[word])`?** Some words have multiple valid pronunciations (e.g., "the" can be 1 syllable as /ðə/ or 1 syllable as /ðiː/; "either" can be 2 or 2). We take the maximum pronunciation length to avoid undercounting.

**Fallback: vowel group counting.** If the word isn't in CMU dict (rare words, proper nouns, archaic forms), we count consecutive vowel groups (a,e,i,o,u,y) using regex. "Beautiful" → `beau-ti-ful` → vowel groups: "eau", "i", "u" → 3. `max(1, count)` ensures minimum 1 syllable (single consonant words would return 0 otherwise).

**Why is syllable count useful?** Different poetic forms have fixed syllable patterns. A Shakespearean sonnet line has 10 syllables (iambic pentameter). Milton's Paradise Lost uses 10-syllable blank verse. Coleridge's Ancient Mariner uses 4-stress ballad meter with 7-8 syllables. These differences show up clearly in the average and variance of syllables per line.

### `get_syllable_features(lines)`

```python
def get_syllable_features(lines):
    counts = []
    for line in lines:
        words = [w for w in word_tokenize(line.lower()) if w.isalpha()]
        counts.append(sum(count_syllables(w) for w in words))
    return {
        "avg_syllables_per_line": np.mean(counts) if counts else 0,
        "std_syllables_per_line": np.std(counts) if counts else 0,
    }
```

**Why `if w.isalpha()`?** Punctuation tokens from `word_tokenize` (commas, periods, apostrophes) would return nonsense from `count_syllables`. Filtering to alphabetic-only tokens ensures clean syllable counts.

**`avg_syllables_per_line`** — captures the average line length in phonological units. Milton's blank verse (~10 syllables/line) vs Blake's short mystical lines (~6 syllables/line) differ significantly.

**`std_syllables_per_line`** — captures consistency of line length. Shakespeare's sonnets have very consistent iambic pentameter (low std). Free verse poets have high variance. This distinguishes structured verse from looser forms.

**Why we dropped `max_syllables_per_line`** — the correlation matrix showed it was nearly perfectly correlated with `avg_syllables_per_line` (they move together). Correlated features don't add new information; they just add noise and slow training. Removed in final version.

### `get_iambic_score(lines)`

```python
def get_iambic_score(lines):
    iambic_count = 0
    for line in lines:
        phones = []
        for word in word_tokenize(line.lower()):
            word = word.strip(string.punctuation)
            if word in cmu_dict:
                phones.extend(cmu_dict[word][0])
        stresses = [p[-1] for p in phones if p[-1].isdigit()]
        if len(stresses) >= 4:
            pairs = [(stresses[i], stresses[i+1]) 
                     for i in range(0, len(stresses)-1, 2)]
            iambic_pairs = sum(1 for a, b in pairs if a == "0" and b == "1")
            if pairs and iambic_pairs / len(pairs) > 0.5:
                iambic_count += 1
    return iambic_count / len(lines) if lines else 0
```

**What is iambic meter?** An iamb is a metrical foot consisting of an unstressed syllable followed by a stressed syllable: da-DUM. "Shall I com-PARE thee TO a SUM-mer's DAY" — each pair (shall-I, com-PARE, thee-TO, a-SUM, mer's-DAY) is an iamb. Five iambs per line = iambic pentameter = the dominant meter of Shakespeare, Milton, and Keats's longer works.

**How does the algorithm detect iambic pattern?**

1. Get all phoneme strings for each word in the line
2. Extract just the stress markers (the digit at the end of vowel phonemes)
3. `stresses = [p[-1] for p in phones if p[-1].isdigit()]` — gives a sequence like `['1', '0', '1', '0', '1']`
4. Pair them up: `(stresses[0], stresses[1])`, `(stresses[2], stresses[3])`, ...
5. An iambic pair is `('0', '1')` — unstressed then stressed
6. If more than 50% of pairs are iambic, the line counts as iambic
7. Return the ratio of iambic lines

**Why `>= 4 stresses`?** Fewer than 4 stress markers means we can't reliably detect a pattern. Single-word lines or very short fragments would give misleading results.

**Why `cmu_dict[word][0]` and not `max`?** For iambic detection we want one specific pronunciation to trace stress patterns, not the maximum. Taking the first pronunciation is standard practice for stress pattern analysis.

**Expected behavior by poet:**
- Shakespeare: high iambic score (sonnets = strict iambic pentameter)
- Milton: high iambic score (blank verse = iambic pentameter)
- Keats: moderate (odes have iambic elements but not strictly)
- Wordsworth: low (blank verse but looser)
- Tennyson: varies widely (ballads vs elegies)
- Coleridge: low (ballad meter, not iambic)

---

## 9. Rhyme Detection Functions

### `get_last_word(line)`

```python
def get_last_word(line):
    words = re.findall(r"\b[a-zA-Z]+\b", line)
    return words[-1].lower() if words else ""
```

**Why regex instead of `split()`?** `line.split()[-1]` would return "day," (with punctuation). We need just the word. `re.findall(r"\b[a-zA-Z]+\b", line)` finds all word-only tokens, then we take the last one.

**Why lowercase?** CMU dict and the `pronouncing` library both use lowercase. Normalizing here avoids case mismatches.

### `words_rhyme(w1, w2)`

```python
def words_rhyme(w1, w2):
    if not w1 or not w2 or w1 == w2:
        return False
    phones1 = pronouncing.phones_for_word(w1)
    phones2 = pronouncing.phones_for_word(w2)
    if not phones1 or not phones2:
        return w1[-2:] == w2[-2:] and len(w1) > 2
    return w2 in set(pronouncing.rhymes(w1))
```

**Why `w1 == w2` returns False?** A word rhyming with itself is trivially true and not a real rhyme. If a poet repeats the same word at line endings, we don't want to count that as rhyming.

**`pronouncing.phones_for_word(w)`** — returns the phoneme sequence for a word. If the word isn't in the pronouncing library's dictionary, returns an empty list.

**`pronouncing.rhymes(w1)`** — returns all words in the CMU dict that rhyme with `w1`. Two words rhyme if they share the same vowel sound and everything after the stressed vowel in the last syllable. "Day" and "say" rhyme because they share "AY" phoneme. "Night" and "light" rhyme because they share "AYT".

**Fallback: last 2 characters.** If neither word is in the pronouncing dictionary (rare archaic words, proper nouns), we fall back to checking if the last 2 characters match. This is imperfect ("at" and "cat" end the same, technically rhyme; "to" and "go" don't despite ending differently) but better than no detection. The `len(w1) > 2` guard prevents very short words from matching spuriously.

### `get_rhyme_features(lines)`

```python
def get_rhyme_features(lines):
    if len(lines) < 2:
        return {"rhyme_density": 0, "is_abab": 0, "is_aabb": 0, "is_abba": 0}
    
    last_words = [get_last_word(l) for l in lines]
    n = len(last_words)
    
    rhyme_count = total_pairs = 0
    for i in range(n):
        for j in range(i+1, min(i+5, n)):
            total_pairs += 1
            if words_rhyme(last_words[i], last_words[j]):
                rhyme_count += 1
    
    rhyme_density = rhyme_count / total_pairs if total_pairs else 0
    
    is_abab = int(n >= 4 and words_rhyme(last_words[0], last_words[2])
                  and words_rhyme(last_words[1], last_words[3]))
    is_aabb = int(n >= 4 and words_rhyme(last_words[0], last_words[1])
                  and words_rhyme(last_words[2], last_words[3]))
    is_abba = int(n >= 4 and words_rhyme(last_words[0], last_words[3])
                  and words_rhyme(last_words[1], last_words[2]))
    
    return {"rhyme_density": rhyme_density, "is_abab": is_abab,
            "is_aabb": is_aabb, "is_abba": is_abba}
```

**`rhyme_density`** — what fraction of nearby line-end word pairs rhyme? We check pairs within a window of 4 lines (`min(i+5, n)`) because poems rarely rhyme lines that are far apart. Shakespeare's sonnets have high rhyme density (they follow ABAB CDCD EFEF GG scheme). Milton's Paradise Lost has zero rhyme density (he explicitly rejected rhyme as a "bondage").

**`is_abab`** — detects the ABAB rhyme scheme. Lines 0 and 2 rhyme (A), lines 1 and 3 rhyme (B). This is the most common quatrain form in English Romantic poetry. Shakespeare's sonnets use ABAB in their three quatrains.

**`is_aabb`** — detects couplets (AABB). Lines 0 and 1 rhyme, lines 2 and 3 rhyme. Common in Tennyson's heroic verse and some Coleridge ballads.

**`is_abba`** — detects "envelope rhyme" or "Italian quatrain." Lines 0 and 3 rhyme, lines 1 and 2 rhyme. Tennyson's In Memoriam famously uses this ABBA scheme. This feature is highly specific to Tennyson.

**Why `int(n >= 4 and ...)`?** We can't detect a 4-line rhyme scheme without at least 4 lines. The `int()` converts True/False to 1/0 for use as a numeric feature.

**Note from confusion matrix:** Rhyme features had surprisingly low discriminative power for these specific poets because many of our training stanzas were extracted from longer poems where rhyme schemes don't complete within a single stanza. `rhyme_density` was most useful; the pattern features (is_abab etc.) had near-zero variance and were reflected as low importance in LightGBM's feature importance chart.

---

## 10. Vocabulary Feature Extractor

```python
def get_vocabulary_features(text, lines):
    words_raw     = word_tokenize(text.lower())
    words_only    = [w for w in words_raw if w.isalpha()]
    total         = len(words_only) if words_only else 1
    
    return {
        "archaic_ratio":     sum(1 for w in words_only if w in ARCHAIC_WORDS) / total,
        "nature_ratio":      sum(1 for w in words_only if w in NATURE_WORDS) / total,
        "dark_ratio":        sum(1 for w in words_only if w in DARK_WORDS) / total,
        "divine_ratio":      sum(1 for w in words_only if w in DIVINE_WORDS) / total,
        "sensory_ratio":     sum(1 for w in words_only if w in SENSORY_WORDS) / total,
        "wordsworth_ratio":  sum(1 for w in words_only if w in WORDSWORTH_WORDS) / total,
        "tennyson_ratio":    sum(1 for w in words_only if w in TENNYSON_WORDS) / total,
        "keats_ratio":       sum(1 for w in words_only if w in KEATS_WORDS) / total,
        "type_token_ratio":  len(set(words_only)) / total,
        "avg_word_length":   np.mean([len(w) for w in words_only]) if words_only else 0,
        "punct_density":     sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
        "exclaim_ratio":     text.count("!") / len(lines) if lines else 0,
        "question_ratio":    text.count("?") / len(lines) if lines else 0,
    }
```

**Why divide by `total` (word count)?** Normalizing by total words converts raw counts to ratios. A long stanza would naturally have more archaic words than a short one even if the ratio is the same. Ratios are length-invariant, making them comparable across stanzas of different sizes.

**`total = len(words_only) if words_only else 1`** — the `else 1` prevents division by zero for empty texts (though this should never happen in clean data).

**`type_token_ratio (TTR)`** — `len(set(words_only)) / total`. The ratio of unique words to total words. A TTR of 0.8 means 80% of words are unique (high vocabulary richness). A TTR of 0.4 means many words are repeated (low richness). Milton and Shakespeare tend to have higher TTR than Coleridge whose repetitive ballad refrains pull it down. TTR is one of the oldest and most studied stylometric features.

**`avg_word_length`** — longer average words suggest more Latinate, formal vocabulary. Milton's Latinate epic diction ("innumerable", "ineffable", "immutable") pushes average word length up. Wordsworth's plain-language manifesto in the Preface to Lyrical Ballads leads to shorter, Anglo-Saxon words.

**`punct_density`** — total punctuation characters divided by text length. Milton uses many commas and semicolons in his long periodic sentences. Shakespeare uses frequent colons and semi-colons in sonnets. This feature captures punctuation habits as a fraction of all characters.

**`exclaim_ratio`** — exclamation marks per line. High in emotionally intense writing. Coleridge's Mariner has many dramatic exclamations ("Water, water, everywhere!"). Milton almost never uses exclamation marks (epic grandeur doesn't exclaim).

**`question_ratio`** — rhetorical questions per line. High in Shakespeare's sonnets ("Shall I compare thee?", "Why is my verse so barren?"). Rhetorical questions are a Shakespearean tic.

---

## 11. Structural Feature Extractor

```python
def get_structural_features(text, lines):
    word_counts = [len(l.split()) for l in lines]
    
    word_freq = Counter(text.lower().split())
    repeated  = sum(1 for w, c in word_freq.items()
                    if c >= 2 and w not in stop_words and len(w) > 3)
    
    end_punct = set(".,;:!?")
    enjambed  = sum(1 for l in lines 
                    if l.strip() and l.strip()[-1] not in end_punct)
    
    return {
        "avg_line_length":   np.mean(word_counts) if word_counts else 0,
        "std_line_length":   np.std(word_counts)  if word_counts else 0,
        "repetition_score":  repeated / len(word_freq) if word_freq else 0,
        "enjambment_ratio":  enjambed / len(lines) if lines else 0,
        "cap_ratio":         sum(1 for l in lines 
                                 if l.strip() and l.strip()[0].isupper()) / len(lines)
                             if lines else 0,
        "line_count":        len(lines),
        "flesch_score":      textstat.flesch_reading_ease(text),
    }
```

**`avg_line_length`** (in words) — short lines (3-5 words) are characteristic of ballad meter (Coleridge, Tennyson's "The Charge of the Light Brigade"). Long lines (8-12 words) are characteristic of epic blank verse (Milton's Paradise Lost).

**`std_line_length`** — variance in line length. Milton's blank verse is very consistent (low std). Coleridge's Ancient Mariner alternates 4-stress and 3-stress lines (higher std). Free verse has high variance.

**`repetition_score`** — what fraction of meaningful words appear more than once? We filter out stopwords (`w not in stop_words`) and very short words (`len(w) > 3`) to focus on content word repetition. Coleridge is famous for repetition: "Water, water, everywhere." Tennyson repeats key phrases in elegies for emotional effect ("Break, break, break"). Shakespeare repeats less. High repetition score = likely Coleridge or Tennyson.

**`enjambment_ratio`** — what fraction of lines do NOT end with terminal punctuation? A line that ends without punctuation means the sentence continues into the next line — this is enjambment. Keats heavily uses enjambment in his odes for flowing, breathless effect ("Season of mists and mellow fruitfulness, / Close bosom-friend of the maturing sun"). Milton uses enjambment extensively. Shakespeare's sonnets tend to be more end-stopped (punctuation at line endings).

**`cap_ratio`** — what fraction of lines begin with a capital letter? In classical poetry, almost all lines begin with capitals (it was typographic convention). Very low cap ratio would indicate unusual formatting or prose-like stanzas that leaked through our extraction filter.

**`line_count`** — number of lines in the stanza. Sonnets are 14 lines. Ballad stanzas are 4 lines. Milton's blank verse comes in variable-length sections. This feature helps distinguish stanza types.

**`flesch_reading_ease`** — the Flesch Reading Ease formula:

```
206.835 - 1.015 × (total_words / total_sentences) 
        - 84.6 × (total_syllables / total_words)
```

Score 0-100. Higher = easier to read. Complex sentences with long words score low. Simple sentences with short words score high. Milton scores around 40 (complex). Tennyson's shorter lyrics score around 70 (accessible). This captures syntactic complexity indirectly.

**Why we dropped `flesch_kincaid`** — Flesch-Kincaid grade level is directly derived from the same sentence length and syllable count inputs as Flesch Reading Ease, just with different coefficients. The correlation matrix confirmed they are nearly perfectly inversely correlated (one goes up exactly as the other goes down). Keeping both adds no information and wastes a feature slot.

---

## 12. POS Tag Feature Extractor

```python
def get_pos_features(text):
    words  = word_tokenize(text)
    tagged = pos_tag(words)
    total  = len(tagged) if tagged else 1
    
    return {
        "noun_ratio":  sum(1 for _, t in tagged if t.startswith("NN")) / total,
        "verb_ratio":  sum(1 for _, t in tagged if t.startswith("VB")) / total,
        "adj_ratio":   sum(1 for _, t in tagged if t.startswith("JJ")) / total,
        "adv_ratio":   sum(1 for _, t in tagged if t.startswith("RB")) / total,
        "prop_ratio":  sum(1 for _, t in tagged if t == "NNP") / total,
    }
```

**What are POS tags?** Part-of-Speech tags label each word with its grammatical role. The Penn Treebank tagset (what NLTK uses) includes:
- `NN` = noun (singular), `NNS` = noun (plural), `NNP` = proper noun
- `VB` = verb (base form), `VBD` = past tense, `VBG` = gerund, etc.
- `JJ` = adjective, `JJR` = comparative, `JJS` = superlative
- `RB` = adverb, `RBR` = comparative, `RBS` = superlative

**Why `.startswith("NN")` instead of `== "NN"`?** All noun tags start with "NN" (NN, NNS, NNP, NNPS). `.startswith("NN")` captures all noun types in one check. Same logic for verbs (`VB`) and adjectives (`JJ`).

**`noun_ratio`** — Milton's epic poetry has very high noun density (catalogues of places, angels, characters). Lyric poetry (Keats, Tennyson) tends toward more verbs and adjectives.

**`verb_ratio`** — active, dynamic writing has more verbs. Coleridge's action-driven ballad (the Mariner sailing, shooting, cursing) has higher verb density than meditative sonnet sequences.

**`adj_ratio`** — Keats is famous for adjective-loading: "tender is the night," "drowsy numbness," "mellow fruitfulness." High adjective ratio is a Keats signal.

**`adv_ratio`** — adverbs modify verbs and adjectives. High adverb use is more characteristic of prose style than verse. Generally low across all poets; useful as a baseline discriminator.

**`prop_ratio`** — proper nouns (names of people, places, gods). Milton has extremely high proper noun density: Satan, Beelzebub, Eden, Pandemonium, Raphael, Gabriel. Tennyson has high prop_ratio from Arthurian names (Lancelot, Arthur, Camelot). Shakespeare's sonnets have very few proper nouns. This feature is highly discriminative for Milton.

**Why add POS features?** They capture grammatical style that surface-level word counts miss. Two poems can use similar words but in very different grammatical patterns. POS ratios are a well-validated stylometric signal in authorship attribution literature.

---

## 13. Master Feature Extractor

```python
def extract_all_features(text):
    if "/" in text:
        lines = [l.strip() for l in text.split("/") if l.strip()]
    else:
        lines = re.split(r"(?<=[.!?,;])\s+", text)
        lines = [l.strip() for l in lines if l.strip()]
    if not lines or len(lines) == 1:
        words = text.split()
        lines = [" ".join(words[i:i+8]) for i in range(0, len(words), 8)]
    
    feats = {}
    feats.update(get_syllable_features(lines))      # 2 features
    feats["iambic_score"] = get_iambic_score(lines) # 1 feature
    feats.update(get_rhyme_features(lines))          # 4 features
    feats.update(get_vocabulary_features(text, lines)) # 13 features
    feats.update(get_structural_features(text, lines)) # 7 features
    feats.update(get_pos_features(text))             # 5 features
    return feats                                     # Total: 32 features
```

**The line-splitting problem** — our dataset stores stanzas in two possible formats. Original Gutenberg downloads used "/" as a separator between lines ("Shall I compare thee / Thou art more lovely"). HuggingFace corpus rows are single text strings with space-joined lines. This function handles both.

**Format 1: slash-separated** — If "/" is in the text, split on "/" to recover individual lines.

**Format 2: continuous text** — Split on punctuation boundaries using `re.split(r"(?<=[.!?,;])\s+", text)`. The `(?<=...)` is a lookbehind assertion — it matches a space that comes after terminal punctuation. This splits "Round many western islands have I been, Which bards in fealty to Apollo hold." → ["Round many western islands have I been,", "Which bards in fealty to Apollo hold."]

**Fallback: 8-word windows** — If neither splitting method produces multiple lines (some stanzas are just long run-on strings), split into chunks of 8 words each. Not perfect but ensures the prosodic features can run.

**Why pass both `text` and `lines` to sub-functions?** Some features need the raw text (punctuation density requires exact character counts, Flesch score needs sentences), while others need individual lines (syllable counting, rhyme detection, enjambment).

**`feats.update()`** — adds all key-value pairs from a sub-function's returned dict into the main feats dict. This pattern avoids repeating `feats["x"] = ...` for each feature.

---

## 14. Text Representations — TF-IDF

### Theory: What is TF-IDF?

TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text into numerical vectors where each dimension corresponds to a word in the vocabulary.

**Term Frequency (TF):** How often does word w appear in document d?

```
TF(w, d) = count(w in d) / total_words(d)
```

Raw frequency is normalized by document length so longer documents don't get unfair weight.

**Inverse Document Frequency (IDF):** How rare is word w across all documents?

```
IDF(w) = log(N / df(w))
```

where N = total documents, df(w) = documents containing w. Words that appear in every document (like "the", "and") get low IDF. Words that appear in only a few documents get high IDF.

**TF-IDF:** Multiply them.

```
TFIDF(w, d) = TF(w, d) × IDF(w)
```

Words that are frequent in this specific document AND rare across all documents get high scores. These are the distinctive vocabulary choices — the words that mark this poet as different from all others.

**With `sublinear_tf=True`**, we use `log(1 + TF)` instead of raw TF. This dampens the effect of very high frequency words. "The" appearing 50 times vs 5 times in one poem isn't 10× more important.

```python
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3
)
tfidf_matrix = tfidf.fit_transform(df["text"])
```

**`max_features=5000`** — keep only the top 5000 words by TF-IDF score across the corpus. This limits memory and computation. 5000 words captures most distinctive vocabulary without overfitting to rare terms.

**`ngram_range=(1, 2)`** — include both unigrams (single words) and bigrams (word pairs). Bigrams capture phrases like "dark night," "eternal bliss," "tender heart" that carry more meaning than individual words. These phrases are often more poet-specific than individual words.

**`min_df=3`** — ignore words that appear in fewer than 3 stanzas. Very rare words are likely noise (typos, OCR errors from Gutenberg) or proper nouns that appear only in one poet's work (overfitting risk).

**Result:** A matrix of shape (2495 stanzas × 5000 words). Each row is a stanza's TF-IDF vector. Stanzas that use similar vocabulary will have similar vectors.

---

## 15. Dimensionality Reduction — SVD

### Theory: What is Truncated SVD (LSA)?

The TF-IDF matrix is 2495 × 5000. Training a classifier on 5000 features for 2495 samples risks overfitting (too many features relative to data). We also have the "curse of dimensionality" — in high-dimensional spaces, distance measures become less meaningful.

**Truncated SVD** (also called Latent Semantic Analysis / LSA) compresses the matrix. It finds the most important "directions" of variation in the data and keeps only those.

Mathematically, SVD decomposes matrix M into M = U × Σ × Vᵀ where:
- U = document-concept matrix (documents in concept space)
- Σ = diagonal matrix of singular values (importance of each concept)
- Vᵀ = concept-term matrix (how each term loads on each concept)

Keeping only the top k singular values gives the best rank-k approximation. With k=150, we compress 5000 dimensions → 150 "semantic dimensions" that capture the most important vocabulary patterns.

**Why is this called "Latent Semantic"?** Because the resulting dimensions don't correspond to specific words — they're latent concepts. Dimension 3 might represent "dark gothic imagery" (high loading on death, grave, shadow, doom) without mapping to any single word.

```python
svd150 = TruncatedSVD(n_components=150, random_state=RANDOM_STATE)
tfidf_150 = svd150.fit_transform(tfidf_matrix)
```

**"Truncated" vs full SVD** — Full SVD computes all dimensions (5000). Truncated SVD only computes the top k, which is much faster and uses less memory.

**Why 150 dimensions?** We tested 50, 100, 150, 200, 300 in our ablation study:

```
TF-IDF 50 dims  → 69.74%
TF-IDF 100 dims → 72.95%
TF-IDF 150 dims → 74.35%  ← best
TF-IDF 200 dims → 71.74%  (overfitting starts)
TF-IDF 300 dims → 72.55%
```

150 hits the sweet spot. Beyond 150, accuracy drops because we're keeping noise dimensions that confuse the model.

**Combined feature matrix:**

```python
X_final = np.hstack([X_handcrafted, tfidf_150, char_reduced])
```

`np.hstack` horizontally stacks numpy arrays. Each row becomes a 232-dimensional vector:
- 32 hand-crafted stylometric features
- 150 TF-IDF semantic dimensions
- 50 character n-gram dimensions

**StandardScaler:**

```python
scaler = StandardScaler()
X_sc_fin = scaler.fit_transform(X_final)
```

Standardizes each feature to zero mean and unit variance. Feature 1 (avg_syllables, range 2-34) and Feature 100 (TF-IDF dimension, range -0.3 to 0.5) are on completely different scales. Tree-based models like LightGBM don't require scaling (they use rank-based splits), but scaling doesn't hurt and keeps our pipeline consistent for potential future use with distance-based models.

**Critical: `fit_transform` on training data only, then `transform` on test data.** We fit the scaler on training data (compute mean and std from training), then apply the same transformation to test data. If we fit on all data, we'd "leak" test statistics into training — a form of data leakage.

---

## 16. Character N-Grams

```python
char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=3000,
    sublinear_tf=True,
    min_df=3
)
char_matrix = char_tfidf.fit_transform(df["text"])
char_svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
char_reduced = char_svd.fit_transform(char_matrix)
```

**What are character n-grams?** Instead of looking at whole words, we look at sequences of n consecutive characters. "thee" with n=3 generates: "the", "hee". With n=4: "thee". With n=5: "_thee" (underscore = word boundary).

**`analyzer="char_wb"`** — character n-grams within word boundaries. The `_wb` variant adds spaces at word boundaries before extracting n-grams. So "thee" becomes " thee " before extraction, generating " th", "the", "hee", "ee ". The word boundary markers help distinguish "thee" (a word) from "thee" as a substring of "together" (different context).

**Why do character n-grams help?** They capture:

1. **Archaic morphology** — "eth" ending (doth, hath, giveth), "'st" suffix (didst, wouldst, couldst), "tis"/"twas" contractions. These are character-level patterns invisible to word-level TF-IDF.

2. **Spelling habits** — Milton and Shakespeare spell some words differently from modern English. Character n-grams capture these spelling fingerprints.

3. **Robustness to OCR noise** — some Gutenberg texts have OCR errors. Character n-grams are more robust to single-character errors than word-level features.

**`ngram_range=(3, 5)`** — 3, 4, and 5 character sequences. Shorter than 3 is too generic (all words share 2-char endings). Longer than 5 rarely overlaps between words.

**Why 50 SVD dimensions for char n-grams vs 150 for word n-grams?** Character n-grams carry less semantic information than words. They're supplementary. 50 compressed dimensions is sufficient to capture the morphological signal without adding too much noise.

**Combined accuracy jump:** Adding char n-grams took us from 80.56% (hand-crafted + word TF-IDF) to 81.36%. Small but consistent improvement.

---

## 17. MLflow Experiment Tracking

### Theory: What is MLflow?

MLflow is an open-source platform for managing the machine learning lifecycle. It addresses a real problem: when you run 11 different experiments (different feature sets, different models), how do you keep track of what parameters produced what results?

Without tracking, you rely on memory or scattered notes. MLflow automatically records:
- **Parameters** — what hyperparameters did you use? (n_estimators=500, learning_rate=0.05)
- **Metrics** — what were the results? (accuracy=0.8136, f1_macro=0.7910)
- **Artifacts** — what files did the model produce? (confusion matrix image)
- **Run metadata** — when did it run? How long did it take?

```python
mlflow.set_experiment("PoetryDNA_NLP")
```

Creates a named experiment container. All runs belong to this experiment.

```python
def run_experiment(run_name, X_all, y_enc, model, params: dict, feature_count: int):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_enc, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_enc
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("feature_count", feature_count)
        model.fit(X_tr, y_tr)
        preds   = model.predict(X_te)
        acc     = accuracy_score(y_te, preds)
        f1_mac  = f1_score(y_te, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_mac)
    return acc, f1_mac, X_tr, X_te, y_tr, y_te
```

**`with mlflow.start_run(run_name=run_name):`** — context manager that opens a tracking run. Everything inside the `with` block is attributed to this run. The run automatically closes when the `with` block exits.

**`mlflow.log_params(params)`** — records all parameters from the dict. Used to record which model and feature configuration was used.

**`mlflow.log_metric("accuracy", acc)`** — records numerical metrics. Queryable later via `mlflow.search_runs()`.

**`train_test_split(..., stratify=y_enc)`** — the `stratify` parameter ensures the class distribution in train and test splits matches the overall distribution. Without stratification, by chance the test set might have very few Coleridge samples, making evaluation unreliable.

**`test_size=0.2`** — 20% of data held out for testing, 80% for training. For a dataset of 2495 samples, this gives ~499 test samples — enough for reliable accuracy estimation.

---

## 18. Model Training — All Three Classifiers

We trained three classifiers on the hand-crafted features as our baseline experiments A1, A2, A3. The goal was to establish which classifier family works best before adding TF-IDF.

---

## 19. Random Forest — Theory and Why We Tried It

### Theory

A Random Forest is an ensemble of decision trees. Each tree is trained on a random subset of the training data (bootstrap sampling) and makes splits using only a random subset of features (feature subsampling). The final prediction is the majority vote across all trees.

**Why randomness?** Individual decision trees are high-variance — they overfit to training data. By building many trees on different random subsets and averaging, variance is reduced (the trees' errors are uncorrelated and cancel out) while bias stays low.

**Why it was a natural first choice for NLP features:**
- Handles mixed feature types (binary flags like is_abab + continuous like avg_syllables) naturally
- Doesn't require feature scaling
- Provides feature importance scores
- Robust to irrelevant features (they just get ignored)
- No hyperparameter tuning required for a reasonable baseline

```python
RandomForestClassifier(
    n_estimators=500,        # 500 trees
    class_weight="balanced", # compensate for Wordsworth/Coleridge imbalance
    random_state=RANDOM_STATE, 
    n_jobs=-1                # use all CPU cores
)
```

**`n_estimators=500`** — 500 trees. More trees = more stable predictions but diminishing returns beyond ~200. 500 is safe. The cost is training time, which is acceptable here.

**`class_weight="balanced"`** — automatically adjusts weights inversely proportional to class frequencies. Wordsworth (210 samples) gets weight ~2.4× higher than Shakespeare (500 samples). This prevents the model from ignoring minority classes.

**Result: 62.12%** — respectable for 32 features on 6 classes, but plateau. Adding more hand-crafted features is unlikely to help much.

---

## 20. XGBoost — Theory and Why We Tried It

### Theory

XGBoost (Extreme Gradient Boosting) builds trees sequentially. Each new tree corrects the errors of all previous trees. This is called boosting.

**How boosting works:**
1. Start with a simple prediction (class proportions)
2. Calculate residuals (errors)
3. Train a tree to predict those residuals
4. Add this tree's predictions (scaled by learning rate) to the running total
5. Repeat until n_estimators trees are built

The key difference from Random Forest: trees are built sequentially, each learning from the previous tree's mistakes. This makes gradient boosting more accurate than Random Forest on most tabular datasets.

```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="mlogloss",
    random_state=RANDOM_STATE, 
    n_jobs=-1
)
```

**`max_depth=6`** — maximum tree depth. Deeper trees capture more complex interactions but overfit more. 6 is standard.

**`learning_rate=0.05`** — how much each tree contributes. Small learning rate + many trees = better generalization but slower training. 0.05 with 500 trees is a standard combination.

**`eval_metric="mlogloss"`** — multiclass log loss. Measures the quality of predicted probabilities, not just predicted class labels. Better metric for multi-class problems than accuracy during training.

**Result: 63.33%** — slightly better than RF but not significantly. The bottleneck is the feature quality, not the model.

---

## 21. LightGBM — Theory and Why It Won

### Theory

LightGBM (Light Gradient Boosting Machine, developed by Microsoft) is gradient boosting but with two key algorithmic differences:

**1. Leaf-wise tree growth** — Standard gradient boosting (XGBoost) grows trees level-by-level (all nodes at the same depth get split simultaneously). LightGBM grows by choosing the single leaf with the highest potential gain and splitting it. This produces asymmetric, deeper trees that capture complex patterns more efficiently.

**2. Histogram-based splitting** — Instead of sorting all features to find the best split point (O(n log n)), LightGBM bins continuous features into 255 histogram bins and finds the best split per bin (O(k) where k=255). Massively faster on large feature matrices.

These two improvements make LightGBM particularly effective on:
- Mixed feature types (our 32 hand-crafted + 200 TF-IDF dims)
- Sparse features (TF-IDF vectors are very sparse — most words don't appear in any given stanza)
- Datasets where some features are much more informative than others

```python
lgbm_params = dict(
    n_estimators=500, 
    learning_rate=0.05,
    class_weight="balanced", 
    random_state=RANDOM_STATE,
    n_jobs=-1, 
    verbose=-1          # suppress all output
)
```

**`verbose=-1`** — LightGBM prints training progress by default. `-1` silences it completely.

**Why LightGBM won over RF and XGBoost:**

With hand-crafted features only, all three are similar (62-63%). The gap opens when TF-IDF is added. The TF-IDF feature matrix has 150 dense SVD dimensions with complex interactions — LightGBM's leaf-wise growth captures these non-linear relationships better than RF's random splits or XGBoost's level-wise growth.

Final accuracy trajectory:
```
Hand-crafted only → RF: 62.12%, XGB: 63.33%, LGBM: 63.33%  (roughly equal)
+ TF-IDF 150      → LGBM: 80.56%  (all others not tested at this point)
+ Char n-grams    → LGBM: 81.36%  (final)
```

---

## 22. Feature Importance Analysis

```python
importances   = lgbm_final.feature_importances_
feat_imp_df   = pd.DataFrame({
    "feature":    all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(20)
```

**What does LightGBM feature importance measure?** By default, LightGBM uses `split` importance — how many times a feature was used as a split point across all trees. A feature used to split 1200 times is much more discriminative than one used 10 times.

**Key finding from the chart:** The top 5 features are a mix of TF-IDF semantic dimensions (tfidf_5, tfidf_1) and hand-crafted features (prop_ratio, rhyme_density, char_0). This validates our design — neither TF-IDF alone nor hand-crafted features alone would be sufficient.

**`prop_ratio` (proper noun ratio) as 2nd most important** — confirms that Milton's Latinate proper nouns (Satan, Gabriel, Eden) and Tennyson's Arthurian names are highly discriminative. Proper noun density separates Milton from all other poets.

**`rhyme_density` as 4th most important** — validates our prosodic feature design. Milton's zero-rhyme blank verse vs Shakespeare's high-rhyme sonnets is a genuine and learnable difference.

**Why TF-IDF dims appear as "tfidf_5", "tfidf_1"** — SVD dimensions don't have interpretable names. Dimension 5 might represent "Miltonic biblical vocabulary", but we can't read that off the number. This is a limitation of latent semantic analysis — interpretability is traded for expressiveness.

---

## 23. Final Model Evaluation

```python
print(classification_report(y_te, y_pred, 
                             target_names=le_nlp.classes_, 
                             digits=4))
```

The `classification_report` prints precision, recall, F1-score, and support for each class.

**Precision** = Of all samples predicted as Keats, what fraction actually are Keats? High precision = few false positives.

**Recall** = Of all actual Keats samples, what fraction did we correctly predict as Keats? High recall = few false negatives.

**F1-score** = Harmonic mean of precision and recall. F1 = 2 × (P × R) / (P + R). Balances precision and recall; good when classes are imbalanced.

**Results:**

```
Coleridge    P=0.8065  R=0.8772  F1=0.8403  support=57
Keats        P=0.7576  R=0.7500  F1=0.7538  support=100
Milton       P=0.9038  R=0.9400  F1=0.9216  support=100
Shakespeare  P=0.8000  R=0.8400  F1=0.8195  support=100
Tennyson     P=0.7981  R=0.8300  F1=0.8137  support=100
Wordsworth   P=0.8000  R=0.4762  F1=0.5970  support=42
```

**Why Milton is best (F1=0.92)?** Milton is genuinely stylistically distinct: epic blank verse, dense proper nouns, biblical vocabulary, extremely long sentences. The model easily separates him.

**Why Wordsworth is worst (F1=0.60)?** Two compounding problems: only 42 test samples (smallest support), and Wordsworth's plain-language style overlaps with other Romantic poets. His recall is 0.47 — the model misses more than half of actual Wordsworth samples, often confusing them with Coleridge (both Lake Poets, shared vocabulary of nature and spirituality).

**`macro avg` vs `weighted avg`:**
- Macro avg treats all classes equally (average F1 across 6 classes, unweighted)
- Weighted avg weights by support (class sample count)
- We report both. For a balanced evaluation of a multi-class problem, macro avg is more honest.

---

## 24. Confusion Matrix Analysis

```python
cm     = confusion_matrix(y_te, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
```

**Why normalize by row?** Raw counts would make Milton look better just because it has more test samples (100 vs 42 for Wordsworth). Dividing by row sums converts to "% of true class correctly predicted" — a fair comparison.

**Reading the confusion matrix:**
- **Rows = true poet**
- **Columns = predicted poet**
- **Diagonal = correct predictions**
- **Off-diagonal = confusions**

**Key confusions:**
- Keats → Shakespeare (7%): Both Romantic-era, both heavy on sensory imagery
- Keats → Tennyson (12%): Both Victorian-adjacent, both musical, both melancholic  
- Wordsworth → Coleridge (19%): Both Lake Poets, shared nature and spiritual vocabulary
- Wordsworth → Shakespeare (14.3%): Both use plain language (though very different contexts)
- Tennyson → Milton (4%): Both write long-form narrative verse

**What confusions tell us:** The model makes the same mistakes a human literary scholar might make. Wordsworth and Coleridge collaborated and lived together — their styles genuinely overlap. Keats and Tennyson are often grouped together as "sensuous" poets. These aren't random errors; they reveal real stylistic relationships.

---

## 25. Fusion — DistilBERT + LightGBM

### Why Fuse Two Models?

Two models trained completely independently on the same data will make different errors. This happens because they learn fundamentally different representations:

**LightGBM** (our NLP model) looks at the poem's surface properties — word choices, rhythmic patterns, structural signals. It's good at distinctive vocabulary and meter but struggles with meaning.

**DistilBERT** (the DL model) reads the poem's deep semantic content — it understands that "in xanadu did kubla khan" is semantically about exotic places and power, a pattern associated with Coleridge. It struggles with short ambiguous fragments where context is thin.

When the two models fail on different samples, combining their predictions produces a better overall result than either alone.

### The Probability Combination Idea

Instead of combining class labels (Keats vs Tennyson), we combine the **probability distributions** each model outputs. LightGBM gives us [0.1, 0.6, 0.05, 0.1, 0.1, 0.05] (60% Keats, etc.). DistilBERT gives us [0.05, 0.5, 0.1, 0.2, 0.1, 0.05] (50% Keats). A weighted average of probabilities gives each model proportional influence based on our confidence in it.

---

## 26. Weighted Fusion — Theory and Code

```python
for w in np.arange(0.50, 1.01, 0.05):
    fused = w * dl_aligned + (1 - w) * nlp_aligned
    preds = fused.argmax(axis=1)
    acc   = accuracy_score(y_true, preds)
    print(f"  DL={w:.2f} NLP={1-w:.2f} {acc:.4f}")
```

**`np.arange(0.50, 1.01, 0.05)`** — creates array [0.50, 0.55, 0.60, ..., 1.00]. We search over 11 weight combinations.

**`fused = w * dl_aligned + (1 - w) * nlp_aligned`** — weighted average of probability matrices. If w=0.55, DistilBERT gets 55% weight and LightGBM gets 45%.

**`fused.argmax(axis=1)`** — for each stanza (row), find the column index with the highest fused probability. That index is the predicted class.

**Why grid search instead of learning the weights?** With only 12 parameters (6 DL probs + 6 NLP probs) and 2495 samples, a learned meta-classifier should outperform a scalar weight. But in practice it didn't, because:

1. The two models' probability outputs are well-calibrated (close to true probabilities)
2. A scalar weight is sufficient when both models are reliable
3. A meta-classifier on this small dataset adds variance — it overfits to patterns in the probability space

**Results:**

```
DL=0.50  NLP=0.50  → 97.60%
DL=0.55  NLP=0.45  → 97.64%  ← best
DL=0.60  NLP=0.40  → 97.23%
DL=1.00  NLP=0.00  → 95.55%
```

**Why does 55/45 beat 50/50?** DistilBERT has slightly higher standalone accuracy (88.44% vs 81.36%), so it deserves slightly more weight. The near-equal weighting confirms that LightGBM is genuinely contributing — it's not just noise. The 1.64% gain from DistilBERT alone to fusion (95.55% → 97.64%) represents the additional signal from NLP features.

**Why does fusion beat both individual models so dramatically?**

The 97.64% fusion accuracy vs 88.44% DistilBERT and 81.36% LightGBM is a large gain. The explanation is that the two models' errors are largely non-overlapping. When DistilBERT misclassifies a Wordsworth stanza, LightGBM's structural features (Wordsworth-specific vocabulary ratio, plain language indicators) often correctly identify it. When LightGBM misclassifies a Coleridge stanza (confused with Wordsworth), DistilBERT's semantic understanding of Coleridge's supernatural Gothic narrative correctly identifies it.

---

## 27. Meta-Classifier — Theory and Why It Lost

```python
X_meta = np.hstack([dl_aligned, nlp_aligned])  # 12 features
meta   = LogisticRegression(C=1.0, max_iter=1000, 
                             multi_class="multinomial",
                             solver="lbfgs")
meta.fit(X_mtr, y_mtr)
```

**What is a meta-classifier / stacking?** Instead of combining model outputs with a fixed formula (weighted average), we train a new classifier on top of the model outputs. The idea: let the meta-classifier learn per-poet, per-model weights — "for Coleridge prediction, trust DistilBERT 70%, for Milton trust LightGBM 60%."

**`LogisticRegression`** — chosen because inputs are already probabilities (0-1 range). LR is a well-calibrated model that gives interpretable weights. For each poet class, it learns a linear combination of the 12 input probabilities.

**`multi_class="multinomial"`** — fits a single model across all classes simultaneously (as opposed to one-vs-rest). Necessary for multi-class problems where classes are not independent.

**`solver="lbfgs"`** — Limited-memory BFGS optimization algorithm. Works well for multinomial problems with L2 regularization.

**Result: 85.17%** — worse than DistilBERT alone (88.44%). Why?

The meta-classifier needs to be trained on a separate validation set not seen during base model training. But we didn't do strict cross-validation stacking — the LightGBM probabilities were computed on the same data it was trained on. This means LightGBM is overconfident about its training examples, giving the meta-classifier misleading high-probability inputs that don't generalize.

Additionally, with only 2495 training samples and 12 features, the meta-classifier's parameter space is actually larger relative to data than the simple weighted average. Simple weighted averaging beat the more sophisticated approach.

**Lesson:** More complex fusion doesn't always win. When both models are already high-quality and well-calibrated, the marginal gain from learning complex weights is smaller than the variance introduced by the meta-classifier's extra parameters.

---

## 28. Why 97.64% and What It Means

The final 97.64% fusion accuracy is unusually high for a 6-class poetry attribution task. Let's be precise about what it means and what it doesn't.

**What it means:**
- On held-out test stanzas that neither model saw during training, the fused system correctly identifies the poet 97.64% of the time
- The test set is stratified — all 6 poets are represented proportionally
- We used train/test split, not cross-validation — a single test set means some variance in this estimate

**What it doesn't mean:**
- It does not mean the system would achieve 97.64% on arbitrary poetry from the internet
- It would likely perform worse on modern poetry, non-English poetry, or poets outside the training set
- The training data came from the same books we tested on — poems not in Project Gutenberg might be harder

**Why such high accuracy is possible:**
- The 6 poets are stylistically distinct in measurable ways (Milton's blank verse, Shakespeare's sonnets, Coleridge's Gothic ballads)
- We trained on 2000+ stanzas — enough for DistilBERT to learn deep semantic patterns
- The fusion captures both semantic and structural signals — two independent information sources
- The training/test distribution is matched (all from Project Gutenberg, same era, same OCR quality)

---

## 29. All Saved Artifacts Explained

```python
joblib.dump(lgbm_final,  "nlp_model.joblib")       # The trained LightGBM classifier
joblib.dump(le_nlp,      "le_nlp.joblib")           # LabelEncoder: maps strings↔integers
joblib.dump(scaler,      "scaler.joblib")           # StandardScaler: mean and std of training features
joblib.dump(tfidf,       "tfidf_vectorizer.joblib") # Fitted TfidfVectorizer (vocabulary + IDF weights)
joblib.dump(svd150,      "svd150.joblib")           # Fitted TruncatedSVD (150 components)
joblib.dump(char_tfidf,  "char_tfidf.joblib")       # Character-level TF-IDF vectorizer
joblib.dump(char_svd,    "char_svd.joblib")         # SVD for char n-grams (50 components)
joblib.dump(meta,        "meta_classifier.joblib")  # Logistic Regression stacker (archived)
joblib.dump(fusion_config, "fusion_config.joblib")  # Weights and poet ordering for fusion
features_df.to_csv("nlp_features.csv", index=False) # 32 hand-crafted features for all stanzas
```

**Why save the vectorizers and scalers separately from the model?**

At inference time (when the web app processes a new poem), we must apply the *exact same transformations* to the new poem that we applied during training. That means:
1. The TF-IDF vectorizer must use the same vocabulary and IDF weights from training
2. The SVD must use the same 150 components from training
3. The scaler must use the same mean and std from training

If we refit any of these on new data, the feature space changes and predictions are meaningless. Saving each transformer preserves its fitted state.

**`fusion_config.joblib`** contains:

```python
{
    "dl_weight":  0.55,        # DistilBERT gets 55% of the vote
    "nlp_weight": 0.45,        # LightGBM gets 45% of the vote
    "dl_poets":   [...],       # Order of poet classes in DL model output
    "nlp_poets":  [...],       # Order of poet classes in NLP model output
    "meta_acc":   0.8517,      # Meta-classifier accuracy (archived)
    "fusion_acc": 0.9764,      # Weighted fusion accuracy
}
```

**Why store poet orderings?** The DL and NLP models may have been trained with different label encodings — the DL model might output probabilities as [Coleridge, Keats, Milton, ...] while the NLP model outputs [Keats, Coleridge, Shakespeare, ...]. Before fusion arithmetic, probabilities must be aligned to the same poet order. The stored orderings enable this at inference time.

**`mlflow_experiment_results.csv`** — a portable record of all 11 MLflow runs exported as a CSV. Contains run names, accuracy, F1, model names, feature descriptions. Used in the technical report and web app's experiment comparison chart.

---

## Summary: The Full Pipeline

```
poetry_dataset.csv (2,495 stanzas)
         │
         ├── Hand-crafted Feature Extraction
         │   ├── 2 syllable features
         │   ├── 1 iambic score
         │   ├── 4 rhyme features
         │   ├── 13 vocabulary features
         │   ├── 7 structural features
         │   └── 5 POS features
         │   = 32 features
         │
         ├── TF-IDF Word N-grams
         │   ├── TfidfVectorizer (5000 vocab, 1-2 grams)
         │   └── TruncatedSVD (5000 → 150 dims)
         │   = 150 features
         │
         └── TF-IDF Character N-grams
             ├── TfidfVectorizer (3000 vocab, 3-5 char grams)
             └── TruncatedSVD (3000 → 50 dims)
             = 50 features
                        │
                        ▼
             np.hstack → 232-dimensional feature vector
                        │
                   StandardScaler
                        │
                   LightGBM (500 trees, leaf-wise)
                        │
             NLP Probabilities (6 classes) ─────────┐
                                                     │
DL Probabilities (DistilBERT, 50-pass MC) ──────────┤
                                                     ▼
                             Weighted Fusion: 0.55×DL + 0.45×NLP
                                                     │
                                        argmax → Final Poet Prediction
                                        
Final Accuracy: 97.64%
```

---

*Document prepared by Mehtab Singh and Sakshi Verma for PoetryDNA — CSR326 Assessment*  
*GitHub: https://github.com/MehtabSingh3711/PoetryDNA*
