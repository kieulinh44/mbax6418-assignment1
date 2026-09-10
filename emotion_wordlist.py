"""
Step 5 deliverable -- NRC word-list emotion scorer (no model calls).

Scores a review's words against the NRC Emotion Lexicon (word -> set of emotion
labels) and takes the label with the highest score as the primary emotion.

The lexicon is the NRC Word-Emotion Association Lexicon (EmoLex), which the
NRClex package bundles (nrclex/data/nrc_en.json). We read that bundled file so
no download or nltk corpus is required at runtime -- reproducible and offline.

Only the 8 assignment emotions are considered (anger, anticipation, disgust,
fear, joy, sadness, surprise, trust); the lexicon's own positive/negative tags
are ignored here so the "primary emotion" is chosen purely among the 8.
Expect quite a lot of "no match" (None) cases on short/terse reviews: EmoLex is
word-level, so slang, names, and sparse text often produce no emotion hits.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

from prompt import EMOTIONS

# Canonical display order used for ties and consistent column ordering.
_EMOTION_SET = frozenset(EMOTIONS)
_TOKEN_RE = re.compile(r"[a-z']+", re.UNICODE)

# Light suffix lemmatizer. The lexicon stores base forms ("happy", "love",
# "run"), so we also probe common inflected endings before giving up on a word.
_SUFFIXES = ("ies", "es", "ing", "ed", "s")


@lru_cache(maxsize=1)
def _lexicon() -> dict:
    """word -> list of emotion labels (from the NRC lexicon bundled with NRClex)."""
    try:
        f = resources.files("nrclex.data").joinpath("nrc_en.json")
    except Exception:
        f = resources.files("nrclex").joinpath("data", "nrc_en.json")
    with f.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _candidate_forms(word: str):
    """Yield the word plus light stemming candidates (deduped, original first)."""
    yield word
    lower = word.lower()
    seen = {word}
    for suf, rep in (("ies", "y"), ("es", ""), ("s", ""), ("ing", ""), ("ed", "")):
        if len(lower) > len(suf) + 1 and lower.endswith(suf):
            cand = lower[: -len(suf)] + rep
            if cand not in seen:
                seen.add(cand)
                yield cand


def score_text(text: str) -> tuple:
    """
    Score a review's words against the NRC lexicon.

    Returns: (primary_emotion, per_emotion_counts, n_matched_words, matched_words)
      - primary_emotion: str or None (None when no word in the review matched any
        of the 8 emotions -- the review may still be clearly positive/negative
        without hitting a lexicon emotion word).
      *per_emotion_counts is a dict over the 8 emotions (count of word->emotion
       associations; a single word can carry several emotion tags).
    """
    if not text:
        return None, {e: 0 for e in EMOTIONS}, 0, {}

    lex = _lexicon()
    counts = {e: 0 for e in EMOTIONS}
    matched: dict[str, int] = {}

    for token in _TOKEN_RE.findall(text.lower()):
        form = None
        for cand in _candidate_forms(token):
            if cand in lex:
                form = cand
                break
        if form is None:
            continue
        for emo in lex[form]:
            if emo in _EMOTION_SET:
                counts[emo] += 1
        matched[form] = matched.get(form, 0) + 1

    n_matched = len(matched)
    best = max(counts.values())
    primary = next((e for e in EMOTIONS if counts[e] == best), None) if best else None
    return primary, counts, n_matched, matched


def overview(records) -> dict:
    """Quick cross-check stats across already-classified records (each record has
    emotion_wl populated)."""
    from collections import Counter
    dist = Counter(r.get("emotion_wl") for r in records)
    return {"distribution": dict(dist), "n": len(records)}
