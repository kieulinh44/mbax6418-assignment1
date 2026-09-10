"""
Step 1 deliverable -- the reusable prompt.

Takes a review's title and text and asks the LLM to return, as parseable JSON:
  - the OVERALL sentiment: one of POSITIVE | NEUTRAL | NEGATIVE (three-class)
    or POSITIVE | NEGATIVE (binary), and
  - the PRIMARY emotion: one of the eight NRC labels.

The prompt is rating-blind by design -- the model is told it will never receive
a star rating, so it must judge the words alone.
"""

# The 8 NRC emotion labels the assignment names.
EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def system_prompt(three_class: bool = True) -> str:
    """Return the system prompt. three_class=True -> POSITIVE/NEUTRAL/NEGATIVE;
    three_class=False -> POSITIVE/NEGATIVE only."""
    if three_class:
        sentiment_classes = "POSITIVE, NEUTRAL, NEGATIVE"
        note = (
            "Use NEUTRAL only when the review is genuinely mixed/indifferent and not "
            "clearly for or against; otherwise commit to POSITIVE or NEGATIVE."
        )
    else:
        sentiment_classes = "POSITIVE, NEGATIVE"
        note = (
            "There is no 'neutral' option. If the review is not clearly positive, "
            "choose NEGATIVE."
        )

    return (
        "You are a sentiment and emotion classifier for written product reviews.\n"
        "Read the review's title and body and judge ONLY the words you are given. "
        "You will NOT receive a star rating, and you must never assume one "
        "(positive reviews, neutral reviews, and negative reviews can all exist "
        "here).\n"
        f"\n"
        f"1) Sentiment -- label the OVERALL sentiment as exactly one of: "
        f"{sentiment_classes}. {note}\n"
        f"2) Emotion  -- pick the SINGLE primary emotion the reviewer most strongly "
        f"expresses, from exactly one of: {', '.join(EMOTIONS)}.\n"
        f"\n"
        "Reply with ONLY a single JSON object and no other text, in exactly this "
        'shape:\n{"sentiment": "POSITIVE", "emotion": "joy"}'
    )


def build_user_prompt(title: str, text: str, max_text: int = 600) -> str:
    """One-turn user message for a single review. Title is often empty or terse,
    so the body is the main evidence; both are included when present."""
    text = (text or "").strip()
    title = (title or "").strip()
    if max_text and len(text) > max_text:
        text = text[:max_text] + "…"
    if title:
        return f"TITLE: {title}\nREVIEW: {text}"
    return f"REVIEW: {text}"
