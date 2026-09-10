"""
Steps 1, 2, 5, 6 -- scoring pipeline for the Amazon 'Gift Cards' review data.

Subcommands
-----------
  python classify.py step2
      First 100 rows in file order, BINARY sentiment vs the rating
      (answer = POSITIVE if rating>=4 else NEGATIVE), LLM emotions +
      NRC word-list emotions. Demonstrates the lopsided 5-star skew.

  python classify.py step6
      CLASS-BALANCED sample, ~50 samples per class (POSITIVE = 4-5*,
      NEUTRAL = 3*, NEGATIVE = 1-2*), fixed random seed, three-class
      sentiment + emotions. This is the main "balanced run".

Flags:  --max-text N   truncate the review body sent to the LLM (default 600)
        --seed N       override the fixed random seed (default 42)
        --dry-run      run the pipeline without calling the API
                       (predictions are mocked to ground truth / 'joy')

The endpoint, model and API key are discovered from this machine's Hermes
config (cmd: model + custom_providers) so the classifier talks to the same
OpenAI-compatible endpoint Hermes uses. The model NEVER sees the rating.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from openai import OpenAI

from prompt import EMOTIONS, build_user_prompt, system_prompt
from emotion_wordlist import score_text

DATA_PATH = r"C:/Users/kieul/AppData/Local/hermes/attachments/Gift_Cards.jsonl (1).gz"
DEFAULT_MODEL = "DeepSeek-V4-Flash-0731"
DEFAULT_BASE_URL = "http://dobolyi.com:9000/v1"
OUT_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Endpoint discovery (mirrors Hermes config so we talk to the same LLM)
# --------------------------------------------------------------------------- #
def _hermes_config():
    cands = []
    if os.environ.get("HERMES_HOME"):
        cands.append(Path(os.environ["HERMES_HOME"]) / "config.yaml")
    cands.append(Path.home() / ".hermes" / "config.yaml")
    if sys.platform.startswith("win"):
        cands.append(Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "config.yaml")
    for c in cands:
        if c.exists():
            try:
                import yaml
                return yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            except Exception:
                return json.loads(c.read_text(encoding="utf-8"))
    return {}


def resolve_endpoint():
    cfg = _hermes_config()
    ms = cfg.get("model", {}) or {}
    key = os.environ.get("OPENAI_API_KEY") or (
        (cfg.get("custom_providers") or [{}])[0].get("api_key")
        or ms.get("api_key")
    )
    base_url = os.environ.get("OPENAI_BASE_URL") or ms.get("base_url") or DEFAULT_BASE_URL
    model = os.environ.get("LLM_MODEL") or ms.get("default") or DEFAULT_MODEL
    return key, base_url, model


# --------------------------------------------------------------------------- #
# Data + sampling
# --------------------------------------------------------------------------- #
def load_reviews(path=DATA_PATH):
    with gzip_open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield {
                "title": (r.get("title") or "").strip(),
                "text": (r.get("text") or "").strip(),
                "rating": r.get("rating"),
                "verified_purchase": bool(r.get("verified_purchase")),
                "helpful_vote": r.get("helpful_vote", 0),
                "timestamp": r.get("timestamp"),
                "asin": r.get("asin"),
            }


def gzip_open(path):
    import gzip
    return gzip.open(path, "rt", encoding="utf-8")


def answer_from_rating(rating, three_class):
    if three_class:
        if rating >= 4:
            return "POSITIVE"
        if rating == 3:
            return "NEUTRAL"
        return "NEGATIVE"
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def balanced_sample(reviews, per_class, three_class, seed):
    """~per_class from each answer class, fixed seed -> reproducible."""
    rng = random.Random(seed)
    buckets = {}
    for r in reviews:
        buckets.setdefault(answer_from_rating(r["rating"], three_class), []).append(r)
    out = []
    for cls in (["POSITIVE", "NEUTRAL", "NEGATIVE"] if three_class else ["POSITIVE", "NEGATIVE"]):
        pool = buckets.get(cls, [])
        take = min(per_class, len(pool))
        out += rng.sample(pool, take)
        print(f"    {cls:9s}: {take} sampled ({len(pool)} available)")
    rng_assign = random.Random(seed)
    rng_assign.shuffle(out)
    return out


# --------------------------------------------------------------------------- #
# LLM call + parse
# --------------------------------------------------------------------------- #
def _extract_json_object(s):
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return s[i : j + 1]


def parse_response(content, three_class):
    """Parse sentiment + primary emotion out of the model reply."""
    data = None
    frag = _extract_json_object(content)
    if frag:
        try:
            data = json.loads(frag)
        except Exception:
            data = None
    # Fallback: siblings without braces.
    low = (content or "").lower()
    if data is None:
        def has(x):
            return f'"{x}"' in low or "'" + x + "'" in low
        if has("positive") or "positive" in low:
            return "POSITIVE", None
        if has("negative") or "negative" in low:
            return "NEGATIVE", None
        return None, None

    sentiment = str(data.get("sentiment") or "").strip().upper()
    if sentiment in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        if not three_class and sentiment == "NEUTRAL":
            sentiment = "NEGATIVE"  # binary mode has no neutral bucket
    else:
        sentiment = None

    emotion = str(data.get("emotion") or "").strip().lower()
    if emotion not in EMOTIONS:
        emotion = None
    return sentiment, emotion


def classify_one(client, model, system_p, user_p, three_class, retries=3):
    last = (None, None)
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user", "content": user_p},
                ],
                temperature=0,
                max_tokens=512,  # this reasoning model spends tokens reasoning first
            )
            content = (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                print(f"    [API error] {type(e).__name__}: {e}", file=sys.stderr)
                return None, None
            continue
        sent, emo = parse_response(content, three_class)
        if sent is None:
            last = (sent, emo)
            continue  # unparseable -> retry once more
        return sent, emo
    return last


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def confusion_matrix(answers, predictions, classes):
    m = {a: {b: 0 for b in classes} for a in classes}
    for a, p in zip(answers, predictions):
        if p is None:      # unparseable prediction: excluded from the matrix
            continue
        m[a][p] += 1
    return m


def report(records, classes, three_class, title):
    """Print a readable scoring summary and return summary dict. A record has
    answer, sentiment (prediction), correct. Rows with sentiment=None (API/parse
    failures) are excluded from scoring but their count is reported."""
    scored = [r for r in records if r.get("sentiment") is not None]
    skipped = len(records) - len(scored)
    correct = sum(1 for r in scored if r["correct"])
    total = len(scored)
    acc = correct / total if total else 0.0
    mat = confusion_matrix(
        [r["answer"] for r in scored],
        [r["sentiment"] for r in scored],
        classes,
    )
    per_class = {}
    for c in classes:
        grp = [r for r in scored if r["answer"] == c]
        ok = sum(1 for r in grp if r["sentiment"] == c)
        per_class[c] = {"n": len(grp), "correct": ok,
                        "accuracy": ok / len(grp) if grp else 0.0}
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"reviews attempted   : {len(records)}   (skipped/unparseable: {skipped})")
    print(f"reviews scored      : {total}")
    print(f"overall accuracy    : {acc:.3f} ({correct}/{total})")
    print("per-class (answer => how often the model picked the right class):")
    for c in classes:
        p = per_class[c]
        print(f"    {c:9s} n={p['n']:3d}  correct={p['correct']:3d}  class-acc={p['accuracy']:.3f}")
    print("\nconfusion matrix  (rows=correct answer, cols=model prediction):")
    print(f"            " + "".join(f"{c[:4]:>8}" for c in classes))
    for a in classes:
        print(f"  {a[:9]:<9}" + "".join(f"{mat[a][b]:>8}" for b in classes))
    print("\nprimary emotion — LLM vs NRC word list:")
    ea = [r for r in scored if r["emotion_llm"] and r["emotion_wl"]]
    agree = sum(1 for r in ea if r["emotion_llm"] == r["emotion_wl"])
    print(f"    records with both : {len(ea)}")
    print(f"    agree             : {agree}  ({(agree/len(ea)*100) if ea else 0:.1f}%)")

    return {
        "title": title, "attempted": len(records), "skipped": skipped,
        "n": total, "accuracy": acc,
        "per_class": per_class, "confusion": mat,
        "emotion_n": len(ea), "emotion_agree": agree,
        "emotion_agree_rate": (agree / len(ea)) if ea else None,
    }


def write_output(records, path):
    out_dir = Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            # drop heavy raw-match internals not needed downstream
            slim = {k: r.get(k) for k in (
                "idx", "asin", "title", "text", "rating", "verified_purchase",
                "helpful_vote", "timestamp", "answer", "sentiment", "correct",
                "emotion_llm", "emotion_wl", "emotion_counts",
            )}
            fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} records -> {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=["step2", "step6"])
    ap.add_argument("--max-text", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    three_class = args.which == "step6"
    classes = (["POSITIVE", "NEUTRAL", "NEGATIVE"] if three_class
               else ["POSITIVE", "NEGATIVE"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (OUT_DIR / ("step6_balanced.jsonl" if three_class else "step2_first100.jsonl"))

    print(f"Loading reviews from {DATA_PATH}")
    reviews = list(load_reviews())
    print(f"  {len(reviews)} reviews loaded.")

    if three_class:
        sample = balanced_sample(reviews, per_class=50, three_class=True, seed=args.seed)
        title = "STEP 6 -- balanced three-class run (seed=%d)" % args.seed
    else:
        sample = list(reviews)[:100]
        title = "STEP 2 -- first 100 rows in file order (binary)"
        print(f"  using first {len(sample)} rows (mostly 5-star, deliberately lopsided)")

    # Catalog
    for i, r in enumerate(sample):
        r["idx"] = i
        r["answer"] = answer_from_rating(r["rating"], three_class)

    system_p = system_prompt(three_class=three_class)

    if args.dry_run:
        print("\n[DRY RUN] mocking LLM as ground truth + emotion='joy' to test the path.")
        for r in sample:
            r["sentiment"] = r["answer"]
            r["emotion_llm"] = "joy"
    else:
        key, base_url, model = resolve_endpoint()
        if not key:
            sys.exit("No API key found (Hermes config or OPENAI_API_KEY) -- use --dry-run to test.")
        print(f"\nEndpoint : {base_url}   Model: {model}   Key: {'set' if key else 'MISSING'}")
        client = OpenAI(api_key=key, base_url=base_url)
        n = len(sample)
        for i, r in enumerate(sample, 1):
            user_p = build_user_prompt(r["title"], r["text"], args.max_text)
            sent, emo = classify_one(client, model, system_p, user_p, three_class)
            r["sentiment"] = sent
            r["emotion_llm"] = emo
            print(f"  [{i}/{n}] truth={r['answer']}  pred={sent}  emo={emo}")

    # Word list emotions (always, no model calls) + correctness
    for r in sample:
        primary, counts, nmatch, _ = score_text(r["title"] + " " + r["text"])
        r["emotion_wl"] = primary
        r["emotion_counts"] = counts
        r["correct"] = (r["sentiment"] == r["answer"]) if r["sentiment"] else False

    summary = report(sample, classes, three_class, title)
    write_output(sample, out_path)

    # persist summary JSON for the dashboard + README
    sum_path = OUT_DIR / ("step6_summary.json" if three_class else "step2_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  summary -> {sum_path}")


if __name__ == "__main__":
    main()
