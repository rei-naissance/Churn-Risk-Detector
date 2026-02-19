"""
External Dataset Loader for Logistics Churn Risk Model
=======================================================
Downloads and preprocesses complaint datasets from Hugging Face, then merges
them with the hand-curated training data in ``churn_model.py``.

Sources
-------
1. **hblim/customer-complaints** — 1 682 short, labelled complaints in three
   categories (billing, delivery, product).  We keep *all* ``delivery``
   complaints as churn-positive, and *all* ``billing``/``product`` as
   churn-negative (they describe product or account issues, not logistics
   failure).

2. **aciborowska/customers-complaints** — 30 000 CFPB consumer complaint
   narratives.  Most are finance-related (credit cards, loans), but a handful
   mention shipping, delivery, or logistics themes.  We keyword-filter the
   narrative text and label matches as churn-positive; a random subsample of
   non-matching rows becomes churn-negative.

The combined dataset is cached to ``data/external_complaints.npz`` so repeated
runs skip the download.

Author : Senior AI Engineer
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Directory that holds the local cache file.
_CACHE_DIR: Path = Path(__file__).resolve().parent / "data"
_CACHE_FILE: Path = _CACHE_DIR / "external_complaints.npz"

# Keywords used to identify logistics-relevant complaints inside the CFPB
# dataset.  Any narrative that contains *at least one* of these tokens (case-
# insensitive) is labelled as a logistics complaint and therefore churn = 1.
_LOGISTICS_FILTER_KEYWORDS: list[str] = [
    "shipping",
    "delivery",
    "delivered",
    "courier",
    "package",
    "parcel",
    "shipment",
    "freight",
    "carrier",
    "tracking",
    "in transit",
    "lost package",
    "damaged package",
    "late delivery",
    "logistics",
    "warehouse",
    "dispatch",
    "postage",
    "usps",
    "fedex",
    "ups",
    "dhl",
]

# Compile one big alternation pattern for speed.
_LOGISTICS_RE: re.Pattern[str] = re.compile(
    "|".join(re.escape(kw) for kw in _LOGISTICS_FILTER_KEYWORDS),
    flags=re.IGNORECASE,
)

# Maximum number of *negative* (non-logistics) samples to keep from the CFPB
# dataset so we don't swamp the corpus with unrelated finance text.
_CFPB_NEG_CAP: int = 200

# Maximum text length (chars) — very long CFPB narratives are trimmed.
_MAX_TEXT_LEN: int = 500


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    """Trim *text* to *max_len* characters on a word boundary."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + " …"


def _load_hblim() -> tuple[list[str], list[int]]:
    """
    Load the ``hblim/customer-complaints`` dataset from Hugging Face.

    Returns
    -------
    texts : list[str]
    labels : list[int]  (1 = delivery = churn, 0 = billing/product = retained)
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Downloading hblim/customer-complaints …")
    ds = load_dataset("hblim/customer-complaints", split="train")

    texts: list[str] = []
    labels: list[int] = []

    # label mapping: 0=billing, 1=delivery, 2=product (ClassLabel order)
    label_names = ds.features["label"].names  # type: ignore[union-attr]
    delivery_idx = label_names.index("delivery")

    for row in ds:
        txt: str = row["text"].strip()  # type: ignore[index]
        if not txt:
            continue
        lbl: int = 1 if row["label"] == delivery_idx else 0  # type: ignore[index]
        texts.append(txt)
        labels.append(lbl)

    logger.info(
        "hblim: loaded %d rows (%d delivery / %d other)",
        len(texts),
        sum(labels),
        len(labels) - sum(labels),
    )
    return texts, labels


def _load_cfpb() -> tuple[list[str], list[int]]:
    """
    Load the ``aciborowska/customers-complaints`` dataset (CFPB) from
    Hugging Face, keyword-filter for logistics themes, and return balanced
    positive + negative samples.

    Returns
    -------
    texts : list[str]
    labels : list[int]  (1 = logistics match = churn, 0 = non-match = retained)
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Downloading aciborowska/customers-complaints …")
    ds = load_dataset("aciborowska/customers-complaints", split="train")

    # The CFPB dataset uses a column called "Consumer complaint narrative"
    # (or similar).  Identify the right text column.
    text_col: str | None = None
    for candidate in (
        "Consumer complaint narrative",
        "consumer_complaint_narrative",
        "complaint_narrative",
        "narrative",
        "text",
    ):
        if candidate in ds.column_names:
            text_col = candidate
            break

    if text_col is None:
        # Fallback: pick the first column whose sample value is a long string.
        for col in ds.column_names:
            sample = ds[0][col]  # type: ignore[index]
            if isinstance(sample, str) and len(sample) > 100:
                text_col = col
                break

    if text_col is None:
        logger.warning("Could not identify text column in CFPB dataset. Skipping.")
        return [], []

    logger.info("CFPB text column identified as '%s'", text_col)

    pos_texts: list[str] = []
    neg_texts: list[str] = []

    for row in ds:
        raw: Any = row[text_col]  # type: ignore[index]
        if not isinstance(raw, str) or len(raw.strip()) < 20:
            continue
        txt = _truncate(raw.strip())

        if _LOGISTICS_RE.search(txt):
            pos_texts.append(txt)
        else:
            neg_texts.append(txt)

    # Cap negative samples so logistics-relevant texts aren't drowned out.
    rng = np.random.default_rng(42)
    if len(neg_texts) > _CFPB_NEG_CAP:
        idx = rng.choice(len(neg_texts), _CFPB_NEG_CAP, replace=False)
        neg_texts = [neg_texts[i] for i in sorted(idx)]

    texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    logger.info(
        "CFPB: kept %d logistics-positive + %d negative = %d total",
        len(pos_texts),
        len(neg_texts),
        len(texts),
    )
    return texts, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_external_data(
    *, use_cache: bool = True
) -> tuple[list[str], list[int]]:
    """
    Download, preprocess, and return external complaint data.

    The result is cached to disk (``data/external_complaints.npz``) so
    subsequent calls are near-instant.

    Parameters
    ----------
    use_cache : bool
        If *True* (default), return the cached copy when available.

    Returns
    -------
    texts : list[str]
        Complaint / feedback strings.
    labels : list[int]
        Binary labels (1 = churn / logistics issue, 0 = retained).
    """
    # --- Try cache first ---------------------------------------------------
    if use_cache and _CACHE_FILE.exists():
        logger.info("Loading cached external data from %s", _CACHE_FILE)
        data = np.load(_CACHE_FILE, allow_pickle=True)
        return data["texts"].tolist(), data["labels"].tolist()

    # --- Download & process ------------------------------------------------
    all_texts: list[str] = []
    all_labels: list[int] = []

    try:
        t, l = _load_hblim()
        all_texts.extend(t)
        all_labels.extend(l)
    except Exception:
        logger.exception("Failed to load hblim dataset — skipping.")

    try:
        t, l = _load_cfpb()
        all_texts.extend(t)
        all_labels.extend(l)
    except Exception:
        logger.exception("Failed to load CFPB dataset — skipping.")

    if not all_texts:
        logger.warning("No external data loaded.")
        return [], []

    # --- Persist cache -----------------------------------------------------
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _CACHE_FILE,
        texts=np.array(all_texts, dtype=object),
        labels=np.array(all_labels, dtype=np.int32),
    )
    logger.info(
        "External data cached → %s  (%d samples)", _CACHE_FILE, len(all_texts)
    )
    return all_texts, all_labels


# ---------------------------------------------------------------------------
# Quick diagnostic when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    texts, labels = load_external_data(use_cache=False)
    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"\nTotal external samples : {len(texts)}")
    print(f"  churn (1)            : {n1}")
    print(f"  retained (0)         : {n0}")
    print(f"\nSample churn texts:")
    for t in [t for t, l in zip(texts, labels) if l == 1][:5]:
        print(f"  • {t[:120]}")
    print(f"\nSample retained texts:")
    for t in [t for t, l in zip(texts, labels) if l == 0][:5]:
        print(f"  • {t[:120]}")
