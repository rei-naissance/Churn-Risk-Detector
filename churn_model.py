"""
Logistics-Specific Churn Risk Model
====================================
A scikit-learn Pipeline that predicts customer churn risk from free-text
shipping complaints.  The pipeline uses TF-IDF (uni+bigrams) fed into a
Random Forest, then applies a *post-processing* boost whenever the text
contains high-priority logistics keywords (e.g. "lost", "refund", "damaged").

Training data comes from *two* sources:

1. **Hand-curated corpus** — 120 short complaint / praise sentences inspired
   by Trustpilot reviews of FedEx, UPS, and DHL.
2. **External datasets** (downloaded on first run via ``data_loader.py``):
   - *hblim/customer-complaints* — 1 260 labelled delivery / billing /
     product complaints from Hugging Face.
   - *aciborowska/customers-complaints* — CFPB narratives keyword-filtered
     for logistics themes.

The merged dataset (~2 100 samples) gives the Random Forest far more signal
than the original 120-sample set alone.

Author : Senior AI Engineer
Created: 2026-02-20
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data_loader import load_external_data

# ---------------------------------------------------------------------------
# Logging — library code must NOT call logging.basicConfig().
# The calling application (app.py, CLI scripts, test suite) is responsible
# for configuring the root logger to the desired level and format.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
__all__ = [
    # Data constants
    "COMPLAINTS",
    "LABELS",
    "HIGH_PRIORITY_KEYWORDS",
    # Tuning / threshold constants
    "KEYWORD_BOOST",
    "RISK_THRESHOLD_CRITICAL",
    "RISK_THRESHOLD_HIGH",
    "RISK_THRESHOLD_MEDIUM",
    "INPUT_MAX_CHARS",
    # Functions
    "build_pipeline",
    "get_training_data",
    "train_pipeline",
    "detect_keywords",
    "apply_keyword_boost",
    "get_pipeline",
    "analyze_complaint",
]

# ---------------------------------------------------------------------------
# 1. Hand-Curated Seed Data (120 samples)
# ---------------------------------------------------------------------------
# These 120 sentences are inspired by *real* Trustpilot reviews of FedEx, UPS,
# and DHL.  They act as a high-quality logistics-specific seed that is later
# merged with thousands of external complaint samples (see §4).
# Label 1 = churned customer, 0 = retained customer.

COMPLAINTS: list[str] = [
    # =================================================================
    # POSITIVE / NEUTRAL  (label 0)  — 60 samples
    # =================================================================
    # -- On-time delivery --
    "The driver was great and very polite",
    "Package arrived on time, well packed",
    "Friendly staff, smooth delivery experience",
    "Got my order a day early, very happy",
    "Customer support resolved my query quickly",
    "Delivery was on schedule, no issues at all",
    "Everything arrived in perfect condition",
    "Quick shipping and the box was intact",
    "Impressed with the real-time tracking updates",
    "The courier was professional and courteous",
    "My parcel arrived ahead of schedule, great service",
    "Smooth delivery, no complaints whatsoever",
    "Excellent packaging, nothing was broken",
    "Helpful support team, issue resolved in minutes",
    "Fast shipping and the item was well protected",
    # -- Praise for communication & tracking --
    "Tracking was accurate down to the minute, really appreciated it",
    "Got an SMS notification right before the driver arrived, very convenient",
    "The live map showed exactly where my package was, awesome feature",
    "Email updates at every hub were clear and timely",
    "Estimated delivery window was spot on, arrived within the first hour",
    "Customer service replied within ten minutes on the chat, super helpful",
    "The support agent walked me through the customs forms patiently",
    "Called the hotline once and my issue was solved on the first try",
    "The chatbot actually understood my question and gave a useful answer",
    "Proactive delay notification saved me a wasted trip home",
    # -- Driver & staff compliments --
    "Our regular driver always rings the bell and waits, much appreciated",
    "The courier carried the heavy box all the way to my third-floor apartment",
    "He went out of his way to deliver during a snowstorm, incredible dedication",
    "Delivery person was friendly and even helped me carry in the furniture",
    "The warehouse staff double-wrapped my fragile vase, arrived perfect",
    "Driver left the package in the covered porch as requested, great attention",
    "Courier followed my safe-place instructions exactly, no issues",
    "Night-shift team processed my urgent shipment on time, thank you",
    "Polite driver asked for a signature and wished me a good day",
    "Staff at the drop-off center were quick, professional, and kind",
    # -- General satisfaction --
    "Two-day ground shipping actually arrived in one day, pleasantly surprised",
    "International shipment cleared customs in under 24 hours, impressive",
    "Returned an item and the prepaid label process was seamless",
    "Free packaging materials at the store saved me a trip, great perk",
    "Express shipping was worth the cost, everything arrived next morning",
    "Consistent reliable service, I have shipped fifty parcels without a problem",
    "Good value for the price, no hidden fees or surprise charges",
    "Scheduled pickup was on time and the driver scanned everything properly",
    "All five boxes in my bulk shipment arrived together on the same day",
    "Fragile electronics arrived bubble-wrapped and undamaged, great care",
    "Temperature-controlled shipping kept my chocolate from melting, perfect",
    "Saturday delivery option is a lifesaver for working professionals",
    "Easy online booking, label printed in seconds, parcel collected same day",
    "The app let me redirect to a neighbor while the driver was en route, handy",
    "Claims process was straightforward and I was reimbursed within a week",
    "Overnight priority actually arrived by 9 AM as promised, reliable",
    "Bulk discount for my small business shipments saved us a lot, grateful",
    "Same driver for six months, knows my building code, zero missed deliveries",
    "Holiday rush season and they still delivered on the exact promised date",
    "Packaging was so good that even the glass candle holders were intact",
    "Their eco-friendly packaging option is great, less waste and still sturdy",
    "Return pickup was scheduled and collected from my door, no hassle at all",
    "The proof-of-delivery photo showed exactly where the parcel was placed",
    "Cross-border shipment had no surprise duties, pricing was transparent",
    "Contacted support about a delay and they upgraded me to express for free",

    # =================================================================
    # NEGATIVE — CHURN RISK  (label 1)  — 60 samples
    # =================================================================
    # -- Late / delayed delivery (sourced themes: FedEx, UPS, DHL reviews) --
    "My package is 4 days late and nobody responds",
    "Paid for overnight shipping and it took five days to arrive",
    "Express delivery due yesterday, postponed until today, still not here",
    "Estimated delivery was Monday, it is now Friday and still waiting",
    "Three different delivery dates were given and none were honored",
    "Package sat at the local hub eight miles away for five days straight",
    "Waited in all day for a delivery that never showed up",
    "Delivery window was 10 AM to 2 PM but the driver never came",
    "Parcel has been out for delivery for three consecutive days now",
    "Paid extra for expedited shipping and it is still stuck in transit",
    "Shipment has not moved from the regional sort facility in a week",
    "They keep updating the estimated date but the package never arrives",
    # -- Lost / missing packages --
    "Shipment lost in transit, very frustrated",
    "Package marked as delivered but I never received it",
    "Tracking says delivered, photo shows a house that is not mine",
    "My parcel vanished after the last scan at the distribution center",
    "They lost three packages in the last two weeks alone",
    "Lost a package and spent three months trying to get the claim paid",
    "Parcel was supposed to arrive a week ago, now listed as lost",
    "No scan updates for ten days, the package has simply disappeared",
    "Filed a lost-package claim and it was denied without explanation",
    "UPS lost my package sitting in their facility for seven days",
    # -- Damaged goods --
    "Friendly staff, but my box was crushed on arrival",
    "I want a refund, the product arrived damaged",
    "The box was completely torn open on arrival",
    "Received my order with a cracked screen, packaging was flimsy",
    "Half the package was ripped apart and items were taped to the outside",
    "My fragile artwork case arrived creased with the corner bent",
    "Driver smashed the box onto the ground, dented the appliance inside",
    "Delivery left in the rain, electronics inside were water-damaged",
    "Boxes were soaked and crumbling when they arrived after sitting outside",
    "Contents shattered because there was zero bubble wrap in the box",
    # -- Stolen / porch piracy --
    "Package was stolen from my porch, no help from support",
    "Driver left parcel outside with no notice and it was taken",
    "Package left on the doorstep in plain sight, got stolen overnight",
    "Signature was required but the driver left it without one, now gone",
    # -- Wrong delivery / misdelivery --
    "Received wrong item, need a replacement immediately",
    "Package delivered to the wrong address across town",
    "Delivered to a neighbor three houses down, no notification",
    "Driver delivered to an apartment instead of the business on the label",
    "They delivered someone else's package to me and mine is nowhere",
    # -- Refund / billing issues --
    "Damaged goods, broken seal, I want my money back",
    "Charged twice and still waiting for the shipment",
    "My order was cancelled without any notice",
    "Filed for a refund weeks ago and have heard nothing back",
    "Hidden customs fees added 50 percent to the cost with no warning",
    "Brokerage surcharge was more than the item itself, outrageous",
    "Insurance claim was denied even though the box was clearly crushed",
    # -- Customer service failures --
    "Terrible service, my fragile item was shattered",
    "Worst experience ever, package lost and no refund",
    "Not delivered after two weeks, unacceptable",
    "Called five times and each agent gave me different information",
    "Impossible to reach a human, the automated system is useless",
    "Customer service hung up on me after thirty minutes on hold",
    "Support agent told me the driver has the right not to deliver",
    "Chat support took six days and over a hundred messages to respond",
    "No phone number available, only a chatbot that loops endlessly",
    "Escalated to a supervisor who never called me back as promised",
    "Absolute shambles, nobody at this company takes accountability",
    "Will never use this carrier again, switching to a competitor",
]

LABELS: list[int] = [
    # 60 positive/neutral → 0
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 60 negative → 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]

# ---------------------------------------------------------------------------
# 2. High-Priority Logistics Keywords
# ---------------------------------------------------------------------------
# If the complaint text contains any of these tokens the final churn risk
# score is *boosted* because these words signal high-severity issues that
# strongly correlate with customer churn in the logistics domain.

HIGH_PRIORITY_KEYWORDS: list[str] = [
    # --- Severity: Critical (package gone or unusable) ---
    "lost",
    "stolen",
    "missing",
    "disappeared",
    "vanished",
    # --- Severity: High (physical damage) ---
    "damaged",
    "broken",
    "shattered",
    "destroyed",
    "crushed",
    "cracked",
    "torn open",
    "ripped",
    "dented",
    "water-damaged",
    "smashed",
    # --- Severity: High (financial / refund) ---
    "refund",
    "money back",
    "reimbursed",
    "hidden fees",
    "overcharged",
    "charged twice",
    "insurance claim",
    # --- Severity: High (service failure) ---
    "not delivered",
    "wrong item",
    "wrong address",
    "misdelivered",
    "cancelled",
    "never arrived",
    "never received",
    "never showed",
    # --- Severity: Medium-High (support failure) ---
    "no response",
    "hung up",
    "impossible to reach",
    "never called back",
    "never use again",
    "worst experience",
    "switching to",
    "competitor",
]

# Per-keyword boost applied during post-processing (additive, on a 0-100 scale).
# Lowered from 6.0 → 4.5 because the expanded keyword list now catches more
# matches per complaint; a smaller per-keyword bump avoids over-inflation.
KEYWORD_BOOST: float = 4.5

# ---------------------------------------------------------------------------
# Risk categorisation thresholds (percentage points)
# ---------------------------------------------------------------------------
# Exported so consumer code (e.g. the Gradio UI footer) stays in sync with
# any future changes to these thresholds automatically.
RISK_THRESHOLD_CRITICAL: int = 80
RISK_THRESHOLD_HIGH: int = 55
RISK_THRESHOLD_MEDIUM: int = 30

# Maximum character count accepted by ``analyze_complaint()``.
# Rejects pathologically long inputs to prevent resource exhaustion.
INPUT_MAX_CHARS: int = 2_000

# ---------------------------------------------------------------------------
# 3. Build the scikit-learn Pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Construct and return the churn-prediction pipeline.

    Steps
    -----
    1. **TfidfVectorizer** — converts raw text into TF-IDF features.
       `ngram_range=(1, 2)` lets the model learn from both single words
       ("lost") and two-word phrases ("not delivered").
    2. **RandomForestClassifier** — an ensemble of decision trees that votes
       on the churn probability.  `class_weight='balanced'` handles the small
       dataset gracefully and `random_state` ensures reproducibility.
    """
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),      # capture bigrams like "not delivered"
                    stop_words="english",     # drop common English filler words
                    max_features=5000,        # larger vocab for merged dataset
                    sublinear_tf=True,        # apply log-normalisation to TF
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,         # more trees for larger dataset
                    max_depth=None,           # let trees grow fully
                    min_samples_leaf=2,       # smoother probability estimates
                    class_weight="balanced",  # counteract any class imbalance
                    random_state=42,          # reproducibility
                    n_jobs=-1,                # use all CPU cores
                ),
            ),
        ]
    )
    return pipeline

# ---------------------------------------------------------------------------
# 4. Merge Hand-Curated + External Data
# ---------------------------------------------------------------------------

def get_training_data() -> tuple[list[str], NDArray[np.int_]]:
    """
    Combine the hand-curated seed corpus with externally downloaded datasets.

    The external datasets are downloaded once (cached to disk) via
    ``data_loader.load_external_data()``.  If the download fails the model
    gracefully falls back to the 120 seed samples.

    Returns
    -------
    X : list[str]  — complaint texts
    y : NDArray     — binary labels (0 retained, 1 churned)
    """
    X: list[str] = list(COMPLAINTS)
    y_list: list[int] = list(LABELS)

    try:
        ext_texts, ext_labels = load_external_data()
        if ext_texts:
            X.extend(ext_texts)
            y_list.extend(ext_labels)
            logger.info(
                "Merged %d external samples → total training set: %d",
                len(ext_texts),
                len(X),
            )
    except Exception:
        logger.exception("External data unavailable — using seed data only.")

    return X, np.array(y_list)

# ---------------------------------------------------------------------------
# 5. Train (fit) the Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(pipeline: Pipeline) -> Pipeline:
    """
    Fit the pipeline on the merged dataset (seed + external) and log
    cross-validated accuracy.
    """
    X, y = get_training_data()

    # Stratified CV — 5-fold works well for ~2 000 samples.
    cv_scores: NDArray[np.float64] = cross_val_score(
        pipeline, X, y, cv=5, scoring="accuracy"
    )
    logger.info(
        "5-Fold CV accuracy: %.2f ± %.2f", cv_scores.mean(), cv_scores.std()
    )

    # Final fit on everything.
    pipeline.fit(X, y)
    logger.info("Pipeline trained on %d samples.", len(X))
    return pipeline

# ---------------------------------------------------------------------------
# 6. Post-Processing — Keyword Boost
# ---------------------------------------------------------------------------

def detect_keywords(text: str) -> list[str]:
    """
    Return all high-priority logistics keywords found in *text*.

    Uses case-insensitive substring matching so that "LOST" and "lost" both
    trigger.  Multi-word keywords like "not delivered" are also detected.
    """
    text_lower: str = text.lower()
    found: list[str] = [kw for kw in HIGH_PRIORITY_KEYWORDS if kw in text_lower]
    return found


def apply_keyword_boost(base_risk: float, keywords: list[str]) -> float:
    """
    Increase the base risk score for each detected keyword.

    The boost is additive: each keyword adds `KEYWORD_BOOST` percentage
    points.  The result is clamped to [0, 100].

    Why additive?
    -------------
    In logistics, a single high-severity keyword (e.g. "lost") already
    warrants escalation.  An additive boost ensures even a modest model
    probability gets pushed into the "high risk" band when the language is
    clearly alarming.
    """
    boosted: float = base_risk + len(keywords) * KEYWORD_BOOST
    return float(np.clip(boosted, 0.0, 100.0))

# ---------------------------------------------------------------------------
# 7. Public API — analyze_complaint()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7a. Lazy-loaded pipeline singleton (thread-safe double-checked locking)
# ---------------------------------------------------------------------------
# Training — including 5-fold cross-validation — is deferred until the first
# call to ``get_pipeline()`` or ``analyze_complaint()``.  This avoids
# expensive model training on every module import, which matters most during
# pytest runs that only exercise helper functions like ``detect_keywords``.
_pipeline: Pipeline | None = None
_pipeline_lock: threading.Lock = threading.Lock()


def get_pipeline() -> Pipeline:
    """Return the trained pipeline, building it on the first call (thread-safe).

    Uses double-checked locking so concurrent requests in a multi-worker
    Gradio deployment do not race to start parallel training jobs.
    """
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:  # re-check inside the lock
                _pipeline = train_pipeline(build_pipeline())
    return _pipeline


def analyze_complaint(text: str) -> dict[str, Any]:
    """
    Analyse a single customer complaint and return a churn-risk report.

    Parameters
    ----------
    text : str
        Free-text complaint or feedback from a logistics customer.

    Returns
    -------
    dict with keys
        - ``input_text``             : the original complaint
        - ``sentiment_score``        : model probability of churn (0–100 %)
        - ``detected_keywords``      : list of high-priority keywords found
        - ``keyword_boost_applied``  : total boost added (percentage points)
        - ``final_churn_risk_pct``   : sentiment + keyword boost, clamped 0–100
        - ``risk_level``             : categorical label (Low / Medium / High / Critical)

    Example
    -------
    >>> result = analyze_complaint("My package was lost and I want a refund")
    >>> result["risk_level"]
    'Critical'
    """
    # --- Input validation -----------------------------------------------
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}.")
    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty.")
    if len(text) > INPUT_MAX_CHARS:
        raise ValueError(
            f"Input exceeds maximum length of {INPUT_MAX_CHARS:,} characters "
            f"(received {len(text):,})."
        )

    # --- Model inference ------------------------------------------------
    # predict_proba returns [[P(class 0), P(class 1)]].
    # Class 1 = "churn", so we take index 1.
    proba: float = float(get_pipeline().predict_proba([text])[0][1]) * 100.0
    logger.info("Raw model churn probability for input: %.2f%%", proba)

    # --- Post-processing ------------------------------------------------
    keywords: list[str] = detect_keywords(text)
    final_risk: float = apply_keyword_boost(proba, keywords)
    boost: float = len(keywords) * KEYWORD_BOOST  # for reporting only

    # --- Risk categorisation --------------------------------------------
    if final_risk >= RISK_THRESHOLD_CRITICAL:
        risk_level = "Critical"
    elif final_risk >= RISK_THRESHOLD_HIGH:
        risk_level = "High"
    elif final_risk >= RISK_THRESHOLD_MEDIUM:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    logger.info(
        "Keywords=%s | Boost=+%.1f%% | Final risk=%.2f%% (%s)",
        keywords or "none",
        boost,
        final_risk,
        risk_level,
    )

    return {
        "input_text": text,
        "sentiment_score": round(proba, 2),
        "detected_keywords": keywords,
        "keyword_boost_applied": round(boost, 2),
        "final_churn_risk_pct": round(final_risk, 2),
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# 8. Quick demo when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_texts: list[str] = [
        "The driver was great",
        "My package is 4 days late",
        "Shipment lost in transit, I want a refund",
        "Friendly staff, but my box was crushed",
        "Package not delivered and item is damaged",
    ]

    print("\n" + "=" * 72)
    print(" LOGISTICS CHURN RISK DETECTOR — Demo")
    print("=" * 72)

    for complaint in demo_texts:
        result: dict[str, Any] = analyze_complaint(complaint)
        print(f"\n📦  \"{result['input_text']}\"")
        print(f"    Sentiment Score       : {result['sentiment_score']:.2f}%")
        print(f"    Detected Keywords     : {result['detected_keywords'] or '—'}")
        print(f"    Keyword Boost Applied : +{result['keyword_boost_applied']:.2f}%")
        print(f"    ➜  Final Churn Risk   : {result['final_churn_risk_pct']:.2f}%  [{result['risk_level']}]")

    print("\n" + "=" * 72 + "\n")
