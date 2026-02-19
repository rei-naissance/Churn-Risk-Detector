"""
Unit tests for the Logistics Churn Risk Detector.
Run with:  pytest test_churn_model.py -v
"""

from __future__ import annotations

import pytest

from sklearn.pipeline import Pipeline

from churn_model import (
    COMPLAINTS,
    HIGH_PRIORITY_KEYWORDS,
    KEYWORD_BOOST,
    LABELS,
    analyze_complaint,
    apply_keyword_boost,
    build_pipeline,
    detect_keywords,
    get_training_data,
)


# ── Dataset integrity (seed data) ────────────────────────────────────────────

class TestSeedDataset:
    """Validate the 120-sample hand-curated seed corpus."""

    def test_complaints_labels_same_length(self) -> None:
        assert len(COMPLAINTS) == len(LABELS)

    def test_at_least_100_samples(self) -> None:
        assert len(COMPLAINTS) >= 100

    def test_balanced_classes(self) -> None:
        n_pos = sum(1 for l in LABELS if l == 0)
        n_neg = sum(1 for l in LABELS if l == 1)
        assert n_pos == n_neg

    def test_no_empty_strings(self) -> None:
        for i, c in enumerate(COMPLAINTS):
            assert len(c.strip()) > 0, f"COMPLAINTS[{i}] is empty"

    def test_labels_are_binary(self) -> None:
        assert set(LABELS) == {0, 1}


# ── Merged training data ─────────────────────────────────────────────────────

class TestMergedData:
    """Validate the merged seed + external training corpus."""

    def test_merged_larger_than_seed(self) -> None:
        X, y = get_training_data()
        assert len(X) > len(COMPLAINTS), "External data should expand the corpus"

    def test_merged_at_least_1000_samples(self) -> None:
        X, y = get_training_data()
        assert len(X) >= 1000

    def test_merged_labels_match_texts(self) -> None:
        X, y = get_training_data()
        assert len(X) == len(y)

    def test_merged_labels_binary(self) -> None:
        X, y = get_training_data()
        assert set(y.tolist()).issubset({0, 1})

    def test_both_classes_present(self) -> None:
        X, y = get_training_data()
        assert 0 in y and 1 in y


# ── detect_keywords ──────────────────────────────────────────────────────────

class TestDetectKeywords:
    def test_finds_single_keyword(self) -> None:
        assert "lost" in detect_keywords("My shipment was lost")

    def test_finds_multiple_keywords(self) -> None:
        kws = detect_keywords("Package damaged, I want a refund")
        assert "damaged" in kws
        assert "refund" in kws

    def test_case_insensitive(self) -> None:
        assert "lost" in detect_keywords("LOST in transit")

    def test_multi_word_keyword(self) -> None:
        assert "not delivered" in detect_keywords("Package not delivered yet")

    def test_no_keywords(self) -> None:
        assert detect_keywords("Great service, very happy") == []

    def test_new_keywords_from_expansion(self) -> None:
        """Verify several keywords added in the expansion are detected."""
        assert "crushed" in detect_keywords("The box arrived crushed")
        assert "hidden fees" in detect_keywords("There were hidden fees on the invoice")
        assert "never arrived" in detect_keywords("The package never arrived")
        assert "wrong address" in detect_keywords("Delivered to the wrong address")
        assert "hung up" in detect_keywords("The agent hung up on me")
        assert "misdelivered" in detect_keywords("Parcel was misdelivered")


# ── apply_keyword_boost ──────────────────────────────────────────────────────

class TestApplyKeywordBoost:
    def test_no_boost_when_no_keywords(self) -> None:
        assert apply_keyword_boost(50.0, []) == 50.0

    def test_single_keyword_boost(self) -> None:
        assert apply_keyword_boost(50.0, ["lost"]) == 50.0 + KEYWORD_BOOST

    def test_multiple_keyword_boost(self) -> None:
        result = apply_keyword_boost(40.0, ["lost", "damaged", "refund"])
        assert result == 40.0 + 3 * KEYWORD_BOOST

    def test_clamped_at_100(self) -> None:
        assert apply_keyword_boost(99.0, ["lost", "refund"]) == 100.0

    def test_clamped_at_0(self) -> None:
        assert apply_keyword_boost(0.0, []) == 0.0


# ── analyze_complaint ────────────────────────────────────────────────────────

class TestAnalyzeComplaint:
    def test_returns_dict_with_expected_keys(self) -> None:
        result = analyze_complaint("Late delivery")
        expected_keys = {
            "input_text",
            "sentiment_score",
            "detected_keywords",
            "keyword_boost_applied",
            "final_churn_risk_pct",
            "risk_level",
        }
        assert set(result.keys()) == expected_keys

    def test_negative_feedback_high_risk(self) -> None:
        result = analyze_complaint("Shipment lost, I want a refund now")
        assert result["risk_level"] in ("High", "Critical")
        assert result["final_churn_risk_pct"] >= 55

    def test_keyword_boost_reflected(self) -> None:
        result = analyze_complaint("Package lost and damaged, give me a refund")
        assert result["keyword_boost_applied"] > 0
        assert len(result["detected_keywords"]) >= 2

    def test_sentiment_score_range(self) -> None:
        result = analyze_complaint("Some random feedback")
        assert 0 <= result["sentiment_score"] <= 100

    def test_final_risk_range(self) -> None:
        result = analyze_complaint("I need my money back, item was destroyed")
        assert 0 <= result["final_churn_risk_pct"] <= 100

    def test_risk_level_is_valid(self) -> None:
        result = analyze_complaint("Package not delivered")
        assert result["risk_level"] in ("Low", "Medium", "High", "Critical")

    def test_real_world_complaint_late_delivery(self) -> None:
        """Inspired by real FedEx Trustpilot review themes."""
        result = analyze_complaint(
            "Paid for express shipping but the package has been sitting "
            "at the hub for five days with no updates"
        )
        assert result["final_churn_risk_pct"] > 30

    def test_real_world_complaint_misdelivery(self) -> None:
        """Inspired by real UPS Trustpilot review themes."""
        result = analyze_complaint(
            "Delivered to the wrong address across the street and the "
            "photo shows a house that is not mine"
        )
        assert "wrong address" in result["detected_keywords"]

    def test_clear_negative_high_keywords(self) -> None:
        """Multiple severe keywords should push risk >= 55%."""
        result = analyze_complaint(
            "Package was lost, item arrived damaged and I was never refunded"
        )
        assert result["final_churn_risk_pct"] >= 55
        assert len(result["detected_keywords"]) >= 2


# ── Pipeline construction ────────────────────────────────────────────────────

class TestPipeline:
    def test_build_pipeline_returns_pipeline(self) -> None:
        p = build_pipeline()
        assert isinstance(p, Pipeline)

    def test_pipeline_has_tfidf_and_clf(self) -> None:
        p = build_pipeline()
        step_names = [name for name, _ in p.steps]
        assert "tfidf" in step_names
        assert "clf" in step_names


# ── External data loader (integration) ───────────────────────────────────────

class TestDataLoader:
    def test_load_external_data_returns_data(self) -> None:
        from data_loader import load_external_data
        texts, labels = load_external_data()
        assert len(texts) > 0
        assert len(texts) == len(labels)

    def test_external_data_has_both_classes(self) -> None:
        from data_loader import load_external_data
        _, labels = load_external_data()
        assert 0 in labels and 1 in labels

    def test_cache_file_exists_after_load(self) -> None:
        from data_loader import _CACHE_FILE, load_external_data
        load_external_data()
        assert _CACHE_FILE.exists()
        pipe = build_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "tfidf" in step_names
        assert "clf" in step_names
