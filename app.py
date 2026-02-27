"""
Gradio Web UI for the Logistics Churn Risk Detector
====================================================
Designed for deployment on **Hugging Face Spaces**.

Launch locally:
    python app.py

The interface exposes:
- A text box for entering a customer complaint.
- A submit button that calls ``analyze_complaint()``.
- A styled results panel with risk gauge, keyword tags, and score breakdown.

Author : Senior AI Engineer
Created: 2026-02-20
"""

from __future__ import annotations

import functools
import logging
import os

import gradio as gr

from churn_model import (
    INPUT_MAX_CHARS,
    POSITIVE_DAMPENER,
    RISK_THRESHOLD_CRITICAL,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    analyze_complaint,
    get_training_data,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_RISK_COLOURS: dict[str, str] = {
    "Critical": "#dc2626",  # red-600
    "High": "#ea580c",      # orange-600
    "Medium": "#ca8a04",    # yellow-600
    "Low": "#16a34a",       # green-600
}

_RISK_EMOJI: dict[str, str] = {
    "Critical": "🔴",
    "High": "🟠",
    "Medium": "🟡",
    "Low": "🟢",
}


# ---------------------------------------------------------------------------
# Core prediction function (wraps analyze_complaint for Gradio)
# ---------------------------------------------------------------------------

def predict(complaint: str) -> str:
    """Run the model and return an HTML-formatted results card."""
    if not complaint or not complaint.strip():
        return _placeholder_html()

    complaint = complaint.strip()
    if len(complaint) > INPUT_MAX_CHARS:
        return _error_html(
            f"Complaint is too long ({len(complaint):,} characters). "
            f"Please keep it under {INPUT_MAX_CHARS:,} characters."
        )

    try:
        result = analyze_complaint(complaint)
    except Exception as exc:
        logger.exception("Unexpected error while analysing complaint.")
        return _error_html(f"Analysis failed: {exc}")

    risk = result["risk_level"]
    colour = _RISK_COLOURS[risk]
    emoji = _RISK_EMOJI[risk]
    pct = result["final_churn_risk_pct"]
    sentiment = result["sentiment_score"]
    boost = result["keyword_boost_applied"]
    keywords = result["detected_keywords"]
    pos_keywords = result["positive_keywords"]
    pos_dampening = result["positive_dampening_applied"]

    kw_tags = "" .join(
        f'<span style="display:inline-block;background:{colour}22;'
        f"color:{colour};border:1px solid {colour};border-radius:12px;"
        f'padding:2px 10px;margin:2px 4px 2px 0;font-size:0.85em;">'
        f"{kw}</span>"
        for kw in keywords
    ) or '<span style="color:#888;font-style:italic;">none detected</span>'

    _POS_COLOUR = "#16a34a"  # green-600
    pos_kw_tags = "".join(
        f'<span style="display:inline-block;background:#16a34a22;'
        f'color:#16a34a;border:1px solid #16a34a;border-radius:12px;'
        f'padding:2px 10px;margin:2px 4px 2px 0;font-size:0.85em;">'
        f"{kw}</span>"
        for kw in pos_keywords
    ) or '<span style="color:#888;font-style:italic;">none detected</span>'

    # Risk gauge bar
    bar_width = max(min(pct, 100), 0)
    gauge = (
        f'<div style="background:#e5e7eb;border-radius:8px;height:24px;'
        f'width:100%;margin:8px 0;overflow:hidden;">'
        f'<div style="background:{colour};height:100%;width:{bar_width}%;'
        f'border-radius:8px;transition:width 0.4s ease;"></div></div>'
    )

    html = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;max-width:560px;">
        <div style="text-align:center;margin-bottom:12px;">
            <span style="font-size:2.4em;">{emoji}</span>
            <div style="font-size:1.6em;font-weight:700;color:{colour};margin-top:4px;">
                {pct:.1f}% &mdash; {risk} Risk
            </div>
        </div>

        {gauge}

        <table style="width:100%;border-collapse:collapse;margin-top:16px;font-size:0.95em;">
            <tr style="border-bottom:1px solid #e5e7eb;">
                <td style="padding:8px 4px;color:#6b7280;">Raw Model Score</td>
                <td style="padding:8px 4px;text-align:right;font-weight:600;">{sentiment:.1f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
                <td style="padding:8px 4px;color:#6b7280;">Keyword Boost</td>
                <td style="padding:8px 4px;text-align:right;font-weight:600;">+{boost:.1f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
                <td style="padding:8px 4px;color:#6b7280;">Positive Signal Dampening</td>
                <td style="padding:8px 4px;text-align:right;font-weight:600;color:#16a34a;">&minus;{pos_dampening:.1f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #e5e7eb;">
                <td style="padding:8px 4px;color:#6b7280;">Final Churn Risk</td>
                <td style="padding:8px 4px;text-align:right;font-weight:700;color:{colour};">{pct:.1f}%</td>
            </tr>
        </table>

        <div style="margin-top:16px;">
            <div style="color:#6b7280;font-size:0.85em;margin-bottom:4px;">Risk Signals (churn indicators)</div>
            {kw_tags}
        </div>
        <div style="margin-top:10px;">
            <div style="color:#6b7280;font-size:0.85em;margin-bottom:4px;">Positive Signals (satisfaction indicators)</div>
            {pos_kw_tags}
        </div>
    </div>
    """
    return html


def _placeholder_html() -> str:
    return (
        '<div style="text-align:center;color:#9ca3af;padding:40px 0;'
        'font-family:system-ui,sans-serif;">'
        "<p>✏️ Enter a customer complaint above and click <b>Analyze</b></p>"
        "</div>"
    )


def _error_html(message: str) -> str:
    """Return an HTML error card displayed in the results panel."""
    return (
        '<div style="text-align:center;color:#dc2626;padding:40px 0;'
        'font-family:system-ui,sans-serif;">'
        f"<p>⚠️ {message}</p>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Training stats (shown in the sidebar / description)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _get_stats_md() -> str:
    X, y = get_training_data()
    n_total = len(X)
    n_churn = int(y.sum())
    n_retained = n_total - n_churn
    return (
        f"Trained on **{n_total:,}** complaint samples "
        f"({n_churn:,} churn · {n_retained:,} retained) from hand-curated "
        f"logistics data + two Hugging Face datasets."
    )


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------

EXAMPLES: list[str] = [
    "Shipment lost in transit, I want a refund",
    "My package is 4 days late and nobody responds",
    "The driver was great and very polite",
    "Package not delivered and item is damaged",
    "Friendly staff, smooth delivery experience",
    "Tracking says delivered, photo shows a house that is not mine",
    "Called five times and each agent gave me different information",
    "Got my order a day early, very happy",
]

with gr.Blocks(
    title="📦 Logistics Churn Risk Detector",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 📦 Logistics Churn Risk Detector\n"
        "Predict customer churn risk from a free-text shipping complaint.\n\n"
        + _get_stats_md()
    )

    with gr.Row():
        with gr.Column(scale=3):
            txt_input = gr.Textbox(
                label="Customer Complaint",
                placeholder="e.g. My package was lost and I want a refund…",
                lines=3,
            )
            btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        with gr.Column(scale=4):
            html_output = gr.HTML(value=_placeholder_html(), label="Results")

    btn.click(fn=predict, inputs=txt_input, outputs=html_output)
    txt_input.submit(fn=predict, inputs=txt_input, outputs=html_output)

    gr.Examples(
        examples=[[e] for e in EXAMPLES],
        inputs=txt_input,
        outputs=html_output,
        fn=predict,
        cache_examples=False,
    )

    gr.Markdown(
        "---\n"
        "**How it works:** TF-IDF (uni+bigrams, 5 000 features) → "
        "Random Forest (300 trees) → keyword boost (+4.5 % per match). "
        f"Risk bands: 🟢 <{RISK_THRESHOLD_MEDIUM} % Low · "
        f"🟡 ≥{RISK_THRESHOLD_MEDIUM} % Medium · "
        f"🟠 ≥{RISK_THRESHOLD_HIGH} % High · "
        f"🔴 ≥{RISK_THRESHOLD_CRITICAL} % Critical.\n\n"
        "Data: [hblim/customer-complaints](https://huggingface.co/datasets/hblim/customer-complaints) · "
        "[aciborowska/customers-complaints](https://huggingface.co/datasets/aciborowska/customers-complaints) · "
        "Trustpilot-inspired seed corpus"
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("SERVER_PORT", "7860")),
    )
