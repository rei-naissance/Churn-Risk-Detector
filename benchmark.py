"""
Performance benchmark for the Logistics Churn Risk Detector.

Run with:
    python benchmark.py
"""

from __future__ import annotations

import logging
import time


def main() -> None:
    # Silence model-training logs so benchmark output is easy to read.
    logging.disable(logging.CRITICAL)

    _SAMPLE_TEXTS = [
        "Shipment lost in transit, I want a refund",
        "Where is my package? The tracking hasn't updated in 3 days",
        "The delivery driver threw my box over the fence and damaged the contents.",
        "Everything arrived perfectly on time, thank you!",
        "I was charged hidden fees for customs which were not disclosed.",
    ]

    print("Loading model...", flush=True)
    start_load = time.perf_counter()
    from churn_model import analyze_complaint, get_pipeline  # noqa: PLC0415
    end_load = time.perf_counter()
    print(f"Model Import / Load Time: {(end_load - start_load) * 1000:.2f} ms")

    # 10 000 samples (5 texts × 2 000)
    texts = _SAMPLE_TEXTS * 2_000

    # Single-instance inference latency
    print("Running single-instance inference (1 000 requests)...", flush=True)
    start_infer = time.perf_counter()
    for t in texts[:1_000]:
        analyze_complaint(t)
    end_infer = time.perf_counter()
    total_time = end_infer - start_infer
    print(f"Average Latency (Single): {total_time / 1_000 * 1_000:.3f} ms/req")
    print(f"Throughput   (Single):   {1_000 / total_time:.2f} req/s")

    # Batch inference throughput
    print("Running batch inference (10 000 samples)...", flush=True)
    pipeline = get_pipeline()
    start_batch = time.perf_counter()
    pipeline.predict_proba(texts)  # 10 000 samples
    end_batch = time.perf_counter()
    batch_time = end_batch - start_batch
    print(f"Throughput   (Batch):    {len(texts) / batch_time:.2f} req/s")


if __name__ == "__main__":
    main()
