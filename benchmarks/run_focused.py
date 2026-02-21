#!/usr/bin/env python3
"""
Focused benchmark: 8 representative configs × strategic pairs.

Tests the critical transfer scenarios:
  ● Same-arch width scaling (64→256, 256→64)
  ● Same-arch depth scaling (2L→6L, 6L→2L)
  ● Cross-arch same-dim (transformer↔mlp↔conv at 128d)
  ● Cross-arch + width gap (transformer 64d→conv 256d, etc.)
"""

import sys, os, time, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import DataLoader
from tessera.utils import setup_logging, set_seed

from cross_arch_benchmark import (
    ARCH_FACTORY, create_dataset, run_single_transfer,
    generate_html_report, BenchmarkResult,
)

logger = setup_logging("focused")


# 8 representative configs
CONFIGS = [
    ("tf_64d_2L",   "transformer", 64,  2),
    ("tf_128d_4L",  "transformer", 128, 4),
    ("tf_256d_6L",  "transformer", 256, 6),
    ("mlp_64d_2L",  "mlp",         64,  2),
    ("mlp_128d_4L", "mlp",         128, 4),
    ("mlp_256d_6L", "mlp",         256, 6),
    ("conv_64d_2L", "conv",        64,  2),
    ("conv_128d_4L","conv",        128, 4),
]

# Strategic pairs (not full combinatorial)
PAIRS = [
    # ── Same-arch width scaling ──
    (0, 1), (1, 2), (2, 0),  # tf: 64→128, 128→256, 256→64
    (3, 4), (4, 5), (5, 3),  # mlp: 64→128, 128→256, 256→64
    (6, 7),                    # conv: 64→128

    # ── Same-arch depth scaling ──
    (0, 2), (2, 0),           # tf: 2L→6L, 6L→2L

    # ── Cross-arch same dimension ──
    (1, 4), (4, 1),           # tf 128↔mlp 128
    (1, 7), (7, 1),           # tf 128↔conv 128
    (4, 7), (7, 4),           # mlp 128↔conv 128

    # ── Cross-arch + width gap ──
    (0, 5), (5, 0),           # tf 64→mlp 256, mlp 256→tf 64
    (0, 7), (7, 0),           # tf 64→conv 128, conv 128→tf 64
    (2, 3), (3, 2),           # tf 256→mlp 64, mlp 64→tf 256
    (6, 2), (2, 6),           # conv 64→tf 256, tf 256→conv 64
    (5, 6), (6, 5),           # mlp 256→conv 64, conv 64→mlp 256
]


def main():
    set_seed(42)

    print("\n" + "=" * 70)
    print("  TESSERA — Focused Cross-Architecture Benchmark")
    print("=" * 70 + "\n")

    # Deduplicate pairs
    seen = set()
    unique_pairs = []
    for i, j in PAIRS:
        key = (i, j)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((CONFIGS[i], CONFIGS[j]))

    logger.info(f"Configs: {len(CONFIGS)}")
    logger.info(f"Pairs: {len(unique_pairs)}")

    train_data = create_dataset(400, 16, seed=42)
    val_data = create_dataset(150, 16, seed=99)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    results = []
    t0 = time.time()

    for idx, (tx, rx) in enumerate(unique_pairs):
        logger.info(f"[{idx+1}/{len(unique_pairs)}]")
        r = run_single_transfer(tx, rx, train_loader, val_loader, uhs_epochs=8, finetune_epochs=4)
        results.append(r)
        logger.info(f"    Δ={r.accuracy_delta:+.1f}%  drift={r.drift:.2f}  time={r.elapsed_seconds:.1f}s")

    total = time.time() - t0

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "focused_results.json"), "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": 42, "device": "cpu", "mode": "focused",
                "num_pairs": len(unique_pairs),
                "total_seconds": round(total, 1),
            },
            "results": [{k: v for k, v in r.__dict__.items()} for r in results],
        }, f, indent=2)

    generate_html_report(results, os.path.join(out_dir, "focused_report.html"))

    successful = [r for r in results if r.status == "success"]
    pos = sum(1 for r in successful if r.accuracy_delta > 0)
    avg_d = sum(r.accuracy_delta for r in successful) / len(successful) if successful else 0

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print(f"  Pairs: {len(results)} | Successful: {len(successful)} | Failed: {len(results)-len(successful)}")
    print(f"  Positive transfers: {pos}/{len(successful)} ({pos/len(successful)*100:.0f}%)")
    print(f"  Avg Δ: {avg_d:+.1f}% | Best: {max(r.accuracy_delta for r in successful):+.1f}%")
    print(f"  Total time: {total:.0f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
