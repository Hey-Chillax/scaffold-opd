# Copyright 2026 Yi Zhang
# SPDX-License-Identifier: Apache-2.0
"""Per-model, per-benchmark statistics from opd/eval/math_benchmarks.py outputs.

Usage: python3 tools/analysis/eval_stats.py <eval root> > eval_stats.json
Reads <root>/<tag>/<bench>.jsonl (one record per sample) and prints
{tag: {bench: {acc, se, pass_at_k, n_problems, n_samples, mean_len, eos_rate, length_cut_rate}}}.
acc is avg@k; se is the standard error over problems (std of per-problem accuracy / sqrt(#problems)).
"""

import glob
import json
import math
import os
import statistics
import sys

root = sys.argv[1]
stats = {}
for tag_dir in sorted(glob.glob(os.path.join(root, "*", ""))):
    tag = os.path.basename(os.path.dirname(tag_dir))
    for path in sorted(glob.glob(os.path.join(tag_dir, "*.jsonl"))):
        bench = os.path.basename(path)[: -len(".jsonl")]
        if "." in bench or "trace" in bench or "stop_events" in bench:
            continue
        per_problem, lengths, eos, cut = {}, [], 0, 0
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                per_problem.setdefault(rec["problem_idx"], []).append(int(rec["correct"]))
                lengths.append(rec["gen_len"])
                eos += bool(rec.get("eos_reached"))
                cut += rec.get("finish_reason") == "length"
        if not per_problem:
            continue
        means = [sum(v) / len(v) for v in per_problem.values()]
        n = len(means)
        stats.setdefault(tag, {})[bench] = {
            "acc": sum(means) / n,
            "se": statistics.stdev(means) / math.sqrt(n) if n > 1 else 0.0,
            "pass_at_k": sum(1 for v in per_problem.values() if any(v)) / n,
            "n_problems": n,
            "n_samples": len(lengths),
            "mean_len": sum(lengths) / len(lengths),
            "eos_rate": eos / len(lengths),
            "length_cut_rate": cut / len(lengths),
        }
json.dump(stats, sys.stdout, indent=1)
