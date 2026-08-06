"""Statistical honesty: paired bootstrap 95% CI + Wilcoxon signed-rank.

Reads results/per_query.json (per-query nDCG/MRR per row), computes for the
comparisons that matter (row5 vs row3, row4 vs row3):
  - paired bootstrap 95% CI on the mean difference (10k resamples, fixed seed)
  - two-sided Wilcoxon signed-rank p-value

The README wording is chosen from this output: 'beats' only if CI excludes 0
AND p<0.05; otherwise 'tracks with' / 'directionally better'.
"""
import json

import numpy as np
from scipy.stats import wilcoxon

DATA = "results/per_query.json"
N_BOOT = 10_000
SEED = 42


def bootstrap_diff(a, b, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    nq = len(a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, nq, nq)
        diffs.append(np.mean(b[idx] - a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi


def main():
    with open(DATA, encoding="utf-8") as f:
        pq = json.load(f)

    for metric in ["ndcg", "mrr"]:
        print(f"\n=== {metric}@10 ===")
        base3 = pq["3_off_rerank"][metric]
        base4 = pq["4_tr_rrf"][metric]
        r5 = pq["5_tr_rerank"][metric]
        n = len(base3)

        for name, a, b in [("row5 vs row3 (trained stack vs off-the-shelf full)", base3, r5),
                           ("row4 vs row3 (trained bi, no rerank vs off-the-shelf full)", base3, base4)]:
            lo, hi = bootstrap_diff(a, b)
            try:
                w = wilcoxon(a, b)
                p = w.pvalue
            except ValueError:
                p = float("nan")
            mean_diff = float(np.mean(np.asarray(b) - np.asarray(a)))
            signif = "SIGNIFICANT" if lo > 0 and p < 0.05 else "not significant"
            print(f"  {name}: mean_diff={mean_diff:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  wilcoxon_p={p:.4f}  -> {signif}")
        print(f"  (n={n})")


if __name__ == "__main__":
    main()
