# Statistical summary (paired bootstrap + Wilcoxon)

Computed on 99 golden queries, per-query metrics, paired design.
Bootstrap: 10,000 resamples, fixed seed 42. Wilcoxon: two-sided signed-rank.

## Trained stack vs off-the-shelf full stack (row 5 vs row 3)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| MRR@10  | +0.059 | [+0.005, +0.112] | 0.026 | significant |
| nDCG@10 | +0.044 | [-0.004, +0.089] | 0.052 | borderline / not significant |

## Trained bi-encoder alone vs off-the-shelf full stack (row 4 vs row 3)

| metric | mean diff | 95% bootstrap CI | Wilcoxon p | verdict |
|---|---|---|---|---|
| nDCG@10 | -0.001 | [-0.063, +0.058] | 0.990 | indistinguishable |
| MRR@10  | -0.000 | [-0.074, +0.070] | 0.985 | indistinguishable |

## Honest wording (chosen by the data)

- **Headline (no p-value needed):** a 29-second fine-tune of a 110M bi-encoder
  *matches* an off-the-shelf stack that requires a frozen 278M cross-encoder
  reranker at query time (nDCG@10 0.753 vs 0.754; p=0.99 → statistically
  indistinguishable). The fine-tune internalized ranking ability.
- **Secondary:** training both models beats the off-the-shelf stack
  *significantly on MRR* (+0.059, p=0.026); on nDCG the gain is directional
  (+0.044, p=0.05, CI includes zero) — reported as near-significant, not "beats".
- **Recall is saturated** at 0.909 (9/99 golds unreachable even for the best row
  after OCR), so the comparison is about *ranking quality*, not recall.
