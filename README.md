# ExtraSensory paper reproduction

This repository reproduces the linear context-recognition baseline from:

> Yonatan Vaizman, Katherine Ellis, and Gert Lanckriet, “Recognizing Detailed
> Human Context In-the-Wild from Smartphones and Smartwatches,” *IEEE Pervasive
> Computing* 16(4), 62–74 (2017).
> [doi:10.1109/MPRV.2017.3971131](https://doi.org/10.1109/MPRV.2017.3971131),
> [arXiv:1609.06354](https://arxiv.org/abs/1609.06354).

The primary target is the paper’s headline 25-label, five-fold,
user-independent balanced-accuracy result for early sensor fusion (`EF`). The
training CLI also supports each of the six individual sensor groups.

## Result

The executed reproduction obtains **0.7683 macro balanced accuracy**, compared
with **0.7688** from the paper’s rounded per-label table (difference −0.0005).
At the published precision, 21 of 25 per-label scores match exactly; all 25 are
within 0.02, with a mean absolute gap of 0.0034. See the
[executed notebook](notebooks/01_paper_reproduction.ipynb) for the population
checks, per-label comparison, and findings.

## Model-improvement study

The follow-up study evaluates temporal, multi-task, label-dependency,
robustness, and sensor-specialized models. Its confirmatory candidate is a
causal temporal convolutional network over right-aligned 30-minute histories.
Histories never cross users or gaps longer than 90 seconds.

The official five-fold TCN score is **0.7979**, or **+0.0296** above the
existing paper-logistic reproduction. On the sealed 12-user audit fold, its
paired gain over threshold-tuned logistic regression is **+0.0159** with a 95%
user-cluster bootstrap interval of **[−0.0085, +0.0372]**. The interval crosses
zero, so the paired audit evidence is promising but not independently
conclusive.

Model selection is staged to reduce overfitting to the official benchmark:

- folds 0–2 are development data;
- fold 4 is a single-use promotion fold;
- fold 3 remains sealed until promotion passes;
- the official five-fold evaluation runs only after the sealed audit passes.

Acceptance also bounds calibration and low-performing-user regressions.
The [executed improvement notebook](notebooks/02_model_improvement_study.ipynb)
contains the final comparison, user-cluster bootstrap confidence interval,
per-label effects, ablations, missing-sensor stress tests, and negative results.
Compact source tables live in `results/research/`; fitted checkpoints and
row-level predictions are reproducible but intentionally ignored by Git.

## What is reproduced

- The exact official subject-level five-fold partition (48 train users and 12
  test users per fold).
- The six paper sensor groups: phone accelerometer, phone gyroscope, watch
  accelerometer, relative location, privacy-preserving audio features, and
  phone state/time of day.
- One class-balanced binary logistic-regression classifier per context label.
- Training-only feature standardization.
- The paper’s cost grid, `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`, selected on a
  stratified one-third validation split by F1.
- Evaluation on examples with all six core sensors and fold-level confusion
  counts pooled before balanced accuracy is calculated.

The current public files add `NaN` markers for labels inferred to be missing in
a later ExtraSensory paper and changed three of the paper’s label columns.
`prepare` combines the current feature archive with the official
`original_labels` archive and the paper’s stated rules to recover all 25
published positive counts exactly. The 2017 paper treated unreported labels as
negative, so `--label-policy legacy-negative` is the reproduction default. Use
`--label-policy exclude-missing` for a modern sensitivity analysis.

## Quick start

```bash
uv sync --extra dev
uv run extrasensory-reproduce download
uv run extrasensory-reproduce prepare
uv run extrasensory-reproduce profile
uv run extrasensory-reproduce train --models EF --jobs 4
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/01_paper_reproduction.ipynb
```

The official feature archive is about 215 MiB; the original-label archive is
about 1 MiB. `download` verifies both SHA-256 values recorded in
`src/extrasensory_reproduction/constants.py`. Raw and processed data are
intentionally ignored by Git.

## Repository layout

- `paper/`: official paper, supplement, and tutorial links with version hashes.
- `data/reference/`: official README/folds and the transcribed published
  25-label balanced-accuracy table.
- `src/extrasensory_reproduction/`: data validation, training, and evaluation.
- `results/`: compact, tracked metrics and figures from the executed run.
- `notebooks/01_paper_reproduction.ipynb`: reader-facing executed analysis.
- `notebooks/02_model_improvement_study.ipynb`: executed gated improvement study.
- `configs/research/`: immutable model and bounded-sweep configurations.
- `tests/`: unit tests for folds, metrics, label semantics, and feature routing.

## Reproduction contract and caveats

The paper does not publish its random seed or scikit-learn version. This repo
uses a visible deterministic seed and the `liblinear` solver, which matches the
default solver used by the official 2017 scikit-learn tutorial. Exact
per-label equality is therefore not expected; agreement is assessed against the
published two-decimal values and the headline macro average.

The public archive contains precomputed features, not raw audio, and does not
state a formal software/data license. It says the dataset is public and
requires the original paper to be cited in publications. This repository does
not redistribute the primary data archive.

## Running the improvement workflow

The bounded search loop borrows the useful part of Karpathy's
`autoresearch` pattern—small config-only mutations, an append-only ledger, and
explicit keep/reject gates—without allowing the agent to edit training code
during a sweep.

```bash
uv run extrasensory-reproduce research sweep \
  --spec configs/research/causal_tcn_sweep.toml \
  --max-trials 30 --budget-hours 12
uv run extrasensory-reproduce research promote \
  --spec configs/research/causal_tcn_sweep.toml
uv run extrasensory-reproduce research audit \
  --spec configs/research/causal_tcn_sweep.toml
uv run python scripts/build_research_report.py
uv run python scripts/build_research_notebook.py
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/02_model_improvement_study.ipynb
```

Each configuration caps CPU workers at four. Neural models use Apple MPS when
available and otherwise fall back to CPU. A dependent family may unlock from
completed development evidence, but only one family should consume the
promotion fold in a study.
