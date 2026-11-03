"""Build the reader-facing reproduction notebook from executed result files."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results"
    notebook_path = root / "notebooks" / "01_paper_reproduction.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    paper = pd.read_csv(root / "data/reference/paper_balanced_accuracy_5fold_headline.csv")
    reproduced = pd.read_csv(result_dir / "label_metrics.csv").query("model == 'EF'").copy()
    metadata = json.loads((result_dir / "run_metadata.json").read_text())
    comparison = reproduced.merge(
        paper[["label", "EF", "n_examples"]],
        on="label",
        validate="one_to_one",
    )
    comparison["delta"] = comparison["balanced_accuracy"] - comparison["EF"]

    published_macro = float(paper["EF"].mean())
    reproduced_macro = float(comparison["balanced_accuracy"].mean())
    macro_delta = reproduced_macro - published_macro
    rounded_matches = int(
        (comparison["balanced_accuracy"].round(2) == comparison["EF"].round(2)).sum()
    )
    within_two_points = int((comparison["delta"].abs() <= 0.02).sum())
    largest_gap_row = comparison.loc[comparison["delta"].abs().idxmax()]
    direction = "above" if macro_delta >= 0 else "below"

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }

    cells = [
        nbf.v4.new_markdown_cell(
            "# Reproducing the ExtraSensory 2017 Early-Fusion Baseline\n\n"
            "Vaizman, Ellis, and Lanckriet, “Recognizing Detailed Human Context "
            "In-the-Wild from Smartphones and Smartwatches.”"
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"The paper reports a 25-label macro balanced accuracy of **"
            f"{published_macro:.3f}** (shown as 0.77); this run obtains **"
            f"{reproduced_macro:.3f}**, {abs(macro_delta):.3f} {direction}. "
            f"**{rounded_matches}/25** per-label scores match the published "
            f"two-decimal value exactly and **{within_two_points}/25** are within "
            "two percentage points. The evaluation population, sensor counts, "
            "folds, and every positive-label total reconcile exactly to the "
            "paper, so the remaining gaps are model-fit variation rather than a "
            "data-split mismatch."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This is an analysis-report notebook for a technical reader who wants "
            "to inspect the evidence and rerun the baseline. The target is the "
            "paper’s first 25 labels and early feature fusion (`EF`). Each label "
            "gets an independent class-balanced logistic-regression classifier. "
            "Features from the six paper sensor groups are standardized using the "
            "outer training fold, missing derived features are set to the training "
            "mean (zero after standardization), and `C` is chosen from "
            "`{0.001, 0.01, 0.1, 1, 10, 100}` on a stratified one-third validation "
            "split by F1. Confusion counts are pooled over the official five "
            "user-disjoint folds before balanced accuracy is calculated.\n\n"
            "### Key Assumptions\n\n"
            "1. The 2017 baseline treated unreported labels as negative. The newer "
            "missing-label mask is retained only for optional sensitivity analysis.\n"
            "2. The paper omits its random seed and software version. This run uses "
            f"seed `{metadata['seed']}` and scikit-learn "
            f"`{metadata['scikit_learn']}` with `liblinear`, matching the solver "
            "used by the official 2017 tutorial.\n\n"
            "Sources: [paper](https://arxiv.org/abs/1609.06354), "
            "[official dataset](http://extrasensory.ucsd.edu/), "
            "[paper PDF](https://arxiv.org/pdf/1609.06354), and "
            "[supplement](http://extrasensory.ucsd.edu/papers/"
            "vaizman2017Pervasive_supplementary.pdf)."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "from IPython.display import display\n\n"
            "sns.set_theme(style='whitegrid', context='notebook')\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'results').exists():\n"
            "    ROOT = ROOT.parent\n"
            "RESULTS = ROOT / 'results'\n"
            "FIGURES = RESULTS / 'figures'\n"
            "FIGURES.mkdir(parents=True, exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            "profile = json.loads((RESULTS / 'data_profile.json').read_text())\n"
            "paper = pd.read_csv(\n"
            "    ROOT / 'data/reference/paper_balanced_accuracy_5fold_headline.csv'\n"
            ")\n"
            "reproduced = pd.read_csv(RESULTS / 'label_metrics.csv').query("
            "\"model == 'EF'\").copy()\n"
            "fold_metrics = pd.read_csv(RESULTS / 'fold_metrics.csv').query("
            "\"model == 'EF'\").copy()\n"
            "run_metadata = json.loads((RESULTS / 'run_metadata.json').read_text())\n\n"
            "population = pd.DataFrame({\n"
            "    'check': ['participants', 'labeled minutes', "
            "'six-sensor evaluation minutes', 'six-sensor participants', "
            "'core features', 'duplicate user-timestamps'],\n"
            "    'observed': [profile['users'], profile['paper_labeled_examples'], "
            "profile['core_complete_examples'], profile['core_complete_users'], "
            "profile['core_features'], profile['duplicate_user_timestamps']],\n"
            "    'paper / expected': [60, 308320, 176941, 51, 175, 0],\n"
            "})\n"
            "assert (population['observed'] == population['paper / expected']).all()\n"
            "display(population)"
        ),
        nbf.v4.new_code_cell(
            "sensor_coverage = (\n"
            "    pd.DataFrame(profile['sensor_coverage']).T\n"
            "    .reset_index(names='sensor')\n"
            "    [['sensor', 'features', 'available_examples', "
            "'available_users', 'missing_rate']]\n"
            ")\n"
            "sensor_coverage['missing_rate'] = sensor_coverage['missing_rate'].map(\n"
            "    lambda value: f'{value:.1%}'\n"
            ")\n"
            "display(sensor_coverage)"
        ),
        nbf.v4.new_code_cell(
            "label_count_check = reproduced.merge(\n"
            "    paper[['label', 'pretty_label', 'n_examples']],\n"
            "    on=['label', 'pretty_label'],\n"
            "    validate='one_to_one',\n"
            ")\n"
            "label_count_check['delta'] = (\n"
            "    label_count_check['test_positive'] - "
            "label_count_check['n_examples']\n"
            ")\n"
            "assert label_count_check['delta'].eq(0).all()\n"
            "display(label_count_check[\n"
            "    ['pretty_label', 'n_examples', 'test_positive', 'delta']\n"
            "].head(10))\n"
            "print('All 25 positive-label counts match the paper exactly.')"
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            "comparison = reproduced.merge(\n"
            "    paper[['label', 'EF']], on='label', validate='one_to_one'\n"
            ").rename(columns={\n"
            "    'EF': 'published_ba',\n"
            "    'balanced_accuracy': 'reproduced_ba',\n"
            "})\n"
            "comparison['delta'] = (\n"
            "    comparison['reproduced_ba'] - comparison['published_ba']\n"
            ")\n"
            "summary = pd.DataFrame({\n"
            "    'metric': ['macro balanced accuracy', 'mean absolute per-label gap', "
            "'two-decimal matches', 'within 0.02'],\n"
            "    'published / target': [comparison['published_ba'].mean(), 0.0, "
            "25, 25],\n"
            "    'reproduced': [comparison['reproduced_ba'].mean(), "
            "comparison['delta'].abs().mean(), "
            "(comparison['reproduced_ba'].round(2) == "
            "comparison['published_ba']).sum(), "
            "(comparison['delta'].abs() <= 0.02).sum()],\n"
            "})\n"
            "display(summary)"
        ),
        nbf.v4.new_code_cell(
            "plot_data = comparison.sort_values('published_ba').reset_index(drop=True)\n"
            "y = np.arange(len(plot_data))\n"
            "fig, ax = plt.subplots(figsize=(9, 10))\n"
            "ax.hlines(y, plot_data['published_ba'], plot_data['reproduced_ba'], "
            "color='#b8b8b8', linewidth=1.5)\n"
            "ax.scatter(plot_data['published_ba'], y, label='Published', "
            "color='#4c78a8', s=45, zorder=3)\n"
            "ax.scatter(plot_data['reproduced_ba'], y, label='Reproduced', "
            "color='#f58518', s=45, zorder=3)\n"
            "ax.set_yticks(y, plot_data['pretty_label'])\n"
            "ax.set_xlim(0.55, 0.91)\n"
            "ax.set_xlabel('Balanced accuracy')\n"
            "ax.set_title('Published and reproduced early-fusion scores')\n"
            "ax.legend(loc='lower right')\n"
            "fig.tight_layout()\n"
            "fig.savefig(FIGURES / 'published_vs_reproduced_ef.png', dpi=160)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "gap_table = comparison.assign(\n"
            "    absolute_gap=comparison['delta'].abs()\n"
            ").sort_values('absolute_gap', ascending=False)[\n"
            "    ['pretty_label', 'published_ba', 'reproduced_ba', "
            "'delta', 'absolute_gap']\n"
            "]\n"
            "display(gap_table.head(10).style.format({\n"
            "    'published_ba': '{:.3f}', 'reproduced_ba': '{:.3f}', "
            "'delta': '{:+.3f}', 'absolute_gap': '{:.3f}'\n"
            "}))"
        ),
        nbf.v4.new_code_cell(
            "selected_c = (\n"
            "    fold_metrics.groupby(['fold', 'selected_c']).size()\n"
            "    .unstack(fill_value=0)\n"
            "    .sort_index(axis=1)\n"
            ")\n"
            "display(selected_c)\n"
            "print('Grid-selected C values across 125 fold-label fits:', "
            "fold_metrics['selected_c'].value_counts().sort_index().to_dict())"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            f"- The headline result is reproduced directionally and numerically: "
            f"macro BA is **{reproduced_macro:.3f}** versus **"
            f"{published_macro:.3f}** from the published rounded per-label table "
            f"(difference {macro_delta:+.3f}).\n"
            f"- **{rounded_matches}/25** labels match at the precision published "
            f"in the supplement; **{within_two_points}/25** differ by no more than "
            "0.02.\n"
            f"- The largest gap is **{largest_gap_row['pretty_label']}** "
            f"({largest_gap_row['EF']:.2f} published versus "
            f"{largest_gap_row['balanced_accuracy']:.3f} reproduced). Since the "
            "data counts and folds reconcile exactly, the most plausible remaining "
            "causes are the unpublished random split seed and numerical/software "
            "version differences.\n"
            "- This run supports the paper’s claim for early sensor fusion. It does "
            "not reproduce the late-fusion (`LFA`/`LFL`) columns; the CLI includes "
            "the six single-sensor baselines, while a faithful learned late-fusion "
            "stack would require an additional choice not fully specified in the "
            "paper (how meta-model training probabilities were generated)."
        ),
        nbf.v4.new_markdown_cell(
            "### Re-run\n\n"
            "From the repository root:\n\n"
            "```bash\n"
            "uv sync --extra dev\n"
            "uv run extrasensory-reproduce download\n"
            "uv run extrasensory-reproduce prepare\n"
            "uv run extrasensory-reproduce profile\n"
            "uv run extrasensory-reproduce train --models EF --jobs 4\n"
            "uv run python scripts/build_notebook.py\n"
            "uv run jupyter nbconvert --execute --to notebook --inplace "
            "notebooks/01_paper_reproduction.ipynb\n"
            "```\n\n"
            f"Recorded run: Python `{metadata['python'].split()[0]}`, "
            f"NumPy `{metadata['numpy']}`, pandas `{metadata['pandas']}`, "
            f"scikit-learn `{metadata['scikit_learn']}`."
        ),
    ]
    notebook["cells"] = cells
    nbf.write(notebook, notebook_path)
    print(f"Wrote {notebook_path}")


if __name__ == "__main__":
    main()
