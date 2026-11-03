"""Build the reader-facing model-improvement study notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results" / "research"
    findings = json.loads((result_dir / "final_findings.json").read_text())
    primary = findings["primary_finding"]
    audit = findings["paired_audit_finding"]
    moe = findings["moe_development_only"]

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Improving ExtraSensory Context Recognition Without Subject Leakage\n\n"
            "A gated, user-independent comparison of temporal, multi-task, robust, "
            "and ensemble methods on the ExtraSensory paper protocol."
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"The promoted **causal TCN** reaches **"
            f"{primary['official_candidate_macro_balanced_accuracy']:.3f}** macro "
            "balanced accuracy on the official five-fold evaluation, versus **"
            f"{primary['official_baseline_macro_balanced_accuracy']:.3f}** for the "
            "existing paper logistic-regression reproduction, a benchmark gain of "
            f"**{primary['official_delta']:+.3f}**. On the sealed audit fold, the "
            f"paired gain over threshold-tuned logistic regression is **"
            f"{audit['delta']:+.3f}** (95% user-cluster bootstrap CI **"
            f"[{audit['user_cluster_bootstrap_95_ci'][0]:+.3f}, "
            f"{audit['user_cluster_bootstrap_95_ci'][1]:+.3f}]**). The interval "
            "crosses zero, so the audit supports the preregistered point-estimate "
            "gate but is not independently conclusive with only 12 users.\n\n"
            "The sensor-gated mixture of experts is even stronger on development "
            f"folds ({moe['macro_balanced_accuracy']:.3f}), but remains a "
            "development-only lead: the TCN had already consumed the one promotion "
            "fold, so promoting MoE afterward would be repeated holdout testing. "
            "Its missing-sensor degradation reduction was only "
            f"{moe['relative_degradation_reduction']:.1%}, below the preregistered "
            "20% gate, so reconstruction pretraining was not escalated."
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Context & Methods

                The original 2017 system fits one class-balanced logistic classifier
                per label on independent minutes. This study tests whether causal
                temporal context, shared multi-label representations, and
                sensor-specialized experts add signal while preserving the paper's
                user-independent evaluation.

                The workflow is deliberately gated:

                1. Development uses official folds 0–2 with short three-epoch neural
                   screens and config-only, single-coordinate mutations.
                2. One selected family may use fold 4 for promotion at 12 epochs.
                3. A passing promotion unlocks the sealed fold-3 audit.
                4. Only a passing audit unlocks the official five-fold run.

                Selection requires macro balanced-accuracy improvement while bounding
                calibration and low-performing-user regressions. Every train/test split
                is user-disjoint; causal histories are reset across users and gaps
                longer than 90 seconds.

                ### Key Assumptions

                1. The legacy-negative label policy is retained so results remain
                   comparable with the reproduced 2017 target.
                2. The archived one-minute features are the unit of analysis. The TCN
                   models up to 30 prior contiguous minutes, but it cannot recover
                   information absent from the public precomputed features.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                import json
                from pathlib import Path

                import matplotlib.pyplot as plt
                import numpy as np
                import pandas as pd
                import seaborn as sns
                from IPython.display import display

                sns.set_theme(style="whitegrid", context="notebook")
                ROOT = Path.cwd()
                if not (ROOT / "results").exists():
                    ROOT = ROOT.parent
                RESULTS = ROOT / "results" / "research"
                FIGURES = RESULTS / "figures"
                FIGURES.mkdir(parents=True, exist_ok=True)

                models = pd.read_csv(RESULTS / "model_comparison.csv")
                ablations = pd.read_csv(RESULTS / "ablation_comparison.csv")
                missing = pd.read_csv(RESULTS / "missing_sensor_comparison.csv")
                per_label = pd.read_csv(RESULTS / "official_per_label_comparison.csv")
                capabilities = pd.read_csv(RESULTS / "capability_matrix.csv")
                findings = json.loads((RESULTS / "final_findings.json").read_text())
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell(
            dedent(
                """
                profile = json.loads((ROOT / "results" / "data_profile.json").read_text())
                data_checks = pd.DataFrame(
                    {
                        "quantity": [
                            "participants",
                            "labeled minutes",
                            "six-sensor evaluation minutes",
                            "six-sensor participants",
                            "input features",
                        ],
                        "value": [
                            profile["users"],
                            profile["paper_labeled_examples"],
                            profile["core_complete_examples"],
                            profile["core_complete_users"],
                            profile["core_features"],
                        ],
                    }
                )
                display(data_checks)
                assert data_checks["value"].tolist() == [60, 308320, 176941, 51, 175]
                print("All official-population checks match the paper reproduction.")
                """
            )
        ),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            dedent(
                """
                official = models.query(
                    "stage == 'official' and model != "
                    "'Paper logistic regression reproduction'"
                ).copy()
                display(
                    official[
                        [
                            "model",
                            "macro_balanced_accuracy",
                            "macro_ece",
                            "user_macro_class_recall_p20",
                            "elapsed_seconds",
                        ]
                    ].style.format(
                        {
                            "macro_balanced_accuracy": "{:.3f}",
                            "macro_ece": "{:.3f}",
                            "user_macro_class_recall_p20": "{:.3f}",
                            "elapsed_seconds": "{:.1f}",
                        }
                    )
                )

                primary = findings["primary_finding"]
                assert np.isclose(
                    primary["official_delta"],
                    primary["official_candidate_macro_balanced_accuracy"]
                    - primary["official_baseline_macro_balanced_accuracy"],
                )
                print(
                    "Official benchmark delta versus the paper LR reproduction:",
                    f"{primary['official_delta']:+.4f}",
                )
                audit = findings["paired_audit_finding"]
                print(
                    "Paired sealed-audit delta:",
                    f"{audit['delta']:+.4f}",
                    "95% user-cluster CI:",
                    tuple(round(value, 4) for value in audit["user_cluster_bootstrap_95_ci"]),
                )
                print(primary["comparison_note"])
                print(primary["execution_note"])
                """
            )
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                plot_official = pd.concat(
                    [
                        pd.DataFrame(
                            {
                                "model": ["Original paper LR reproduction"],
                                "macro_balanced_accuracy": [
                                    float(
                                        pd.read_csv(ROOT / "results" / "model_averages.csv")
                                        .query("model == 'EF'")
                                        ["macro_balanced_accuracy"]
                                        .iloc[0]
                                    )
                                ],
                            }
                        ),
                        official[["model", "macro_balanced_accuracy"]],
                    ],
                    ignore_index=True,
                )
                fig, ax = plt.subplots(figsize=(9, 4.5))
                sns.barplot(
                    data=plot_official,
                    x="macro_balanced_accuracy",
                    y="model",
                    hue="model",
                    legend=False,
                    palette=["#9ca3af", "#4c78a8"],
                    ax=ax,
                )
                ax.set_xlim(0.74, 0.81)
                ax.set_xlabel("Macro balanced accuracy")
                ax.set_ylabel("")
                ax.set_title("Official five-fold performance")
                for container in ax.containers:
                    ax.bar_label(container, fmt="%.3f", padding=3)
                fig.tight_layout()
                fig.savefig(FIGURES / "official_model_comparison.png", dpi=160)
                plt.show()
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            "### Evidence gates\n\n"
            "The table separates stages intentionally. Development is for choosing "
            "ideas; promotion and audit are single-use checks, not extra tuning data."
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                gates = pd.DataFrame(
                    [
                        {
                            "stage": stage,
                            "candidate": row["candidate"],
                            "baseline": row["baseline"],
                            "delta": row["candidate"] - row["baseline"],
                        }
                        for stage, row in {
                            "development": {
                                "candidate": float(
                                    models.query(
                                        "stage == 'development' and model == 'Causal TCN'"
                                    )["macro_balanced_accuracy"].iloc[0]
                                ),
                                "baseline": float(
                                    models.query(
                                        "stage == 'development' and "
                                        "model == 'Threshold-tuned logistic regression'"
                                    )["macro_balanced_accuracy"].iloc[0]
                                ),
                            },
                            "promotion": {
                                "candidate": float(
                                    models.query(
                                        "stage == 'promotion' and model == 'Causal TCN'"
                                    )["macro_balanced_accuracy"].iloc[0]
                                ),
                                "baseline": float(
                                    models.query(
                                        "stage == 'promotion' and "
                                        "model == 'Threshold-tuned logistic regression'"
                                    )["macro_balanced_accuracy"].iloc[0]
                                ),
                            },
                            "audit": {
                                "candidate": float(
                                    models.query(
                                        "stage == 'audit' and model == 'Causal TCN'"
                                    )["macro_balanced_accuracy"].iloc[0]
                                ),
                                "baseline": float(
                                    models.query(
                                        "stage == 'audit' and "
                                        "model == 'Threshold-tuned logistic regression'"
                                    )["macro_balanced_accuracy"].iloc[0]
                                ),
                            },
                            "official": {
                                "candidate": primary[
                                    "official_candidate_macro_balanced_accuracy"
                                ],
                                "baseline": primary[
                                    "official_baseline_macro_balanced_accuracy"
                                ],
                            },
                        }.items()
                    ]
                )
                display(gates.style.format({column: "{:.3f}" for column in gates.columns[1:]}))
                """
            )
        ),
        nbf.v4.new_markdown_cell("### What helped on development folds"),
        nbf.v4.new_code_cell(
            dedent(
                """
                development = models.query("stage == 'development'").sort_values(
                    "macro_balanced_accuracy", ascending=False
                )
                display(
                    development[
                        [
                            "model",
                            "macro_balanced_accuracy",
                            "macro_ece",
                            "user_macro_class_recall_p20",
                            "protocol_note",
                        ]
                    ].style.format(
                        {
                            "macro_balanced_accuracy": "{:.3f}",
                            "macro_ece": "{:.3f}",
                            "user_macro_class_recall_p20": "{:.3f}",
                        }
                    )
                )

                fig, ax = plt.subplots(figsize=(9, 5))
                sns.barplot(
                    data=development,
                    x="macro_balanced_accuracy",
                    y="model",
                    hue="model",
                    legend=False,
                    color="#4c78a8",
                    ax=ax,
                )
                ax.set_xlim(0.64, 0.81)
                ax.set_xlabel("Development macro balanced accuracy")
                ax.set_ylabel("")
                ax.set_title("Development comparison (not confirmatory evidence)")
                fig.tight_layout()
                fig.savefig(FIGURES / "development_model_comparison.png", dpi=160)
                plt.show()
                """
            )
        ),
        nbf.v4.new_markdown_cell("### Temporal and robustness ablations"),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(
                    ablations.sort_values("macro_balanced_accuracy", ascending=False).style.format(
                        {
                            "macro_balanced_accuracy": "{:.3f}",
                            "delta_vs_default_tcn": "{:+.3f}",
                            "macro_ece": "{:.3f}",
                            "user_macro_class_recall_p20": "{:.3f}",
                            "elapsed_seconds": "{:.1f}",
                        }
                    )
                )
                print(
                    "The 30-minute history matters; label refinement is effectively neutral. "
                    "GroupDRO raises aggregate BA slightly but lowers the user P20 metric, "
                    "so it does not clearly achieve its robustness objective."
                )
                """
            )
        ),
        nbf.v4.new_markdown_cell("### Per-label official effects"),
        nbf.v4.new_code_cell(
            dedent(
                """
                ordered = per_label.sort_values("delta")
                fig, ax = plt.subplots(figsize=(9, 9))
                colors = np.where(ordered["delta"] >= 0, "#54a24b", "#e45756")
                ax.barh(ordered["label"], ordered["delta"], color=colors)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_xlabel("Causal TCN − paper LR reproduction balanced accuracy")
                ax.set_title("Official per-label change")
                fig.tight_layout()
                fig.savefig(FIGURES / "official_per_label_delta.png", dpi=160)
                plt.show()
                display(
                    pd.concat([ordered.head(5), ordered.tail(5)])
                    .sort_values("delta")
                    .style.format(
                        {
                            "causal_tcn_balanced_accuracy": "{:.3f}",
                            "paper_lr_reproduction_balanced_accuracy": "{:.3f}",
                            "delta": "{:+.3f}",
                        }
                    )
                )
                """
            )
        ),
        nbf.v4.new_markdown_cell("### Missing-sensor robustness"),
        nbf.v4.new_code_cell(
            dedent(
                """
                development_missing = missing.query("stage == 'development'").copy()
                robustness = (
                    development_missing.groupby("model", as_index=False)["degradation"]
                    .mean()
                    .rename(columns={"degradation": "average_degradation"})
                )
                display(robustness.style.format({"average_degradation": "{:.3f}"}))

                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=development_missing,
                    x="missing_sensor_group",
                    y="degradation",
                    hue="model",
                    ax=ax,
                )
                ax.axhline(0, color="black", linewidth=1)
                ax.set_ylabel("Macro BA degradation")
                ax.set_xlabel("Removed sensor group")
                ax.set_title("Development missing-sensor stress test")
                ax.tick_params(axis="x", rotation=35)
                fig.tight_layout()
                fig.savefig(FIGURES / "missing_sensor_robustness.png", dpi=160)
                plt.show()
                """
            )
        ),
        nbf.v4.new_markdown_cell("### Implemented versus executed"),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(capabilities)
                print(
                    "Implementation-only entries are not claimed as empirical wins. "
                    "They remain available for a future compute-authorized study."
                )
                """
            )
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                f"""
                ## Takeaways

                - The official five-fold causal TCN score is **{primary['official_delta']:+.3f}**
                  above the existing paper logistic-regression reproduction. This is a
                  benchmark comparison, not a paired confidence interval, because the
                  earlier reproduction did not retain row-level probabilities.
                - Against threshold-tuned logistic regression on the sealed audit,
                  the TCN improves the point estimate, calibration, and low-user P20.
                  Its 12-user cluster-bootstrap interval crosses zero, so uncertainty
                  remains and the result is not independently conclusive.
                - A 30-minute causal history carries useful information; shortening it
                  to five minutes costs about 0.008 macro BA. The low-rank label
                  refinement layer adds essentially nothing here.
                - The mixture of experts is the best development result, but it is
                  **not** a confirmed winner. A clean follow-up must start with a fresh
                  promotion protocol or an external dataset.
                - The missing-sensor robustness gate prevented an unsupported SSL
                  escalation. This is a useful negative result: more machinery was not
                  allowed to consume confirmatory evaluation merely because it existed.

                The main practical value is therefore specific: temporal modeling of
                contiguous, user-safe minute histories improves the ExtraSensory
                benchmark. Claims about MoE superiority, general deployment robustness,
                or real-time raw-sensor systems require new confirmatory evidence.
                """
            ).strip()
        ),
    ]

    output = root / "notebooks" / "02_model_improvement_study.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
