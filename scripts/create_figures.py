"""Generate PNG/SVG companion figures from checked results; never new experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE, TEAL, ORANGE, GRAY = "#28588f", "#187b70", "#b86127", "#677684"
VARIANTS = ["generic", "knowledge_only", "emotion_only", "knowledge_plus_predicted_emotion"]
NAMES = ["Generic", "Knowledge only", "Emotion only", "Combined"]
CRITERIA = ["correctness", "helpfulness", "knowledge_appropriateness", "affective_appropriateness"]
LABELS = ["Correctness", "Helpfulness", "Knowledge fit", "Affective fit"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "savefig.facecolor": "white"})


def save(fig, name, note):
    fig.text(.02, .02, note, fontsize=9, color="#455463", va="bottom")
    fig.tight_layout(rect=(0, .09, 1, .92))
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=220, metadata={"Creator": "AKC-Tutor reproducibility scripts"})
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = json.loads((ROOT / "analysis/reanalysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    fig.suptitle("Emotion classification: saved held-out predictions", fontsize=16, x=.02, ha="left")
    for i, (phase, label, color) in enumerate([("1", "TF-IDF + logistic regression", GRAY), ("2", "RoBERTa", BLUE)]):
        r = data["phase12"][phase]
        ci = r["macro_f1_bootstrap_95ci"]
        for ax, metric in zip(axes, ("accuracy", "macro_f1")):
            value = r[metric]
            ax.scatter(value, i, color=color, s=90, zorder=3)
            if metric == "macro_f1":
                ax.plot(ci, [i, i], color=color, linewidth=3)
            ax.text(value + .018, i, f"{value:.4f}", va="center", color=color)
    for ax, title in zip(axes, ("Accuracy", "Macro-F1 (95% bootstrap interval)")):
        ax.set(xlim=(0, .8), ylim=(-.6, 1.6), yticks=[0, 1],
               yticklabels=["TF-IDF + LR", "RoBERTa"], title=title)
        ax.grid(axis="x", alpha=.2)
    save(fig, "emotion_classification", "n = 4,590 per model; 28 labels; 1,000 row-bootstrap resamples, seed 42. No retraining.")

    table = pd.read_csv(ROOT / "research/results/phase3/model_comparison.csv").sort_values("auc")
    model_names = {
        "knowledge_plus_behavior_proxy": "Knowledge + behavior proxies",
        "knowledge_only": "Knowledge only", "behavior_proxy_only": "Behavior proxies only",
        "knowledge_plus_shuffled_behavior_control": "Knowledge + shuffled behavior",
        "constant_training_prevalence": "Constant training prevalence",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.suptitle("Educational prediction: learner-disjoint test results", fontsize=16, x=.02, ha="left")
    for i, row in enumerate(table.itertuples()):
        color = TEAL if row.model == "knowledge_plus_behavior_proxy" else GRAY
        ax.scatter(row.auc, i, s=85, color=color, zorder=3)
        ax.text(row.auc + .008, i, f"{row.auc:.4f}", va="center", color=color)
    ax.set(yticks=np.arange(len(table)), yticklabels=[model_names[m] for m in table.model],
           xlim=(.45, .9), xlabel="ROC-AUC (point axis shown from 0.45)", ylim=(-.6, 4.6))
    ax.axvline(.5, color="#bac2cb", linestyle="--")
    ax.grid(axis="x", alpha=.2)
    save(fig, "educational_prediction", "57,690 interactions / 624 held-out learners. Behavior proxies are not measured emotion.\n"
         "Supplied AUCs independently checked against privately held saved probabilities; no model retraining.")

    responses = pd.read_csv(ROOT / "research/results/phase4/all_variant_responses.csv")
    selected = responses[responses.variant == "knowledge_only"]
    levels = ["low", "medium", "high"]
    counts = pd.crosstab(selected.scenario_knowledge_level, selected.knowledge_level).reindex(index=levels, columns=levels, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("Policy audit: nominal conditions versus executed behavior", fontsize=16, x=.02, ha="left")
    axes[0].imshow(counts, cmap="Blues", vmin=0, vmax=13)
    for i in range(3):
        for j in range(3):
            value = counts.iloc[i, j]
            axes[0].text(j, i, str(value), ha="center", va="center", color="white" if value > 7 else "#243647", fontsize=16)
    axes[0].set(xticks=range(3), xticklabels=levels, yticks=range(3), yticklabels=levels,
                xlabel="Executed knowledge band", ylabel="Nominal scenario band")
    bottom = np.zeros(4)
    for action, color, label in [("worked_example", BLUE, "Worked example"), ("hint", GRAY, "Hint"), ("challenge", TEAL, "Challenge")]:
        values = [int(((responses.variant == v) & (responses.action == action)).sum()) for v in VARIANTS]
        axes[1].barh(np.arange(4), values, left=bottom, color=color, label=label)
        for i, value in enumerate(values):
            if value:
                axes[1].text(bottom[i] + value / 2, i, str(value), ha="center", va="center", color="white")
        bottom += values
    axes[1].set(yticks=range(4), yticklabels=NAMES, xlabel="Responses", xlim=(0, 24))
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, frameon=False, fontsize=9)
    save(fig, "policy_audit", "Nominal medium probability 0.8367 exceeds the code's high threshold (> 0.75).\n"
         "A distress override exists, but combined and knowledge-only select identical actions in all 24 saved scenarios.")

    rows = data["phase5"]["criteria"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    fig.suptitle("Combined minus knowledge-only: scenario-paired rating effects", fontsize=16, x=.02, ha="left")
    for i, row in enumerate(rows):
        ci, value = row["ci95_recomputed"], row["mean_difference"]
        color = TEAL if row["exact_holm_p"] < .05 else GRAY
        ax.plot(ci, [i, i], linewidth=3, color=color)
        ax.scatter(value, i, s=80, color=color, zorder=3)
        ax.text(.86, i, f'{value:+.3f}   Holm p = {row["exact_holm_p"]:.4f}', va="center", fontsize=10)
    ax.axvline(0, color="#738292", linestyle="--")
    ax.set(yticks=range(4), yticklabels=LABELS, xlabel="Mean rating-point difference on a 1–5 scale", xlim=(-.3, 1.6), ylim=(3.6, -.6))
    ax.grid(axis="x", alpha=.2)
    save(fig, "paired_rating_effects", "24 scenario pairs, ratings averaged across two supplied evaluators. 30,000 scenario-bootstrap resamples.\n"
         "New declared seeds 4200–4203; exact two-sided sign-flip tests and Holm across four criteria. Not learning gains.")

    means = pd.DataFrame(data["phase5"]["variant_means"]).T.reindex(index=VARIANTS, columns=CRITERIA)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.suptitle("Descriptive response ratings", fontsize=16, x=.02, ha="left")
    image = ax.imshow(means, cmap="Blues", vmin=1, vmax=5, aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{means.iloc[i,j]:.3f}", ha="center", va="center", color="white" if means.iloc[i,j] > 3.6 else "#243647", fontsize=13)
    ax.set(yticks=range(4), yticklabels=NAMES, xticks=range(4), xticklabels=LABELS)
    fig.colorbar(image, ax=ax, label="Mean score (1–5)", ticks=range(1, 6), shrink=.8)
    save(fig, "rating_means", "48 response-rating rows per variant: 24 scenarios × two evaluators. Descriptive averages, not independent learner outcomes.")
    print("Generated five companion figures, each as PNG and SVG.")


if __name__ == "__main__":
    main()
