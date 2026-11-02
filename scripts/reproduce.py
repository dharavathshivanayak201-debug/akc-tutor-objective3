"""Recompute saved-prediction metrics, replay policies, and reanalyse ratings.

Does not train models or unpickle joblib artifacts. A privately held Phase III
prediction table may be supplied for aggregate verification; it is never copied
into outputs. Original source files are never overwritten.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from akc_tutor.policy import VARIANTS, respond

RESEARCH = ROOT / "research"
CRITERIA = ["correctness", "helpfulness", "knowledge_appropriateness", "affective_appropriateness"]


def classification(y, pred, labels=28):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    cm = np.bincount(y * labels + pred, minlength=labels**2).reshape(labels, labels)
    tp = cm.diagonal()
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(labels), where=denominator != 0)
    return {"accuracy": float(np.mean(y == pred)), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.average(f1, weights=cm.sum(1))), "confusion_matrix": cm.tolist()}


def auc(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if not n1 or not n0:
        return float("nan")
    return float((rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary_metrics(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    pred = score >= .5
    tp = np.sum((y == 1) & pred)
    denom = np.sum(y == 1) + np.sum(pred)
    p = np.clip(score, 1e-15, 1 - 1e-15)
    return {"auc": auc(y, score), "accuracy": float(np.mean(pred == y)),
            "f1": float(2 * tp / denom) if denom else 0.,
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((score - y)**2))}


def exact_signflip(differences):
    # Ratings averaged across two raters lie on half-point steps.
    values = np.rint(2 * np.asarray(differences)).astype(int)
    if not np.allclose(values, 2 * np.asarray(differences), rtol=0, atol=1e-12):
        raise ValueError("Exact rating test expects half-point differences.")
    distribution = Counter({0: 1})
    for value in values[values != 0]:
        new = Counter()
        for total, count in distribution.items():
            new[total + int(value)] += count
            new[total - int(value)] += count
        distribution = new
    target = abs(int(values.sum()))
    return sum(c for total, c in distribution.items() if abs(total) >= target) / sum(distribution.values())


def holm(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    return adjusted


def phase12():
    output = {}
    for phase in (1, 2):
        path = RESEARCH / f"results/phase{phase}"
        data = pd.read_csv(path / "test_predictions_anonymized.csv")
        assert len(data) == 4590
        original_cm = pd.read_csv(path / "confusion_matrix.csv", index_col=0)
        ids = {label: i for i, label in enumerate(original_cm.index)}
        if phase == 1:
            y = data.true_label.map(ids).to_numpy()
            pred = data.predicted_label.map(ids).to_numpy()
        else:
            y = data.true_label_id.to_numpy()
            pred = data.pred_label_id.to_numpy()
            assert np.array_equal(data.true_emotion.map(ids), y)
            assert np.array_equal(data.pred_emotion.map(ids), pred)
        fresh = classification(y, pred)
        saved = json.loads((path / "test_metrics.json").read_text())
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            assert np.isclose(fresh[name], saved[name], rtol=1e-12, atol=1e-12), (phase, name)
        cm = original_cm.to_numpy()
        assert np.array_equal(cm, fresh["confusion_matrix"])
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(1000):
            ix = rng.integers(0, len(y), size=len(y))
            boot.append(classification(y[ix], pred[ix])["macro_f1"])
        ci = np.quantile(boot, [.025, .975]).tolist()
        assert np.allclose(ci, saved["macro_f1_bootstrap_95ci"], rtol=1e-12, atol=1e-12)
        fresh["macro_f1_bootstrap_95ci"] = ci
        fresh["bootstrap_replicates"] = 1000
        fresh["labels"] = list(ids)
        output[str(phase)] = {"n": len(data), **fresh, "scope": "Saved hard predictions; no model retraining."}
    return output


def phase3(path):
    if path is None:
        return {"status": "not rerun", "reason": "Learner-level prediction table intentionally withheld; supply --phase3-predictions from an authorized source."}
    expected = json.loads((ROOT / "provenance/release-exclusions.json").read_text())[0]["sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Unexpected private prediction input hash."
    data = pd.read_csv(path)
    table = pd.read_csv(RESEARCH / "results/phase3/model_comparison.csv").set_index("model")
    fresh = {}
    for model in table.index:
        metrics = binary_metrics(data.correct, data["p_" + model])
        for name, value in metrics.items():
            assert np.isclose(value, table.loc[model, name], rtol=1e-10, atol=1e-12), (model, name)
        fresh[model] = metrics
    delta = fresh["knowledge_plus_behavior_proxy"]["auc"] - fresh["knowledge_only"]["auc"]
    saved = json.loads((RESEARCH / "results/phase3/paired_bootstrap.json").read_text())
    assert np.isclose(delta, saved["delta_auc_fusion_minus_knowledge"], rtol=1e-12, atol=1e-12)
    y = data.correct.to_numpy()
    p1 = data.p_knowledge_plus_behavior_proxy.to_numpy()
    p0 = data.p_knowledge_only.to_numpy()
    learners = np.unique(data.user_id)
    indices = {u: np.flatnonzero(data.user_id.to_numpy() == u) for u in learners}
    rng = np.random.default_rng(44)
    boot = []
    for _ in range(500):
        sampled = rng.choice(learners, size=len(learners), replace=True)
        ix = np.concatenate([indices[u] for u in sampled])
        boot.append(auc(y[ix], p1[ix]) - auc(y[ix], p0[ix]))
    return {"status": "aggregate metrics matched", "n": len(data), "learners": int(data.user_id.nunique()),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "models": fresh,
            "delta_auc_fusion_minus_knowledge": delta,
            "cluster_bootstrap_ci95_recomputed": np.quantile(boot, [.025, .975]).tolist(),
            "cluster_bootstrap_replicates": 500, "cluster_bootstrap_seed": 44,
            "cluster_order": "Lexicographic order of supplied pseudonymous user_id values.",
            "reported_cluster_bootstrap_ci95": saved["learner_cluster_bootstrap_95pct_ci"],
            "scope": "Saved probabilities only. Training not rerun; no individual rows exported. Bootstrap reanalysis uses declared pseudonymous learner ordering, not an exact claim about the original pre-anonymization RNG sequence."}


def phase4():
    data = pd.read_csv(RESEARCH / "results/phase4/all_variant_responses.csv")
    assert len(data) == 96 and data.scenario_id.nunique() == 24
    for _, row in data.iterrows():
        kwargs = dict(zip(("use_knowledge", "use_emotion"), VARIANTS[row.variant]))
        fresh = respond(row.skill, row.student_text, row.knowledge_probability, row.pred_emotion, **kwargs)
        for name in ("response", "action", "knowledge_level"):
            assert fresh[name] == row[name], (row.scenario_id, row.variant, name)
    selected = data[data.variant == "knowledge_only"]
    observed = pd.crosstab(selected.scenario_knowledge_level, selected.knowledge_level)
    actions = data.pivot(index="scenario_id", columns="variant", values="action")
    changed = actions.knowledge_only != actions.knowledge_plus_predicted_emotion
    knowledge = json.loads((RESEARCH / "results/phase4/knowledge_state_provenance.json").read_text())
    return {"responses_replayed": len(data), "scenarios": 24,
            "knowledge_source": knowledge, "nominal_to_actual_counts": observed.to_dict(orient="index"),
            "combined_vs_knowledge_action_changes": int(changed.sum()),
            "scope": "Template/action replay from supplied probabilities and saved labels; no transformer inference."}


def phase5():
    ratings = pd.read_csv(RESEARCH / "evaluation/verified_human_ratings_anonymized.csv")
    assert len(ratings) == 192 and not ratings[CRITERIA].isna().any().any()
    assert not ratings.duplicated(["rater_id", "response_id"]).any()
    assert len(ratings.rater_id.unique()) == 2 and ratings.response_id.nunique() == 96
    for column in CRITERIA:
        assert ratings[column].isin(range(1, 6)).all()
    key = pd.read_csv(RESEARCH / "results/phase5/PRIVATE_variant_key.csv")
    assert len(key) == 96 and not key.response_id.duplicated().any()
    joined = ratings.merge(key, on="response_id", suffixes=("", "_key"), validate="many_to_one")
    assert len(joined) == len(ratings)
    assert not key.duplicated(["scenario_id", "variant"]).any()
    assert key.scenario_id.nunique() == 24 and key.variant.nunique() == 4
    for col in ("scenario_id", "variant"):
        if col + "_key" in joined:
            assert (joined[col] == joined[col + "_key"]).all()
    means = ratings.groupby(["scenario_id", "variant"])[CRITERIA].mean()
    saved = pd.read_csv(RESEARCH / "evaluation/paired_ablation_scenario_level.csv").set_index("criterion")
    output = []
    for i, column in enumerate(CRITERIA):
        pivot = means[column].unstack("variant")
        delta = (pivot.knowledge_plus_predicted_emotion - pivot.knowledge_only).to_numpy()
        assert len(delta) == 24
        assert np.isclose(delta.mean(), saved.loc[column, "mean_difference"], atol=1e-12)
        rng = np.random.default_rng(4200 + i)
        boot = delta[rng.integers(0, len(delta), size=(30000, len(delta)))].mean(axis=1)
        ci = np.quantile(boot, [.025, .975])
        output.append({"criterion": column, "mean_difference": float(delta.mean()),
                       "ci95_recomputed": ci.tolist(), "bootstrap_seed": 4200 + i,
                       "exact_signflip_p": exact_signflip(delta),
                       "reported_monte_carlo_p": float(saved.loc[column, "p_value"]),
                       "reported_holm_p": float(saved.loc[column, "holm_adjusted_p"]),
                       "positive": int((delta > 0).sum()), "negative": int((delta < 0).sum()),
                       "ties": int((delta == 0).sum())})
    for row, adjusted in zip(output, holm([r["exact_signflip_p"] for r in output])):
        row["exact_holm_p"] = float(adjusted)
    reliability = []
    saved_reliability = pd.read_csv(RESEARCH / "evaluation/inter_rater_reliability.csv").set_index("criterion")
    for column in CRITERIA:
        pair = ratings.pivot(index="response_id", columns="rater_id", values=column)
        a, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
        observed = np.mean((a - b)**2) / 16
        expected = np.mean((a[:, None] - b[None, :])**2) / 16
        kappa = float(1 - observed / expected) if expected else float("nan")
        assert np.isclose(kappa, saved_reliability.loc[column, "quadratic_weighted_kappa"], atol=1e-12)
        assert np.isclose(np.mean(a == b), saved_reliability.loc[column, "exact_agreement"], atol=1e-12)
        reliability.append({"criterion": column, "quadratic_weighted_kappa": kappa,
                            "exact_agreement": float(np.mean(a == b)),
                            "within_one_point": float(np.mean(abs(a - b) <= 1))})
    return {"ratings": len(ratings), "responses": 96, "scenarios": 24, "criteria": output,
            "variant_means": ratings.groupby("variant")[CRITERIA].mean().to_dict(orient="index"),
            "reliability": reliability,
            "scope": "Independent reanalysis of supplied anonymized ratings, not new ratings or independent authentication of their collection. Exact sign-flip p differs from historical Monte Carlo p; CIs use a new declared seed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-predictions", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/reanalysis.json")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((ROOT / "research").resolve()):
        raise ValueError("Do not overwrite preserved research inputs.")
    results = {"phase12": phase12(), "phase3": phase3(args.phase3_predictions),
               "phase4": phase4(), "phase5": phase5()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("Phase I/II predictions verified; 96 policy responses replayed; 192 ratings reanalysed.")
    print("Phase III:", results["phase3"]["status"])
    print("Output:", args.output)


if __name__ == "__main__":
    main()
