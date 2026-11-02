# Reproduction levels and operating instructions

This repository separates arithmetic verification, policy replay and complete experimental retraining. Those activities have different inputs and cannot be represented as interchangeable evidence. Original inherited files are immutable within this release.

## Public saved-output verification

Install `requirements.txt` in an isolated Python 3.11+ environment. These dependencies intentionally exclude training frameworks and serialized-model loading.

```bash
python scripts/verify_integrity.py
python scripts/reproduce.py --output analysis/public-reanalysis.json
python scripts/create_figures.py
python -m unittest discover -s tests -v
```

The integrity script checks all 98 preserved public files against their recorded SHA-256 values, checks the supplied original manifest while accounting for one declared exclusion, and rejects the withheld prediction table if it appears in its original public location. The reanalysis script uses assertions to stop on mismatched metrics or response replay rather than silently accepting disagreement.

The public output marks Phase III prediction verification “not rerun.” The committed `analysis/reanalysis.json` records the maintainer-side aggregate check performed using the privately retained table; it is an audit result, not a substitute for making that table publicly reproducible.

`scripts/create_figures.py` reads the committed aggregate report and preserved aggregate model table. It writes the five PNG/SVG pairs under `figures/`; figure generation is not a new experiment. The original graphs and vector paper figures remain under `research/`.

## Authorized private Phase III verification

If you lawfully hold the exact excluded prediction table, keep it outside the repository. Supply its path explicitly and write only aggregate output:

```bash
python scripts/reproduce.py \
  --phase3-predictions /authorized/private/test_predictions_anonymized.csv \
  --output analysis/private-input-aggregate-check.json
```

The excluded table's expected SHA-256 is `cd638c5f398a3b2d1a1f3868ff4c8e337a86478190135f9c090e8893cde90b40`. The script computes its hash and exports no learner identifiers, skills, source row indices, outcomes or per-row probabilities. The declared release exclusion is in `provenance/release-exclusions.json`.

The new 500-replicate cluster bootstrap uses seed 44 and lexicographic pseudonym order. It need not reproduce the old pre-anonymization RNG sequence; compare the estimand, interval and interpretation, not only the last decimal.

## Complete experimental retraining

Open the preserved notebooks in phase order after reviewing their cells. They are Colab-oriented, contain download/install cells, and may write to `/content` or request artifacts from previous phases. They are not claimed to be unattended local command-line programs.

- **Phase I:** obtain the official simplified GoEmotions files, apply the recorded single-label filter, run validation C selection and the final baseline fit, and save the generated archive.
- **Phase II:** obtain the same data and the base RoBERTa model, use a compatible PyTorch/Transformers environment and suitable compute, and retain the selected checkpoint. This repository does not contain that checkpoint.
- **Phase III:** independently verify permission to obtain/use the educational source, verify its content hash, then run cleaning, lagged-feature construction, learner splits, ablations and cluster bootstrap.
- **Phase IV:** supply the relevant model/checkpoint artifacts according to notebook prompts. Preserve the original probability conditions if reproducing the submitted responses; changing thresholds or quantiles defines a new experiment.
- **Phase V:** the supplied workbooks are completed historical evaluations. Reusing them reproduces their analysis, not a new blinded study. New studies need new response IDs/order, a separately controlled key and an appropriate consent/ethics process.

No GPU training, dataset downloads, checkpoint recovery or historical joblib deserialization was performed by the added verification layer. Model training should be rerun in a separately recorded environment, not inferred from the existence of saved prediction files.

## Environment boundaries

The inherited `research/environment/requirements.txt` is preserved as evidence but is not a lockfile and contains `python>=3.10`, which is not a valid Python-runtime installation instruction for pip. Do not install it blindly as the new analysis environment.

The new root `requirements.txt` defines compatible dependency ranges. `provenance/analysis-environment.txt` records the locally observed versions used for the added check; it is not an independently established reconstruction of the original training runtime. Historical training version/GPU claims are preserved in each phase's `run_metadata.json`, not certified by this repository.

For stronger archival reproducibility, rerun training, record exact package wheels and CUDA/runtime details, preserve the RoBERTa checkpoint with permission, and archive an approved immutable release. A fixed Git commit and an eventual DOI improve citation stability, but do not resolve data rights or scientific validity.

## Safety and failure modes

- **Serialized pipelines:** `.joblib` files can execute code when loaded and are version-sensitive. They are preserved, not unpickled automatically. Load only trusted artifacts in an isolated compatible environment.
- **Historical support scripts:** `research/code/support/` contains paper/notebook builders and preparation-stage analyses. Some refer to obsolete paths or inputs; they are not the supported reproduction entry points.
- **Missing private table:** do not fabricate rows, silently substitute another dataset or imply public full reconstruction. The public script reports the missing verification level explicitly.
- **Numerical variation:** percentile intervals and floating-point results can vary by RNG ordering or package implementation. New seeds and test definitions are stated in `scripts/reproduce.py` and `docs/RESULTS.md`.
- **Changing policy behavior:** correcting the nominal-medium mismatch or removing the distress override would change the evaluated system. Keep corrections on a new version and regenerate responses and ratings if making claims about that version.
