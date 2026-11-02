# Data dictionary, provenance and public-release boundary

The uploaded reproducibility archive is the source of the inherited evidence. Its SHA-256 is `77c24c2798c89ea3211bb15d2a407b3092f5b84a1dd2d5ff495c8f0cf1c966ff`; it contained 99 files, including the original manifest. The public `research/` subtree preserves 98 files byte-for-byte and deliberately excludes one learner-level prediction table.

## Provenance records

- **`provenance/preserved-files.json`:** relative public path, byte count and SHA-256 for every inherited public file, including the original manifest.
- **`research/MANIFEST_SHA256.json`:** the supplied archive manifest, retained unchanged. It still lists the excluded table; this is expected and explicitly handled by the new verifier.
- **`provenance/release-exclusions.json`:** excluded path, byte count, hash and release rationale.
- **`provenance/workbook-consistency.json`:** workbook sheet/range checks against corresponding CSVs. `R1_ratings!A1:I97` and `R2_ratings!A1:I97` each contain one header plus 96 matching rating rows, with no formulas.
- **`analysis/reanalysis.json`:** newly generated aggregate numerical checks, methods, seeds and scope qualifiers. This is separate from the unchanged historical outputs.

## Emotion prediction tables

Phase I `test_predictions_anonymized.csv` contains `prediction_row`, `true_label` and `predicted_label`. Phase II contains row number, true/predicted IDs, corresponding emotion names and confidence. There are 4,590 rows in each; original Reddit text and usernames are not included in these prediction tables.

`prediction_row` preserves test ordering and is not a guarantee of unlinkability to an externally available dataset. Labels use the original 28-label order, with neutral last. These derived tables are released for arithmetic verification, not as a claim that all upstream data rights or privacy risks have been independently resolved.

The official source describes GoEmotions as Reddit-derived and documents the benchmark taxonomy and preparation; it should be consulted for use restrictions and limitations ([official GoEmotions repository](https://github.com/google-research/google-research/tree/master/goemotions)). This release does not treat a software license as proof of dataset redistribution permission.

## Educational interaction data and exclusion

The recorded Phase III source is `as.csv` from the pyBKT examples project, SHA-256 `f22e3fb7872c1784ce93b0f9ebabbe0cbcac4f896fd8b4a11667b9715d77dbdc` ([recorded source file](https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv)). The examples repository is a provenance lead, not a substitute for verifying the underlying educational dataset's identity and terms ([pyBKT examples repository](https://github.com/CAHLR/pyBKT-examples)).

The withheld `results/phase3/test_predictions_anonymized.csv` contains:

| Field | Meaning | Release concern |
|---|---|---|
| `user_id` | Pseudonymous learner cluster | Repeated interaction linkage |
| `skill_id` | Educational skill name | Contextual linkage |
| `row_order` | Original source row index | Direct source-record matching |
| `correct` | Observed response correctness | Learner-level outcome |
| Five `p_*` columns | Saved probability for each ablation | Individual prediction disclosure |

Pseudonymization does not remove the linkage provided by source indices. Raw educational records and this prediction table are not included publicly while redistribution and privacy review remain unresolved. Five-model aggregate scores, split counts, feature definitions, aggregate uncertainty and fitted pipelines remain included; pipeline availability does not resolve training-data licensing.

## Scenario and response tables

The scenarios are researcher-authored, not transcripts from observed learners. Relevant columns in `research/results/phase4/` are:

- **`scenario_id`:** authored scenario identifier; 24 unique IDs.
- **`skill` and `student_text`:** topic and authored utterance.
- **`scenario_knowledge_level`:** nominal condition label; do not confuse it with the executed band.
- **`knowledge_probability`:** illustrative quantile-derived value, not an estimate from this scenario speaker's history.
- **`author_intended_affect`:** design annotation, not observed psychological ground truth.
- **`pred_emotion` and `emotion_confidence`:** saved classifier output on the authored text; confidence is not used by the deterministic policy.
- **`variant`:** generic, knowledge-only, emotion-only or combined.
- **`response`, `action`, `knowledge_level`, `used_emotion`:** realized template output and actual policy state.

The four response variants produce 96 records. Automated scored tables add mechanical indicators and response word counts; those columns do not establish human-rated correctness or educational effectiveness.

## Rating files and completed blinding key

Each final evaluator workbook/CSV has nine columns: rater ID, response ID, scenario ID, skill, nominal scenario knowledge level and four integer rating criteria. The combined CSV additionally contains the unblinded variant. It has 192 rows and no missing criterion scores.

`R1` and `R2` are evaluator pseudonyms. Their identities, expertise, independence, evaluation times, consent and ethics status cannot be authenticated from the supplied anonymized files. Authors must verify those matters before submission.

`research/results/phase5/PRIVATE_variant_key.csv` is named as a private preparation artifact but maps response IDs to experimental conditions only. It is included for completed-study auditability; no evaluator identities are present in the mapping. This public release permanently unblinds these responses, so a future blinded study must use a new key and new presentation materials.

## Manuscript and contact information

The paper retains its supplied author names, affiliations, ORCIDs and manuscript contact addresses as publication-facing author information. No login passwords, access tokens or account credentials are part of this repository. The release does not assert that author metadata or declarations have been independently authenticated.
