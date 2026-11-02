# Limitations and manuscript correction checklist

The repository improves traceability and calculation checks, not the experimental design after the fact. A journal submission should distinguish verified saved-output consistency, historical metadata claims, author attestations and evidence that is still missing. The supplied manuscript and results remain unchanged so disagreements are visible.

## Scientific interpretation

- **Unpaired datasets:** emotion classification uses Reddit-derived text; educational prediction uses separate learner interactions. There are no matched learner-level affect/knowledge observations establishing real multimodal personalization.
- **Behavior is not emotion:** recent errors, changes in correctness and previous response time are behavioral proxies. The Phase III ROC-AUC improvement is not evidence of emotion-aware fusion.
- **Template scope:** responses are deterministic authored templates over three topics. There is no evaluated generative LLM, open-ended dialogue capability or autonomous pedagogical planner.
- **Synthetic scenarios:** the 24 utterances are researcher-authored. They are not an authentic student dialogue test set, a population sample, or evidence of affect-recognition accuracy in education.
- **Outcomes:** rubric ratings assess response quality. There is no measured learning gain, retention, transfer, engagement improvement, classroom deployment or prospective safety validation.
- **Generalization:** one saved model run, one educational data source, a small scenario set and two submitted evaluator workbooks do not establish broad robustness.

## Policy corrections needed before submission

### Nominal medium versus executed high

The supplied quantiles are low 0.4306987486, medium 0.8367195490 and high 0.9102973387. The code's high threshold is greater than 0.75, so every nominal-medium condition executes the high branch when knowledge is enabled. The nominal scenario counts are 11 low, 6 medium and 7 high; actual knowledge-conditioned bands are 11 low, 0 medium and 13 high.

Describe nominal and executed bands separately. Do not claim that the saved knowledge-conditioned study covers all three executed policy levels. Changing quantiles/thresholds is a legitimate follow-up experiment, but requires regeneration and reevaluation; it is not a documentation-only correction to these results.

### Distress override and action-independence wording

The original policy contains a negative-emotion high-band override that replaces a challenge with a hint. All 24 saved combined-versus-knowledge-only action comparisons are identical because this override does not activate in those cases.

The manuscript's equation `a = g(q)` and broad action-independent wording omit this implemented branch. A faithful description is “knowledge normally selects action; predicted distress can override a high-band challenge, although no such override occurred in the evaluated cases.” Do not infer a universal guarantee from an inactive branch.

### Endpoint and statistical provenance

Historical Phase V preparation metadata declares helpfulness, combined versus knowledge-only, as the primary endpoint. Helpfulness is not statistically significant in the supplied final analysis. Affective appropriateness is the significant outcome among the four tested criteria; treat its status transparently rather than retrospectively relabeling it as the original primary endpoint.

The supplied notebook uses 5,000 bootstrap and 20,000 Monte Carlo sign-flip draws. The final manuscript states 30,000 and 200,000, but the exact corresponding final script and RNG contract are absent. Preserve the reported historical numbers as reported, and either recover that script or explicitly use the new declared reanalysis. Exact sign-flip results are an independent sensitivity analysis, not an exact recreation of historical Monte Carlo p-values.

## Evaluation provenance

- **Final versus preparatory evidence:** `research/evaluation/` contains complete final workbooks/CSVs. `research/results/phase5/run_metadata.json` still says no ratings and zero raters; that file describes the preparation stage and is superseded for final rating counts, not silently rewritten.
- **Historical support scripts:** earlier scripts discuss missing/duplicate/95-row preparation problems. The final supplied workbooks each contain 96 rows and match their final CSVs. Do not merge evidence from different stages into a single alleged collection history.
- **Evaluator authentication:** confirm and document recruitment, expertise, independence, instructions, blinding, conflicts, rating dates and any revision procedure. Consistent files do not prove independent collection.
- **Ethics and consent:** authors must establish relevant institutional approval/exemption and consent obligations. This repository supplies no invented approval number or retrospective consent statement.
- **Small paired sample:** inference has 24 scenario pairs, not 192 independent experimental units or 768 independent observations. Ratings share scenarios and templates; findings should not be generalized to student populations without new data.

## Reproducibility and release gaps

- **Missing checkpoint:** trained RoBERTa weights are absent, limiting direct reconstruction of Phase II/IV inference.
- **Private table:** withholding the educational prediction rows limits public Phase III score/cluster-bootstrap reconstruction; aggregate checks do not remove that limitation.
- **Environment:** inherited versions are recorded claims, not verified lockfiles. Training has not been rerun across seeds or hardware.
- **Model artifact safety:** joblib availability is not a guarantee of compatibility, harmless deserialization or dataset/model distribution rights.
- **References:** the supplied bibliography and reference-verification notes are preserved, but no independent full bibliography audit, SCI indexing verification or Q1 certification is asserted.
- **Archival status:** a public GitHub repository is not an immutable DOI-bearing archive. Select a final reviewed commit and approved release before adding a permanent manuscript citation.

## Author review before journal submission

- [ ] Correct policy/action wording and report actual executed knowledge bands.
- [ ] Preserve helpfulness as the historical primary endpoint and label affective-fit inference appropriately.
- [ ] Reconcile statistical draw counts, scripts, seeds and final manuscript tables.
- [ ] Verify evaluator provenance, ethics, consent, blinding and conflicts.
- [ ] Verify educational dataset identity and redistribution terms; review derived artifacts as needed.
- [ ] Confirm software, manuscript and model licensing with all rights holders.
- [ ] Supply or explain missing checkpoints and rerun full training if exact reconstruction is claimed.
- [ ] Audit all references, author metadata, affiliations, declarations and venue requirements.
- [ ] Freeze a reviewed commit/release and update the manuscript code-availability statement.

Completing repository engineering alone does not establish journal readiness or acceptance. The remaining items require scientific, legal or author-level judgment.
