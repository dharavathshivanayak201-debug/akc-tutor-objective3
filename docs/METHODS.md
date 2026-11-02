# Methods and executable code map

The five inherited notebooks define the supplied experiments. The portable scripts add narrowly scoped verification of their outputs; they do not replace the notebooks with an undocumented new training pipeline. Refer to `research/paper/AKC_Tutor_IEEE_Full_Paper.pdf` and `.tex` for the preserved manuscript.

## Experimental architecture

```text
Official single-label GoEmotions subset
    -> TF-IDF + logistic regression / RoBERTa
    -> text-label predictions

Separate educational interaction records
    -> past-only knowledge and behavior-history features
    -> next-correctness probability

Researcher-authored scenario text + illustrative probability
    + saved predicted emotion
    -> four deterministic template conditions
    -> blinded response IDs
    -> two submitted rating workbooks
    -> scenario-paired response-quality comparisons
```

The scenario interface does not create matched learner-level emotion/knowledge observations. The educational and emotion classifiers therefore cannot be described as a jointly validated affect-aware learning model.

## Phase I: sparse-text emotion baseline

- **Source:** `research/code/phase1/AKC_Tutor_Phase1_Automatic_Colab.ipynb`.
- **Task:** 28-way, single-label emotion classification after excluding multilabel examples. The supplied subset sizes are 36,308 train, 4,548 development and 4,590 test.
- **Features and classifier:** TF-IDF word unigrams/bigrams, minimum document frequency 2 and maximum 100,000 features; logistic regression. Consult the notebook for complete vectorizer and solver settings.
- **Selection protocol:** compare regularization values by validation macro-F1, then refit the selected pipeline on train plus development. The saved selected C is 10. Test predictions are not used for this selection.
- **Outputs:** `results/phase1/` contains class counts, validation comparisons, a fitted joblib pipeline, confusion matrix, classification report and test predictions.
- **Added check:** `scripts/reproduce.py::phase12` maps string labels to the saved confusion-matrix label order, recomputes 28-label scores, and checks the seeded row-bootstrap interval against the saved interval.

The official GoEmotions repository describes the benchmark as Reddit-derived and documents 27 emotion categories plus neutral. This repository's single-label filtering is a local experimental choice, not the size or task definition of the full multilabel benchmark ([official GoEmotions repository](https://github.com/google-research/google-research/tree/master/goemotions)).

## Phase II: transformer emotion classifier

- **Source:** `research/code/phase2/AKC_Tutor_Phase2_RoBERTa_Automatic_Colab.ipynb`.
- **Configuration:** `FacebookAI/roberta-base`, maximum length 128, batch size 16, gradient accumulation 2, requested epochs 3, learning rate 0.00002, weight decay 0.01, warm-up ratio 0.1 and seed 42.
- **Selection protocol:** train on the training split and select the checkpoint using validation macro-F1. The supplied run metadata identifies epoch 3 as the selected checkpoint.
- **Protocol distinction:** the sparse baseline is refit on train plus development after C selection, while RoBERTa uses validation checkpoint selection without that refit. This is documented rather than silently treated as identical training exposure.
- **Outputs:** saved predictions, confidence values, training history, validation trajectory, confusion matrix and metric summaries. The trained checkpoint and tokenizer export are not present.
- **Added check:** confirm emotion names agree with label IDs, reconstruct scores/confusion counts, and reproduce the 1,000-resample macro-F1 interval from hard predictions.

This check does not establish that saved probabilities came from a particular checkpoint, that rerunning training will produce identical weights, or that a Reddit classifier is calibrated for learner affect.

## Phase III: educational correctness prediction

- **Source:** `research/code/phase3/AKC_Tutor_Phase3_Knowledge_Behavior_Fusion_Colab.ipynb`.
- **Recorded input:** the notebook records `as.csv` from the pyBKT examples repository. The exact recorded download and hash are in `research/results/phase3/run_metadata.json`; neither the filename nor the hosting repository alone establishes redistribution rights ([recorded data URL](https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv)).
- **Cleaning and ordering:** the saved metadata reports 525,534 raw and 446,844 cleaned records, 4,151 learners and 110 skills. The notebook sorts learner records by event order and original row order, then constructs lagged and cumulative history features.
- **Knowledge features:** `prior_attempts`, `prior_correct`, `prior_accuracy`, `prior_skill_attempts`, `prior_skill_correct`, `prior_skill_accuracy` and `prev_skill_correct`.
- **Behavior proxies:** recent error (`1 - recent2_accuracy`), change between preceding correctness values, and log-transformed previous response time when available. These are behavioral-history summaries, not emotion observations.
- **Pipeline:** median imputation with missingness indicators, standard scaling and logistic regression. C is selected from 0.01, 0.1, 1 and 10 using validation ROC-AUC.
- **Ablations:** knowledge only; behavior proxy only; combined; knowledge plus shuffled behavior; and constant training prevalence. Thresholded classification uses probability at least 0.5.
- **Split:** learner-disjoint train/validation/test, with 2,905/622/624 learners and 331,474/57,680/57,690 interaction rows. Histories of earlier test-learner interactions are available sequentially for later test predictions; “learner-disjoint” does not mean “no within-test history.”
- **Uncertainty:** the historical summary uses 500 learner-cluster bootstrap replicates for combined-minus-knowledge ROC-AUC.
- **Added check:** `phase3` recomputes ROC-AUC, accuracy, F1, log loss and Brier score for all five saved probability columns. It also resamples whole pseudonymous learners 500 times with seed 44 and publishes only aggregates.

The new bootstrap orders supplied pseudonyms lexicographically. Original pre-anonymization learner order is not retained as an RNG contract, so its numerical interval is a sensitivity rerun rather than an asserted exact replication of the historical resampling sequence.

## Phase IV: deterministic tutoring templates

- **Source:** `research/code/phase4/AKC_Tutor_Phase4_Automated_Colab.ipynb`.
- **Portable implementation:** `akc_tutor/policy.py` transcribes the supplied `LESSONS`, `ACK`, `NEG` and `respond` objects. It does not load emotion or knowledge models.
- **Inputs:** three authored mathematical topics, saved predicted emotion labels, and illustrative low/medium/high probabilities selected from Phase III held-out prediction quantiles. The probability is not estimated from the scenario speaker.
- **Knowledge rule:** probability below 0.45 selects low/worked-example; above 0.75 selects high/challenge; otherwise medium/hint. Values exactly 0.45 and 0.75 are medium.
- **Override:** when emotion conditioning is enabled and the label is in `NEG`, a high-band challenge is replaced with a hint. The reported band remains high; only content/action changes.
- **Tone:** an emotion-specific acknowledgement is prepended when the label is in `ACK`; otherwise a fixed neutral acknowledgement is used. Confidence is not consulted by `respond`.
- **Non-input:** `text` is retained in the function signature for parity, but the template function does not consume it. A separate classifier produced the saved emotion label.
- **Four conditions:** generic uses neither conditioning source; knowledge-only uses probability; emotion-only uses a fixed medium-band hint plus emotion wording; combined uses both sources and the possible override.
- **Added check:** `phase4` regenerates each of the 96 texts, actions and executed bands exactly from the saved records. Boundary and hypothetical distress-override cases have separate unit tests.

The original mechanical checks for topic grounding, follow-up question and policy conformity are template assertions. They are not independent measures of instructional validity or learning benefit.

## Phase V: paired response-quality evaluation

- **Source:** `research/code/phase5/AKC_Tutor_Phase5_Blinded_Evaluation_Colab.ipynb`; final supplied evidence is in `research/evaluation/`.
- **Unit:** 24 scenarios, each with four responses. Two workbooks supply one rating per response per evaluator, yielding 192 response-rating rows and 768 individual criterion scores.
- **Rubric:** integer scores 1–5 for correctness, helpfulness, knowledge appropriateness and affective appropriateness.
- **Blinding:** pseudonymous response IDs are mapped to variants using `research/results/phase5/PRIVATE_variant_key.csv`. This is a completed-study unblinding key, not evaluator identity information. Never distribute this public key to future blinded evaluators.
- **Paired estimand:** average the two evaluators within scenario/variant first; subtract knowledge-only from combined; analyze the resulting 24 scenario differences.
- **Agreement:** compare evaluators across the same 96 response IDs using exact agreement and quadratic-weighted Cohen's kappa. Kappa is agreement beyond marginally expected agreement, not evaluator accuracy.
- **New inference:** 30,000 scenario bootstrap draws per criterion, seeds 4200–4203 in the listed criterion order, percentile 95% intervals, exact two-sided sign-flip tests, and Holm adjustment across all four outcomes.
- **Exact test:** average ratings lie on half-point steps, so a signed-sum frequency table enumerates the sign-flip distribution without Monte Carlo error. Zeros do not alter that distribution.

Independence, expertise and collection procedure remain author-attested. The public verification checks consistency of the supplied records; it does not authenticate those attestations.
