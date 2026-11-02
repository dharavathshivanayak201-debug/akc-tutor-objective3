# Dataset provenance and access

## GoEmotions (Phases I-II)
Source: https://github.com/google-research/google-research/tree/master/goemotions
Publication: https://aclanthology.org/2020.acl-main.372/
Retrieve from the official repository and review the dataset's licensing, annotation provenance, and applicable Reddit data-use conditions before redistribution. Raw text and dataset copies are deliberately omitted from this package. Preserve the original train/dev/test partitions.

## pyBKT educational interactions (Phase III)
Project: https://github.com/CAHLR/pyBKT
Retrieve the example data using the notebook's stated source, and separately verify the underlying dataset's terms of access, privacy and redistribution; the pyBKT software license does not automatically license underlying student records. The cleaned interaction table is omitted because it contains learner-level records.

## Phase IV-V
24 researcher-authored scenarios and 96 generated responses are included in results/phase4. These are constructed scenarios, not real matched student-emotion interactions. Human ratings are included as anonymized workbooks with free-text comments and response text removed; use response IDs and phase4 responses to join.

## Model checkpoints
No large model weights are bundled. Phase II/IV notebook and emotion_checkpoint_provenance.json describe model initialization/provenance. Verify any referenced checkpoint is actually accessible; no persistent user-owned checkpoint URL has been independently verified.
