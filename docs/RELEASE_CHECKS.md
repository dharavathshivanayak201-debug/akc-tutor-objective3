# Prepared-release verification record

This record describes checks completed on the prepared Objective 3 package. It does not imply that a public upload, GitHub CI run, journal review, independent training rerun or ethics audit has occurred.

## Completed locally

- **Input integrity:** all 98 inherited public files match their recorded byte counts and SHA-256 values. The supplied original manifest is preserved unchanged, with one documented withheld-table exception.
- **Emotion results:** both 4,590-row prediction sets reproduce accuracy, macro-F1, weighted F1, confusion matrices and the supplied 1,000-replicate macro-F1 bootstrap intervals.
- **Educational results:** an authorized private-input check of the exact withheld prediction table reproduces all five metrics for each of five ablations. A new 500-replicate learner-cluster sensitivity bootstrap uses declared ordering and seed; only aggregates are committed.
- **Response replay:** all 96 response texts, actions and executed knowledge bands match the deterministic portable policy.
- **Ratings:** 192 complete response-rating rows, two evaluators and 96 response IDs; no duplicate rater/response pairs and integer criterion scores from 1 to 5. Workbook cells match corresponding CSVs.
- **Statistics:** all four paired mean differences, exact agreement and quadratic-weighted kappa values match saved results. New declared-seed bootstrap intervals and exact sign-flip/Holm tests are included separately from the reported historical Monte Carlo results.
- **Tests:** 20 unit/integration checks pass locally, including policy threshold boundaries, the inactive distress-override branch, metric cases, preserved-input integrity, prediction bootstrap reconstruction, response replay and rating checks.
- **Notebook syntax:** Python cells from all five supplied notebooks compile after excluding single-line Colab shell/magic commands. No training cells were executed by this check.
- **Public-only dry run:** saved-output verification completes without the excluded educational table and explicitly reports Phase III prediction verification as “not rerun.”
- **Figures:** five companion figures were generated in PNG and SVG; all five PNGs were visually inspected for labeling and clipping.
- **Release boundary:** the original uploaded ZIP is not republished wholesale. The excluded learner-level table and local cache files are not included in the prepared archive.

## Not completed by these checks

Full training, GPU inference, checkpoint recovery, historical joblib loading, external evaluator authentication, upstream license clearance, complete bibliography verification, prospective learner evaluation and journal-readiness certification remain outside this verification.

The GitHub CI workflow is supplied but its successful execution must be verified after files are published. The prepared archive is a usable fallback if browser/CLI access prevents upload; a repository scaffold alone should not yet be cited as a populated code-availability resource.
