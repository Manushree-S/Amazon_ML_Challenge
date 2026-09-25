# Experiment & Validation Notes

## 1. Training Summary
- **S1 Training Sample**: 3,000 reference entities loaded from ground truth.
- **Total Training Pairs**: 19,199 pairs.
  - **Positive pairs**: 10,305 pairs (true matches verified in S2/S3).
  - **Hard negative pairs**: 8,894 pairs (mined directly from unmatched blocking candidates produced by the inverted index, not random negatives).
- **Model**: LightGBM Classifier (`LGBMClassifier`, 300 estimators, learning rate 0.05, max depth 6, class_weight='balanced').
- **Top 5 Feature Importances**:
  1. `combined_token_jaccard`: 883
  2. `name_jaro_winkler`: 882
  3. `name_token_sort_ratio`: 857
  4. `name_char3_jaccard`: 837
  5. `tfidf_cosine_sim`: 828

---

## 2. Holdout Validation & Threshold Sweep
- **Holdout Set**: 500 S1 entities held out from training (`offset=3500`, `size=500`).
- **Blocking Candidates Generated**: Average 15.21 candidates per S1 entity.
- **Blocking Recall Ceiling**: **84.80%** (1,473 true matches captured out of 1,737 total true matches).

### Threshold Sweep on Holdout Set:
| Threshold | Macro \(F_{0.5}\) | Precision | Recall | Singleton Accuracy | Matched \(F_{0.5}\) |
|---|---|---|---|---|---|
| 0.50 | 0.9241 | 0.9643 | 0.8505 | 0.9615 | 0.9220 |
| 0.55 | 0.9241 | 0.9643 | 0.8505 | 0.9615 | 0.9220 |
| 0.60 | 0.9250 | 0.9654 | 0.8505 | 0.9615 | 0.9230 |
| 0.65 | 0.9249 | 0.9654 | 0.8502 | 0.9615 | 0.9229 |
| 0.70 | 0.9253 | 0.9659 | 0.8502 | 0.9615 | 0.9233 |
| **0.75** | **0.9258** | **0.9666** | **0.8502** | **0.9615** | **0.9239** |
| 0.80 | 0.9256 | 0.9666 | 0.8497 | 0.9615 | 0.9236 |
| 0.85 | 0.9252 | 0.9666 | 0.8490 | 0.9615 | 0.9232 |

- **Chosen Optimal Threshold**: **0.75** (maximizes Macro \(F_{0.5}\) by favoring high precision of 0.9666 over spurious merges).
- **Singleton Accuracy**: 96.15% (25 of 26 true singletons correctly predicted as empty).
- **Matched Entity \(F_{0.5}\)**: 0.9239.

---

## 3. Test Inference & Output Validation
- **Outputs Generated**:
  - `output/matching_results.tsv`: 1,732,544 rows (1,148 non-empty, 1,731,396 empty singletons).
  - `output/candidate_pairs.tsv`: 1,732,544 rows (4,987 non-empty, 1,727,557 empty singletons).
- **Submission Validation (`utils/validate_submission.py`)**:
  - Standard run: **`PASS — no blocking issues found. Safe to submit.`**
  - Full ID-existence check (`--check-ids` against all 9,969,589 test S2/S3 IDs): **`PASS`**.
  - All matched IDs verified to be strict subsets of candidates and exist in test source files.
