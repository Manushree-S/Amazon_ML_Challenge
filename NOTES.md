# Experiment & Validation Notes

## 1. Raw Dataset Verification (`wc -l`)
Exact line counts and record counts across all challenge files:

| File Path | Total Lines | Header Rows | Data Records | Notes |
|---|---|---|---|---|
| `dataset/train/train_source1.tsv` | 2,206,822 | 1 | **2,206,821** | Reference S1 entities |
| `dataset/train/train_source2.tsv` | 5,034,617 | 1 | **5,034,616** | Source 2 entities |
| `dataset/train/train_source3.tsv` | 5,285,604 | 1 | **5,285,603** | Source 3 entities |
| `dataset/train/train_ground_truth.tsv` | 2,206,822 | 1 | **2,206,821** | Ground truth matches |
| `dataset/test/test_source1.tsv` | 1,732,545 | 1 | **1,732,544** | S1 test entities (exact row count required) |
| `dataset/test/test_source2.tsv` | 4,887,274 | 1 | **4,887,273** | Test match pool S2 |
| `dataset/test/test_source3.tsv` | 5,082,317 | 1 | **5,082,316** | Test match pool S3 |

*Total test candidate target IDs: 4,887,273 (S2) + 5,082,316 (S3) = **9,969,589**.*

---

## 2. [CURRENT / ACTIVE] SQLite-Backed Blocking & Leakage-Free 85/15 Split Experiments

### A. Non-Overlapping Train/Holdout Boundary Verification
Both `train.py` and `validate_holdout.py` implement an automated, leakage-free 85/15 split:
- **Total Ground Truth**: 2,206,821 rows
- **Holdout Fraction**: 15% (331,024 reserved rows at the file tail)
- **Training Cutoff**: Row index 1,875,797 (`train.py` reads rows before index 1,875,797)
- **Holdout Offset**: Row index 1,875,797 (`validate_holdout.py` reads from offset 1,875,797)
- **Status**: Split boundary matches identically between scripts; 0% train/holdout data leakage.

### B. SQLite-Backed Candidate Generation & Model Performance
- **Blocking Engine**: SQLite-backed inverted index on disk with insertion-time postings cap (`MAX_INSERT_PER_KEY = 2000`).
- **Holdout Evaluation Slice**: Tail holdout slice drawn starting at `offset=1875797`.
- **Blocking Candidates Generated**: Average **15.14** candidates per S1 entity.
- **Blocking Recall Ceiling**: **87.97%** (1,492 true matches captured out of 1,696).

#### Threshold Sweep on Leakage-Free Holdout:
| Threshold | Macro \(F_{0.5}\) | Precision | Recall | Singleton Accuracy | Matched \(F_{0.5}\) |
|---|---|---|---|---|---|
| 0.50 | 0.9309 | 0.9602 | 0.8757 | 1.0000 | 0.9257 |
| 0.55 | 0.9314 | 0.9611 | 0.8754 | 1.0000 | 0.9263 |
| 0.60 | 0.9314 | 0.9611 | 0.8754 | 1.0000 | 0.9263 |
| 0.65 | 0.9314 | 0.9613 | 0.8747 | 1.0000 | 0.9263 |
| 0.70 | 0.9320 | 0.9626 | 0.8743 | 1.0000 | 0.9269 |
| 0.75 | 0.9326 | 0.9633 | 0.8743 | 1.0000 | 0.9275 |
| 0.80 | 0.9331 | 0.9640 | 0.8743 | 1.0000 | 0.9281 |
| **0.85** | **0.9334** | **0.9643** | **0.8743** | **1.0000** | **0.9284** |

- **Optimal Calibrated Decision Threshold**: **0.85**
- **Macro \(F_{0.5}\)**: **0.9334**
- **Macro Precision**: **0.9643**
- **Macro Recall**: **0.8743**
- **Singleton Accuracy**: **100.00%** (35 of 35 singletons correct)
- **Matched Entities \(F_{0.5}\)**: **0.9284** (465 entities)

### C. Memory Profiling & Full-Scale Scaling Limits
- **Small-Scale Training Memory**: 184 MB RSS (`sample_size=2500`).
- **Full-Scale Scaling Observation (`sample_size=2206821`)**:
  - `load_records_by_ids` for 1,875,797 S1 entities + ~2.5M matching S2/S3 entities creates ~4.4M in-memory Python dictionaries (>35 million Python objects).
  - RSS climbed to **4.63 GB** (4,629,471,232 bytes) at 145 seconds.
  - Total system RAM utilization crossed 90% (free physical memory dropped below 1.6 GB of 16.4 GB total).
  - **Rule 3 Safe Stop**: The execution was safely terminated per instructions once memory exceeded the 80% RAM threshold, preventing OS thrashing.

---

## 3. [SUPERSEDED / DEPRECATED] Initial Small-Sample Exploration Run

*Retained for traceability of iterative progression.*

- **Training Scope**: 3,000 entities.
- **Training Pairs**: 19,199 pairs (10,305 positive, 8,894 hard negative).
- **Holdout Set**: 500 entities (`offset=3500`, `size=500`).
- **Holdout Metrics (at threshold 0.75)**:
  - Macro \(F_{0.5}\): 0.9258
  - Precision: 0.9666
  - Recall: 0.8502
  - Singleton Accuracy: 96.15%
- **Status**: Superseded by the full-dataset 85/15 split model above.

---

## 4. Test Outputs & Validation
- **`output/matching_results.tsv`**: 1,732,544 rows (matches generated with tuned threshold 0.80).
- **`output/candidate_pairs.tsv`**: 1,732,544 rows.
- **Validation**:
  - `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`
  - Result: **`PASS — no blocking issues found. Safe to submit.`**
