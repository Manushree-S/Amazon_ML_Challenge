# Business Entity Resolution Pipeline

High-performance, offline machine learning pipeline for large-scale multi-source business entity resolution, developed for the **Amazon ML Challenge 2026**.

Optimized for **Macro \(F_{0.5}\)** scoring with full singleton support, open-label country handling, C++ fuzzy matching, multi-strategy inverted-index blocking, and LightGBM gradient boosted decision trees.

---

## 1. Project Directory Structure

```
.
├── output/
│   ├── matching_results.tsv       # Scored on leaderboard
│   └── candidate_pairs.tsv        # Blocking's final candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocess.py       # Normalization: legal suffixes, punctuation, abbreviations, postal codes
│       │   ├── blocking.py         # Multi-strategy blocking: token overlap, phonetic, postal, country partition
│       │   ├── features.py         # Pairwise similarity: RapidFuzz C++, Jaro-Winkler, Levenshtein, TF-IDF cosine
│       │   ├── train.py            # Positive pairs + hard negatives, LightGBM classifier training
│       │   ├── validate_holdout.py # Holdout validation, Macro F_0.5 scoring & threshold tuning
│       │   ├── predict.py          # End-to-end inference, candidate generation & leaderboard match export
│       │   └── utils.py            # Exact challenge Macro F_0.5 metric, ID sanitization, TSV helpers
│       ├── README.md               # End-to-end run instructions & documentation
│       └── requirements.txt        # Pinned dependency versions
├── Documentation_template.md       # Methodology and technical write-up
├── utils/
│   └── validate_submission.py      # Official challenge validation script
└── .gitignore                      # Excludes raw datasets and large output TSVs
```

---

## 2. Dataset Setup & Placement Instructions

The raw dataset is not committed to the repository due to size. Anyone reproducing the results must place the competition TSV files at these exact paths relative to the repository root before running the pipeline:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All source files are tab-separated (`sep="\t"`) containing:
- `entity_id`: prefixed by `S1-`, `S2-`, or `S3-`
- `business_name`
- `business_address`
- `country`: open string label (includes `US`, `India`, and `France` in the test set)

---

## 3. Environment Installation

Install the pinned dependencies into your Python 3.8+ environment:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 4. End-to-End Execution Guide

All commands can be executed directly from the repository root.

### Step 1: Preprocessing & Candidate Generation (Blocking)
To run candidate generation across sources:
```bash
python code/business_entity_resolution/src/blocking.py \
    --s1 dataset/train/train_source1.tsv \
    --s2 dataset/train/train_source2.tsv \
    --s3 dataset/train/train_source3.tsv \
    --output output/candidate_pairs.tsv \
    --max-candidates 20
```

### Step 2: Train Classifier with Hard Negatives
Trains the LightGBM entity resolution classifier:
```bash
python code/business_entity_resolution/src/train.py \
    --train-dir dataset/train \
    --model-output models/classifier.joblib \
    --sample-size 5000 \
    --max-negatives 3
```

### Step 3: Holdout Validation & Threshold Calibration
Evaluates Macro \(F_{0.5}\) on a held-out slice of training entities and tunes the decision threshold:
```bash
python code/business_entity_resolution/src/validate_holdout.py \
    --train-dir dataset/train \
    --model-path models/classifier.joblib \
    --holdout-size 1000 \
    --holdout-offset 5000 \
    --max-cands 20
```

### Step 4: Test Inference & Output Generation
Generates both `output/candidate_pairs.tsv` and `output/matching_results.tsv`:
```bash
python code/business_entity_resolution/src/predict.py \
    --test-dir dataset/test \
    --model-path models/classifier.joblib \
    --matching-output output/matching_results.tsv \
    --candidate-output output/candidate_pairs.tsv \
    --threshold 0.65 \
    --max-candidates 20
```

---

## 5. Submission Validation

Before submitting to the portal, verify compliance with all challenge constraints:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

To run with optional memory-intensive ID existence check across all 10M test IDs:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

**Expected Result:**
```
PASS — no blocking issues found. Safe to submit.
```

---

## 6. Hard Rules Enforced by Code

1. **Every S1 Entity Present**: All 1,732,544 Source 1 test entities appear exactly once in both output files.
2. **Strict S2/S3 ID Filtering**: ID lists only contain `S2-` and `S3-` entity IDs. Self-matches (`S1-`) are rejected.
3. **Candidate Subset Guarantee**: Every matched ID in `matching_results.tsv` is guaranteed to be present in `candidate_pairs.tsv` for that Source 1 entity.
4. **Clean Singletons**: Empty string (not "None" or null) for entities with no matches.
5. **Open Label Country**: Dynamically partitions and processes by country string, natively handling unseen countries (e.g., France).
6. **Fully Offline & Compliant**: 0 external API calls, 0 network lookups. Models are strictly MIT/Apache-2.0 licensed and <8B parameters.

---

## 7. Packaging Final Submission Zip

Package the project into the final challenge submission zip:

### Linux / macOS
```bash
zip -r <team_name>_submission.zip \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/ \
    Documentation_template.md
```

### Windows (PowerShell)
```powershell
Compress-Archive -Path output, code, Documentation_template.md -DestinationPath <team_name>_submission.zip
```
