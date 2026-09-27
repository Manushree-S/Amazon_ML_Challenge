"""
ML Challenge 2026 - Business Entity Resolution
Model Training Module

- Builds positive pairs from ground truth (S1, matched_id)
- Retrieves true matching records across Source 2 and Source 3
- Generates hard negative pairs from unmatched candidates produced by blocking
- Extracts rich pairwise similarity features (RapidFuzz C++, TF-IDF cosine)
- Trains a LightGBM classifier optimized for business entity resolution
- Serializes trained model to disk
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from tqdm import tqdm
import psutil

process = psutil.Process(os.getpid())
peak_rss_mb = 0.0

def track_memory() -> float:
    global peak_rss_mb
    rss = process.memory_info().rss / (1024 * 1024)
    if rss > peak_rss_mb:
        peak_rss_mb = rss
    return rss

try:
    from .preprocess import normalize_business_name, normalize_address
    from .blocking import CandidateIndex, generate_candidates_for_s1
    from .features import FeatureExtractor, FEATURE_NAMES
    from .utils import parse_ground_truth
except ImportError:
    from preprocess import normalize_business_name, normalize_address
    from blocking import CandidateIndex, generate_candidates_for_s1
    from features import FeatureExtractor, FEATURE_NAMES
    from utils import parse_ground_truth


def load_records_by_ids(
    file_path: str,
    target_ids: Set[str],
    additional_pool_size: int = 5000
) -> Dict[str, Dict[str, str]]:
    """
    Scan a TSV source file to extract all target_ids plus up to additional_pool_size
    records for background blocking and negative sampling.
    """
    records = {}
    found_target = 0
    pool_added = 0

    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) < 4:
                parts = parts + [""] * (4 - len(parts))
            eid, name, addr, country = parts[0].strip(), parts[1], parts[2], parts[3].strip()

            if eid in target_ids:
                records[eid] = {
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                }
                found_target += 1
            elif pool_added < additional_pool_size:
                records[eid] = {
                    "business_name": name,
                    "business_address": addr,
                    "country": country,
                }
                pool_added += 1

            if found_target >= len(target_ids) and pool_added >= additional_pool_size:
                break

    return records


def build_training_pairs(
    s1_records: Dict[str, Dict[str, str]],
    s2_records: Dict[str, Dict[str, str]],
    s3_records: Dict[str, Dict[str, str]],
    gt_map: Dict[str, Set[str]],
    max_hard_negatives: int = 3,
    max_cands_blocking: int = 15
) -> Tuple[List[Tuple[str, str, int]], Dict[str, Dict[str, str]]]:
    """
    Construct positive pairs from ground truth and hard negative pairs from blocking candidates.
    Returns:
    - pairs: list of (s1_id, match_id, label)
    - record_lookup: dict mapping entity_id -> {business_name, business_address, country}
    """
    record_lookup = {}
    record_lookup.update(s1_records)
    record_lookup.update(s2_records)
    record_lookup.update(s3_records)

    # Build country-partitioned candidate indices for negative generation
    unique_countries = {r["country"] for r in s1_records.values() if r["country"]}
    indices_by_country: Dict[str, CandidateIndex] = {}

    for c_str in unique_countries:
        index = CandidateIndex()
        for eid, r in record_lookup.items():
            if eid.startswith(("S2-", "S3-")) and r["country"] == c_str:
                index.add_record(eid, r["business_name"], r["business_address"], c_str)
        indices_by_country[c_str] = index

    pairs = []
    total_pos = 0
    total_neg = 0

    pbar = tqdm(s1_records.items(), desc="Generating training pairs")
    for idx_item, (s1_id, s1_data) in enumerate(pbar):
        if idx_item % 5000 == 0:
            rss_mb = track_memory()
            pbar.set_postfix(rss_mb=f"{rss_mb:.1f}")
        s1_country = s1_data["country"]
        true_matches = gt_map.get(s1_id, set())

        # 1. Add positive pairs
        for mid in true_matches:
            if mid in record_lookup:
                pairs.append((s1_id, mid, 1))
                total_pos += 1

        # 2. Add hard negative pairs from blocking candidates
        index = indices_by_country.get(s1_country)
        if index is not None:
            cands = generate_candidates_for_s1(
                s1_id=s1_id,
                s1_name=s1_data["business_name"],
                s1_address=s1_data["business_address"],
                s1_country=s1_country,
                index=index,
                max_candidates=max_cands_blocking
            )

            # Filter candidates that are NOT in true matches
            hard_negs = [c for c in cands if c not in true_matches and c in record_lookup]
            selected_negs = hard_negs[:max_hard_negatives]
            for nid in selected_negs:
                pairs.append((s1_id, nid, 0))
                total_neg += 1

    print(f"Constructed {len(pairs)} pairs: {total_pos} positives, {total_neg} hard negatives.")
    return pairs, record_lookup


def count_gt_rows(gt_path: str) -> int:
    """Count data rows (excluding header) in the ground truth TSV without
    loading it fully into a DataFrame."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def train_model(
    train_dir: str = "dataset/train",
    model_output_path: str = "models/classifier.joblib",
    sample_size: int = 2500,
    max_negatives_per_entity: int = 3,
    holdout_fraction: float = 0.15
):
    """
    Train LightGBM entity resolution classifier.

    To prevent train/holdout leakage, a fixed fraction (holdout_fraction) of
    ground truth rows is ALWAYS reserved at the tail of the file and never
    read for training, regardless of sample_size. validate_holdout.py uses
    the same convention by default, so the two scripts agree on a
    non-overlapping split without needing manually-coordinated offsets.
    """
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    total_gt_rows = count_gt_rows(gt_path)
    train_cutoff = int(total_gt_rows * (1 - holdout_fraction))
    print(
        f"Ground truth has {total_gt_rows} total rows. Reserving last "
        f"{holdout_fraction*100:.0f}% ({total_gt_rows - train_cutoff} rows) as holdout "
        f"-- training will only draw from rows before index {train_cutoff}."
    )

    # HARD CAP on actual training entities used, independent of what sample_size
    # requests. Loading full record data (name/address/country) for every
    # eligible S1 entity plus its true matches does not scale to millions of
    # entities in plain Python dicts -- this is what caused the 4.6GB+ OOM at
    # full-dataset scale. A gradient-boosted classifier over ~21 hand-crafted
    # similarity features saturates well before millions of examples, so a
    # large *random* sample gives comparable model quality at a fraction of
    # the memory. MAX_TRAINABLE_ENTITIES below is deliberately generous
    # (200k) but still bounded; raise it only if you've confirmed available
    # RAM can absorb it (~2-3KB per entity across the record dicts + pairs).
    MAX_TRAINABLE_ENTITIES = 200000
    effective_sample_size = min(sample_size, train_cutoff, MAX_TRAINABLE_ENTITIES)
    if sample_size > MAX_TRAINABLE_ENTITIES:
        print(
            f"WARNING: requested sample_size={sample_size} exceeds the safe cap "
            f"of {MAX_TRAINABLE_ENTITIES} trainable entities. Capping to "
            f"{MAX_TRAINABLE_ENTITIES} and drawing a RANDOM sample from the "
            f"{train_cutoff} eligible rows (not just the first N) so the training "
            f"set stays representative. Override MAX_TRAINABLE_ENTITIES in code "
            f"if you have confirmed enough RAM for a larger run."
        )

    print(f"Reading eligible ground truth rows (up to cutoff {train_cutoff})...")
    gt_eligible_df = pd.read_csv(gt_path, sep="\t", nrows=train_cutoff)

    if effective_sample_size < len(gt_eligible_df):
        gt_df = gt_eligible_df.sample(n=effective_sample_size, random_state=42)
        print(f"Randomly sampled {effective_sample_size} of {len(gt_eligible_df)} "
              f"eligible rows for training.")
    else:
        gt_df = gt_eligible_df
    del gt_eligible_df

    gt_map = {}
    target_s1_ids = set()
    target_match_ids = set()

    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        ids = {x.strip() for x in matched.split(",") if x.strip()} if matched.strip() else set()
        gt_map[s1] = ids
        target_s1_ids.add(s1)
        target_match_ids.update(ids)

    print(f"Target S1 entities: {len(target_s1_ids)}, Target match IDs from S2/S3: {len(target_match_ids)}")

    print("Retrieving S1 records from train_source1.tsv...")
    s1_records = load_records_by_ids(s1_path, target_s1_ids, additional_pool_size=0)
    print(f"Retrieved {len(s1_records)} S1 records.")

    # Cap the background/negative pool at a fixed size regardless of sample_size.
    # This pool exists only to give the blocking index enough S2/S3 volume to
    # mine hard negatives from -- it does not need to scale with the number of
    # S1 training entities, and letting it do so (sample_size * 2, previously)
    # caused unbounded memory growth at full-dataset scale.
    pool_size = min(sample_size * 2, 50000)
    print(f"Scanning S2 and S3 for matched entities and background pool (pool cap={pool_size})...")
    s2_records = load_records_by_ids(s2_path, target_match_ids, additional_pool_size=pool_size)
    s3_records = load_records_by_ids(s3_path, target_match_ids, additional_pool_size=pool_size)
    print(f"Retrieved {len(s2_records)} S2 records, {len(s3_records)} S3 records.")

    pairs, record_lookup = build_training_pairs(
        s1_records, s2_records, s3_records, gt_map,
        max_hard_negatives=max_negatives_per_entity
    )

    if not pairs:
        raise ValueError("No training pairs generated!")

    print("Extracting pairwise features...")
    fe = FeatureExtractor()
    X = []
    y = []

    pbar_feat = tqdm(pairs, desc="Extracting features")
    for idx_pair, (s1_id, cand_id, label) in enumerate(pbar_feat):
        if idx_pair % 10000 == 0:
            rss_mb = track_memory()
            pbar_feat.set_postfix(rss_mb=f"{rss_mb:.1f}")
        r1 = record_lookup[s1_id]
        r2 = record_lookup[cand_id]
        feat = fe.extract_pair_features(
            name1=r1["business_name"],
            addr1=r1["business_address"],
            country1=r1["country"],
            name2=r2["business_name"],
            addr2=r2["business_address"],
            country2=r2["country"],
        )
        X.append(feat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Feature matrix shape: {X.shape}, Label distribution: {np.bincount(y)}")

    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X, y)

    # Display feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances:")
    for i in sorted_idx[:10]:
        print(f"  {FEATURE_NAMES[i]:25s}: {importances[i]}")

    os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"\nModel saved successfully to {model_output_path}!")
    print(f"Peak memory during training: {peak_rss_mb:.1f} MB")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Resolution Classifier")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory with train TSVs")
    parser.add_argument("--model-output", default="models/classifier.joblib", help="Output path for model")
    parser.add_argument("--sample-size", type=int, default=2500, help="Number of S1 training records to use (capped to respect holdout reserve)")
    parser.add_argument("--max-negatives", type=int, default=3, help="Hard negatives per entity")
    parser.add_argument("--holdout-fraction", type=float, default=0.15, help="Fraction of ground truth reserved as holdout tail, never used for training")
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        model_output_path=args.model_output,
        sample_size=args.sample_size,
        max_negatives_per_entity=args.max_negatives,
        holdout_fraction=args.holdout_fraction
    )
