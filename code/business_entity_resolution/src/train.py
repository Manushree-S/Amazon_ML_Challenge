"""
ML Challenge 2026 - Business Entity Resolution
Model Training Module

- Builds positive pairs from ground truth (S1, matched_id)
- Efficiently retrieves true matching records across Source 2 and Source 3
- Generates hard negative pairs from unmatched candidates produced by blocking
- Extracts rich pairwise similarity features
- Trains a LightGBM classifier optimized for business entity resolution
- Serializes trained model to disk
"""

import os
import sys
import random
import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from tqdm import tqdm

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
        header_line = next(f, None)
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) < 4:
                # pad missing fields
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
    s1_df: pd.DataFrame,
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
    print("Building entity record lookup...")

    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"]).strip()
        record_lookup[eid] = {
            "business_name": str(row["business_name"]) if pd.notna(row["business_name"]) else "",
            "business_address": str(row["business_address"]) if pd.notna(row["business_address"]) else "",
            "country": str(row["country"]) if pd.notna(row["country"]) else "",
        }

    record_lookup.update(s2_records)
    record_lookup.update(s3_records)

    # Build country-partitioned candidate indices for negative generation
    unique_countries = s1_df["country"].fillna("").unique()
    indices_by_country: Dict[str, CandidateIndex] = {}

    for country in unique_countries:
        c_str = str(country)
        index = CandidateIndex()
        for eid, r in record_lookup.items():
            if eid.startswith(("S2-", "S3-")) and r["country"] == c_str:
                index.add_record(eid, r["business_name"], r["business_address"], c_str)
        indices_by_country[c_str] = index

    pairs = []
    total_pos = 0
    total_neg = 0

    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="Generating training pairs"):
        s1_id = str(row["entity_id"]).strip()
        s1_country = str(row["country"]) if pd.notna(row["country"]) else ""
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
                s1_name=str(row["business_name"]) if pd.notna(row["business_name"]) else "",
                s1_address=str(row["business_address"]) if pd.notna(row["business_address"]) else "",
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


def train_model(
    train_dir: str = "dataset/train",
    model_output_path: str = "models/classifier.joblib",
    sample_size: int = 5000,
    max_negatives_per_entity: int = 3
):
    """
    Train LightGBM entity resolution classifier.
    """
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Reading training S1 sample (sample_size={sample_size})...")
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=sample_size)
    gt_map = parse_ground_truth(gt_path, nrows=sample_size)

    # Collect all needed positive match IDs
    target_ids = set()
    for s1_id in s1_df["entity_id"]:
        s1_id_str = str(s1_id).strip()
        target_ids.update(gt_map.get(s1_id_str, set()))
    print(f"Target match IDs to retrieve from S2/S3: {len(target_ids)}")

    print("Scanning S2 and S3 for matched entities and background pool...")
    s2_records = load_records_by_ids(s2_path, target_ids, additional_pool_size=sample_size * 2)
    s3_records = load_records_by_ids(s3_path, target_ids, additional_pool_size=sample_size * 2)
    print(f"Retrieved {len(s2_records)} S2 records, {len(s3_records)} S3 records.")

    pairs, record_lookup = build_training_pairs(
        s1_df, s2_records, s3_records, gt_map,
        max_hard_negatives=max_negatives_per_entity
    )

    if not pairs:
        raise ValueError("No training pairs generated!")

    print("Extracting pairwise features...")
    fe = FeatureExtractor()
    X = []
    y = []

    for s1_id, cand_id, label in tqdm(pairs, desc="Extracting features"):
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
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Resolution Classifier")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory with train TSVs")
    parser.add_argument("--model-output", default="models/classifier.joblib", help="Output path for model")
    parser.add_argument("--sample-size", type=int, default=3000, help="Number of S1 training records to use")
    parser.add_argument("--max-negatives", type=int, default=3, help="Hard negatives per entity")
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        model_output_path=args.model_output,
        sample_size=args.sample_size,
        max_negatives_per_entity=args.max_negatives
    )
