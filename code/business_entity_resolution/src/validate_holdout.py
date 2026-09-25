"""
ML Challenge 2026 - Business Entity Resolution
Holdout Validation Module

- Holds out a slice of train data (stratified by S1 entities)
- Runs multi-strategy blocking to generate candidate pairs
- Extracts pairwise features and computes classifier prediction probabilities
- Optimizes decision threshold for Macro F_0.5 (precision weighted 2x over recall)
- Evaluates Macro F_0.5 strictly adhering to challenge specifications (including singletons)
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

try:
    from .preprocess import normalize_business_name, normalize_address
    from .blocking import CandidateIndex, generate_candidates_for_s1
    from .features import FeatureExtractor
    from .utils import compute_macro_f05, parse_ground_truth, sanitize_id_list
    from .train import load_records_by_ids
except ImportError:
    from preprocess import normalize_business_name, normalize_address
    from blocking import CandidateIndex, generate_candidates_for_s1
    from features import FeatureExtractor
    from utils import compute_macro_f05, parse_ground_truth, sanitize_id_list
    from train import load_records_by_ids


def evaluate_threshold(
    s1_ids: List[str],
    candidate_probs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    threshold: float
) -> Dict[str, float]:
    """
    Apply decision threshold to candidate match probabilities and compute Macro F_0.5.
    """
    predictions: Dict[str, Set[str]] = {}
    for s1 in s1_ids:
        cands_with_probs = candidate_probs.get(s1, [])
        matches = {cid for cid, p in cands_with_probs if p >= threshold}
        predictions[s1] = matches

    return compute_macro_f05(ground_truth, predictions, s1_ids)


def validate_pipeline(
    train_dir: str = "dataset/train",
    model_path: str = "models/classifier.joblib",
    holdout_size: int = 1000,
    holdout_offset: int = 5000,
    max_cands: int = 20,
    optimize_threshold: bool = True
) -> Tuple[float, float]:
    """
    Run holdout validation and threshold tuning.
    Returns: (best_threshold, best_macro_f05)
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first!")

    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    fe = FeatureExtractor()

    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    print(f"Loading holdout S1 entities (offset={holdout_offset}, size={holdout_size})...")
    s1_df = pd.read_csv(s1_path, sep="\t", skiprows=range(1, holdout_offset), nrows=holdout_size)
    gt_full_df = pd.read_csv(gt_path, sep="\t", skiprows=range(1, holdout_offset), nrows=holdout_size)

    gt_map = {}
    target_match_ids = set()
    for _, row in gt_full_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        ids = {x.strip() for x in matched.split(",") if x.strip()} if matched.strip() else set()
        gt_map[s1] = ids
        target_match_ids.update(ids)

    print(f"Holdout S1 entities: {len(s1_df)}, True matches to retrieve: {len(target_match_ids)}")
    print("Loading candidate records from S2 and S3...")
    s2_records = load_records_by_ids(s2_path, target_match_ids, additional_pool_size=holdout_size * 2)
    s3_records = load_records_by_ids(s3_path, target_match_ids, additional_pool_size=holdout_size * 2)

    record_lookup = {}
    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"]).strip()
        record_lookup[eid] = {
            "business_name": str(row["business_name"]) if pd.notna(row["business_name"]) else "",
            "business_address": str(row["business_address"]) if pd.notna(row["business_address"]) else "",
            "country": str(row["country"]) if pd.notna(row["country"]) else "",
        }
    record_lookup.update(s2_records)
    record_lookup.update(s3_records)

    # Build country-partitioned candidate indices
    unique_countries = s1_df["country"].fillna("").unique()
    indices_by_country = {}
    for country in unique_countries:
        c_str = str(country)
        index = CandidateIndex()
        for eid, r in record_lookup.items():
            if eid.startswith(("S2-", "S3-")) and r["country"] == c_str:
                index.add_record(eid, r["business_name"], r["business_address"], c_str)
        indices_by_country[c_str] = index

    # Generate candidates & predict probabilities
    print("\nRunning blocking and ML inference on holdout set...")
    candidate_probs: Dict[str, List[Tuple[str, float]]] = {}
    s1_ids = []

    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc="Scoring holdout"):
        s1_id = str(row["entity_id"]).strip()
        s1_country = str(row["country"]) if pd.notna(row["country"]) else ""
        s1_ids.append(s1_id)

        index = indices_by_country.get(s1_country)
        if index is None:
            candidate_probs[s1_id] = []
            continue

        cands = generate_candidates_for_s1(
            s1_id=s1_id,
            s1_name=str(row["business_name"]) if pd.notna(row["business_name"]) else "",
            s1_address=str(row["business_address"]) if pd.notna(row["business_address"]) else "",
            s1_country=s1_country,
            index=index,
            max_candidates=max_cands
        )
        valid_cands = sanitize_id_list(cands, s1_id)

        if not valid_cands:
            candidate_probs[s1_id] = []
            continue

        r1 = record_lookup[s1_id]
        batch_feats = []
        for cid in valid_cands:
            r2 = record_lookup[cid]
            feat = fe.extract_pair_features(
                name1=r1["business_name"],
                addr1=r1["business_address"],
                country1=r1["country"],
                name2=r2["business_name"],
                addr2=r2["business_address"],
                country2=r2["country"],
            )
            batch_feats.append(feat)

        probs = model.predict_proba(np.array(batch_feats, dtype=np.float32))[:, 1]
        candidate_probs[s1_id] = list(zip(valid_cands, probs))

    # Threshold sweep
    thresholds = np.arange(0.30, 0.95, 0.05) if optimize_threshold else [0.65]
    best_thresh = 0.65
    best_res = None
    best_f05 = -1.0

    print("\n--- Threshold Tuning for Macro F_0.5 ---")
    print(f"{'Threshold':10s} | {'Macro F0.5':12s} | {'Precision':10s} | {'Recall':10s} | {'Singleton Acc':14s}")
    print("-" * 65)

    for th in thresholds:
        res = evaluate_threshold(s1_ids, candidate_probs, gt_map, th)
        print(f"{th:<10.2f} | {res['macro_f05']:<12.4f} | {res['macro_precision']:<10.4f} | {res['macro_recall']:<10.4f} | {res['singleton_accuracy']:<14.4f}")
        if res["macro_f05"] > best_f05:
            best_f05 = res["macro_f05"]
            best_thresh = th
            best_res = res

    print("-" * 65)
    print(f"\nOptimal Decision Threshold: {best_thresh:.2f}")
    print(f"Optimal Macro F_0.5:        {best_res['macro_f05']:.4f}")
    print(f"Macro Precision:            {best_res['macro_precision']:.4f}")
    print(f"Macro Recall:               {best_res['macro_recall']:.4f}")
    print(f"Singleton Accuracy:         {best_res['singleton_accuracy']:.4f}")

    return best_thresh, best_res["macro_f05"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holdout Validation & Threshold Tuning")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory with train TSVs")
    parser.add_argument("--model-path", default="models/classifier.joblib", help="Path to trained model")
    parser.add_argument("--holdout-size", type=int, default=500, help="Number of holdout entities")
    parser.add_argument("--holdout-offset", type=int, default=2000, help="Offset row index for holdout slice")
    parser.add_argument("--max-cands", type=int, default=20, help="Max candidates per entity")
    args = parser.parse_args()

    validate_pipeline(
        train_dir=args.train_dir,
        model_path=args.model_path,
        holdout_size=args.holdout_size,
        holdout_offset=args.holdout_offset,
        max_cands=args.max_cands,
    )
