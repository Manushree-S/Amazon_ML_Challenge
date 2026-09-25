"""
ML Challenge 2026 - Business Entity Resolution
Holdout Validation Module

- Holds out a slice of train data (stratified by S1 entities)
- Runs multi-strategy blocking to generate candidate pairs
- Extracts pairwise features and computes classifier prediction probabilities
- Optimizes decision threshold for Macro F_0.5 (precision weighted 2x over recall)
- Evaluates Macro F_0.5 strictly adhering to challenge specifications (including singletons)
- Reports breakdown by singletons vs entities with true matches
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple, Any
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
    holdout_size: int = 500,
    holdout_offset: int = 3000,
    max_cands: int = 20,
    optimize_threshold: bool = True
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Run holdout validation and threshold tuning.
    Returns: (best_threshold, best_macro_f05, best_results_dict)
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

    print(f"Loading holdout ground truth (offset={holdout_offset}, size={holdout_size})...")
    gt_full_df = pd.read_csv(gt_path, sep="\t", skiprows=range(1, holdout_offset), nrows=holdout_size)

    gt_map = {}
    target_s1_ids = []
    target_match_ids = set()

    for _, row in gt_full_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        ids = {x.strip() for x in matched.split(",") if x.strip()} if matched.strip() else set()
        gt_map[s1] = ids
        target_s1_ids.append(s1)
        target_match_ids.update(ids)

    print(f"Holdout S1 entities: {len(target_s1_ids)}, True matches to retrieve: {len(target_match_ids)}")

    print("Retrieving S1 records from train_source1.tsv...")
    s1_records = load_records_by_ids(s1_path, set(target_s1_ids), additional_pool_size=0)
    print(f"Retrieved {len(s1_records)} S1 records.")

    print("Loading candidate records from S2 and S3...")
    s2_records = load_records_by_ids(s2_path, target_match_ids, additional_pool_size=holdout_size * 2)
    s3_records = load_records_by_ids(s3_path, target_match_ids, additional_pool_size=holdout_size * 2)

    record_lookup = {}
    record_lookup.update(s1_records)
    record_lookup.update(s2_records)
    record_lookup.update(s3_records)

    # Build country-partitioned candidate indices
    unique_countries = {r["country"] for r in s1_records.values() if r["country"]}
    indices_by_country = {}
    for c_str in unique_countries:
        index = CandidateIndex()
        for eid, r in record_lookup.items():
            if eid.startswith(("S2-", "S3-")) and r["country"] == c_str:
                index.add_record(eid, r["business_name"], r["business_address"], c_str)
        indices_by_country[c_str] = index

    # Generate candidates & predict probabilities
    print("\nRunning blocking and ML inference on holdout set...")
    candidate_probs: Dict[str, List[Tuple[str, float]]] = {}
    eval_s1_ids = [s1 for s1 in target_s1_ids if s1 in s1_records]

    total_candidates_count = 0
    true_matches_captured_by_blocking = 0
    total_true_matches_in_eval = 0

    for s1_id in tqdm(eval_s1_ids, desc="Scoring holdout"):
        s1_data = s1_records[s1_id]
        s1_country = s1_data["country"]
        true_matches = gt_map.get(s1_id, set())
        total_true_matches_in_eval += len(true_matches)

        index = indices_by_country.get(s1_country)
        if index is None:
            candidate_probs[s1_id] = []
            continue

        cands = generate_candidates_for_s1(
            s1_id=s1_id,
            s1_name=s1_data["business_name"],
            s1_address=s1_data["business_address"],
            s1_country=s1_country,
            index=index,
            max_candidates=max_cands
        )
        valid_cands = sanitize_id_list(cands, s1_id)
        total_candidates_count += len(valid_cands)

        # Blocking recall ceiling computation
        captured = len(set(valid_cands) & true_matches)
        true_matches_captured_by_blocking += captured

        if not valid_cands:
            candidate_probs[s1_id] = []
            continue

        batch_feats = []
        for cid in valid_cands:
            r2 = record_lookup[cid]
            feat = fe.extract_pair_features(
                name1=s1_data["business_name"],
                addr1=s1_data["business_address"],
                country1=s1_country,
                name2=r2["business_name"],
                addr2=r2["business_address"],
                country2=r2["country"],
            )
            batch_feats.append(feat)

        probs = model.predict_proba(np.array(batch_feats, dtype=np.float32))[:, 1]
        candidate_probs[s1_id] = list(zip(valid_cands, probs))

    # Blocking statistics
    avg_cands = total_candidates_count / len(eval_s1_ids) if eval_s1_ids else 0.0
    blocking_recall_ceiling = (true_matches_captured_by_blocking / total_true_matches_in_eval) if total_true_matches_in_eval > 0 else 1.0

    print(f"\n--- Blocking Performance on Holdout ---")
    print(f"Average candidates generated per S1 entity: {avg_cands:.2f}")
    print(f"Blocking recall ceiling (captured / true):  {blocking_recall_ceiling * 100:.2f}% ({true_matches_captured_by_blocking}/{total_true_matches_in_eval})")

    # Threshold sweep
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85] if optimize_threshold else [0.65]
    best_thresh = 0.65
    best_res = None
    best_f05 = -1.0

    print("\n--- Threshold Tuning for Macro F_0.5 ---")
    print(f"{'Threshold':10s} | {'Macro F0.5':12s} | {'Precision':10s} | {'Recall':10s} | {'Singleton Acc':14s} | {'Matched F0.5':12s}")
    print("-" * 80)

    for th in thresholds:
        res = evaluate_threshold(eval_s1_ids, candidate_probs, gt_map, th)
        print(f"{th:<10.2f} | {res['macro_f05']:<12.4f} | {res['macro_precision']:<10.4f} | {res['macro_recall']:<10.4f} | {res['singleton_f05']:<14.4f} | {res['matched_macro_f05']:<12.4f}")
        if res["macro_f05"] > best_f05:
            best_f05 = res["macro_f05"]
            best_thresh = th
            best_res = res

    print("-" * 80)
    print(f"\nOptimal Decision Threshold: {best_thresh:.2f}")
    print(f"Overall Macro F_0.5:        {best_res['macro_f05']:.4f}")
    print(f"Macro Precision:            {best_res['macro_precision']:.4f}")
    print(f"Macro Recall:               {best_res['macro_recall']:.4f}")
    print(f"Singleton Score (Empty=1):  {best_res['singleton_f05']:.4f} (count: {best_res['singleton_count']})")
    print(f"Matched Entities F_0.5:     {best_res['matched_macro_f05']:.4f} (count: {best_res['matched_entity_count']})")
    print(f"Matched Entities Precision: {best_res['matched_macro_precision']:.4f}")
    print(f"Matched Entities Recall:    {best_res['matched_macro_recall']:.4f}")

    best_res["avg_candidates"] = avg_cands
    best_res["blocking_recall_ceiling"] = blocking_recall_ceiling
    return best_thresh, best_res["macro_f05"], best_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holdout Validation & Threshold Tuning")
    parser.add_argument("--train-dir", default="dataset/train", help="Directory with train TSVs")
    parser.add_argument("--model-path", default="models/classifier.joblib", help="Path to trained model")
    parser.add_argument("--holdout-size", type=int, default=500, help="Number of holdout entities")
    parser.add_argument("--holdout-offset", type=int, default=3000, help="Offset row index for holdout slice")
    parser.add_argument("--max-cands", type=int, default=20, help="Max candidates per entity")
    args = parser.parse_args()

    validate_pipeline(
        train_dir=args.train_dir,
        model_path=args.model_path,
        holdout_size=args.holdout_size,
        holdout_offset=args.holdout_offset,
        max_cands=args.max_cands,
    )
