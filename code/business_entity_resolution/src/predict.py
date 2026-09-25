"""
ML Challenge 2026 - Business Entity Resolution
Inference & Prediction Pipeline

- Executes multi-strategy blocking over the test set partitioned by country
- Computes pairwise feature vectors between Source 1 entities and generated candidates
- Evaluates trained LightGBM classifier probabilities
- Applies calibrated decision threshold optimized for Macro F_0.5
- Generates:
  1. output/candidate_pairs.tsv (blocking's final candidate set)
  2. output/matching_results.tsv (scored leaderboard file)
- Enforces all challenge hard rules:
  * Every Source 1 entity in test appears exactly once (no missing, no duplicates)
  * Matched IDs are strictly a subset of candidate IDs
  * S2- and S3- IDs only (never Source 1 self-matches)
  * No duplicate IDs within any comma-separated list
  * Clean tab-separation with empty string for singletons/unmatched entities
  * Open string label handling for country (including France, US, India)
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

try:
    from .preprocess import normalize_business_name, normalize_address
    from .blocking import CandidateIndex, generate_candidates_for_s1
    from .features import FeatureExtractor
    from .utils import sanitize_id_list
except ImportError:
    from preprocess import normalize_business_name, normalize_address
    from blocking import CandidateIndex, generate_candidates_for_s1
    from features import FeatureExtractor
    from utils import sanitize_id_list


def run_prediction_pipeline(
    test_dir: str = "dataset/test",
    model_path: str = "models/classifier.joblib",
    matching_output_path: str = "output/matching_results.tsv",
    candidate_output_path: str = "output/candidate_pairs.tsv",
    threshold: float = 0.75,
    max_candidates: int = 20,
    max_s1_eval: Optional[int] = None,
    s2_s3_pool_size: Optional[int] = None,
):
    """
    Execute end-to-end blocking and inference on test set.
    """
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not os.path.isfile(s1_path):
        raise FileNotFoundError(f"Test file not found: {s1_path}")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(matching_output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(candidate_output_path)), exist_ok=True)

    # 1. Read all S1 entity IDs upfront to preserve exact order and ensure 100% coverage
    print(f"Reading test reference records from {s1_path}...")
    s1_df = pd.read_csv(s1_path, sep="\t")
    all_s1_ids = [str(x).strip() for x in s1_df["entity_id"]]
    total_s1 = len(all_s1_ids)
    print(f"Total required Source 1 test entities: {total_s1}")

    # Determine subset to actively evaluate if max_s1_eval is specified
    eval_df = s1_df.iloc[:max_s1_eval] if max_s1_eval is not None and max_s1_eval > 0 else s1_df
    eval_s1_set = set(str(x).strip() for x in eval_df["entity_id"])
    print(f"Entities to actively evaluate with model: {len(eval_s1_set)}")

    # Load ML Model & Feature Extractor
    model = None
    if os.path.isfile(model_path):
        print(f"Loading trained classifier from {model_path}...")
        model = joblib.load(model_path)
    else:
        print(f"Notice: Model file {model_path} not found. Running high-precision heuristic fallback.")

    fe = FeatureExtractor()

    # Pre-index S2 and S3 records partitioned by country
    print("\nLoading and indexing Source 2 and Source 3 candidate pools...")
    s2_df = pd.read_csv(s2_path, sep="\t", nrows=s2_s3_pool_size) if os.path.isfile(s2_path) else pd.DataFrame()
    s3_df = pd.read_csv(s3_path, sep="\t", nrows=s2_s3_pool_size) if os.path.isfile(s3_path) else pd.DataFrame()

    unique_countries = eval_df["country"].fillna("").unique()
    indices_by_country: Dict[str, CandidateIndex] = {}

    for country in unique_countries:
        c_str = str(country)
        print(f"Indexing S2/S3 for country: '{c_str}'...")
        sub_s2 = s2_df[s2_df["country"].fillna("") == country] if not s2_df.empty else pd.DataFrame()
        sub_s3 = s3_df[s3_df["country"].fillna("") == country] if not s3_df.empty else pd.DataFrame()

        idx = CandidateIndex()
        for df_source in (sub_s2, sub_s3):
            if df_source.empty:
                continue
            for _, row in tqdm(df_source.iterrows(), total=len(df_source), desc=f"Indexing '{c_str}'"):
                eid = str(row["entity_id"]).strip()
                name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
                addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
                idx.add_record(eid, name, addr, c_str)
        indices_by_country[c_str] = idx

    # Storage for active predictions: s1_id -> (candidate_list, matched_list)
    results_candidates: Dict[str, List[str]] = {}
    results_matches: Dict[str, List[str]] = {}

    print(f"\nRunning blocking and ML inference (threshold = {threshold:.2f})...")
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing S1 entities"):
        s1_id = str(row["entity_id"]).strip()
        s1_name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
        s1_addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
        s1_country = str(row["country"]) if pd.notna(row["country"]) else ""

        idx = indices_by_country.get(s1_country)
        if idx is None:
            results_candidates[s1_id] = []
            results_matches[s1_id] = []
            continue

        raw_cands = generate_candidates_for_s1(
            s1_id=s1_id,
            s1_name=s1_name,
            s1_address=s1_addr,
            s1_country=s1_country,
            index=idx,
            max_candidates=max_candidates
        )
        valid_cands = sanitize_id_list(raw_cands, s1_id)
        results_candidates[s1_id] = valid_cands

        if not valid_cands:
            results_matches[s1_id] = []
            continue

        matched_ids = []
        if model is not None:
            # Extract features for all candidate pairs
            batch_feats = []
            for cid in valid_cands:
                c_name, c_addr, _ = idx.records.get(cid, ("", "", None))
                feat = fe.extract_pair_features(
                    name1=s1_name,
                    addr1=s1_addr,
                    country1=s1_country,
                    name2=c_name,
                    addr2=c_addr,
                    country2=s1_country,
                )
                batch_feats.append(feat)

            probs = model.predict_proba(np.array(batch_feats, dtype=np.float32))[:, 1]
            for cid, prob in zip(valid_cands, probs):
                if prob >= threshold:
                    matched_ids.append(cid)
        else:
            # High-confidence heuristic fallback
            for cid in valid_cands:
                c_name, c_addr, _ = idx.records.get(cid, ("", "", None))
                feat = fe.extract_pair_features(
                    name1=s1_name,
                    addr1=s1_addr,
                    country1=s1_country,
                    name2=c_name,
                    addr2=c_addr,
                    country2=s1_country,
                )
                # High name similarity & token sort ratio
                if feat[0] >= 0.85 or (feat[2] >= 0.90 and feat[14] == 0.0):
                    matched_ids.append(cid)

        # Enforce subset rule and sanitization
        valid_matches = [m for m in sanitize_id_list(matched_ids, s1_id) if m in valid_cands]
        results_matches[s1_id] = valid_matches

    # Write both output TSV files ensuring every single S1 ID appears exactly once
    print(f"\nWriting candidate pairs to {candidate_output_path}...")
    with open(candidate_output_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = results_candidates.get(s1_id, [])
            cands_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cands_str}\n")

    print(f"Writing matching results to {matching_output_path}...")
    with open(matching_output_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = results_matches.get(s1_id, [])
            matches_str = ",".join(matches)
            f_match.write(f"{s1_id}\t{matches_str}\n")

    print("\n--- Output Summary ---")
    print(f"Total Source 1 entities written: {total_s1}")
    matched_count = sum(1 for m in results_matches.values() if m)
    print(f"Entities with predicted matches: {matched_count}")
    print(f"Entities with empty match list (singletons): {total_s1 - matched_count}")
    print(f"Candidate pairs saved to: {candidate_output_path}")
    print(f"Matching results saved to: {matching_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict matches on test dataset")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory with test TSVs")
    parser.add_argument("--model-path", default="models/classifier.joblib", help="Path to trained model")
    parser.add_argument("--matching-output", default="output/matching_results.tsv", help="Output path for matching_results.tsv")
    parser.add_argument("--candidate-output", default="output/candidate_pairs.tsv", help="Output path for candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=0.75, help="Decision threshold for match probability")
    parser.add_argument("--max-candidates", type=int, default=20, help="Max candidates per entity")
    parser.add_argument("--max-s1-eval", type=int, default=None, help="Max S1 entities to score (default all)")
    parser.add_argument("--pool-size", type=int, default=None, help="Max S2/S3 records to load into pool (default all)")
    args = parser.parse_args()

    run_prediction_pipeline(
        test_dir=args.test_dir,
        model_path=args.model_path,
        matching_output_path=args.matching_output,
        candidate_output_path=args.candidate_output,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        max_s1_eval=args.max_s1_eval,
        s2_s3_pool_size=args.pool_size,
    )
