"""
ML Challenge 2026 - Business Entity Resolution
Inference & Prediction Pipeline (Memory-Safe, Streaming & Incremental)

- Streams Source 2 and Source 3 candidate records line-by-line directly into SQLite-backed indices
  (eliminates pd.read_csv + iterrows() overhead and memory consumption)
- Evaluates Source 1 test entities using trained LightGBM classifier probabilities
- Streams output incrementally to output/matching_results.tsv and output/candidate_pairs.tsv
  (each row is appended as computed, preserving progress and surviving early stops)
- Implements strict time-box protection (--max-runtime-minutes): gracefully halts and pads
  any remaining S1 entities with valid empty rows so output files ALWAYS contain all 1,732,544
  entities and pass utils/validate_submission.py
- Enforces all challenge hard rules:
  * Every Source 1 entity in test appears exactly once (no missing, no duplicates)
  * Matched IDs are strictly a subset of candidate IDs
  * S2- and S3- IDs only (never Source 1 self-matches)
  * No duplicate IDs within any comma-separated list
  * Clean tab-separation with empty string for singletons/unmatched entities
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import psutil

process = psutil.Process(os.getpid())

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


def stream_source_file_to_indices(
    file_path: str,
    indices_by_country: Dict[str, CandidateIndex],
    pool_limit: Optional[int] = None,
    source_name: str = "Source"
):
    """
    Stream a TSV candidate file line-by-line directly into SQLite indices without
    materializing large pandas DataFrames in RAM.
    """
    if not os.path.isfile(file_path):
        print(f"Warning: {file_path} not found, skipping.")
        return

    print(f"Streaming and indexing records from {file_path} ({source_name})...")
    count = 0
    t0 = time.time()

    with open(file_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip TSV header
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) < 4:
                parts = parts + [""] * (4 - len(parts))
            eid, name, addr, country = parts[0].strip(), parts[1], parts[2], parts[3].strip()

            idx = indices_by_country.get(country)
            if idx is not None:
                idx.add_record(eid, name, addr, country)
                count += 1

            if pool_limit is not None and count >= pool_limit:
                print(f"Reached pool limit of {pool_limit:,} records for {source_name}.")
                break

            if count > 0 and count % 100000 == 0:
                elapsed = time.time() - t0
                rss_mb = process.memory_info().rss / (1024 * 1024)
                print(f"  Indexed {count:,} {source_name} records ({count/elapsed:.0f} rec/s, RSS: {rss_mb:.1f} MB)...")

    elapsed = time.time() - t0
    print(f"Completed {source_name}: {count:,} records indexed in {elapsed:.1f}s.")


def run_prediction_pipeline(
    test_dir: str = "dataset/test",
    model_path: str = "models/classifier.joblib",
    matching_output_path: str = "output/matching_results.tsv",
    candidate_output_path: str = "output/candidate_pairs.tsv",
    threshold: float = 0.85,
    max_candidates: int = 20,
    max_s1_eval: Optional[int] = None,
    s2_s3_pool_size: Optional[int] = None,
    max_runtime_minutes: float = 33.0,
):
    """
    Execute streaming blocking and inference on test set with incremental disk writing.
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
    all_s1_records: List[Tuple[str, str, str, str]] = []
    unique_countries: Set[str] = set()

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) < 4:
                parts = parts + [""] * (4 - len(parts))
            eid, name, addr, country = parts[0].strip(), parts[1], parts[2], parts[3].strip()
            all_s1_records.append((eid, name, addr, country))
            unique_countries.add(country)

    total_s1 = len(all_s1_records)
    print(f"Total required Source 1 test entities: {total_s1:,} across {len(unique_countries)} country partitions.")

    # Load ML Model & Feature Extractor
    model = None
    if os.path.isfile(model_path):
        print(f"Loading trained classifier from {model_path}...")
        model = joblib.load(model_path)
    else:
        print(f"Notice: Model file {model_path} not found. Running high-precision heuristic fallback.")

    fe = FeatureExtractor()

    # Pre-index S2 and S3 candidate records partitioned by country using SQLite-backed indices
    print("\nInitializing SQLite candidate indices for country partitions...")
    indices_by_country: Dict[str, CandidateIndex] = {c: CandidateIndex() for c in unique_countries}

    try:
        # Stream S2 and S3 directly without pandas overhead
        stream_source_file_to_indices(s2_path, indices_by_country, pool_limit=s2_s3_pool_size, source_name="Source 2")
        stream_source_file_to_indices(s3_path, indices_by_country, pool_limit=s2_s3_pool_size, source_name="Source 3")

        # Build SQLite B-tree indexes for fast queries
        for c_str, idx in indices_by_country.items():
            print(f"Building SQLite indexes for country partition '{c_str}'...")
            idx.build_indexes()

        print(f"\n--- Starting Streaming Inference & Incremental Writing (threshold={threshold:.2f}, time-box={max_runtime_minutes:.0f}m) ---")
        processed_count = 0
        matched_entities_count = 0
        total_matched_ids_count = 0
        start_time = time.time()
        max_runtime_seconds = max_runtime_minutes * 60.0

        # Open both output files and stream results incrementally (append mode)
        with open(candidate_output_path, "w", encoding="utf-8") as f_cand, \
             open(matching_output_path, "w", encoding="utf-8") as f_match:

            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

            try:
                for s1_id, s1_name, s1_addr, s1_country in all_s1_records:
                    # Check max S1 entity cap if specified
                    if max_s1_eval is not None and processed_count >= max_s1_eval:
                        print(f"\nReached max S1 evaluation limit of {max_s1_eval:,} entities.")
                        break

                    # Check time box limit
                    if time.time() - start_time >= max_runtime_seconds:
                        print(f"\nReached time-box limit of {max_runtime_minutes:.0f} minutes. Halting inference gracefully...")
                        break

                    idx = indices_by_country.get(s1_country)
                    valid_cands: List[str] = []
                    valid_matches: List[str] = []

                    if idx is not None:
                        raw_cands = generate_candidates_for_s1(
                            s1_id=s1_id,
                            s1_name=s1_name,
                            s1_address=s1_addr,
                            s1_country=s1_country,
                            index=idx,
                            max_candidates=max_candidates
                        )
                        valid_cands = sanitize_id_list(raw_cands, s1_id)

                        if valid_cands:
                            if model is not None:
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
                                matched_ids = [cid for cid, prob in zip(valid_cands, probs) if prob >= threshold]
                            else:
                                matched_ids = []
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
                                    if feat[0] >= 0.85 or (feat[2] >= 0.90 and feat[14] == 0.0):
                                        matched_ids.append(cid)

                            valid_matches = [m for m in sanitize_id_list(matched_ids, s1_id) if m in valid_cands]

                    # INCREMENTAL STREAMING WRITE: write row immediately
                    f_cand.write(f"{s1_id}\t{','.join(valid_cands)}\n")
                    f_match.write(f"{s1_id}\t{','.join(valid_matches)}\n")

                    if valid_matches:
                        matched_entities_count += 1
                        total_matched_ids_count += len(valid_matches)

                    processed_count += 1

                    if processed_count % 2000 == 0:
                        f_cand.flush()
                        f_match.flush()
                        elapsed = time.time() - start_time
                        rss_mb = process.memory_info().rss / (1024 * 1024)
                        speed = processed_count / elapsed if elapsed > 0 else 0
                        print(f"Processed {processed_count:,}/{total_s1:,} ({processed_count/total_s1*100:.1f}%) | "
                              f"Matched entities: {matched_entities_count:,} ({total_matched_ids_count:,} IDs) | "
                              f"Speed: {speed:.1f} it/s | RSS: {rss_mb:.1f} MB")

            finally:
                # CRITICAL: Pad all un-evaluated entities with empty match lists so output files
                # ALWAYS contain all 1,732,544 rows and 100% pass utils/validate_submission.py!
                remaining_entities = all_s1_records[processed_count:]
                if remaining_entities:
                    print(f"\nPadding remaining {len(remaining_entities):,} un-evaluated S1 entities with empty match lists...")
                    for rem_id, _, _, _ in remaining_entities:
                        f_cand.write(f"{rem_id}\t\n")
                        f_match.write(f"{rem_id}\t\n")
                f_cand.flush()
                f_match.flush()

        total_elapsed = time.time() - start_time
        print("\n--- Inference Output Summary ---")
        print(f"Total Source 1 entities in test: {total_s1:,}")
        print(f"Actively scored S1 entities:    {processed_count:,}")
        print(f"Entities with predicted matches: {matched_entities_count:,}")
        print(f"Total matched IDs predicted:     {total_matched_ids_count:,}")
        print(f"Entities with empty match list:  {total_s1 - matched_entities_count:,}")
        print(f"Total elapsed inference time:    {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        print(f"Candidate pairs saved to:        {candidate_output_path}")
        print(f"Matching results saved to:       {matching_output_path}")

    finally:
        # Clean up SQLite temporary database files
        for idx in indices_by_country.values():
            idx.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict matches on test dataset (Streaming & Incremental)")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory with test TSVs")
    parser.add_argument("--model-path", default="models/classifier.joblib", help="Path to trained model")
    parser.add_argument("--matching-output", default="output/matching_results.tsv", help="Output path for matching_results.tsv")
    parser.add_argument("--candidate-output", default="output/candidate_pairs.tsv", help="Output path for candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=0.85, help="Decision threshold for match probability")
    parser.add_argument("--max-candidates", type=int, default=20, help="Max candidates per entity")
    parser.add_argument("--max-s1-eval", type=int, default=None, help="Max S1 entities to actively score (default all)")
    parser.add_argument("--pool-size", type=int, default=None, help="Max S2/S3 records to index per source (default all)")
    parser.add_argument("--max-runtime-minutes", type=float, default=33.0, help="Hard time-box limit in minutes (default 33)")
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
        max_runtime_minutes=args.max_runtime_minutes,
    )
