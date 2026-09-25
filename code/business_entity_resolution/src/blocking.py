"""
ML Challenge 2026 - Business Entity Resolution
Blocking & Candidate Generation Module

Design & Tradeoff Analysis (Recall vs. Reduction Ratio):
- Problem: S1 (~1.7M) x (S2 + S3) (~10M) = ~1.7e13 pairwise comparisons.
- Reduction Ratio target: >99.999% reduction to make feature computation & inference tractable.
- Recall Ceiling target: Maximize retention of true matches (>90% recall).
- Multi-strategy Union Approach:
  1. Country Partition: Strict partitioning by open-label country string (zero cross-country loss).
  2. Significant Name Tokens: Inverted index on non-stopword, non-legal-suffix tokens.
  3. Phonetic Keys: Soundex and Metaphone indexing on primary name tokens (resilient to typos like 'Williams' vs 'Wilblims').
  4. Postal Code + Address Token: Recovers matches where name is corrupted/domain-based but address matches.
  5. Prefix/Trigram Match: Recovers compound domain names and truncated names.
  6. Multi-Vote Candidate Capping: Candidates ranked by match signals and capped at top-K (default K=20).
- Guarantee: Writes the FINAL candidate_pairs.tsv scored by the ML model.
"""

import os
import sys
import collections
from typing import Dict, List, Set, Tuple, Optional, Iterator
import pandas as pd
from tqdm import tqdm

try:
    from .preprocess import (
        normalize_business_name,
        normalize_address,
        get_core_name_tokens,
        extract_postal_code,
        soundex,
        simplified_metaphone,
    )
except ImportError:
    from preprocess import (
        normalize_business_name,
        normalize_address,
        get_core_name_tokens,
        extract_postal_code,
        soundex,
        simplified_metaphone,
    )

# Common generic stopwords to avoid over-indexing
COMMON_STOPWORDS = {
    "the", "and", "of", "in", "for", "on", "at", "to", "a", "an", "is",
    "by", "with", "from", "as", "into", "near", "opp", "road", "street",
    "avenue", "lane", "floor", "suite", "unit", "block", "sector", "plot",
    "shop", "bldg", "building", "hospital", "nagar", "market", "bazaar"
}


class CandidateIndex:
    """
    Inverted indices for Source 2 and Source 3 entities within a single country partition.
    """
    def __init__(self):
        # Maps token -> set of entity_ids
        self.token_index: Dict[str, List[str]] = collections.defaultdict(list)
        # Maps phonetic key -> set of entity_ids
        self.phonetic_index: Dict[str, List[str]] = collections.defaultdict(list)
        # Maps postal_code -> set of entity_ids
        self.postal_index: Dict[str, List[str]] = collections.defaultdict(list)
        # Maps 4-char name prefix -> set of entity_ids
        self.prefix_index: Dict[str, List[str]] = collections.defaultdict(list)
        # Store metadata for quick verification: id -> (norm_name, norm_addr, postal)
        self.records: Dict[str, Tuple[str, str, Optional[str]]] = {}

    def add_record(self, entity_id: str, name: str, address: str, country: str):
        # Only index S2 and S3 IDs
        if not (entity_id.startswith("S2-") or entity_id.startswith("S3-")):
            return

        norm_name = normalize_business_name(name)
        norm_addr = normalize_address(address)
        postal = extract_postal_code(address, country)

        self.records[entity_id] = (norm_name, norm_addr, postal)

        # 1. Significant token index
        core_tokens = get_core_name_tokens(norm_name)
        for token in core_tokens:
            if token not in COMMON_STOPWORDS and len(token) >= 3:
                self.token_index[token].append(entity_id)

        # 2. Phonetic index on primary name tokens
        for token in core_tokens[:2]:
            if len(token) >= 3 and token not in COMMON_STOPWORDS:
                sx = soundex(token)
                meta = simplified_metaphone(token)
                if sx != "0000":
                    self.phonetic_index["sx:" + sx].append(entity_id)
                if meta:
                    self.phonetic_index["mt:" + meta].append(entity_id)

        # 3. Postal code index
        if postal:
            self.postal_index[postal].append(entity_id)

        # 4. Name prefix index (first 4 letters)
        if norm_name:
            first_word = norm_name.split()[0]
            if len(first_word) >= 4:
                self.prefix_index[first_word[:4]].append(entity_id)


def generate_candidates_for_s1(
    s1_id: str,
    s1_name: str,
    s1_address: str,
    s1_country: str,
    index: CandidateIndex,
    max_candidates: int = 20
) -> List[str]:
    """
    Multi-strategy candidate retrieval for a single Source 1 entity.
    Returns ranked list of candidate IDs (at most max_candidates).
    """
    norm_name = normalize_business_name(s1_name)
    norm_addr = normalize_address(s1_address)
    postal = extract_postal_code(s1_address, s1_country)

    scores: Dict[str, float] = collections.defaultdict(float)
    core_tokens = get_core_name_tokens(norm_name)
    s1_addr_tokens = set(norm_addr.split()) if norm_addr else set()

    # 1. Token overlap matching
    for token in core_tokens:
        if token not in COMMON_STOPWORDS and len(token) >= 3:
            matched_ids = index.token_index.get(token, [])
            # Inverted frequency weighting: rarer tokens contribute higher score
            token_weight = 4.0 if len(matched_ids) < 500 else (2.0 if len(matched_ids) < 5000 else 0.8)
            for cid in matched_ids:
                scores[cid] += token_weight

    # 2. Phonetic matching
    for token in core_tokens[:2]:
        if len(token) >= 3 and token not in COMMON_STOPWORDS:
            sx = soundex(token)
            meta = simplified_metaphone(token)
            if sx != "0000":
                for cid in index.phonetic_index.get("sx:" + sx, []):
                    scores[cid] += 1.5
            if meta:
                for cid in index.phonetic_index.get("mt:" + meta, []):
                    scores[cid] += 1.5

    # 3. Postal code matching with address overlap confirmation
    if postal and postal in index.postal_index:
        for cid in index.postal_index[postal]:
            c_name, c_addr, _ = index.records.get(cid, ("", "", None))
            # Address token overlap boost
            c_addr_tokens = set(c_addr.split()) if c_addr else set()
            overlap = len(s1_addr_tokens & c_addr_tokens - COMMON_STOPWORDS)
            if overlap >= 1:
                scores[cid] += 3.0 + 0.5 * overlap

    # 4. Name prefix matching
    if norm_name:
        first_word = norm_name.split()[0]
        if len(first_word) >= 4:
            pfx = first_word[:4]
            for cid in index.prefix_index.get(pfx, []):
                scores[cid] += 1.0

    if not scores:
        return []

    # Rank candidates by total match score descending
    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    # Deduplicate and cap to max_candidates
    final_candidates = [cid for cid, _ in sorted_candidates[:max_candidates]]
    return final_candidates


def run_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates_per_s1: int = 20,
    output_candidate_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Run multi-strategy blocking partitioned by country.
    Returns: mapping s1_entity_id -> list of candidate_entity_ids.
    Writes: output_candidate_path if provided.
    """
    # Enforce open-set country grouping
    unique_countries = s1_df["country"].fillna("").unique()

    candidates_map: Dict[str, List[str]] = {}

    for country in unique_countries:
        country_str = str(country)
        print(f"\n--- Blocking for country partition: '{country_str}' ---")

        # Partition dataframes for current country
        sub_s1 = s1_df[s1_df["country"].fillna("") == country]
        sub_s2 = s2_df[s2_df["country"].fillna("") == country] if s2_df is not None else pd.DataFrame()
        sub_s3 = s3_df[s3_df["country"].fillna("") == country] if s3_df is not None else pd.DataFrame()

        print(f"Entities in partition: S1={len(sub_s1)}, S2={len(sub_s2)}, S3={len(sub_s3)}")

        # Build candidate index for S2 and S3 in this country
        index = CandidateIndex()
        for df_source in (sub_s2, sub_s3):
            if df_source.empty:
                continue
            for _, row in tqdm(df_source.iterrows(), total=len(df_source), desc="Indexing S2/S3"):
                eid = str(row["entity_id"]).strip()
                name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
                addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
                index.add_record(eid, name, addr, country_str)

        # Retrieve candidates for all S1 entities in this country
        for _, row in tqdm(sub_s1.iterrows(), total=len(sub_s1), desc="Retrieving S1 candidates"):
            s1_id = str(row["entity_id"]).strip()
            name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
            addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""

            cands = generate_candidates_for_s1(
                s1_id=s1_id,
                s1_name=name,
                s1_address=addr,
                s1_country=country_str,
                index=index,
                max_candidates=max_candidates_per_s1
            )
            # Enforce S2/S3 ID validity only
            valid_cands = [c for c in cands if c.startswith(("S2-", "S3-")) and c != s1_id]
            # Deduplicate preserving order
            seen = set()
            dedup_cands = [c for c in valid_cands if not (c in seen or seen.add(c))]
            candidates_map[s1_id] = dedup_cands

    # Write output/candidate_pairs.tsv if path specified
    if output_candidate_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_candidate_path)), exist_ok=True)
        print(f"\nWriting final candidate pairs to {output_candidate_path}...")
        with open(output_candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            # Write in exact order of S1 records
            for s1_id in s1_df["entity_id"]:
                s1_id_str = str(s1_id).strip()
                cands = candidates_map.get(s1_id_str, [])
                f.write(f"{s1_id_str}\t{','.join(cands)}\n")
        print(f"Successfully saved {len(candidates_map)} candidate rows.")

    return candidates_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-strategy Candidate Generation (Blocking)")
    parser.add_argument("--s1", default="dataset/train/train_source1.tsv", help="Path to Source 1 TSV")
    parser.add_argument("--s2", default="dataset/train/train_source2.tsv", help="Path to Source 2 TSV")
    parser.add_argument("--s3", default="dataset/train/train_source3.tsv", help="Path to Source 3 TSV")
    parser.add_argument("--output", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=20, help="Max candidate pairs per S1 entity")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row limit for quick testing")
    args = parser.parse_args()

    print(f"Loading data from {args.s1}...")
    df_s1 = pd.read_csv(args.s1, sep="\t", nrows=args.nrows)
    df_s2 = pd.read_csv(args.s2, sep="\t", nrows=args.nrows)
    df_s3 = pd.read_csv(args.s3, sep="\t", nrows=args.nrows)

    run_blocking(df_s1, df_s2, df_s3, max_candidates_per_s1=args.max_candidates, output_candidate_path=args.output)
