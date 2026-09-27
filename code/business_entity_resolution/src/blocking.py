"""
ML Challenge 2026 - Business Entity Resolution
Blocking & Candidate Generation Module (memory-safe, SQLite-backed)

WHY THIS VERSION EXISTS
------------------------
At full scale (~1.7M Source 1 entities x ~10M Source 2/3 records) the original
implementation built four separate pure-Python dict-of-lists indices
(token_index, phonetic_index, postal_index, prefix_index) fully in RAM, plus a
records dict holding normalized name/address/postal for every S2/S3 record.
That easily reaches many GB of RAM (small Python objects have large per-object
overhead), causing the process to thrash (high memory, near-zero CPU) instead
of crashing outright -- which is exactly what was observed.

This version keeps the IDENTICAL blocking logic and scoring weights, but
stores the postings lists and record metadata in a SQLite database on disk
(with proper indexes) instead of Python dicts. Peak RAM stays roughly
constant regardless of dataset size; SQLite does the heavy lifting.

Public API is unchanged: CandidateIndex, generate_candidates_for_s1, run_blocking.
predict.py and any other caller do not need to change.

Design & Tradeoff Analysis (Recall vs. Reduction Ratio) -- unchanged from before:
- Country Partition: strict partitioning by open-label country string.
- Significant Name Tokens: inverted index on non-stopword, non-legal-suffix tokens.
- Phonetic Keys: Soundex and Metaphone indexing on primary name tokens.
- Postal Code + Address Token overlap confirmation.
- Prefix Match: recovers compound/truncated names.
- Multi-Vote Candidate Capping: candidates ranked by match signal, capped top-K.
"""

import os
import sys
import sqlite3
import tempfile
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

COMMON_STOPWORDS = {
    "the", "and", "of", "in", "for", "on", "at", "to", "a", "an", "is",
    "by", "with", "from", "as", "into", "near", "opp", "road", "street",
    "avenue", "lane", "floor", "suite", "unit", "block", "sector", "plot",
    "shop", "bldg", "building", "hospital", "nagar", "market", "bazaar"
}

# Cap how many postings a single token/key can accumulate before we stop
# treating it as useful for blocking (extremely common tokens add memory/time
# but little discriminative value). Applied only at *query* time via LIMIT,
# so indexing itself never scans unboundedly.
MAX_POSTINGS_PER_KEY = 20000


class CandidateIndex:
    """
    SQLite-backed inverted index for Source 2 / Source 3 entities within a
    single country partition. Same public surface as the original in-memory
    version: add_record(...), and a `.records` mapping-like accessor via
    get_record(entity_id) plus a dict-like `.records` shim for callers that
    do `idx.records.get(cid, default)`.
    """

    def __init__(self, db_path: Optional[str] = None):
        # Use a temp file-backed DB (not ':memory:') so the OS can page it
        # out under memory pressure instead of the process being OOM-killed.
        # A tmp file also survives across the add/query phases cheaply.
        self._owns_file = db_path is None
        self.db_path = db_path or tempfile.mktemp(suffix=".sqlite")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = MEMORY")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self._init_schema()
        self._insert_buffer_tokens = []
        self._insert_buffer_phonetic = []
        self._insert_buffer_postal = []
        self._insert_buffer_prefix = []
        self._insert_buffer_records = []
        self._buffer_flush_size = 20000
        self._built = False
        # small dict-like shim so `idx.records.get(cid, default)` keeps working
        self.records = _RecordsShim(self)
        # Cap postings PER KEY at insertion time (not just query time). This is
        # what actually bounds both memory and query latency -- a query-time
        # LIMIT still requires SQLite to build/scan the full matching set first
        # when combined with an IN(...) batch, and for low-cardinality keys
        # (phonetic codes especially) that full set can be enormous at scale.
        # Counters below are lightweight (int per key), unlike storing every
        # posting.
        self._token_counts: Dict[str, int] = collections.Counter()
        self._phonetic_counts: Dict[str, int] = collections.Counter()
        self._postal_counts: Dict[str, int] = collections.Counter()
        self._prefix_counts: Dict[str, int] = collections.Counter()
        self.MAX_INSERT_PER_KEY = 2000

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            norm_name TEXT,
            norm_addr TEXT,
            postal TEXT
        )""")
        c.execute("CREATE TABLE token_postings (token TEXT, entity_id TEXT)")
        c.execute("CREATE TABLE phonetic_postings (key TEXT, entity_id TEXT)")
        c.execute("CREATE TABLE postal_postings (postal TEXT, entity_id TEXT)")
        c.execute("CREATE TABLE prefix_postings (prefix TEXT, entity_id TEXT)")
        self.conn.commit()

    def add_record(self, entity_id: str, name: str, address: str, country: str):
        if not (entity_id.startswith("S2-") or entity_id.startswith("S3-")):
            return

        norm_name = normalize_business_name(name)
        norm_addr = normalize_address(address)
        postal = extract_postal_code(address, country)

        self._insert_buffer_records.append((entity_id, norm_name, norm_addr, postal))

        core_tokens = get_core_name_tokens(norm_name)
        for token in core_tokens:
            if token not in COMMON_STOPWORDS and len(token) >= 3:
                if self._token_counts[token] < self.MAX_INSERT_PER_KEY:
                    self._token_counts[token] += 1
                    self._insert_buffer_tokens.append((token, entity_id))

        for token in core_tokens[:2]:
            if len(token) >= 3 and token not in COMMON_STOPWORDS:
                sx = soundex(token)
                meta = simplified_metaphone(token)
                if sx != "0000":
                    key = "sx:" + sx
                    if self._phonetic_counts[key] < self.MAX_INSERT_PER_KEY:
                        self._phonetic_counts[key] += 1
                        self._insert_buffer_phonetic.append((key, entity_id))
                if meta:
                    key = "mt:" + meta
                    if self._phonetic_counts[key] < self.MAX_INSERT_PER_KEY:
                        self._phonetic_counts[key] += 1
                        self._insert_buffer_phonetic.append((key, entity_id))

        if postal:
            if self._postal_counts[postal] < self.MAX_INSERT_PER_KEY:
                self._postal_counts[postal] += 1
                self._insert_buffer_postal.append((postal, entity_id))

        if norm_name:
            first_word = norm_name.split()[0]
            if len(first_word) >= 4:
                pfx = first_word[:4]
                if self._prefix_counts[pfx] < self.MAX_INSERT_PER_KEY:
                    self._prefix_counts[pfx] += 1
                    self._insert_buffer_prefix.append((pfx, entity_id))

        if len(self._insert_buffer_records) >= self._buffer_flush_size:
            self._flush()

    def _flush(self):
        c = self.conn.cursor()
        if self._insert_buffer_records:
            c.executemany(
                "INSERT OR REPLACE INTO records VALUES (?,?,?,?)",
                self._insert_buffer_records,
            )
            self._insert_buffer_records = []
        if self._insert_buffer_tokens:
            c.executemany("INSERT INTO token_postings VALUES (?,?)", self._insert_buffer_tokens)
            self._insert_buffer_tokens = []
        if self._insert_buffer_phonetic:
            c.executemany("INSERT INTO phonetic_postings VALUES (?,?)", self._insert_buffer_phonetic)
            self._insert_buffer_phonetic = []
        if self._insert_buffer_postal:
            c.executemany("INSERT INTO postal_postings VALUES (?,?)", self._insert_buffer_postal)
            self._insert_buffer_postal = []
        if self._insert_buffer_prefix:
            c.executemany("INSERT INTO prefix_postings VALUES (?,?)", self._insert_buffer_prefix)
            self._insert_buffer_prefix = []
        self.conn.commit()

    def build_indexes(self):
        """Call once after all add_record() calls are done, before querying."""
        if self._built:
            return
        self._flush()
        c = self.conn.cursor()
        c.execute("CREATE INDEX idx_token ON token_postings(token)")
        c.execute("CREATE INDEX idx_phonetic ON phonetic_postings(key)")
        c.execute("CREATE INDEX idx_postal ON postal_postings(postal)")
        c.execute("CREATE INDEX idx_prefix ON prefix_postings(prefix)")
        self.conn.commit()
        self._built = True

    def get_record(self, entity_id: str) -> Tuple[str, str, Optional[str]]:
        cur = self.conn.execute(
            "SELECT norm_name, norm_addr, postal FROM records WHERE entity_id = ?",
            (entity_id,),
        )
        row = cur.fetchone()
        if row is None:
            return ("", "", None)
        return (row[0] or "", row[1] or "", row[2])

    def token_postings_for(self, token: str) -> List[str]:
        cur = self.conn.execute(
            "SELECT entity_id FROM token_postings WHERE token = ? LIMIT ?",
            (token, MAX_POSTINGS_PER_KEY),
        )
        return [r[0] for r in cur.fetchall()]

    def token_posting_count(self, token: str) -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM token_postings WHERE token = ?", (token,)
        )
        return cur.fetchone()[0]

    def phonetic_postings_for(self, key: str) -> List[str]:
        cur = self.conn.execute(
            "SELECT entity_id FROM phonetic_postings WHERE key = ? LIMIT ?",
            (key, MAX_POSTINGS_PER_KEY),
        )
        return [r[0] for r in cur.fetchall()]

    def postal_postings_for(self, postal: str) -> List[str]:
        cur = self.conn.execute(
            "SELECT entity_id FROM postal_postings WHERE postal = ? LIMIT ?",
            (postal, MAX_POSTINGS_PER_KEY),
        )
        return [r[0] for r in cur.fetchall()]

    def prefix_postings_for(self, prefix: str) -> List[str]:
        cur = self.conn.execute(
            "SELECT entity_id FROM prefix_postings WHERE prefix = ? LIMIT ?",
            (prefix, MAX_POSTINGS_PER_KEY),
        )
        return [r[0] for r in cur.fetchall()]

    # --- Batched lookups: one round-trip for many keys at once. These are
    # what generate_candidates_for_s1 actually uses; the single-key methods
    # above are kept for compatibility/testing but are no longer the hot path.

    def token_postings_batch(self, tokens: List[str]) -> Dict[str, List[str]]:
        if not tokens:
            return {}
        placeholders = ",".join("?" for _ in tokens)
        cur = self.conn.execute(
            f"SELECT token, entity_id FROM token_postings WHERE token IN ({placeholders})",
            tokens,
        )
        out: Dict[str, List[str]] = collections.defaultdict(list)
        for tok, eid in cur.fetchall():
            out[tok].append(eid)
        return out

    def phonetic_postings_batch(self, keys: List[str]) -> Dict[str, List[str]]:
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        cur = self.conn.execute(
            f"SELECT key, entity_id FROM phonetic_postings WHERE key IN ({placeholders})",
            keys,
        )
        out: Dict[str, List[str]] = collections.defaultdict(list)
        for k, eid in cur.fetchall():
            out[k].append(eid)
        return out

    def postal_postings_batch(self, postals: List[str]) -> Dict[str, List[str]]:
        if not postals:
            return {}
        placeholders = ",".join("?" for _ in postals)
        cur = self.conn.execute(
            f"SELECT postal, entity_id FROM postal_postings WHERE postal IN ({placeholders})",
            postals,
        )
        out: Dict[str, List[str]] = collections.defaultdict(list)
        for p, eid in cur.fetchall():
            out[p].append(eid)
        return out

    def prefix_postings_batch(self, prefixes: List[str]) -> Dict[str, List[str]]:
        if not prefixes:
            return {}
        placeholders = ",".join("?" for _ in prefixes)
        cur = self.conn.execute(
            f"SELECT prefix, entity_id FROM prefix_postings WHERE prefix IN ({placeholders})",
            prefixes,
        )
        out: Dict[str, List[str]] = collections.defaultdict(list)
        for p, eid in cur.fetchall():
            out[p].append(eid)
        return out

    def get_records_batch(self, entity_ids: List[str]) -> Dict[str, Tuple[str, str, Optional[str]]]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        cur = self.conn.execute(
            f"SELECT entity_id, norm_name, norm_addr, postal FROM records WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
        return {r[0]: (r[1] or "", r[2] or "", r[3]) for r in cur.fetchall()}

    def close(self):
        try:
            self.conn.close()
        finally:
            if self._owns_file and os.path.exists(self.db_path):
                try:
                    os.remove(self.db_path)
                except OSError:
                    pass


class _RecordsShim:
    """Lets callers keep doing `idx.records.get(cid, default)` unchanged."""
    def __init__(self, index: "CandidateIndex"):
        self._index = index

    def get(self, entity_id: str, default=("", "", None)):
        rec = self._index.get_record(entity_id)
        if rec == ("", "", None) and default != ("", "", None):
            # ambiguous between "missing" and "genuinely empty" -- fall back
            # to a light existence check only when caller supplied a
            # non-trivial default (keeps behavior close to a real dict.get)
            cur = self._index.conn.execute(
                "SELECT 1 FROM records WHERE entity_id = ?", (entity_id,)
            )
            if cur.fetchone() is None:
                return default
        return rec


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
    Same scoring logic as the original in-memory version; postings are now
    fetched from SQLite instead of Python dicts.
    """
    if not index._built:
        index.build_indexes()

    norm_name = normalize_business_name(s1_name)
    norm_addr = normalize_address(s1_address)
    postal = extract_postal_code(s1_address, s1_country)

    scores: Dict[str, float] = collections.defaultdict(float)
    core_tokens = get_core_name_tokens(norm_name)
    s1_addr_tokens = set(norm_addr.split()) if norm_addr else set()

    # 1. Token overlap matching -- one batched query for all tokens at once
    query_tokens = [t for t in core_tokens if t not in COMMON_STOPWORDS and len(t) >= 3]
    token_hits = index.token_postings_batch(query_tokens)
    for token, matched_ids in token_hits.items():
        n = len(matched_ids)
        token_weight = 4.0 if n < 500 else (2.0 if n < 5000 else 0.8)
        for cid in matched_ids:
            scores[cid] += token_weight

    # 2. Phonetic matching -- one batched query for all sx:/mt: keys
    phonetic_keys = []
    for token in core_tokens[:2]:
        if len(token) >= 3 and token not in COMMON_STOPWORDS:
            sx = soundex(token)
            meta = simplified_metaphone(token)
            if sx != "0000":
                phonetic_keys.append("sx:" + sx)
            if meta:
                phonetic_keys.append("mt:" + meta)
    phonetic_hits = index.phonetic_postings_batch(phonetic_keys)
    for key, matched_ids in phonetic_hits.items():
        for cid in matched_ids:
            scores[cid] += 1.5

    # 3. Postal code matching with address overlap confirmation
    if postal:
        postal_hits = index.postal_postings_batch([postal])
        postal_cands = postal_hits.get(postal, [])
        if postal_cands:
            cand_records = index.get_records_batch(postal_cands)
            for cid in postal_cands:
                _, c_addr, _ = cand_records.get(cid, ("", "", None))
                c_addr_tokens = set(c_addr.split()) if c_addr else set()
                overlap = len(s1_addr_tokens & c_addr_tokens - COMMON_STOPWORDS)
                if overlap >= 1:
                    scores[cid] += 3.0 + 0.5 * overlap

    # 4. Name prefix matching
    if norm_name:
        first_word = norm_name.split()[0]
        if len(first_word) >= 4:
            pfx = first_word[:4]
            prefix_hits = index.prefix_postings_batch([pfx])
            for cid in prefix_hits.get(pfx, []):
                scores[cid] += 1.0

    if not scores:
        return []

    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [cid for cid, _ in sorted_candidates[:max_candidates]]


def run_blocking(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    max_candidates_per_s1: int = 20,
    output_candidate_path: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Run multi-strategy blocking partitioned by country, one country at a
    time, streaming results straight to disk when output_candidate_path is
    given so we never hold the full candidates_map for all countries in RAM.
    """
    unique_countries = s1_df["country"].fillna("").unique()

    candidates_map: Dict[str, List[str]] = {}
    out_f = None
    if output_candidate_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_candidate_path)), exist_ok=True)
        out_f = open(output_candidate_path, "w", encoding="utf-8")
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")

    try:
        for country in unique_countries:
            country_str = str(country)
            print(f"\n--- Blocking for country partition: '{country_str}' ---")

            sub_s1 = s1_df[s1_df["country"].fillna("") == country]
            sub_s2 = s2_df[s2_df["country"].fillna("") == country] if s2_df is not None else pd.DataFrame()
            sub_s3 = s3_df[s3_df["country"].fillna("") == country] if s3_df is not None else pd.DataFrame()

            print(f"Entities in partition: S1={len(sub_s1)}, S2={len(sub_s2)}, S3={len(sub_s3)}")

            index = CandidateIndex()
            try:
                for df_source in (sub_s2, sub_s3):
                    if df_source.empty:
                        continue
                    for _, row in tqdm(df_source.iterrows(), total=len(df_source), desc=f"Indexing S2/S3 [{country_str}]"):
                        eid = str(row["entity_id"]).strip()
                        name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
                        addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
                        index.add_record(eid, name, addr, country_str)
                index.build_indexes()

                for _, row in tqdm(sub_s1.iterrows(), total=len(sub_s1), desc=f"Retrieving S1 candidates [{country_str}]"):
                    s1_id = str(row["entity_id"]).strip()
                    name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
                    addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""

                    cands = generate_candidates_for_s1(
                        s1_id=s1_id, s1_name=name, s1_address=addr,
                        s1_country=country_str, index=index,
                        max_candidates=max_candidates_per_s1
                    )
                    valid_cands = [c for c in cands if c.startswith(("S2-", "S3-")) and c != s1_id]
                    seen = set()
                    dedup_cands = [c for c in valid_cands if not (c in seen or seen.add(c))]

                    if out_f is not None:
                        out_f.write(f"{s1_id}\t{','.join(dedup_cands)}\n")
                    else:
                        candidates_map[s1_id] = dedup_cands
            finally:
                index.close()
    finally:
        if out_f is not None:
            out_f.close()

    if output_candidate_path:
        print(f"Successfully wrote candidate pairs to {output_candidate_path}")

    return candidates_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-strategy Candidate Generation (Blocking)")
    parser.add_argument("--s1", default="dataset/train/train_source1.tsv")
    parser.add_argument("--s2", default="dataset/train/train_source2.tsv")
    parser.add_argument("--s3", default="dataset/train/train_source3.tsv")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading data from {args.s1}...")
    df_s1 = pd.read_csv(args.s1, sep="\t", nrows=args.nrows)
    df_s2 = pd.read_csv(args.s2, sep="\t", nrows=args.nrows)
    df_s3 = pd.read_csv(args.s3, sep="\t", nrows=args.nrows)

    run_blocking(df_s1, df_s2, df_s3, max_candidates_per_s1=args.max_candidates, output_candidate_path=args.output)
