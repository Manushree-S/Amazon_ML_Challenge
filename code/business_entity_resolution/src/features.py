"""
ML Challenge 2026 - Business Entity Resolution
Pairwise Feature Extraction Module

Extracts rich pairwise similarity features between Source 1 and candidate records:
1. Name similarities:
   - Levenshtein normalized similarity
   - Jaro-Winkler similarity
   - Token Sort Ratio & Token Set Ratio (rapidfuzz C++)
   - Word-level Jaccard similarity
   - Character 3-gram Jaccard similarity
   - Name length difference & ratio
   - Name prefix match flag
2. Address similarities:
   - Levenshtein normalized similarity
   - Jaro-Winkler similarity
   - Token Sort Ratio & Token Set Ratio
   - Word-level Jaccard similarity
   - Missing address indicator flag
3. Location & Identity features:
   - Country exact match (binary, handles open set)
   - Postal code exact match / mismatch / missing indicators
4. Overall text & TF-IDF similarity:
   - Combined token Jaccard similarity
   - Character n-gram TF-IDF cosine similarity (via HashingVectorizer, fully offline)
   - Optional sentence-transformer embedding cosine similarity hook
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer

try:
    from .preprocess import (
        normalize_business_name,
        normalize_address,
        extract_postal_code,
    )
except ImportError:
    from preprocess import (
        normalize_business_name,
        normalize_address,
        extract_postal_code,
    )

FEATURE_NAMES = [
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_char3_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_prefix_match",
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaccard",
    "addr_is_empty",
    "country_exact_match",
    "postal_match",
    "postal_mismatch",
    "postal_missing",
    "combined_token_jaccard",
    "tfidf_cosine_sim",
]


def word_jaccard(tokens1: set, tokens2: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def char_ngrams(text: str, n: int = 3) -> set:
    """Generate character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


class FeatureExtractor:
    """
    Computes pairwise feature vectors between (S1, S2/S3) entity pairs.
    """
    def __init__(self, use_sentence_transformer: bool = False):
        self.use_sentence_transformer = use_sentence_transformer
        # Fast, memory-bounded, fully offline hashing vectorizer for TF-IDF cosine
        self.hasher = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            n_features=2048,
            alternate_sign=False,
            norm="l2"
        )
        self.st_model = None
        if use_sentence_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: SentenceTransformer not available ({e}), skipping embedding similarity.")
                self.st_model = None

    def extract_pair_features(
        self,
        name1: str,
        addr1: str,
        country1: str,
        name2: str,
        addr2: str,
        country2: str,
    ) -> List[float]:
        """
        Extract numerical features for a single entity pair.
        """
        norm_name1 = normalize_business_name(name1)
        norm_name2 = normalize_business_name(name2)
        norm_addr1 = normalize_address(addr1)
        norm_addr2 = normalize_address(addr2)

        # 1. Name Features
        name_lev = Levenshtein.normalized_similarity(norm_name1, norm_name2)
        name_jw = JaroWinkler.similarity(norm_name1, norm_name2)
        name_sort = fuzz.token_sort_ratio(norm_name1, norm_name2) / 100.0
        name_set = fuzz.token_set_ratio(norm_name1, norm_name2) / 100.0

        name_toks1 = set(norm_name1.split())
        name_toks2 = set(norm_name2.split())
        name_jacc = word_jaccard(name_toks1, name_toks2)

        char3_1 = char_ngrams(norm_name1, 3)
        char3_2 = char_ngrams(norm_name2, 3)
        name_char3_jacc = word_jaccard(char3_1, char3_2)

        len1, len2 = len(norm_name1), len(norm_name2)
        max_len = max(len1, len2, 1)
        min_len = min(len1, len2)
        name_len_diff = abs(len1 - len2) / max_len
        name_len_ratio = min_len / max_len

        # Name prefix match (first 4 characters)
        pfx1 = norm_name1[:4] if len(norm_name1) >= 4 else norm_name1
        pfx2 = norm_name2[:4] if len(norm_name2) >= 4 else norm_name2
        name_prefix = 1.0 if pfx1 and pfx1 == pfx2 else 0.0

        # 2. Address Features
        addr_empty = 1.0 if (not norm_addr1 or not norm_addr2) else 0.0
        if addr_empty == 1.0:
            addr_lev = 0.0
            addr_jw = 0.0
            addr_sort = 0.0
            addr_set = 0.0
            addr_jacc = 0.0
        else:
            addr_lev = Levenshtein.normalized_similarity(norm_addr1, norm_addr2)
            addr_jw = JaroWinkler.similarity(norm_addr1, norm_addr2)
            addr_sort = fuzz.token_sort_ratio(norm_addr1, norm_addr2) / 100.0
            addr_set = fuzz.token_set_ratio(norm_addr1, norm_addr2) / 100.0
            addr_toks1 = set(norm_addr1.split())
            addr_toks2 = set(norm_addr2.split())
            addr_jacc = word_jaccard(addr_toks1, addr_toks2)

        # 3. Country & Postal Code Features
        c1 = (country1 or "").strip().lower()
        c2 = (country2 or "").strip().lower()
        country_match = 1.0 if c1 and c2 and c1 == c2 else 0.0

        post1 = extract_postal_code(addr1, country1)
        post2 = extract_postal_code(addr2, country2)

        if post1 and post2:
            postal_match = 1.0 if post1 == post2 else 0.0
            postal_mismatch = 1.0 if post1 != post2 else 0.0
            postal_missing = 0.0
        else:
            postal_match = 0.0
            postal_mismatch = 0.0
            postal_missing = 1.0

        # 4. Combined & TF-IDF Cosine Features
        comb_toks1 = name_toks1 | (set(norm_addr1.split()) if norm_addr1 else set())
        comb_toks2 = name_toks2 | (set(norm_addr2.split()) if norm_addr2 else set())
        comb_jacc = word_jaccard(comb_toks1, comb_toks2)

        # TF-IDF Char n-gram Cosine
        s1_full = f"{norm_name1} {norm_addr1}".strip()
        s2_full = f"{norm_name2} {norm_addr2}".strip()
        if s1_full and s2_full:
            vecs = self.hasher.transform([s1_full, s2_full])
            tfidf_sim = float((vecs[0].multiply(vecs[1])).sum())
        else:
            tfidf_sim = 0.0

        features = [
            name_lev,
            name_jw,
            name_sort,
            name_set,
            name_jacc,
            name_char3_jacc,
            name_len_diff,
            name_len_ratio,
            name_prefix,
            addr_lev,
            addr_jw,
            addr_sort,
            addr_set,
            addr_jacc,
            addr_empty,
            country_match,
            postal_match,
            postal_mismatch,
            postal_missing,
            comb_jacc,
            tfidf_sim,
        ]
        return features

    def extract_batch_features(
        self,
        pairs: List[Tuple[Dict[str, str], Dict[str, str]]]
    ) -> np.ndarray:
        """
        Extract features for a batch of (s1_dict, s2_dict) pairs.
        Each dict has keys: 'business_name', 'business_address', 'country'.
        """
        feats = []
        for s1, s2 in pairs:
            f = self.extract_pair_features(
                name1=s1.get("business_name", ""),
                addr1=s1.get("business_address", ""),
                country1=s1.get("country", ""),
                name2=s2.get("business_name", ""),
                addr2=s2.get("business_address", ""),
                country2=s2.get("country", ""),
            )
            feats.append(f)
        return np.array(feats, dtype=np.float32)
