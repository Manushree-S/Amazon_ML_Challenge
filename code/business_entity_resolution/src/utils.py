"""
ML Challenge 2026 - Business Entity Resolution
Utility Functions Module

Contains:
- Exact Macro F_0.5 evaluation metric implementation (challenge specified)
- TSV readers & writers conforming to strict format rules
- Validation and formatting utilities
"""

import sys
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd
import numpy as np


def compute_f05_single(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_0.5 for a single Source 1 entity.
    Includes singleton scoring rules:
    - true empty and pred empty => 1.0
    - true empty and pred non-empty => 0.0
    - true non-empty and pred empty => 0.0
    - true non-empty and pred non-empty => standard F_0.5
    """
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return precision, recall, f05


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_entities: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities in the evaluation set.
    """
    if s1_entities is None:
        s1_entities = list(ground_truth.keys())

    precisions = []
    recalls = []
    f05_scores = []
    singleton_count = 0
    singleton_correct = 0

    for s1_id in s1_entities:
        true_matches = ground_truth.get(s1_id, set())
        pred_matches = predictions.get(s1_id, set())

        p, r, f = compute_f05_single(true_matches, pred_matches)
        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f)

        if len(true_matches) == 0:
            singleton_count += 1
            if len(pred_matches) == 0:
                singleton_correct += 1

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "total_evaluated": len(s1_entities),
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
    }


def parse_ground_truth(gt_path: str, nrows: Optional[int] = None) -> Dict[str, Set[str]]:
    """
    Parse train_ground_truth.tsv into a mapping:
    source1_entity_id -> set of matched_entity_ids.
    """
    df = pd.read_csv(gt_path, sep="\t", nrows=nrows)
    gt_map = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if matched and matched.strip():
            ids = {x.strip() for x in matched.split(",") if x.strip()}
            gt_map[s1] = ids
        else:
            gt_map[s1] = set()
    return gt_map


def sanitize_id_list(id_list: List[str], s1_id: str) -> List[str]:
    """
    Strict validation rule enforcer:
    - S2- and S3- prefixes only
    - No self-matches (never an S1- ID)
    - Deduplicated preserving order
    """
    seen = set()
    cleaned = []
    for mid in id_list:
        mid_clean = mid.strip()
        if not mid_clean:
            continue
        if mid_clean == s1_id or mid_clean.startswith("S1-"):
            continue  # Reject self match
        if not mid_clean.startswith(("S2-", "S3-")):
            continue  # Reject invalid prefix
        if mid_clean not in seen:
            seen.add(mid_clean)
            cleaned.append(mid_clean)
    return cleaned
