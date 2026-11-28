from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ALL_BUG_IDS = (
    "fraction_add",
    "binomial_expand",
    "operator_precedence",
    "unary_minus_distrib",
    "transposition_sign",
    "slip",
)

RAW_EEDI_DIR = Path("data/raw/eedi")
FROZEN_MAP_PATH = Path("data/processed/eedi_to_buglens_map.csv")

EEDI_KEYWORD_RULES: list[tuple[str, str]] = [
    (r"adds?\s+numerators|adds?\s+denominators", "fraction_add"),
    (r"expand.*squared|squared.*expand|\(a\s*\+\s*b\)\^?2|middle\s+term", "binomial_expand"),
    (r"order\s+of\s+operations|left\s+to\s+right|bidmas|bodmas|pemdas", "operator_precedence"),
    (r"negative.*brackets?|minus.*brackets?|distribute.*negative", "unary_minus_distrib"),
    (r"changes?\s+side|other\s+side.*sign|transpos", "transposition_sign"),
]

OUT_OF_SCOPE_TOPIC_PATTERN = re.compile(
    r"angle|triangle|polygon|circle|area|perimeter|volume|graph|coordinate|gradient|axis|axes|symmetry|bearing|trigonometry|pythagoras",
    re.IGNORECASE,
)

IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[.*?\]\(.*?\)")


def normalized_text_hash(text: str) -> str:
    """Near-duplicate text hash for lineage-group deduplication (Step 1.4)."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+\-*/=^().]", " ", text.lower())).strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]


def unpivot_and_filter_eedi(raw_train_csv: Path = RAW_EEDI_DIR / "train.csv") -> pd.DataFrame:
    """Step 1.1: Reshape Eedi train.csv into long format and drop NaN / image rows."""
    df = pd.read_csv(raw_train_csv)
    rows: list[dict[str, object]] = []

    for _, r in df.iterrows():
        q_text = str(r.get("QuestionText", "")).strip()
        if IMAGE_MARKDOWN_PATTERN.search(q_text):
            continue

        correct_alpha = str(r.get("CorrectAnswer", "")).strip()
        if correct_alpha not in {"A", "B", "C", "D"}:
            continue
        correct_text = str(r.get(f"Answer{correct_alpha}Text", "")).strip()
        if IMAGE_MARKDOWN_PATTERN.search(correct_text):
            continue

        subject_name = str(r.get("SubjectName", ""))
        construct_name = str(r.get("ConstructName", ""))
        q_id = int(r["QuestionId"])

        for alpha in ("A", "B", "C", "D"):
            if alpha == correct_alpha:
                continue
            misc_id = r.get(f"Misconception{alpha}Id")
            if pd.isna(misc_id):
                continue
            distractor_text = str(r.get(f"Answer{alpha}Text", "")).strip()
            if IMAGE_MARKDOWN_PATTERN.search(distractor_text):
                continue

            rows.append(
                {
                    "QuestionId": q_id,
                    "AnswerAlphabet": alpha,
                    "QuestionText": q_text,
                    "CorrectAnswerText": correct_text,
                    "DistractorText": distractor_text,
                    "MisconceptionId": int(misc_id),
                    "SubjectName": subject_name,
                    "ConstructName": construct_name,
                    "group_id": f"eedi_q_{q_id}_{normalized_text_hash(q_text)}",
                }
            )

    return pd.DataFrame(rows)


def audit_misconception_mapping(
    raw_mapping_csv: Path = RAW_EEDI_DIR / "misconception_mapping.csv",
    output_map_csv: Path = FROZEN_MAP_PATH,
) -> pd.DataFrame:
    """Step 1.2: Search misconception_mapping.csv with targeted algebraic keywords."""
    output_map_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_mapping_csv)
    candidates: list[dict[str, object]] = []

    for _, r in df.iterrows():
        desc = str(r["MisconceptionName"]).strip()
        desc_lower = desc.lower()
        matched_bug = None
        for pattern, bug_id in EEDI_KEYWORD_RULES:
            if re.search(pattern, desc_lower):
                matched_bug = bug_id
                break
        if matched_bug is not None:
            candidates.append(
                {
                    "MisconceptionId": int(r["MisconceptionId"]),
                    "MisconceptionName": desc,
                    "bug_id": matched_bug,
                    "verified": False,
                }
            )

    out_df = pd.DataFrame(candidates)
    out_df.to_csv(output_map_csv, index=False)
    return out_df


def build_curated_eedi_dataset(
    long_df: pd.DataFrame,
    frozen_map_df: pd.DataFrame,
    max_slip_ratio: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Step 1.2 + 1.3: Join frozen B1-B5 map and sample in-domain numerical/algebraic S_slip."""
    active = frozen_map_df[frozen_map_df["bug_id"].isin(ALL_BUG_IDS)].copy()
    id_to_bug = dict(zip(active["MisconceptionId"].astype(int), active["bug_id"], strict=True))

    b1_b5 = long_df[long_df["MisconceptionId"].isin(id_to_bug)].copy()
    b1_b5["bug_id"] = b1_b5["MisconceptionId"].map(id_to_bug)
    b1_b5["source"] = "eedi_b1_b5"

    unmapped = long_df[~long_df["MisconceptionId"].isin(id_to_bug)].copy()
    topic_text = (
        unmapped["SubjectName"].fillna("")
        + " "
        + unmapped["ConstructName"].fillna("")
        + " "
        + unmapped["QuestionText"].fillna("")
    )
    in_domain_neg = unmapped[~topic_text.str.contains(OUT_OF_SCOPE_TOPIC_PATTERN)].copy()

    n_slip = max(1, int(len(b1_b5) * max_slip_ratio)) if len(b1_b5) > 0 else len(in_domain_neg)
    if len(in_domain_neg) > n_slip:
        in_domain_neg = in_domain_neg.sample(n=n_slip, random_state=seed)
    in_domain_neg["bug_id"] = "slip"
    in_domain_neg["source"] = "eedi_slip"

    return pd.concat([b1_b5, in_domain_neg], ignore_index=True)


def verify_disjoint_groups(
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "group_id",
) -> None:
    """Step 1.4: Assert zero group/template/QuestionId leakage across train, cal, and test."""
    tr = set(train_df[group_col].dropna().unique())
    ca = set(cal_df[group_col].dropna().unique())
    te = set(test_df[group_col].dropna().unique())
    if tr & ca or tr & te or ca & te:
        raise AssertionError("Leakage detected across train / cal / test group_ids!")


def group_split_train_cal_test(
    df: pd.DataFrame,
    group_col: str = "group_id",
    label_col: str = "bug_id",
    cal_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 1.4: Partition dataset into group-disjoint train, cal, and test splits."""
    groups = np.array(sorted(df[group_col].dropna().unique()))
    if len(groups) < 3:
        raise ValueError("Need at least 3 distinct groups to split into train, cal, and test.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    n_test = max(1, int(round(len(shuffled) * test_size)))
    n_cal = max(1, int(round(len(shuffled) * cal_size)))
    test_groups = set(shuffled[:n_test])
    cal_groups = set(shuffled[n_test : n_test + n_cal])
    train_groups = set(shuffled[n_test + n_cal :])

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    cal_df = df[df[group_col].isin(cal_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    verify_disjoint_groups(train_df, cal_df, test_df, group_col=group_col)
    return train_df, cal_df, test_df
