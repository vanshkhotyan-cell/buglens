from pathlib import Path

import pandas as pd
import pytest

import buglens
from buglens.dataset import (
    audit_misconception_mapping,
    build_curated_eedi_dataset,
    group_split_train_cal_test,
    unpivot_and_filter_eedi,
    verify_disjoint_groups,
)
from buglens.tracking import get_dvc_lock_hash, get_git_commit


def test_package_imports():
    assert buglens is not None


def test_lineage_helpers():
    assert isinstance(get_git_commit(), str)
    assert get_dvc_lock_hash("nonexistent.lock") == "untracked"


def test_eedi_curation_and_group_splits(tmp_path: Path):
    raw_train = tmp_path / "train.csv"
    raw_map = tmp_path / "misconception_mapping.csv"
    out_map = tmp_path / "eedi_to_buglens_map.csv"

    pd.DataFrame(
        [
            {
                "MisconceptionId": 10,
                "MisconceptionName": "Adds numerators and denominators when adding fractions",
            },
            {
                "MisconceptionId": 20,
                "MisconceptionName": "Carries out operations left to right regardless of priority",
            },
            {
                "MisconceptionId": 30,
                "MisconceptionName": "General arithmetic slip when multiplying",
            },
            {"MisconceptionId": 40, "MisconceptionName": "Confuses interior angles in a triangle"},
        ]
    ).to_csv(raw_map, index=False)

    rows = []
    for qid in range(1, 13):
        rows.append(
            {
                "QuestionId": qid,
                "ConstructName": "Algebraic manipulation",
                "SubjectName": "Number and Algebra",
                "CorrectAnswer": "A",
                "QuestionText": f"Evaluate expression {qid}: 1/2 + 1/3",
                "AnswerAText": "5/6",
                "AnswerBText": "2/5",
                "AnswerCText": "1/6",
                "AnswerDText": "2/6",
                "MisconceptionAId": None,
                "MisconceptionBId": 10 if qid % 2 == 0 else 20,
                "MisconceptionCId": 30,
                "MisconceptionDId": None,
            }
        )
    rows.append(
        {
            "QuestionId": 99,
            "ConstructName": "Angles",
            "SubjectName": "Geometry",
            "CorrectAnswer": "A",
            "QuestionText": "Find angle x in the triangle",
            "AnswerAText": "60",
            "AnswerBText": "120",
            "AnswerCText": "90",
            "AnswerDText": "30",
            "MisconceptionAId": None,
            "MisconceptionBId": 40,
            "MisconceptionCId": None,
            "MisconceptionDId": None,
        }
    )
    rows.append(
        {
            "QuestionId": 100,
            "ConstructName": "Fractions",
            "SubjectName": "Number",
            "CorrectAnswer": "A",
            "QuestionText": "Look at ![diagram](img.png)",
            "AnswerAText": "1",
            "AnswerBText": "2",
            "AnswerCText": "3",
            "AnswerDText": "4",
            "MisconceptionAId": None,
            "MisconceptionBId": 10,
            "MisconceptionCId": None,
            "MisconceptionDId": None,
        }
    )
    pd.DataFrame(rows).to_csv(raw_train, index=False)

    map_df = audit_misconception_mapping(raw_map, out_map)
    assert set(map_df["bug_id"]) == {"fraction_add", "operator_precedence"}

    long_df = unpivot_and_filter_eedi(raw_train)
    assert 100 not in set(long_df["QuestionId"])

    curated = build_curated_eedi_dataset(long_df, map_df, max_slip_ratio=0.5, seed=42)
    assert 99 not in set(curated["QuestionId"])
    assert "slip" in set(curated["bug_id"])

    train_df, cal_df, test_df = group_split_train_cal_test(
        curated, cal_size=0.2, test_size=0.2, seed=42
    )
    verify_disjoint_groups(train_df, cal_df, test_df)

    with pytest.raises(AssertionError):
        verify_disjoint_groups(train_df, train_df, test_df)
