import buglens
from buglens.tracking import get_dvc_lock_hash, get_git_commit


def test_package_imports():
    assert buglens is not None


def test_lineage_helpers():
    assert isinstance(get_git_commit(), str)
    assert get_dvc_lock_hash("nonexistent.lock") == "untracked"
