from fastapi.testclient import TestClient
import buglens
from buglens.api import app
from buglens.tracking import get_dvc_lock_hash, get_git_commit


def test_package_imports():
    assert buglens is not None


def test_lineage_helpers():
    assert isinstance(get_git_commit(), str)
    assert get_dvc_lock_hash("nonexistent.lock") == "untracked"


def test_api_health_and_ready():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["ready"] is False
