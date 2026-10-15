"""Unit tests for DAComp-DE environment — scoring logic and data integrity.

Run with: uv run pytest tests.py -v
"""

import hashlib
import json
from pathlib import Path

import pytest
from openreward.api.sandboxes.types import FileTransferError, FileTransferResult

import dacomp_de
from evaluate_de import (
    extract_arch_score,
    extract_max_score_from_rubric,
    map_schema,
    weighted_score,
)


class TestMapSchema:
    def test_marts(self):
        assert map_schema("marts") == "mart"

    def test_staging(self):
        assert map_schema("staging") == "staging"

    def test_intermediate(self):
        assert map_schema("intermediate") == "intermediate"


class TestWeightedScore:
    def test_equal_weights(self):
        scores = [(1.0, 10.0), (0.5, 10.0), (0.0, 10.0)]
        result = weighted_score(scores)
        assert abs(result - 50.0) < 1e-6

    def test_unequal_weights(self):
        # staging=15%, intermediate=25%, marts=60%
        scores = [(1.0, 15.0), (0.5, 25.0), (0.0, 60.0)]
        result = weighted_score(scores)
        expected = ((1.0 * 15.0 + 0.5 * 25.0 + 0.0 * 60.0) / 100.0) * 100.0
        assert abs(result - expected) < 1e-6

    def test_empty(self):
        assert weighted_score([]) == 0.0

    def test_perfect(self):
        scores = [(1.0, 15.0), (1.0, 25.0), (1.0, 60.0)]
        assert weighted_score(scores) == 100.0


class TestExtractArchScore:
    def test_simple(self):
        result = json.dumps({"Requirement1": {"Score": 10}, "Total Score": 30})
        assert extract_arch_score(result) == 30

    def test_chinese(self):
        result = json.dumps({"需求1": {"得分": 5}, "总得分": 20})
        assert extract_arch_score(result) == 20

    def test_none(self):
        assert extract_arch_score(None) is None


class TestExtractMaxScore:
    def test_english(self):
        rubric = "# [Total Score | 44 points] The solution must..."
        assert extract_max_score_from_rubric(rubric) == 44.0

    def test_no_score(self):
        assert extract_max_score_from_rubric("No score here") is None


# ---------------------------------------------------------------------------
# Data integrity tests (only run if data is downloaded)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
DE_TASKS_PATH = DATA_DIR / "tasks_de.json"


@pytest.fixture
def de_tasks():
    if not DE_TASKS_PATH.exists():
        pytest.skip("DE task data not downloaded (run prepare_data.py first)")
    with open(DE_TASKS_PATH) as f:
        return json.load(f)


class TestDEDataIntegrity:
    def test_task_count(self, de_tasks):
        assert len(de_tasks) == 110

    def test_required_fields(self, de_tasks):
        for task in de_tasks:
            assert "instance_id" in task

    def test_unique_ids(self, de_tasks):
        ids = [t["instance_id"] for t in de_tasks]
        assert len(ids) == len(set(ids))

    def test_task_type_distribution(self, de_tasks):
        types = {}
        for task in de_tasks:
            t = task.get("task_type", "unknown")
            types[t] = types.get(t, 0) + 1
        assert types.get("impl", 0) == 30
        assert types.get("evol", 0) == 50
        assert types.get("arch", 0) == 30

    def test_stable_ordering(self, de_tasks):
        ids = [t["instance_id"] for t in de_tasks]
        assert ids == sorted(ids)


class _FakeSandbox:
    """Serves the agent's /workspace files. download() refuses files over the
    SDK's default cap, as the real one does; download_to() streams with no cap."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files

    async def run(self, cmd):
        return ("", 0)

    async def download(self, path, max_size=256 * 1024 * 1024):
        if len(self.files[path]) > max_size:
            raise FileTransferError(f"{path} is over the {max_size}-byte download limit")
        return self.files[path]

    async def download_to(self, path, local_path):
        if path not in self.files:
            raise FileTransferError(f"Cannot download {path}: no such file")
        data = self.files[path]
        Path(local_path).write_bytes(data)
        return FileTransferResult(size=len(data), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """A DE-Impl task whose pipeline writes /workspace/shop.duckdb; grading
    records the bytes it was given and scores 80."""
    (tmp_path / "gold" / "de-impl-1").mkdir(parents=True)
    (tmp_path / "gold" / "de-impl-1" / "shop.duckdb").write_bytes(b"gold")
    monkeypatch.setattr(dacomp_de, "_gold_db_dir", tmp_path / "gold")
    monkeypatch.setattr(dacomp_de, "_eval_config", {"de-impl-1": {"database_file": "shop.duckdb"}})
    graded = []

    def fake_eval(pred_db_path, gold_db_path, config):
        graded.append((pred_db_path, pred_db_path.read_bytes()))
        return {"score": 80.0, "layers": []}

    monkeypatch.setattr(dacomp_de, "evaluate_de_pipeline", fake_eval)
    env = dacomp_de.DACompDE.__new__(dacomp_de.DACompDE)
    env.instance_id = "de-impl-1"
    return env, graded


class TestPipelineSubmission:
    @pytest.mark.asyncio
    async def test_database_over_the_sdk_download_cap_is_graded(self, pipeline_env):
        env, graded = pipeline_env
        db = b"D" * (300 * 1024 * 1024)
        env.sandbox = _FakeSandbox({"/workspace/shop.duckdb": db})

        result = await env._eval_pipeline()

        assert result.reward == pytest.approx(0.8)
        assert len(graded) == 1 and graded[0][1] == db
        assert not graded[0][0].exists()  # the temp copy is cleaned up

    @pytest.mark.asyncio
    async def test_missing_database(self, pipeline_env):
        env, graded = pipeline_env
        env.sandbox = _FakeSandbox({})

        result = await env._eval_pipeline()

        assert result.metadata["error"] == "no_database" and result.reward == 0.0
        assert graded == []

    @pytest.mark.asyncio
    async def test_empty_database(self, pipeline_env):
        env, graded = pipeline_env
        env.sandbox = _FakeSandbox({"/workspace/shop.duckdb": b""})

        result = await env._eval_pipeline()

        assert result.metadata["error"] == "empty_database" and result.reward == 0.0
        assert graded == []
