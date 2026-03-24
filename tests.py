"""Unit tests for DAComp-DE environment — scoring logic and data integrity.

Run with: uv run pytest tests.py -v
"""

import json
from pathlib import Path

import pytest

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
