"""DAComp DE evaluation — DuckDB comparison + LLM-judged architecture scoring.

Ported from ByteDance-Seed/DAComp evaluation suite. Supports two evaluation modes:

- DE-Impl / DE-Evol: Deterministic execution-based evaluation.
  Runs agent's pipeline, compares resulting DuckDB against gold using
  row-hash multiset comparison with layer-weighted scoring (CFS mode).

- DE-Arch: LLM-judged blueprint evaluation against rubric.
"""

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import duckdb
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

EVAL_MODEL = "gpt-5-mini"

# ---------------------------------------------------------------------------
# DE-Arch LLM evaluation prompt
# ---------------------------------------------------------------------------

DE_ARCH_PROMPT = """
# Task Description
You are a professional data architect. You will evaluate a model blueprint based on a given user question and a scoring rubric.
Your task is to review a set of scoring criteria for the model blueprint, and then, based on these criteria, assess the blueprint to determine the extent to which it meets the standards.

The scoring rubric provides a total score and various requirements. Where:
- Total Score: Represents the maximum possible score after summing all scoring criteria.
- Requirements: Represent different needs the assistant must satisfy. Each requirement contains multiple scoring criteria. These criteria are divided into two categories:
    - 1. Deterministic criteria: These can be scored directly without considering different implementation paths.
    - 2. Non-deterministic criteria: These usually have multiple implementation paths. When evaluating, first determine the "best matching path" based on the assistant's response, and then score based on the sub-criteria under that path. If there is no clearly matching path, use your own expertise to judge whether the assistant's response correctly meets the requirement's goal and calculate if it is reasonable. If correct, assign points, but the score for this requirement cannot exceed the maximum score of other defined paths.

Final Scoring Logic:
Final Score = Sum of all requirement scores.
Requirement Score = Sum of all criteria scores within that requirement.
Criteria Score = Direct score OR Best matching path score OR Unmatched path score OR Sum of sub-criteria.
Best Matching Path Score = Sum of the scores of the sub-criteria under that path.

Please analyze and score item by item according to the rubric. If you have any hesitation on any point, do not guess or make subjective assumptions—assign 0 points directly. **You must provide evidence; if evidence is missing, assign 0 points.**

[User Question Start]
{user_query}
[User Question End]

[Model Blueprint Start]
{model_blueprint}
[Model Blueprint End]

[Scoring Rubric Start]
{rubric}
[Scoring Rubric End]

You need to analyze and score each item one by one according to the scoring rubric.
# Response format as follows:
```json
{{
    "Requirement1": {{
        "Criterion1.1": {{
            "Analysis": "Carefully read the content of the model blueprint, determine whether it meets Criterion 1.1, and assign a score.",
            "Evidence": [],
            "Score": int
        }},
        "Total Score": int
    }},
    "Total Score": int
}}
```
"""

# ---------------------------------------------------------------------------
# Schema / scoring helpers (ported from DAComp utils.py)
# ---------------------------------------------------------------------------


def map_schema(layer: str) -> str:
    """Map layer name to DuckDB schema name (marts → mart)."""
    return "mart" if layer == "marts" else layer


def weighted_score(scores: list[tuple[float, float]]) -> float:
    """Compute weighted average score * 100."""
    total_w = sum(w for _, w in scores)
    if total_w > 0:
        return (sum(s * w for s, w in scores) / total_w) * 100.0
    elif scores:
        return (sum(s for s, _ in scores) / len(scores)) * 100.0
    return 0.0


# ---------------------------------------------------------------------------
# DuckDB table comparison (row-hash multiset)
# ---------------------------------------------------------------------------


class CoreAccuracyEvaluator:
    """Compare tables between predicted and gold DuckDB databases."""

    def __init__(self, pred_db_path: Path, gold_db_path: Path, config: dict[str, Any]):
        self.pred_db = pred_db_path
        self.gold_db = gold_db_path
        self.config = config

    def compare_table(self, schema: str, table: str) -> bool:
        """Row-hash multiset comparison for a single table.

        Casts all columns to LOWER(TRIM(CAST(col AS VARCHAR))), concatenates with '|',
        computes MD5 hash, then compares hash-count dictionaries.
        """
        try:
            pred_ro = duckdb.connect(str(self.pred_db), read_only=True)
            gold_ro = duckdb.connect(str(self.gold_db), read_only=True)

            pred_schema = self._resolve_schema(pred_ro, schema)
            gold_schema = self._resolve_schema(gold_ro, schema)

            # Check table existence
            if not self._table_exists(pred_ro, pred_schema, table):
                pred_ro.close()
                gold_ro.close()
                return False
            if not self._table_exists(gold_ro, gold_schema, table):
                pred_ro.close()
                gold_ro.close()
                return False

            # Get columns
            gold_cols = self._get_columns(gold_ro, gold_schema, table)
            pred_cols = self._get_columns(pred_ro, pred_schema, table)

            # Determine comparison columns (config-specified or intersection)
            inter_cols = self._get_compare_cols(schema, table, gold_cols, pred_cols)
            if not inter_cols:
                pred_ro.close()
                gold_ro.close()
                return False

            # Row count short-circuit
            pred_count = pred_ro.execute(
                f"SELECT COUNT(*) FROM {pred_schema}.{table}"
            ).fetchone()[0]
            gold_count = gold_ro.execute(
                f"SELECT COUNT(*) FROM {gold_schema}.{table}"
            ).fetchone()[0]
            pred_ro.close()
            gold_ro.close()

            if pred_count != gold_count:
                return False

            # Hash-based comparison
            col_expr = ", ".join(
                f"COALESCE(LOWER(TRIM(CAST({c} AS VARCHAR))), 'null')"
                for c in inter_cols
            )
            hash_query = (
                f"SELECT MD5(CONCAT_WS('|', {col_expr})) AS rh, COUNT(*) AS cnt "
                f"FROM {{schema}}.{table} GROUP BY rh"
            )

            pconn = duckdb.connect(str(self.pred_db), read_only=True)
            gconn = duckdb.connect(str(self.gold_db), read_only=True)
            try:
                p_schema = self._resolve_schema(pconn, schema)
                g_schema = self._resolve_schema(gconn, schema)
                pred_hashes = {
                    r[0]: int(r[1])
                    for r in pconn.execute(
                        hash_query.format(schema=p_schema)
                    ).fetchall()
                }
                gold_hashes = {
                    r[0]: int(r[1])
                    for r in gconn.execute(
                        hash_query.format(schema=g_schema)
                    ).fetchall()
                }
            except Exception as e:
                logger.debug(f"Hash comparison error for {schema}.{table}: {e}")
                return False
            finally:
                pconn.close()
                gconn.close()

            return pred_hashes == gold_hashes

        except Exception as e:
            logger.debug(f"compare_table error for {schema}.{table}: {e}")
            return False

    def _resolve_schema(self, conn: duckdb.DuckDBPyConnection, schema: str) -> str:
        """Resolve mart/marts alias."""
        if schema in ("mart", "marts"):
            try:
                names = {
                    r[0]
                    for r in conn.execute(
                        "SELECT schema_name FROM information_schema.schemata"
                    ).fetchall()
                }
                if "marts" in names:
                    return "marts"
                if "mart" in names:
                    return "mart"
            except Exception:
                pass
        return schema

    def _table_exists(
        self, conn: duckdb.DuckDBPyConnection, schema: str, table: str
    ) -> bool:
        try:
            count = conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables "
                f"WHERE table_schema = '{schema}' AND table_name = '{table}'"
            ).fetchone()[0]
            return count > 0
        except Exception:
            return False

    def _get_columns(
        self, conn: duckdb.DuckDBPyConnection, schema: str, table: str
    ) -> list[str]:
        try:
            rows = conn.execute(
                f"SELECT column_name FROM information_schema.columns "
                f"WHERE table_schema = '{schema}' AND table_name = '{table}' "
                f"ORDER BY ordinal_position"
            ).fetchall()
            return [r[0] for r in rows]
        except Exception:
            return []

    def _get_compare_cols(
        self,
        schema: str,
        table: str,
        gold_cols: list[str],
        pred_cols: list[str],
    ) -> list[str]:
        """Get columns to compare: config-specified or intersection."""
        try:
            cfg_layers = self.config.get("layers", {})
            layer_name = (
                schema
                if schema in cfg_layers
                else ("marts" if schema == "mart" else schema)
            )
            tbl_cfg = cfg_layers.get(layer_name, {}).get("tables", {}).get(table)
            allowed_cols = []
            if isinstance(tbl_cfg, dict):
                allowed_cols = list(tbl_cfg.get("compare_cols") or [])
            if allowed_cols:
                gold_map = {c.lower(): c for c in gold_cols}
                pred_set = {c.lower() for c in pred_cols}
                return [
                    gold_map[a.lower()]
                    for a in allowed_cols
                    if a.lower() in gold_map and a.lower() in pred_set
                ]
        except Exception:
            pass
        # Fallback: intersection preserving gold order
        return [c for c in gold_cols if c in pred_cols]


# ---------------------------------------------------------------------------
# Progressive layer evaluator (CFS mode)
# ---------------------------------------------------------------------------


def evaluate_de_pipeline(
    pred_db_path: Path,
    gold_db_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a DE-Impl/Evol prediction against gold using CFS mode.

    Returns dict with score [0, 100] and detailed layer breakdown.
    """
    evaluator = CoreAccuracyEvaluator(pred_db_path, gold_db_path, config)
    layers = list(config.get("layers", {}).keys())

    layer_results = []
    scores: list[tuple[float, float]] = []

    for layer_name in layers:
        tables_cfg = config.get("layers", {}).get(layer_name, {}).get("tables", {})
        table_names = list(tables_cfg.keys())
        schema = map_schema(layer_name)
        layer_weight = config["layers"].get(layer_name, {}).get("weight", 0)

        def _tbl_weight(val: Any) -> int:
            return int(val) if not isinstance(val, dict) else int(val.get("weight", 0))

        total_points = sum(_tbl_weight(v) for v in tables_cfg.values())
        earned_points = 0
        table_details = {}

        for t in table_names:
            match = evaluator.compare_table(schema, t)
            table_details[t] = match
            if match:
                earned_points += _tbl_weight(tables_cfg[t])

        layer_score = earned_points / total_points if total_points > 0 else 0.0
        layer_results.append({
            "layer_name": layer_name,
            "layer_score": layer_score,
            "layer_weight": layer_weight,
            "tables": table_details,
            "earned": earned_points,
            "total": total_points,
        })
        scores.append((layer_score, float(layer_weight)))

    final_score = weighted_score(scores)

    return {
        "score": final_score,
        "layers": layer_results,
        "n_layers": len(layers),
    }


# ---------------------------------------------------------------------------
# DE-Arch LLM evaluation
# ---------------------------------------------------------------------------

# Score extraction for DE-Arch (same pattern as DA rubric)
TARGET_KEYS = {"总得分", "总分", "score", "total_score", "Score", "Total Score"}


def _strip_json_block(text: str) -> str:
    trimmed = text.strip()
    if not trimmed:
        return ""
    if trimmed.startswith("```"):
        parts = trimmed.split("\n", 1)
        body = parts[1] if len(parts) > 1 else ""
        end_split = body.rsplit("```", 1)
        body = end_split[0] if len(end_split) > 1 else body
        return body.strip()
    match = re.search(r"```json\s*(\{.*?\})\s*```", trimmed, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return trimmed


def _parse_json_response(raw: Optional[str]) -> Optional[Any]:
    if not raw:
        return None
    text = _strip_json_block(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        cleaned = re.sub(r":\s*\+(\d+(?:\.\d+)?)", r": \1", text)
        cleaned = re.sub(r"\\(?![\"\\/bfnrtu0-9])", r"\\\\", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


def extract_arch_score(result: Optional[str]) -> Optional[float]:
    """Extract the total score from a DE-Arch LLM evaluation response."""
    if not result:
        return None
    data = _parse_json_response(result)
    if data is None:
        pattern = re.compile(
            r"[\"']?(?:Total Score|总得分|总分|Score)[\"']?\s*[:=：]\s*(-?\d+(?:\.\d+)?)"
        )
        matches = pattern.findall(result)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                pass
        return None

    # Collect all score values, return the last one (outermost total)
    scores: list[Optional[float]] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in TARGET_KEYS:
                    if isinstance(value, (int, float)):
                        scores.append(float(value))
                    elif isinstance(value, str):
                        try:
                            scores.append(float(value.strip()))
                        except ValueError:
                            pass
                collect(value)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(data)
    for score in reversed(scores):
        if score is not None:
            return score
    return None


def extract_max_score_from_rubric(rubric: str) -> Optional[float]:
    """Extract max possible score from rubric text."""
    # Look for patterns like "Total Score | 44 points" or "总分.*?(\d+)"
    patterns = [
        r"Total Score[^\d]*?(\d+)",
        r"总分[^\d]*?(\d+)",
        r"\[Total Score\s*\|\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, rubric, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


async def evaluate_de_arch(
    client: AsyncOpenAI,
    instruction: str,
    blueprint: str,
    rubric: str,
    max_score: Optional[float] = None,
) -> dict[str, Any]:
    """Evaluate a DE-Arch blueprint submission via LLM judge.

    Returns dict with score [0, 100] and detailed breakdown.
    """
    prompt_text = DE_ARCH_PROMPT.format(
        user_query=instruction,
        model_blueprint=blueprint,
        rubric=rubric,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=EVAL_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=4096,
            )
            content = response.choices[0].message.content
            if content:
                content = content.strip()
                break
        except Exception as e:
            logger.warning(f"DE-Arch eval attempt {attempt + 1} failed: {e}")
            content = None

    actual_score = extract_arch_score(content) if content else None

    if max_score is None:
        max_score = extract_max_score_from_rubric(rubric)
    if max_score is None or max_score <= 0:
        max_score = 100.0

    if actual_score is not None:
        score_pct = min(100.0, max(0.0, (actual_score / max_score) * 100.0))
    else:
        score_pct = 0.0

    return {
        "score": score_pct,
        "raw_score": actual_score,
        "max_score": max_score,
        "raw_response": content,
    }
