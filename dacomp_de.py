"""DAComp DE — OpenReward sandbox environment for data engineering tasks.

Agents work with dbt-style SQL pipelines. Three sub-types:
- DE-Impl: Build a complete SQL pipeline from scratch (deterministic DuckDB grading)
- DE-Evol: Modify/extend an existing pipeline (deterministic DuckDB grading)
- DE-Arch: Design an architecture blueprint (LLM-judged)

Paper: https://arxiv.org/abs/2512.04324
Dataset: https://huggingface.co/DAComp
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from openreward import AsyncOpenReward, SandboxBucketConfig, SandboxSettings
from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from pydantic import BaseModel

from evaluate_de import evaluate_de_arch, evaluate_de_pipeline

logger = logging.getLogger(__name__)

# --- Module-level client cache ---
_openai_clients: dict[str, AsyncOpenAI] = {}


def _get_openai_client(api_key: str) -> AsyncOpenAI:
    if api_key not in _openai_clients:
        _openai_clients[api_key] = AsyncOpenAI(api_key=api_key)
    return _openai_clients[api_key]


# --- Module-level data loading ---

if os.path.exists("/orwd_data"):
    _DATA_DIR = Path("/orwd_data")
else:
    _DATA_DIR = Path(__file__).parent

_all_records: dict[str, dict] = {}
_test_tasks: list[JSONObject] = []

_json_path = _DATA_DIR / "tasks_de.json"
if _json_path.exists():
    with open(_json_path) as _f:
        _records = json.load(_f)
    for _record in _records:
        _instance_id = _record["instance_id"]
        _all_records[_instance_id] = _record
        _test_tasks.append({
            "instance_id": _instance_id,
            "task_type": _record.get("task_type", "unknown"),
        })
else:
    logger.warning(f"DE data file not found: {_json_path}")

# Load DE evaluation config (layers, tables, weights)
_eval_config: dict[str, dict] = {}
_eval_config_path = _DATA_DIR / "eval_data" / "eval_config.yaml"
if _eval_config_path.exists():
    with open(_eval_config_path) as _f:
        _raw_config = yaml.safe_load(_f)
    # Index by instance_id
    for _id, _cfg in _raw_config.get("examples", {}).items():
        _eval_config[_id] = _cfg

# Load DE-Arch gold data (rubrics)
_arch_gold: dict[str, dict] = {}
_arch_gold_path = _DATA_DIR / "eval_data" / "dacomp-arch-gold.jsonl"
if _arch_gold_path.exists():
    with open(_arch_gold_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            _entry = json.loads(_line)
            _arch_gold[_entry.get("id", "")] = _entry

# Gold DuckDB directory
_gold_db_dir = _DATA_DIR / "eval_data" / "gold"


# --- Pydantic parameter models ---

class BashParams(BaseModel, extra="forbid"):
    command: str


class SubmitParams(BaseModel, extra="forbid"):
    """Submit work for evaluation. For DE-Arch tasks, provide a YAML blueprint.
    For DE-Impl/Evol tasks, submit after running your pipeline."""
    answer: str = ""


# --- Environment class ---

class DACompDE(Environment):
    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        instance_id = str(task_spec["instance_id"])
        if instance_id not in _all_records:
            raise ValueError(f"Unknown DE task: {instance_id}")

        record = _all_records[instance_id]
        self.instance_id = instance_id
        self.instruction: str = record.get("instruction", "")
        self.task_type: str = record.get("task_type", self._infer_type(instance_id))

        # OpenReward API key for sandbox
        or_api_key = (
            secrets.get("OPENREWARD_API_KEY")
            or secrets.get("api_key")
            or os.environ.get("OPENREWARD_API_KEY", "").strip('"')
        )
        if not or_api_key:
            raise ValueError("OpenReward API key required (pass as OPENREWARD_API_KEY)")

        # OpenAI API key for DE-Arch evaluation
        openai_api_key = (
            secrets.get("OPENAI_API_KEY")
            or secrets.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY", "").strip('"')
        )
        self.openai_client = (
            _get_openai_client(openai_api_key) if openai_api_key else None
        )

        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/DAComp-DE",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="2:4",
            block_network=True,
            bucket_config=SandboxBucketConfig(
                mount_path="/data",
                read_only=True,
                only_dir=f"{self.instance_id}",
            ),
        )

        or_client = AsyncOpenReward(api_key=or_api_key)
        self.sandbox = or_client.sandbox(self.sandbox_settings)

        self.submitted = False

    @staticmethod
    def _infer_type(instance_id: str) -> str:
        if "impl" in instance_id:
            return "impl"
        elif "evol" in instance_id:
            return "evol"
        elif "arch" in instance_id:
            return "arch"
        return "unknown"

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "test":
            return _test_tasks
        return []

    async def get_prompt(self) -> list[TextBlock]:
        if self.task_type == "arch":
            prompt = self._get_arch_prompt()
        elif self.task_type == "impl":
            prompt = self._get_impl_prompt()
        else:  # evol
            prompt = self._get_evol_prompt()

        return [TextBlock(text=prompt)]

    def _get_arch_prompt(self) -> str:
        return f"""You are a data architect tasked with designing a comprehensive data architecture blueprint.

## Task

{self.instruction}

## Environment

You have access to a Linux sandbox with Python 3.12, DuckDB, and common data tools.
Use the `bash` tool to explore any provided data or context at `/data/`.

## Instructions

1. Analyze the business requirements described in the task.
2. Design a complete data architecture blueprint in YAML format.
3. The blueprint should define data models, staging/intermediate/marts layers, and relationships.
4. When ready, use the `submit` tool with your YAML blueprint as the `answer`.

You get one submission attempt."""

    def _get_impl_prompt(self) -> str:
        return f"""You are a data engineer tasked with building a complete data pipeline from scratch.

## Task

{self.instruction}

## Data

The initial repository with specifications is available at `/data/`. Explore it to understand the requirements.

## Environment

You have access to a Linux sandbox with Python 3.12, DuckDB, pandas, and common data tools.
Use the `bash` tool to run commands, write SQL files, and build your pipeline.

## Instructions

1. Explore the repository at `/data/` to understand the specifications and existing structure.
2. Build a complete multi-stage SQL pipeline (staging → intermediate → marts layers).
3. Create a `run.py` script that, when executed, produces the DuckDB database with all required tables.
4. Test your pipeline by running `run.py` and verifying the output.
5. When ready, use the `submit` tool to trigger evaluation (your pipeline will be run and compared against a gold database).

Your output will be evaluated by:
- Executing your `run.py` to produce a DuckDB database
- Comparing each table against the gold database using row-hash comparison
- Layer-weighted scoring: staging (15%), intermediate (25%), marts (60%)

You get one submission attempt."""

    def _get_evol_prompt(self) -> str:
        return f"""You are a data engineer tasked with extending an existing data pipeline.

## Task

{self.instruction}

## Data

The existing repository is available at `/data/`. It contains a working pipeline that you need to modify.

## Environment

You have access to a Linux sandbox with Python 3.12, DuckDB, pandas, and common data tools.
Use the `bash` tool to explore, modify, and test the pipeline.

## Instructions

1. Explore the existing repository at `/data/` to understand the current pipeline.
2. Read the new specification/requirements carefully.
3. Modify or extend the pipeline to meet the new requirements.
4. Create or update `run.py` to produce the updated DuckDB database.
5. Test your changes by running `run.py` and verifying the output.
6. When ready, use the `submit` tool to trigger evaluation.

Your output will be evaluated by:
- Executing your `run.py` to produce a DuckDB database
- Comparing each table against the gold database using row-hash comparison
- Layer-weighted scoring: staging (15%), intermediate (25%), marts (60%)

You get one submission attempt."""

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command in the sandbox environment."""
        result = await self.sandbox.run(params.command.strip())
        output, code = result

        if result.truncated:
            output = f"...(truncated, output exceeded limit)\n{output}"

        return ToolOutput(
            blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
            metadata={"output": output, "exit_code": code, "truncated": result.truncated},
            reward=0.0,
            finished=False,
        )

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit your work for evaluation.

        For DE-Arch: provide your YAML blueprint as the `answer`.
        For DE-Impl/DE-Evol: your pipeline will be executed and the resulting
        DuckDB database compared against the gold standard. Pass an empty answer
        or a brief description.
        """
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="Already submitted. Only one submission is allowed.")],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )

        self.submitted = True

        if self.task_type == "arch":
            return await self._eval_arch(params.answer)
        else:
            return await self._eval_pipeline()

    async def _eval_arch(self, blueprint: str) -> ToolOutput:
        """Evaluate a DE-Arch submission via LLM judge."""
        if not self.openai_client:
            return ToolOutput(
                blocks=[TextBlock(text="Evaluation error: OpenAI API key required for DE-Arch evaluation.")],
                metadata={"error": "no_openai_key"},
                reward=0.0,
                finished=True,
            )

        gold_entry = _arch_gold.get(self.instance_id, {})
        rubric = gold_entry.get("rubric", "")
        if not rubric:
            return ToolOutput(
                blocks=[TextBlock(text="Evaluation error: no rubric found for this task.")],
                metadata={"error": "no_rubric"},
                reward=0.0,
                finished=True,
            )

        try:
            eval_result = await evaluate_de_arch(
                client=self.openai_client,
                instruction=self.instruction or gold_entry.get("question", ""),
                blueprint=blueprint,
                rubric=rubric,
            )
        except Exception as e:
            logger.exception("DE-Arch evaluation error")
            return ToolOutput(
                blocks=[TextBlock(text=f"Evaluation error: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=True,
            )

        score = eval_result["score"]
        reward = score / 100.0

        result_text = f"""Submission Results:
- Score: {score:.1f}/100
- Raw Score: {eval_result['raw_score']}/{eval_result['max_score']}"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata=eval_result,
            reward=reward,
            finished=True,
        )

    async def _eval_pipeline(self) -> ToolOutput:
        """Evaluate a DE-Impl/Evol submission via DuckDB comparison."""
        config = _eval_config.get(self.instance_id)
        if not config:
            return ToolOutput(
                blocks=[TextBlock(text="Evaluation error: no evaluation config found for this task.")],
                metadata={"error": "no_config"},
                reward=0.0,
                finished=True,
            )

        database_file = config.get("database_file", "")
        if not database_file:
            return ToolOutput(
                blocks=[TextBlock(text="Evaluation error: no database_file in config.")],
                metadata={"error": "no_database_file"},
                reward=0.0,
                finished=True,
            )

        # Run the agent's pipeline in sandbox
        run_result = await self.sandbox.run("cd /workspace && python run.py")
        run_output, run_code = run_result

        if run_code != 0:
            logger.warning(f"run.py failed for {self.instance_id}: exit {run_code}")

        # Download the resulting DuckDB from sandbox
        try:
            pred_db_bytes = await self.sandbox.download(f"/workspace/{database_file}")
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(
                    text=f"Pipeline execution failed: could not find {database_file}.\n"
                    f"run.py output:\n{run_output}\n(exit {run_code})"
                )],
                metadata={"error": "no_database", "run_output": run_output, "run_code": run_code},
                reward=0.0,
                finished=True,
            )

        if not pred_db_bytes:
            return ToolOutput(
                blocks=[TextBlock(text=f"Pipeline produced empty database file: {database_file}")],
                metadata={"error": "empty_database"},
                reward=0.0,
                finished=True,
            )

        # Find gold DuckDB (stored per-task: gold/{instance_id}/{database_file})
        gold_db_path = _gold_db_dir / self.instance_id / database_file
        if not gold_db_path.exists():
            # Fallback: flat layout
            gold_db_path = _gold_db_dir / database_file
        if not gold_db_path.exists():
            return ToolOutput(
                blocks=[TextBlock(text="Evaluation error: gold database not found.")],
                metadata={"error": "no_gold_db"},
                reward=0.0,
                finished=True,
            )

        # Write pred DB to temp file for comparison
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
            tmp.write(pred_db_bytes)
            pred_db_path = Path(tmp.name)

        try:
            eval_result = evaluate_de_pipeline(
                pred_db_path=pred_db_path,
                gold_db_path=gold_db_path,
                config=config,
            )
        except Exception as e:
            logger.exception("DE pipeline evaluation error")
            return ToolOutput(
                blocks=[TextBlock(text=f"Evaluation error: {e}")],
                metadata={"error": str(e)},
                reward=0.0,
                finished=True,
            )
        finally:
            pred_db_path.unlink(missing_ok=True)

        score = eval_result["score"]
        reward = score / 100.0

        # Build layer summary
        layer_lines = []
        for layer in eval_result.get("layers", []):
            name = layer["layer_name"]
            ls = layer["layer_score"]
            lw = layer["layer_weight"]
            layer_lines.append(f"  - {name}: {ls:.1%} (weight: {lw})")

        result_text = f"""Submission Results:
- Score: {score:.1f}/100
- Layers:
{chr(10).join(layer_lines)}"""

        return ToolOutput(
            blocks=[TextBlock(text=result_text)],
            metadata={
                "score": score,
                "layers": eval_result.get("layers", []),
            },
            reward=reward,
            finished=True,
        )
