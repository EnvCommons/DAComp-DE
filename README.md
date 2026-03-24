# DAComp-DE

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/DAComp-DE)

## Description

**DAComp-DE** (Data Agent Competition — Data Engineering) is an environment for evaluating AI agents on multi-stage data engineering tasks. Agents build, extend, or design dbt-style SQL pipelines.

- **DE-Impl (30 tasks)**: Build a complete SQL pipeline from scratch (staging → intermediate → marts).
- **DE-Evol (50 tasks)**: Modify or extend an existing pipeline to meet new requirements.
- **DE-Arch (30 tasks)**: Design a comprehensive data architecture blueprint in YAML.

## Capabilities

- SQL pipeline construction (DuckDB, dbt-style layers)
- Repository exploration and modification
- Data architecture design (YAML blueprints)
- Python scripting and data tooling

## Compute Requirements

- Sandbox: 2 CPU / 4GB memory per session
- LLM evaluation: OpenAI API access (gpt-5-mini) for DE-Arch scoring only

## License

[MIT License](https://github.com/ByteDance-Seed/DAComp/blob/main/LICENCE)

## Tasks

| Sub-type | Split | Count | Description |
|----------|-------|-------|-------------|
| DE-Impl | test | 30 | Build SQL pipeline from scratch |
| DE-Evol | test | 50 | Extend existing SQL pipeline |
| DE-Arch | test | 30 | Design architecture blueprint |

## Reward Structure

### DE-Impl/Evol (Deterministic, 0–100 scale)

Row-hash multiset comparison of each table against gold DuckDB, with layer-weighted scoring:
- Staging: 15%
- Intermediate: 25%
- Marts: 60%

### DE-Arch (LLM-judged, 0–100 scale)

LLM evaluates YAML blueprint against rubric with evidence-based scoring.

## Data

- **Source**: [HuggingFace](https://huggingface.co/DAComp) (dacomp-de, dacomp-de-gold)
- **DE**: 110 task repositories, 80 gold DuckDB databases, 30 architecture rubrics

## Tools

| Tool | Description |
|------|-------------|
| `bash` | Execute bash commands in the sandbox (Python, SQL, DuckDB, file I/O) |
| `submit` | Submit work for evaluation (YAML for DE-Arch, triggers pipeline run for DE-Impl/Evol) |

## Time Horizon

Multi-turn. DE-Impl: 20–50 tool calls, DE-Evol: 10–30, DE-Arch: 5–15.

## Environment Difficulty

Even state-of-the-art agents achieve success rates under 20% on DE-Impl/Evol.

## Other Environment Requirements

- OpenAI API key for DE-Arch LLM evaluation
- OpenReward API key for sandbox access

## Safety

Tasks involve synthetic/public data engineering schemas. No sensitive personal data. Sandboxes are network-isolated.

## Citations

```bibtex
@misc{lei2025dacomp,
      title={DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle},
      author={Fangyu Lei and Jinxiang Meng and Yiming Huang and Junjie Zhao and Yitong Zhang and Jianwen Luo and Xin Zou and Ruiyi Yang and Wenbo Shi and Yan Gao and Shizhu He and Zuo Wang and Qian Liu and Yang Wang and Ke Wang and Jun Zhao and Kang Liu},
      year={2025},
      eprint={2512.04324},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.04324},
}
```
