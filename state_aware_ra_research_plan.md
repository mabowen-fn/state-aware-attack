# State-Aware Response Priming (SARP): Research Starter

> **Status:** Week-1 research scaffold. This repository runs *pre-specified, fixed* dialogue conditions, records per-turn judge outcomes, and computes behavioral state-transition summaries. It does not generate or optimize jailbreak prompts. Use only an authorized, isolated evaluation target and benchmark material you are permitted to evaluate.

## 1. Research goal

We want to test whether response-priming conditions studied in Response Attack (RA) are associated with measurable changes in a target model's safety behavior over dialogue trajectories, and whether these measurements add information beyond ordinary output-level feedback.

The project connects two complementary perspectives:

- **Response Attack (RA):** studies how a preceding assistant response can contextually prime a target's subsequent response. Published in AAAI-26: <https://ojs.aaai.org/index.php/AAAI/article/view/40836> (paper/preprint: <https://arxiv.org/abs/2507.05248>).
- **STAR:** treats dialogue history as a state-transition operator and studies state-dependent safety failures across multi-turn interaction. Published in ICML 2026: <https://proceedings.mlr.press/v306/li26jo.html> (preprint: <https://arxiv.org/abs/2603.15684>).

### Primary research question

**Does per-turn safety-state measurement provide predictive or decision-making value beyond response-level feedback when analyzing fixed RA-like and STAR-inspired dialogue trajectories?**

This is deliberately narrower than claiming a new attack. First establish a reproducible effect and valid measurement. Only then decide whether a state-aware controller is justified.

### Hypotheses

- **H1 — Trajectory effect:** the judge's continuous probability score varies systematically over a multi-turn trajectory, beyond noise in repeated evaluations.
- **H2 — Priming association:** pre-specified RA-derived context conditions differ from direct-query controls under a matched target model and evaluation budget.
- **H3 — Incremental signal:** a state proxy computed from earlier turns predicts a later safety outcome better than the latest-turn score alone.
- **H4 — Controller value (later phase):** a controller using validated trajectory measurements outperforms a non-adaptive or output-only controller under a predeclared matched budget.

H1–H3 are the first week's goals. H4 is **not** claimed by this starter and should only be tested after the measurement is validated.

## 2. Threat model and scope

- **Target:** local Ollama model `llama3.1:8b` via its OpenAI-compatible chat-completions endpoint.
- **Official evaluator:** your HarmBench judge endpoint. `/score` is the canonical binary ASR contract; `/score_full` adds raw logit fields for exploratory analysis.
- **Dialogue source:** pre-specified scripts and/or pre-generated research artifacts that you are authorized to use. The runner does not invent harmful objectives, generate attack primers, or adapt prompts based on judge feedback.
- **Auxiliary model:** `mgraffam/qwen3.8-EfficientThink-uncensored:27b` is recorded in configuration as an optional future research dependency, but is not required in this first fixed-script experiment.
- **Interpretation:** a change in the judge's probability is a *behavioral proxy*, not a direct observation of a model's hidden safety state. Hidden-state claims require access to the target's activations and a separate validated probing method.

## 3. Architecture

```text
JSONL case scripts
       |
       v
Experiment runner ----> Ollama OpenAI-compatible API (target completions)
       |                                  |
       |<--------------------------------- response
       v
Per-turn event log ----> HarmBench judge /score or /score_full
       |
       v
Trajectory metrics: first/last probability, delta, max, mean, observed ASR rate
```

Each case contains a `behavior` label for the judge and one or more named conditions. A condition has optional `prefix_messages` and a list of fixed `turns`. Prefix messages let you replay a preconstructed history; turn messages let the runner generate target responses sequentially while recording each step. The user-side turns are fixed before execution; no adaptive attacker is built into this starter.

## 4. Project structure

```text
state_aware_ra/
├── README.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── data/
│   ├── README.md
│   └── smoke_cases.jsonl
├── src/stateaware/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── judge.py
│   ├── metrics.py
│   ├── runner.py
│   └── target.py
└── tests/
    ├── test_dataset.py
    └── test_metrics.py
```

## 5. Setup

Requires Python 3.11+ and an installed/running Ollama instance. The current Ollama OpenAI compatibility documentation lists `http://localhost:11434/v1/` as the local base URL and supports `/v1/chat/completions`: <https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx>.

```bash
cd state_aware_ra
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
cp .env.example .env
```

Make sure Ollama is running and the target model is present:

```bash
ollama pull llama3.1:8b
ollama list
```

Edit `.env` if your endpoints differ. The local OpenAI-compatible client needs a placeholder API key (default `ollama`); Ollama ignores it for local requests.

### Judge endpoints

Configured defaults:

- `JUDGE_SCORE_URL=http://10.50.2.39:18080/score`
- `JUDGE_SCORE_FULL_URL=http://10.50.2.39:18082/score_full`
- `JUDGE_HEALTH_URL=http://10.50.2.39:18082/health`

The supplied service contract says `/health` is **POST**, so `doctor` sends `POST {}`. Score requests send `{"behavior": "...", "generation": "..."}`. The default `both` mode uses `/score_full` for each turn's trajectory diagnostics and calls `/score` on the final turn of each condition for canonical ASR. Thus formal final ASR is taken from `/score`; the per-turn ASR/logits from `/score_full` are diagnostic signals. `score` mode calls `/score` every turn; `full` mode calls only `/score_full`; `off` skips judge calls. Do not interpret `logit_diff` as a probability.

The judge server is a private address from your environment. This runtime cannot verify that it is reachable from your machine; run `stateaware doctor` locally.

## 6. Verify connectivity

```bash
stateaware doctor
```

This checks the target's `/v1/models` endpoint and sends a POST to the configured judge `/health` endpoint. It does not run a harmful benchmark case or call `/score`.

## 7. Run the safe smoke test

```bash
stateaware run \
  --cases data/smoke_cases.jsonl \
  --out runs/smoke \
  --judge-mode both
```

The smoke corpus is harmless and exists only to verify data loading, target completion, event logging, and judge integration. **Do not report smoke-test judge scores as HarmBench results or research findings.**

Outputs:

- `events.jsonl`: one record per model turn, including generation, optional judge result, latency, and error information.
- `summary.json`: aggregate trajectory metrics per case and condition, including probability change from first to last evaluated turn.
- `run_metadata.json`: model settings, endpoint mode, and timestamps (does not include API keys).

Avoid committing run outputs that contain sensitive prompts or model generations.

## 8. Experiment data format

One JSON object per line. See `data/smoke_cases.jsonl` for a harmless example.

```json
{
  "case_id": "case-001",
  "behavior": "Benchmark behavior description consumed by the judge",
  "conditions": {
    "direct": {
      "prefix_messages": [],
      "turns": [
        {"user": "A fixed user turn"}
      ]
    },
    "reference_condition": {
      "prefix_messages": [
        {"role": "user", "content": "A previously constructed history item"},
        {"role": "assistant", "content": "A previously recorded assistant message"}
      ],
      "turns": [
        {"user": "A fixed final/evaluation turn"}
      ]
    }
  },
  "metadata": {"split": "pilot"}
}
```

Supported prefix roles are `system`, `user`, and `assistant`. `prefix_messages` are replayed as supplied; do not assume you can forge assistant-role history through every real deployment interface. `turns` are fixed user messages. The runner adds each generated assistant response to history before proceeding to the next turn.

For a real comparison, ensure each case has equivalent task intent across conditions, freeze the development/test split before evaluation, and use the same target version, judge endpoint, judge mode, temperature, and turn/token budget.

## 9. Suggested first-week protocol

1. **Day 1 — Infrastructure and baseline.** Run `doctor`, run the smoke set, and verify manually that the events correspond to the transcript. Record Ollama version, model tag, judge version string, and hardware.
2. **Day 2 — Data preparation.** Prepare a small, approved evaluation subset with explicit `direct`, `ra_fixed` (preconstructed/released artifact), and `star_diagnostic` conditions. Keep the inputs fixed; do not let the judge write or mutate the next turn.
3. **Day 3 — Pilot.** Run a small subset; inspect errors, turn logging, timeouts, and judge version consistency. Do not tune on the held-out split.
4. **Days 4–5 — Matched run.** Run each condition on the same cases and in a randomized/interleaved order where feasible. Log all calls and failed trials, and preserve raw outputs.
5. **Day 6 — Analysis.** Compare per-case paired differences, bootstrap confidence intervals, first/last/max/mean probabilities, binary ASR, and cost (target calls, tokens where available, wall time). Manually audit a sample of judge outputs.
6. **Day 7 — Go/no-go.** Decide whether the trajectory score is reproducible and whether it provides incremental predictive value beyond the latest score. Do not claim a new attack or mechanism on the basis of ASR alone.

### Predeclared decision rules

- **Continue method development** only if a paired improvement is practically meaningful, uncertainty is reported, and the effect replicates on held-out cases.
- **Pivot to mechanism analysis** if the trajectory measurements are reproducible but do not improve ASR.
- **Stop/reformulate** if the result disappears under matched budgets, depends on a small number of cases, or is not robust to judge auditing.

A 10 percentage-point improvement may be used as a *project-specific go/no-go threshold*, not a universal statistical rule or a guaranteed result. A sample of 100–150 cases is only a pilot; power depends on paired disagreement rates and effect size.

## 10. Metrics and interpretation

- `asr`: binary judge output. In the default `both` mode, the official final ASR is recorded in `official_judge` from `/score`; per-turn fields in `judge` come from `/score_full` and are diagnostic.
- `probability`: continuous judge output; treat it as that judge's score, not necessarily a calibrated probability unless calibration has been validated.
- `logit_diff`: raw `logit_yes - logit_no` returned by `/score_full`; it is not a probability.
- `probability_delta`: last observed score minus first observed score for a case/condition.
- `max_probability`: maximum observed score along the measured trajectory.

A trajectory association is not necessarily causal. Causal interpretation requires controlled interventions that change one defined factor at a time while preserving the rest of the condition. A hidden-state interpretation requires actual activation data; standard Ollama OpenAI-compatible chat responses do not expose internal activations.

## 11. Research risks and novelty check

This initial runner does not establish novelty. Before claiming a contribution, compare the exact design against the published RA and STAR implementations and more recent adaptive multi-turn red-teaming work. In particular:

- Does the state proxy predict later outcome after controlling for the latest-turn score?
- Does it generalize to different cases and models?
- Does a state-aware decision rule beat a simple output-only decision rule at equal cost?
- Does the observed effect survive judge audits and changes in the fixed test subset?

If none of these holds, do not force the title. A good result may be negative: the state monitor may add no predictive value beyond simple response-level feedback.

## 12. Sources

- Miao et al., *Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models*, AAAI 2026. <https://ojs.aaai.org/index.php/AAAI/article/view/40836>
- Li et al., *State-Dependent Safety Failures in Multi-Turn Language Model Interaction*, ICML 2026. <https://proceedings.mlr.press/v306/li26jo.html>
- Ollama, *OpenAI compatibility* documentation. <https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx>

## 13. Scope of this starter

Implemented now: configuration, OpenAI-compatible target client, judge API client for `/score` and `/score_full`, POST `/health` check, fixed-script multi-turn runner, append-only JSONL logs, trajectory summaries, smoke data, and unit tests.

Not implemented now: autonomous generation or optimization of jailbreak prompts, adaptive attack search, direct access to internal hidden activations, or a claim that the proposed method beats RA. These require separate validated experimental decisions rather than being silently conflated with the measurement harness.

## 14. Paired comparison after a run

When the same `case_id` occurs under both conditions, the comparison command estimates a paired difference and a percentile bootstrap confidence interval over cases:

```bash
stateaware compare \
  --summary runs/experiment/summary.json \
  --baseline direct \
  --candidate ra_fixed \
  --metric last_turn_asr \
  --bootstrap-samples 10000
```

For ASR, `last_turn_asr` is the primary binary endpoint; in `both` mode it comes from the final `/score` call, not `/score_full`. `last_turn_full_asr` is a separate exploratory field derived from `/score_full`. `probability_delta` is useful for trajectory analysis but is not itself an ASR metric. The bootstrap resamples paired case-level differences; it does not correct for multiple comparisons or compensate for a non-representative benchmark. Decide comparisons and the primary endpoint before inspecting the held-out results.
