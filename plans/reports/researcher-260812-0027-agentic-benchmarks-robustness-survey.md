# Agentic/Tool-Use Benchmarks: Robustness & Off-Diagonal Coverage Survey
**As of August 2026**

## Executive Summary

This report catalogs 29+ public agentic/tool-use benchmarks and assesses their coverage of six off-diagonal robustness dimensions that distinguish format-binding (mode-key overfitting) from deliberative tool-use reasoning. Key finding: **most benchmarks remain in-distribution; critical gaps exist in cross-harness transfer and schema perturbation robustness.**

---

## Benchmark Catalog

### Tier 1: Production-Grade, 2026 Standard Evals

| Benchmark | Maintainer | Primary Focus | Scoring Method | Link |
|-----------|-----------|---------------|----------------|------|
| **BFCL v4** | UC Berkeley Gorilla | Function calling (parallel, serial, multi-lang) | AST evaluation, success rate | https://gorilla.cs.berkeley.edu/leaderboard.html |
| **τ²-Bench (Tau2-Bench) Verified** | Amazon AGI / Sierra Research | Tool-agent-user interaction w/ policy adherence | Task success, policy compliance, coverage | https://github.com/amazon-agi/tau2-bench-verified |
| **SWE-Bench Verified** | OpenAI + Princeton | Real GitHub issue resolution | Patch correctness, trajectory traces | https://www.swebench.com |
| **WebArena** | CMU/UC Berkeley | DOM-based web task completion | Deterministic success functions (812 tasks) | https://webarena.dev |
| **OSWorld** | CMU | Desktop agent multi-app orchestration | Cross-app task success rate | https://osworld-benchmark.github.io |
| **GAIA** (Levels 1-3) | Hugging Face GAIA team | General assistant reasoning + tool use | Multi-step task success, reasoning traces | https://huggingface.co/spaces/gaia-benchmark/leaderboard |

### Tier 2: Active Research & Specialized Eval

| Benchmark | Maintainer | Primary Focus | Dimensions Emphasized | Link |
|-----------|-----------|---------------|-----|------|
| **VisualWebArena** | CMU | Screenshot-based web agents | Visual grounding, screenshot matching | https://benchmarks.darvinyi.com/benchmarks/webarena |
| **ComplexFuncBench** | Authors (Alibaba DAMO) | Multi-step constrained function calling | Long-context parameter reasoning | https://llm-stats.com/benchmarks/complexfuncbench |
| **MCP-Bench** | ICLR 2026 | Real MCP servers (28 servers, 250 tools) | Cross-tool coordination, parametrization | https://github.com/ShishirPatil/mcp-bench |
| **ToolSandbox** | Apple ML Research | Stateful, multi-turn tool interaction | Implicit state dependencies, feedback loops | https://machinelearning.apple.com/research/toolsandbox |
| **TIDE-Bench** | Authors | Task-aware diagnostic for tool-integrated reasoning | Process reliability, tool-use efficiency, cost | https://arxiv.org/abs/2605.09544 |
| **ToolMaze** ("When Tools Fail") | Authors | Dynamic replanning under perturbations | Error recovery, implicit semantic failures | https://arxiv.org/abs/2606.05806 |
| **RobustBench-TC** | Authors | Schema/format perturbation robustness | 22 perturbation types (observation, action, reward, transition) | https://arxiv.org/abs/2605.11928 |
| **AgentAbstain** | MIT-IBM Watson AI Lab | Calibrated abstention ("when not to act") | Pre-execution and runtime abstention signals | https://agentabstain.github.io |
| **EgoBench** | Authors | Egocentric multimodal tool-using agents | Visual grounding in real-world scenarios (1,045 tasks) | https://arxiv.org/abs/2605.27820 |
| **AgentEscapeBench** | Authors | Out-of-domain tool-grounded reasoning | Escape-room tasks + real tool execution | https://arxiv.org/abs/2605.07926 |
| **AgentNoiseBench** | Authors | Robustness under environmental noise | User-noise and tool-noise injection | https://arxiv.org/abs/2602.11348 |
| **ToolMisuseBench** | Authors | Tool misuse & recovery under faults | Deterministic fault injection, recovery trajectory | https://arxiv.org/abs/2604.01508 |
| **MINT** | Authors | Multi-turn with natural language feedback | Feedback integration, turn-level reasoning | https://arxiv.org/abs/2312.07322 (estimated) |
| **MetaTool** | Authors | Tool usage awareness, abstention decisions | "None" selection when no tool needed | https://arxiv.org/abs/2310.03128 |
| **AgentBoard** | Authors | Fine-grained progress & sub-skill analysis | 9 tasks, 1,013 environments, trajectory inspection | https://proceedings.neurips.cc/paper_files/paper/2024/file/877b40688e330a0e2a3fc24084208dfa-Paper-Datasets_and_Benchmarks_Track.pdf |
| **ASTRA-Bench** | Authors | Tool use with personal user context | Context awareness, user preference adherence | https://arxiv.org/abs/2603.01357 |
| **ToolBeHonest** | Authors | Multi-level hallucination diagnostics | Hallucination sources in tool-augmented LLMs | https://arxiv.org/abs/2406.20015 |
| **Seal-Tools (SealTool)** | Authors | Extensive API coverage (4,076 APIs) | Data leakage minimization, robustness | https://github.com/sambanova/Seal-Tools |

### Tier 3: Earlier/Specialized/Niche

| Benchmark | Primary Focus | Status 2026 | Coverage Note |
|-----------|---------------|-----------|------------------|
| **ToolBench/ToolEval** | 16,464 RapidAPI endpoints | Fine-tuning data source; expensive eval | Primarily used for dataset generation |
| **API-Bank** | API tool calling | Active research | Traditional baseline |
| **AgentBench** | Multi-environment interaction | Superseded by Tau2/SWE-Bench Verified | Earlier generation |
| **NexusBench** | Function call + agent tasks | Niche/archival | Benchmarking quality comparison |
| **ToolEmu** | Risky agent behaviors (36 tools, 144 cases) | Safety-focused | High-stakes tool misuse scenarios |
| **AgentHarm** | Agent safety evaluation | Safety-focused | Adversarial behavior testing |
| **MCPWorld** | Tool selection/parametrization | Emerging | MCP ecosystem eval |
| **OSWorld-MCP** | MCP tool invocation in computer use | Emerging | Desktop + MCP hybrid |
| **MultiWOZ 2.2** | Dialogue state tracking (8,437 dialogues) | Legacy baseline | Not agentic per se |

---

## Off-Diagonal Dimension Coverage Matrix

### Scoring: ✓ = Direct evaluation, ◐ = Partial/indirect, ✗ = Not covered

| Benchmark | (a) Format/<br/>Schema Perturb | (b) Unseen/<br/>Held-out Tools | (c) Abstention<br/>No Tool | (d) Tool Failure<br/>+ Recovery | (e) Cross-Harness<br/>Transfer | (f) Multi-Turn<br/>State Track |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| **BFCL v4** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **τ²-Bench Verified** | ✗ | ◐ | ◐ | ✓ | ✗ | ✓ |
| **SWE-Bench Verified** | ◐ | ◐ | ✗ | ✓ | ◐ | ✓ |
| **WebArena** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **VisualWebArena** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **OSWorld** | ✗ | ◐ | ✗ | ◐ | ✗ | ✓ |
| **GAIA** | ✗ | ✓ | ✗ | ◐ | ✗ | ✓ |
| **ComplexFuncBench** | ◐ | ✗ | ✗ | ✗ | ✗ | ◐ |
| **MCP-Bench** | ✓ | ✓ | ◐ | ◐ | ◐ | ✓ |
| **ToolSandbox** | ✗ | ◐ | ✗ | ◐ | ✗ | ✓ |
| **TIDE-Bench** | ✗ | ✗ | ✗ | ◐ | ✗ | ✓ |
| **ToolMaze** | ✓ | ✗ | ✗ | ✓ | ✗ | ◐ |
| **RobustBench-TC** | **✓** | ✗ | ✗ | ✓ | ✗ | ✗ |
| **AgentAbstain** | ✗ | ✗ | **✓** | ✗ | ✗ | ◐ |
| **EgoBench** | ✗ | ✗ | ✗ | ◐ | ✗ | ✓ |
| **AgentEscapeBench** | ✗ | **✓** | ✗ | ✓ | ✗ | ✓ |
| **AgentNoiseBench** | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **ToolMisuseBench** | ✗ | ✗ | ✗ | **✓** | ✗ | ◐ |
| **MINT** | ✗ | ✗ | ✗ | ◐ | ✗ | ✓ |
| **MetaTool** | ✗ | ✗ | **✓** | ✗ | ✗ | ✗ |
| **AgentBoard** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| **ASTRA-Bench** | ✗ | ◐ | ✗ | ✗ | ✗ | ✓ |
| **ToolBeHonest** | ✓ | ✗ | ✗ | ✗ | ✗ | ◐ |
| **Seal-Tools** | ✗ | ◐ | ✗ | ✗ | ✗ | ✗ |
| **ToolBench** | ◐ | ✓ | ✗ | ✗ | ✗ | ◐ |
| **API-Bank** | ✗ | ◐ | ✗ | ✗ | ✗ | ✗ |
| **ToolEmu** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **AgentHarm** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **MCPWorld** | ✗ | ✓ | ✗ | ◐ | ✗ | ✗ |

---

## Small Model vs. Large Model Performance Gaps

### Leaderboard Patterns (August 2026)

**BFCL v3/v4:**
- Top frontier cluster: GLM 4.5 (76.7%), Claude Opus 4.7 (76.6%), Gemini 3.1 Flash (76.5%)
- Proprietary vs. open-weight gap: **8–15 percentage points**
- Closed-API vs. top open-weight gap: **3–4 percentage points**
- Gap *largest on complex chains*; most models handle single-function calls; only top 3–5 orchestrate multi-step workflows reliably

**OccuBench (professional task agents):**
- Gemini Pro vs. Flash-Lite: **11.0% gap**
- Claude Opus vs. Sonnet 4.6: **7.1% gap**

**ToolSandbox:**
- Significant proprietary vs. open-source gap observed
- Complex tasks (state dependency, canonicalization) remain challenging for all

**Small Model Struggles (Llama 3B, similar):**
- Working memory bottleneck: dual task context + tool-evaluation frame exceeds capacity
- Structured output formation (JSON with correct schema) is critical blocker
- Potential mitigation: tool-use in context can close gap with frontier models

**Key Insight:** Performance ceiling is *task complexity*, not model size alone. Multi-turn and complex topologies trap smaller models in trial-and-error loops more severely.

---

## Gap Analysis: Off-Diagonal Dimensions Least Covered by Public Benchmarks

### Critical Gaps

**1. Cross-Harness/Scaffold Transfer (Dimension E)**
- **Coverage:** ~3 benchmarks with partial evaluation (SWE-Bench Verified, MCP-Bench, some GAIA splits)
- **Why it matters:** Harness engineering (prompt frames, tool descriptors, execution budgets) dominates agent performance. SFT on fixed harnesses teaches format binding to *that scaffold* specifically.
- **What's missing:** No public benchmark systematically tests harness transfer (e.g., rename parameters, reorder tool order, change API endpoint structure, alter execution budget).
- **Research evidence:** Cross-harness GRPO pools trajectories across harness versions; pure harness-only overfits. Yet no benchmark measures this decay.
- **Status:** Emergent issue in 2026; community aware but no standard eval yet.

**2. Held-Out/Unseen Tool Vocabularies (Dimension B)**
- **Coverage:** ~4 benchmarks directly (AgentEscapeBench ✓, ToolBench ◐, GAIA ◐, MCPWorld ✓); many test domain split but not tool *novelty*.
- **Why it matters:** In-distribution tool memorization is the "easy" case. Real-world agents must adapt to new APIs, libraries, or service changes mid-deployment.
- **What's missing:** Benchmarks with held-out tool subsets (e.g., train on 80% of ToolBench, evaluate on held-out 20% unseen tools in same domains). Currently, most evals use train/test splits on *tasks*, not *tools*.
- **AgentEscapeBench exception:** Explicitly constructs unfamiliar escape-room tasks + real tool execution.
- **Status:** Recognized problem; few solutions deployed.

**3. Schema/Format Perturbation (Dimension A)**
- **Coverage:** ~5 benchmarks with explicit schema tests (RobustBench-TC ✓✓, ToolMaze ✓, AgentNoiseBench ✓, ToolBeHonest ✓, ComplexFuncBench ◐)
- **Why it matters:** "Format binding" is the core hypothesis—models trained on {JSON, OpenAI Schema, type hints} may emit trained *structures* regardless of actual reasoning.
- **RobustBench-TC 22 perturbations:**
  - Schema drift (extra/missing fields in responses)
  - Type mismatches
  - Response format changes (XML vs JSON)
  - Transient vs. permanent failures
- **Gap:** Most benchmarks don't *perturb* schemas during eval. They measure accuracy on fixed schemas. ToolMaze and RobustBench-TC are exceptions.
- **Status:** Emerging as critical post-2024; RobustBench-TC (May 2026) and ToolMaze (June 2026) are newest dedicated evals.

**4. Tool Failure + Error Recovery Trajectory (Dimension D)**
- **Coverage:** ~8 benchmarks cover this (ToolMaze ✓, ToolMisuseBench ✓, Tau2 ✓, AgentNoiseBench ✓, RobustBench-TC ✓, AgentCE-Bench ◐, SWE-Bench ✓, ToolSandbox ◐)
- **Why it matters:** Real deployment injects: API downtime, malformed responses, rate limits, permission errors, network delays.
- **ToolMaze findings:** Perturbation Recovery Rate (PRR) drops ~37% under implicit semantic failures; agents over-trust corrupted outputs.
- **Gap:** Explicit fault injection is now standard, but *recovery trajectory quality* (replanning, context management, coherence) is underspecified. Most evals just measure task success or binary recover/fail.
- **Status:** Active focus; multiple 2026 benchmarks address this.

### Moderate Gaps

**5. Abstention / "No Tool Needed" Cases (Dimension C)**
- **Coverage:** ~3 benchmarks (AgentAbstain ✓✓, MetaTool ✓, Tau2 ◐)
- **Why it matters:** Many tasks don't need tool calls. Models trained on in-distribution tool usage may hallucinate tools. Calibrated refusal is safety-critical.
- **AgentAbstain findings:** Even best models (Gemini 3.1 Pro) only achieve 59.5% paired accuracy on abstain vs. act decision.
- **MetaTool:** Tests "None" selection; finds LLMs struggle with confidence calibration.
- **Gap:** Most production benchmarks (BFCL, WebArena, OSWorld) don't explicitly evaluate abstention. Abstain scenarios are either absent or implicit.
- **Status:** Specialized eval; not yet routine for production benchmarks.

**6. Multi-Turn State Tracking (Dimension F)**
- **Coverage:** ~12 benchmarks cover this (ToolSandbox ✓, Tau2 ✓, MINT ✓, WebArena ✓, OSWorld ✓, GAIA ✓, TIDE-Bench ✓, SWE-Bench ✓, VisualWebArena ✓, ASTRA-Bench ✓, ToolMaze ◐, AgentBoard ✓)
- **Why it matters:** State consistency across turns (context management, memory, variable tracking, dialogue history) is essential for multi-step tasks.
- **Coverage status:** This is well-covered in dialogue + interaction benchmarks; less rigorous in code/API benchmarks (BFCL measures single turn).
- **Gap:** While many measure success, few diagnose *where* state coherence fails—e.g., information loss, inconsistent assumptions, stale context.
- **Status:** Mature topic; not a critical gap.

---

## Format Binding vs. Deliberation: Diagnostic Interpretation

### What the Gaps Reveal

**Hypothesis:** SFT on formatted traces creates **format binding** (mode-key behavior) that conflates with deliberative reasoning.

**Evidence from benchmarks:**
1. **RobustBench-TC:** Format perturbations degrade performance, showing reliance on fixed schemas.
2. **ToolMaze:** Over-trust in corrupted outputs suggests pattern matching rather than verification.
3. **AgentAbstain:** <60% accuracy on abstain pairs indicates miscalibration—likely memorized tool-call patterns rather than genuine task understanding.
4. **Small model gaps:** Working memory bottleneck suggests structural (reasoning) limitations, but gap widens on *structured output* formation, pointing to schema binding.

### Off-Diagonal Tests That Would Distinguish

**Strongest discriminators:**
- **(E) Cross-harness transfer:** If a model learns task reasoning, it should transfer across harness changes (prompt rewording, API renames). Format binding stays local to one harness.
- **(A) Schema perturbation:** Genuine tool-use reasoning should degrade gracefully under schema drift. Format binding fails abruptly.
- **(B) Unseen tools:** Memorized tools won't generalize. Learned tool-use patterns (selection, parameter grounding) should transfer.
- **(D) Error recovery trajectory:** Pattern-matched models may emit trained recovery steps; deliberative models construct context-appropriate repairs.

**Weakest discriminators (already well-covered):**
- (C) Abstention: Specialized but easier to test directly.
- (F) Multi-turn state: Covered by most dialogue evals; less discriminative.

---

## Leaderboard Data: Notable Small-to-Large Gaps

| Benchmark | Small Model | Large Model | Gap | Reference Model Pair |
|-----------|-------------|-------------|-----|----------------------|
| **BFCL v3** | ~55–62% (Gemini Flash-Lite) | ~76% (Claude Opus, GLM 4.5) | ~14–21 pts | Frontier vs. mid-tier |
| **OccuBench** | ~42% (Flash-Lite) | ~53% (Gemini Pro) | **11 pts** | Vision-heavy tasks |
| **OccuBench** | ~57% (Sonnet 4.6) | ~64% (Claude Opus) | **7.1 pts** | Anthropic pair |
| **ToolSandbox** | Large proprietary gap observed | — | ~5–10 pts (est.) | Proprietary vs. open |
| **GAIA L1** | Approaching ceiling | Ceiling | ~3 pts | 2026 saturation |

---

## Summary: Benchmark Saturation & 2026 Frontier

### What's Saturated
- **Single-turn function calling (BFCL):** Frontier models cluster at 76–77%. Additional scaling offers marginal gains.
- **Deterministic browser tasks (WebArena):** Top models >71%. DOM-based + deterministic success functions favor rigid strategies.
- **Dialogue state tracking (MultiWOZ 2.2):** Long-standing baseline; incremental improvements.

### What's Active/Emergent
- **Robustness & perturbation (2026 new):** RobustBench-TC, ToolMaze, AgentNoiseBench push the frontier on fault tolerance.
- **Real-world complexity (MCP-Bench):** 28 live MCP servers + 250 tools; harder coordination requirements.
- **Abstention calibration (AgentAbstain):** New focus; <60% best performance shows significant headroom.
- **Cross-harness transfer:** Research-only; no production eval yet.

---

## Unresolved Questions

1. **Format binding signature:** Can we design a minimal perturbation suite that unambiguously separates memorized format compliance from task reasoning? (RobustBench-TC is ~22 perturbations; is that sufficient/necessary?)

2. **Cross-harness transfer metrics:** How much of the 37% PRR drop in ToolMaze is harness overfitting vs. genuine error recovery failure? What transfer curve should we expect?

3. **Abstention coverage in in-distribution benchmarks:** Are abstention cases *systematically* underrepresented in production benchmarks (e.g., BFCL, WebArena) due to task selection bias, or are they rare in practice?

4. **Held-out tool evaluation standard:** Should a canonical "held-out-tools" split become routine for benchmarks like BFCL (e.g., 80/20 tool train/test)? Who would maintain it?

5. **Leaderboard gaming:** Multiple evals report benchmarks exploitable without genuine reasoning (April 2026 incident: agent scored 100% on 7/8 leading benchmarks via infrastructure flaws). How prevalent is this across public leaderboards?

6. **MCP ecosystem maturity:** Will MCP-Bench, MCPWorld, and MCP-specific evals become standard, or remain niche? What's the adoption trajectory?

7. **Small model scaling wall:** Is the observed tool-calling gap (~7–21 pts) a working-memory bottleneck (recoverable with context) or a fundamental reasoning limitation?

8. **State consistency diagnosis:** Which multi-turn failures are consistency-specific (stale context, information loss) vs. reasoning failures (wrong plan)? Current evals measure task success, not root cause.

---

## Recommendations for Your Glyph Research

**If format binding is your hypothesis:**
1. **Prioritize dimension (A), (E), (B)** in order: Schema perturbation (RobustBench-TC), cross-harness transfer (design custom harness transfer suite), unseen tools (AgentEscapeBench or design held-out subset).
2. **Run comparisons on saturated benchmarks (BFCL, WebArena)** to establish baseline, then probe off-diagonal robustness.
3. **Diagnostic combo:** RobustBench-TC + held-out-tools split + cross-harness GRPO traces. This trio should isolate format binding vs. reasoning.
4. **Note:** No single public benchmark covers all six dimensions. You may need to synthesize or augment.

**Benchmark selection for your work:**
- **For in-distribution confirmation:** BFCL v4 (function calling), Tau2 (policy + state), SWE-Bench Verified (complex reasoning).
- **For off-diagonal probing:** RobustBench-TC (a), AgentEscapeBench (b), MetaTool (c), ToolMaze (d), [design custom] (e), ToolSandbox (f).

---

## Sources

- https://gorilla.cs.berkeley.edu/leaderboard.html (BFCL v4)
- https://github.com/amazon-agi/tau2-bench-verified (Tau2-Bench Verified)
- https://www.swebench.com (SWE-Bench)
- https://webarena.dev (WebArena)
- https://osworld-benchmark.github.io (OSWorld)
- https://huggingface.co/spaces/gaia-benchmark/leaderboard (GAIA)
- https://arxiv.org/abs/2605.09544 (TIDE-Bench)
- https://arxiv.org/abs/2606.05806 (ToolMaze)
- https://arxiv.org/abs/2605.11928 (RobustBench-TC)
- https://agentabstain.github.io (AgentAbstain)
- https://arxiv.org/abs/2605.27820 (EgoBench)
- https://arxiv.org/abs/2605.07926 (AgentEscapeBench)
- https://arxiv.org/abs/2602.11348 (AgentNoiseBench)
- https://arxiv.org/abs/2604.01508 (ToolMisuseBench)
- https://arxiv.org/abs/2310.03128 (MetaTool)
- https://machinelearning.apple.com/research/toolsandbox (ToolSandbox)
- https://arxiv.org/abs/2406.20015 (ToolBeHonest)
- https://benchmarkingagents.com (2026 benchmark survey)
- https://arxiv.org/abs/2607.12227 (Harness Evolution for LLM Agents)
- https://arxiv.org/abs/2504.19277 (Small Models, Big Tasks)
- https://llm-stats.com/benchmarks/complexfuncbench (ComplexFuncBench)
- https://github.com/ShishirPatil/mcp-bench (MCP-Bench)
- https://arxiv.org/abs/2407.12871 (MetaTool meta-task augmentation)
