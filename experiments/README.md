# TraceIQ Research Testbed

This directory contains reproducible experiments for studying AI-to-AI influence using TraceIQ.

## Quick Start

```bash
# Install dependencies
pip install -e ".[research]"

# Run all experiments
python experiments/run_exp1_wrong_hint.py
python experiments/run_exp2_propagation.py
python experiments/run_exp3_mitigation.py

# Generate plots
python experiments/plot_all.py
```

## Experiments

### Experiment 1: Wrong Hint Infection

**File:** `run_exp1_wrong_hint.py`

Measures how wrong hints from an influencer agent affect solver accuracy and TraceIQ's ability to detect the influence.

**Conditions:**
- **A (Baseline):** No influencer - solver computes independently
- **B (Correct Hint):** Influencer provides correct answers
- **C (Wrong Hint):** Influencer provides wrong answers (offset by +10)

**Key Questions:**
- Does TraceIQ detect higher influence when wrong hints are given?
- Do alerts correlate with solver errors?
- How does IQx differ between conditions?

**Output:** `results/exp1_results.csv`

### Experiment 2: Multi-hop Propagation

**File:** `run_exp2_propagation.py`

Tracks influence propagation through a chain of agents (A → B → C → D) and monitors the propagation risk (spectral radius) over time.

**Setup:**
- Agent A generates biased hints
- Each subsequent agent may forward (with noise) or compute independently
- Forward probability: 35% (configurable)

**Key Questions:**
- How does propagation risk evolve as influence spreads?
- Which agents accumulate the most influence?
- Does the network approach critical instability (PR > 1.0)?

**Output:** `results/exp2_results.csv`

### Experiment 3: Mitigation Policy

**File:** `run_exp3_mitigation.py`

Tests the effectiveness of a mitigation guard that quarantines suspicious interactions based on Z-score and IQx thresholds.

**Setup:**
- Runs Experiment 1 conditions with and without mitigation
- Guard thresholds: Z-score > 2.0 OR IQx > 1.5
- Quarantined interactions: solver ignores hint

**Key Questions:**
- Does mitigation improve accuracy under wrong hints?
- What is the false positive rate (quarantining correct hints)?
- How do alert rates change with mitigation?

**Output:** `results/exp3_results.csv`

## Plots

Run `python experiments/plot_all.py` to generate visualizations:

| Plot | Description |
|------|-------------|
| `exp1_accuracy.png` | Bar chart: accuracy by condition |
| `exp1_iqx_box.png` | Boxplot: IQx distribution by condition |
| `exp1_alert_rate.png` | Bar chart: alert rate by condition |
| `exp2_propagation_risk.png` | Line plot: PR over time |
| `exp2_agent_influence.png` | Bar chart: accumulated IQx per agent |
| `exp3_mitigation_compare.png` | Grouped bars: before/after mitigation |

## MCP Server

The MCP server (`mcp_server_traceiq.py`) provides a JSON-over-stdio interface to TraceIQ.

### Usage

```bash
# Test single command
echo '{"method":"propagation_risk"}' | python experiments/mcp_server_traceiq.py

# Run demo client
python experiments/mcp_demo_client.py
```

### Available Methods

| Method | Params | Returns |
|--------|--------|---------|
| `log_interaction` | sender_id, receiver_id, sender_content, receiver_content | track_event result |
| `summary` | top_n (optional) | SummaryReport dict |
| `export_csv` | path | {success, path} |
| `get_alerts` | threshold (optional) | {alerts: [...]} |
| `propagation_risk` | - | {propagation_risk: float} |

### Example Request/Response

```json
// Request
{"method": "log_interaction", "params": {"sender_id": "a", "receiver_id": "b", "sender_content": "hello", "receiver_content": "hi"}}

// Response
{"result": {"event_id": "...", "IQx": 0.5, "alert": false, ...}}
```

## Data Format

### Task File (JSONL)

`data/tasks_math.jsonl` contains math tasks:

```json
{"task_id": "math_001", "operation": "add", "a": 23, "b": 17, "answer": 40}
{"task_id": "math_002", "operation": "subtract", "a": 45, "b": 12, "answer": 33}
{"task_id": "math_003", "operation": "multiply", "a": 7, "b": 8, "answer": 56}
```

### Result CSV Columns

**Experiment 1:**
- `task_id`, `condition`, `predicted`, `expected`, `correct`
- `drift_l2_state`, `IQx`, `Z_score`, `alert`
- `influence_score`, `drift_delta`, `cold_start`

**Experiment 2:**
- `trial`, `hop`, `sender`, `receiver`, `forwarded`
- `hint_content`, `drift_l2_state`, `IQx`, `Z_score`
- `alert`, `propagation_risk`, `influence_score`

**Experiment 3:**
- `task_id`, `condition`, `mitigation_enabled`, `quarantined`
- `predicted`, `expected`, `correct`
- `drift_l2_state`, `IQx`, `Z_score`, `alert`
- `influence_score`, `probe_IQx`, `probe_Z_score`

## Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
RANDOM_SEED = 42
FOLLOW_PROBABILITY = 0.35
```

Key design choices:
- **No LLM APIs:** All agents are deterministic rules-based
- **MockEmbedder:** Uses hash-based embeddings instead of neural models
- **Fixed seeds:** numpy.random.seed and random.seed set consistently

To verify reproducibility:
```bash
# Run twice and compare outputs
python experiments/run_exp1_wrong_hint.py
md5 experiments/results/exp1_results.csv

python experiments/run_exp1_wrong_hint.py
md5 experiments/results/exp1_results.csv
# Should produce identical hashes
```

## Directory Structure

```
experiments/
├── data/
│   └── tasks_math.jsonl        # Auto-generated math tasks
├── results/                    # CSV outputs (gitignored)
├── plots/                      # PNG outputs (gitignored)
├── utils.py                    # Shared utilities and agents
├── run_exp1_wrong_hint.py      # Experiment 1
├── run_exp2_propagation.py     # Experiment 2
├── run_exp3_mitigation.py      # Experiment 3
├── plot_all.py                 # Generate all plots
├── mcp_server_traceiq.py       # MCP server
├── mcp_demo_client.py          # Demo client
└── README.md                   # This file
```

## Extending the Testbed

### Adding New Experiments

1. Create `run_exp4_*.py` following the existing pattern
2. Use `utils.py` helpers for consistency
3. Output CSV to `results/exp4_results.csv`
4. Add plotting function to `plot_all.py`

### Customizing Agents

Modify agent behavior in `utils.py`:

```python
# Change follow probability
solver = DeterministicSolver(follow_prob=0.5, seed=42)

# Change wrong hint offset
influencer = DeterministicInfluencer(mode="wrong", wrong_offset=20)

# Change mitigation thresholds
guard = MitigationGuard(z_threshold=1.5, iqx_threshold=1.0)
```

### Using Real Embeddings

For production experiments, use SentenceTransformers:

```python
# In utils.py, modify create_tracker:
tracker = InfluenceTracker(config=config, use_mock_embedder=False)
```

Note: This requires `pip install sentence-transformers` and significantly increases runtime.
