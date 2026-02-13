# CLI Reference

TraceIQ provides a command-line interface for common operations.

## Installation

The CLI is installed automatically with the package:

```bash
pip install traceiq
```

Verify installation:

```bash
traceiq --version
```

## Commands

### `traceiq init`

Initialize a new TraceIQ database.

```bash
traceiq init [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database (default: `traceiq.db`)
- `--config PATH` - Path to JSON config file

**Example:**
```bash
traceiq init --db myproject.db
```

### `traceiq ingest`

Ingest interactions from a JSONL file.

```bash
traceiq ingest INPUT_FILE [OPTIONS]
```

**Arguments:**
- `INPUT_FILE` - Path to JSONL file with interactions

**Options:**
- `--db PATH` - Path to SQLite database (default: `traceiq.db`)
- `--mock-embedder` - Use mock embedder (no sentence-transformers required)

**Input Format:**
Each line should be a JSON object:
```json
{"sender_id": "agent_a", "receiver_id": "agent_b", "sender_content": "Hello", "receiver_content": "Hi there"}
```

**Example:**
```bash
traceiq ingest interactions.jsonl --db myproject.db --mock-embedder
```

### `traceiq summary`

Show summary of tracked interactions.

```bash
traceiq summary [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database (default: `traceiq.db`)
- `--top-n INT` - Number of top agents to show (default: 10)
- `--json` - Output as JSON

**Example:**
```bash
traceiq summary --db myproject.db --top-n 5

# JSON output
traceiq summary --db myproject.db --json > summary.json
```

### `traceiq export`

Export data to CSV or JSONL.

```bash
traceiq export [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database (default: `traceiq.db`)
- `-o, --output PATH` - Output file path (required)
- `--format [csv|jsonl]` - Export format (default: csv)

**Example:**
```bash
traceiq export --db myproject.db -o data.csv --format csv
traceiq export --db myproject.db -o data.jsonl --format jsonl
```

### `traceiq plot`

Generate visualizations. Requires matplotlib (`pip install traceiq[plot]`).

#### `traceiq plot drift`

Plot drift over time.

```bash
traceiq plot drift [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database
- `-o, --output PATH` - Output image path (required)

**Example:**
```bash
traceiq plot drift --db myproject.db -o drift.png
```

#### `traceiq plot heatmap`

Plot influence heatmap.

```bash
traceiq plot heatmap [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database
- `-o, --output PATH` - Output image path (required)

**Example:**
```bash
traceiq plot heatmap --db myproject.db -o heatmap.png
```

#### `traceiq plot influencers`

Plot top influencers bar chart.

```bash
traceiq plot influencers [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database
- `-o, --output PATH` - Output image path (required)
- `--top-n INT` - Number of top influencers (default: 10)

**Example:**
```bash
traceiq plot influencers --db myproject.db -o top.png --top-n 5
```

#### `traceiq plot network`

Plot influence network graph.

```bash
traceiq plot network [OPTIONS]
```

**Options:**
- `--db PATH` - Path to SQLite database
- `-o, --output PATH` - Output image path (required)
- `--min-weight FLOAT` - Minimum edge weight to display (default: 0.1)

**Example:**
```bash
traceiq plot network --db myproject.db -o network.png --min-weight 0.2
```

## Workflow Example

```bash
# 1. Initialize database
traceiq init --db analysis.db

# 2. Prepare your interactions file (interactions.jsonl)
# Each line: {"sender_id": "...", "receiver_id": "...", "sender_content": "...", "receiver_content": "..."}

# 3. Ingest data
traceiq ingest interactions.jsonl --db analysis.db

# 4. View summary
traceiq summary --db analysis.db

# 5. Export for further analysis
traceiq export --db analysis.db -o results.csv

# 6. Generate visualizations
traceiq plot heatmap --db analysis.db -o heatmap.png
traceiq plot network --db analysis.db -o network.png
traceiq plot influencers --db analysis.db -o influencers.png
```

## Configuration File

You can use a JSON config file:

```json
{
  "storage_backend": "sqlite",
  "storage_path": "traceiq.db",
  "baseline_window": 10,
  "drift_threshold": 0.3,
  "influence_threshold": 0.5
}
```

Use with:
```bash
traceiq init --config config.json
```
