# Integration Guide

This guide provides templates for integrating TraceIQ with different types of AI agent architectures.

## Overview

TraceIQ can track influence across various agent types. The key is providing appropriate context to maximize metric confidence:

| Agent Type | receiver_input_view | receiver_state_before/after | state_quality |
|------------|---------------------|----------------------------|---------------|
| LLM-only | Not needed | Not needed | low |
| RAG | Retrieved chunks | Not needed | medium |
| Tool-using | Tool outputs | Not needed | medium |
| Memory | Memory context | Memory state | high |

## LLM-Only Agent

For simple LLM agents without retrieval or tools, basic tracking is sufficient.

```python
from traceiq import InfluenceTracker, TrackerConfig

config = TrackerConfig(
    storage_backend="sqlite",
    storage_path="influence.db",
    enable_risk_scoring=True,
)

tracker = InfluenceTracker(config=config)

def process_message(sender_agent, receiver_agent, message):
    """Process a message between two LLM agents."""

    # Get response from receiver
    response = receiver_agent.generate(message)

    # Track the interaction
    result = tracker.track_event(
        sender_id=sender_agent.id,
        receiver_id=receiver_agent.id,
        sender_content=message,
        receiver_content=response,
    )

    # Check for anomalies
    if result["alert"]:
        logger.warning(f"Anomaly detected: Z={result['Z_score']:.2f}")

    return response, result

# Usage
response, metrics = process_message(agent_a, agent_b, "What do you think about X?")
print(f"Risk level: {metrics['risk_level']}")
```

**State Quality**: `low` - Only the final output is tracked.

## RAG Agent

For RAG agents, include the retrieved context to improve metric confidence.

```python
from traceiq import InfluenceTracker, TraceIQEvent

tracker = InfluenceTracker()

def process_rag_query(sender_agent, rag_agent, query):
    """Process a query through a RAG agent."""

    # Retrieve relevant documents
    retrieved_chunks = rag_agent.retrieve(query)
    retrieved_text = "\n".join([chunk.text for chunk in retrieved_chunks])

    # Generate response with retrieved context
    response = rag_agent.generate(query, context=retrieved_chunks)

    # Create full event with input view
    event = TraceIQEvent(
        run_id="rag_session_001",
        sender_id=sender_agent.id,
        receiver_id=rag_agent.id,
        sender_content=query,
        receiver_output=response,
        receiver_input_view=retrieved_text,  # What the RAG actually saw
    )

    # Track with full context
    result = tracker.track_event(event=event)

    return response, result

# Usage
response, metrics = process_rag_query(user, rag_agent, "Tell me about climate change")
print(f"Confidence: {metrics['confidence']}")  # Should be "medium"
```

**State Quality**: `medium` - We know what context the agent had access to.

## Tool-Using Agent

For agents that use tools, include tool outputs in the input view.

```python
from traceiq import InfluenceTracker, TraceIQEvent

tracker = InfluenceTracker()

def process_with_tools(sender_agent, tool_agent, instruction):
    """Process an instruction through a tool-using agent."""

    # Agent decides which tools to use
    tool_plan = tool_agent.plan_tools(instruction)

    # Execute tools and collect outputs
    tool_outputs = []
    for tool_call in tool_plan:
        output = tool_agent.execute_tool(tool_call)
        tool_outputs.append(f"[{tool_call.name}]: {output}")

    tool_context = "\n".join(tool_outputs)

    # Generate final response using tool outputs
    response = tool_agent.synthesize(instruction, tool_outputs)

    # Track with tool context
    event = TraceIQEvent(
        run_id="tool_session_001",
        sender_id=sender_agent.id,
        receiver_id=tool_agent.id,
        sender_content=instruction,
        receiver_output=response,
        receiver_input_view=tool_context,  # Tool outputs the agent used
        metadata={"tools_used": [tc.name for tc in tool_plan]},
    )

    result = tracker.track_event(event=event)
    return response, result
```

**State Quality**: `medium` - Tool outputs provide context about what influenced the response.

## Memory Agent

For agents with persistent memory, track memory state changes for highest confidence.

```python
from traceiq import InfluenceTracker, TraceIQEvent
import json

tracker = InfluenceTracker()

def process_with_memory(sender_agent, memory_agent, message):
    """Process a message through an agent with persistent memory."""

    # Capture memory state BEFORE processing
    memory_before = json.dumps(memory_agent.get_memory_state())

    # Retrieve relevant memories for context
    relevant_memories = memory_agent.recall(message)
    memory_context = "\n".join(relevant_memories)

    # Process message and update memory
    response = memory_agent.process(message)

    # Capture memory state AFTER processing
    memory_after = json.dumps(memory_agent.get_memory_state())

    # Track with full state information
    event = TraceIQEvent(
        run_id="memory_session_001",
        sender_id=sender_agent.id,
        receiver_id=memory_agent.id,
        sender_content=message,
        receiver_output=response,
        receiver_input_view=memory_context,        # What memories were recalled
        receiver_state_before=memory_before,       # Memory before interaction
        receiver_state_after=memory_after,         # Memory after interaction
    )

    result = tracker.track_event(event=event)

    # Log if memory changed significantly
    if result["risk_level"] in ("high", "critical"):
        logger.warning(f"High-risk memory modification detected!")

    return response, result
```

**State Quality**: `high` - Full before/after state enables canonical drift computation.

## Multi-Agent Orchestrator

For complex multi-agent systems, track the full conversation flow.

```python
from traceiq import InfluenceTracker, TrackerConfig
from traceiq.report import generate_risk_report
from uuid import uuid4

config = TrackerConfig(
    storage_backend="sqlite",
    storage_path="orchestrator.db",
    enable_risk_scoring=True,
    enable_policy=True,  # Enable mitigation
)

tracker = InfluenceTracker(config=config)
run_id = str(uuid4())

class AgentOrchestrator:
    def __init__(self, agents: list, tracker: InfluenceTracker):
        self.agents = {a.id: a for a in agents}
        self.tracker = tracker
        self.run_id = str(uuid4())
        self.task_counter = 0

    def route_message(self, sender_id: str, receiver_id: str, message: str):
        """Route a message and track the interaction."""
        self.task_counter += 1

        receiver = self.agents[receiver_id]
        response = receiver.process(message)

        result = self.tracker.track_event(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_content=message,
            receiver_content=response,
            run_id=self.run_id,
            task_id=f"task_{self.task_counter:04d}",
        )

        # Check policy decision
        if result["policy_action"] == "quarantine":
            self.handle_quarantine(sender_id, receiver_id, message, result)
            return None

        return response

    def handle_quarantine(self, sender_id, receiver_id, message, result):
        """Handle quarantined interactions."""
        logger.warning(
            f"Interaction quarantined: {sender_id} -> {receiver_id}\n"
            f"  Risk: {result['risk_level']} ({result['risk_score']:.3f})\n"
            f"  Reason: {result.get('policy_reason', 'unknown')}"
        )

    def generate_report(self, output_path: str):
        """Generate a risk report for this orchestration run."""
        from traceiq.schema import TraceIQEvent

        events = self.tracker.get_events()
        scores = self.tracker.get_scores()

        trace_events = [
            TraceIQEvent(
                event_id=str(e.event_id),
                run_id=self.run_id,
                sender_id=e.sender_id,
                receiver_id=e.receiver_id,
                sender_content=e.sender_content,
                receiver_output=e.receiver_content,
                ts=e.timestamp.timestamp(),
            )
            for e in events
        ]

        generate_risk_report(
            events=trace_events,
            scores=scores,
            run_id=self.run_id,
            output_path=output_path,
        )
```

## Best Practices

### 1. Always Check Validity

Don't alert or take action on invalid metrics (cold start):

```python
result = tracker.track_event(...)

if not result["valid"]:
    # Metrics not reliable yet - still warming up
    return response

if result["alert"]:
    # This is a real alert, not cold-start noise
    handle_alert(result)
```

### 2. Use Run IDs for Experiment Tracking

```python
from uuid import uuid4

run_id = str(uuid4())

for interaction in experiment_data:
    result = tracker.track_event(
        ...,
        run_id=run_id,
        task_id=interaction["task_id"],
    )
```

### 3. Export Regularly for Analysis

```python
# Export to CSV for offline analysis
tracker.export_csv("influence_data.csv")

# Generate risk report
from traceiq.report import generate_risk_report
generate_risk_report(events, scores, run_id, "report.md")
```

### 4. Monitor Propagation Risk

For multi-agent systems, monitor PR over time:

```python
pr = tracker.get_propagation_risk()
if pr > 1.0:
    logger.warning(f"PR={pr:.2f} - influence may amplify through network")
```

### 5. Don't Trust IQx on Cold Start

The minimum baseline samples (default: 20) exist for a reason:

```python
# BAD: Alerting on early metrics
if result["IQx"] > 2.0:  # May be noise!
    alert()

# GOOD: Check validity first
if result["valid"] and result["IQx"] > 2.0:
    alert()
```

## MCP Server

An experimental MCP (Model Context Protocol) server is included in `experiments/mcp_server_traceiq.py`. This provides a JSON-over-stdio interface for external tools to log interactions and query metrics.

See [experiments/README.md](../experiments/README.md) for usage details.

## Common Mistakes to Avoid

1. **Alerting on cold start**: Always check `result["valid"]` before taking action
2. **Computing PR per-row**: PR should be computed over windows, not individual events
3. **Treating IQx as causal truth**: IQx measures correlation, not causation
4. **Feeding unbounded IQx into graphs**: Use bounded weights for numerical stability
5. **Ignoring receiver_input_view**: For RAG/tools, this context improves metric quality
