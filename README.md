# Simple Orchestrator

A lightweight multi-agent orchestration engine for coordinating LLM agents through a message bus. Built as an early prototype for the [Emergent Wisdom](https://github.com/emergent-wisdom) project.

> **Note:** This was an early experiment. The iterative understanding loop in [ewa](https://github.com/emergent-wisdom/ewa) supersedes this approach — fresh stateless agents with graph-as-only-memory turned out to be simpler and more reliable than named agents with direct messaging.

## What it does

Multiple named agents (e.g. strategist, explorer, reader, skeptic) coordinate through:
- **Message bus** — agents send messages to each other via an inbox/outbox system
- **MCP tools** — agents share access to Understanding Graph and Sema vocabulary
- **Three orchestration modes** — queue-based, two-phase, and four-phase pipelines

## Architecture

```
MessageBus          Central inbox/outbox with per-agent mailboxes
SwarmAgent          Wraps a Gemini chat session with MCP tool access
MCPClient           Stdio JSON-RPC client with schema sanitization

Orchestration modes:
  run_swarm                  Queue-based round-robin
  run_with_orchestrator      Two-phase: readers then reviewers
  run_four_phase_orchestrator  READ → THINK → FLUID pipeline
```

~1150 lines of Python across 3 files. Gemini-only.

## Usage

```bash
export GOOGLE_API_KEY=your_key
python3 run.py --project-root ../understanding-graph
```

## What we learned

- Direct agent-to-agent messaging creates an invisible communication channel not captured in the graph
- Persistent chat sessions accumulate context until they hit the window limit
- Named roles (reader, skeptic, synthesizer) add complexity without proportional value
- The simpler approach — one anonymous agent per round, graph as the only memory — works better

See [ewa](https://github.com/emergent-wisdom/ewa) for the current approach.

## License

MIT
