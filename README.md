# Simple Orchestrator

A lightweight multi-agent orchestration engine for coordinating LLM agents through a message bus. Built as an early prototype for the [Emergent Wisdom](https://github.com/emergent-wisdom) project.

> **Note:** This library had two lives. Its agent-runtime primitives (`SwarmAgent`, `MCPClient`, and the message bus as prompt dispatch and transcript log) power the published [entangled-alignment](https://github.com/emergent-wisdom/entangled-alignment) case studies, where it is pinned as a submodule — and in that use, agents never message each other: the bus carries orchestrator prompts and the `swarm.jsonl` transcript, while coordination happens stigmergically through the Understanding Graph. The *direct agent-to-agent messaging* was the early experiment that did not survive: the iterative loop in [ewa](https://github.com/emergent-wisdom/ewa) supersedes that part — fresh stateless agents with graph-as-only-memory proved simpler and more reliable than persistent named agents exchanging messages.

## What it does

Multiple named agents (e.g. strategist, explorer, reader, skeptic) coordinate through:
- **Message bus** — per-agent mailboxes the orchestrator uses to dispatch prompts and collect transcripts; agents *can* message each other in `run_swarm` mode, though the published pipelines never use that channel
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
  run_four_phase_orchestrator  READ+UNDERSTAND → THINK → FLUID pipeline
```

~1150 lines of Python across 3 files. Gemini-only.

## Usage

```bash
export GOOGLE_API_KEY=your_key
python3 run.py --project-root ../understanding-graph
```

## What we learned

- Direct agent-to-agent messaging creates an invisible communication channel not captured in the graph — so the published pipelines never use it; coordination goes through the graph
- Persistent chat sessions accumulate context until they hit the window limit
- Named roles earn their keep over a *shared graph*, where their disagreements become typed nodes — the entangled-alignment case studies run eleven named roles stigmergically. Named roles *with direct messaging* added complexity without proportional value
- For autonomous iteration, one stateless agent per round with the graph as the only memory works better

See [ewa](https://github.com/emergent-wisdom/ewa) for the interactive successor, and [entangled-alignment](https://github.com/emergent-wisdom/entangled-alignment) for the pipeline these primitives still power.

## License

MIT
