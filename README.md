# Emergent Swarm

**Emergent Swarm** is a lightweight, high-precision multi-agent engine designed for **Inbox Zero** communication and **stigmergic collaboration**. 

Unlike traditional agent frameworks that rely on endless chat history (leading to context bloat), Emergent Swarm enforces a "Focus + Recall" architecture where agents process their inbox, act on a shared persistent state (like a knowledge graph), and then clear their memory.

Built for **Emergent Wisdom** projects.

## Key Features

- **📬 Inbox Zero Architecture:** Agents consume their entire mailbox every turn. Messages are processed, acted upon, and then archived. Context remains lean and focused.
- **🔌 Native MCP Support:** First-class support for the **Model Context Protocol (MCP)**. Plug in any MCP server (Understanding Graph, Sema Vocabulary, Filesystem, etc.) directly into your agents.
- **✨ Schema Sanitization:** Automatically cleans and validates MCP tool schemas (stripping `additionalProperties`, etc.) to ensure 100% compatibility with strict LLM APIs like Google Gemini.
- **🛡️ Type-Safe:** Built with Pydantic for robust data validation and structural integrity.
- **🕸️ Stigmergy First:** Designed for agents that communicate through *environment modification* (e.g., updating a graph) rather than just chat.

## Installation

```bash
git clone https://github.com/emergent-wisdom/emergent-swarm.git
cd emergent-swarm
pip install -e .
```

## Quick Start

The engine comes with a built-in runner that auto-discovers MCP servers in your workspace.

```bash
# 1. Set your API Key
export GOOGLE_API_KEY=your_key_here

# 2. Run the swarm
# (automatically finds Understanding Graph & Sema MCP if present in ../)
python3 run.py --project-root ../understanding
```

## Architecture: Focus + Recall

1.  **Focus (Working Memory):** At the start of a turn, an agent receives a prompt containing *only* the new messages in its inbox. This ensures they are reacting to the immediate signal.
2.  **Recall (Long-Term Memory):** Agents have access to tools like `read_history()` to retrieve past context if needed, or they query the shared environment (Understanding Graph).
3.  **Action:** Agents execute tools (MCP or local) to modify the world.
4.  **Clear:** The inbox is wiped. The cycle repeats.

## Development

This project uses **Ruff** for linting and formatting. A pre-commit hook is included to ensure code quality.

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter manually
ruff check .
ruff format .
```

## License

MIT
