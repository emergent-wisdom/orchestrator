import argparse
import os
import sys
from pathlib import Path

# Add src to path so we can import swarm without installing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from google import genai

from swarm import MCPClient, MessageBus, SwarmAgent, run_swarm


def main():
    parser = argparse.ArgumentParser()
    # Anchor to the directory of this script (Code/swarm)
    base_dir = Path(__file__).parent.absolute()
    code_dir = base_dir.parent

    parser.add_argument(
        "--project-root",
        default=str(code_dir / "understanding"),
        help="Path to understanding project",
    )
    args = parser.parse_args()

    # Setup Google GenAI
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        env_path = os.path.join(args.project_root, ".env")
        if os.path.exists(env_path):
            print(f"Loading API key from {env_path}")
            with open(env_path) as f:
                for line in f:
                    if line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)
    bus = MessageBus(autosave_path="swarm.jsonl")

    # --- MCP SERVERS ---
    mcp_clients = []

    # 1. Understanding Graph
    ug_server_path = os.path.join(
        args.project_root, "packages/mcp-server/dist/index.js"
    )
    if os.path.exists(ug_server_path):
        print(f"Found Understanding MCP at {ug_server_path}")
        ug_mcp = MCPClient(
            command="node",
            args=[ug_server_path],
            cwd=os.path.dirname(ug_server_path),
            name="understanding",
        )
        mcp_clients.append(ug_mcp)

    # 2. Sema Vocabulary
    sema_src_path = code_dir / "sema-frontend/sema/src"
    if sema_src_path.exists():
        print(f"Found Sema source at {sema_src_path}")
        sema_mcp = MCPClient(
            command="python3",
            args=["-m", "sema.mcp.server"],
            cwd=str(sema_src_path),
            name="sema",
        )
        mcp_clients.append(sema_mcp)

    # Start all
    for mcp in mcp_clients:
        try:
            mcp.start()
            print(f"Connected to {mcp.name}. Found {len(mcp.tools)} tools.")
        except Exception as e:
            print(f"Failed to connect to {mcp.name}: {e}")

    # Create Agents
    strategist = SwarmAgent(
        name="strategist",
        system_instructions="""You are the strategist. 
        Coordinate the team using the understanding graph and Sema vocabulary.
        - Use 'graph_*' tools to manage comprehension.
        - Use 'sema_*' tools to ensure semantic alignment.""",
        bus=bus,
        client=client,
        extra_tools=mcp_clients,
        allowed_to_submit=True,
    )

    explorer = SwarmAgent(
        name="explorer",
        system_instructions="""You are the explorer.
        Investigate topics and verify findings with Sema.
        Record insights in the understanding graph.""",
        bus=bus,
        client=client,
        extra_tools=mcp_clients,
    )

    # Run
    agents = {"strategist": strategist, "explorer": explorer}

    print("\n--- STARTING SWARM ---")
    strategist.kickstart(
        "Check the current graph state and cross-reference with the Sema stats."
    )

    try:
        run_swarm(agents, bus, max_turns=10)
    finally:
        for mcp in mcp_clients:
            mcp.close()


if __name__ == "__main__":
    main()
