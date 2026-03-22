"""
YAML-driven experiment runner for emergent-swarm.

Reads a scenario YAML file and runs the swarm with the specified agents,
model, and configuration.

Usage:
    python run_experiment.py --scenario experiment.yaml --turns 100
    python run_experiment.py --scenario experiment.yaml --output-dir traces/run_001 --subdir-name my_run
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from google import genai
from swarm import MCPClient, MessageBus, SwarmAgent, run_swarm


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run(scenario_path: str, max_turns: int, output_dir: str | None, subdir_name: str | None):
    scenario = load_scenario(scenario_path)
    name = scenario.get("name", "Unnamed Experiment")
    model = os.environ.get("MODEL", scenario.get("model", "gemini-3.1-flash-lite-preview"))
    use_sema = scenario.get("use_sema", False)

    print(f"Experiment: {name}")
    print(f"Model: {model}")
    print(f"Sema: {'enabled' if use_sema else 'disabled'}")

    # Setup output
    if output_dir and subdir_name:
        run_dir = os.path.join(output_dir, subdir_name)
    elif output_dir:
        run_dir = output_dir
    else:
        run_dir = "."
    os.makedirs(run_dir, exist_ok=True)

    trace_path = os.path.join(run_dir, "trace.jsonl")

    # Google GenAI client
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    bus = MessageBus(autosave_path=trace_path)

    # MCP servers
    mcp_clients = []
    if use_sema:
        # Start Sema MCP server
        sema_mcp = MCPClient(
            command="sema",
            args=["mcp"],
            name="sema",
        )
        mcp_clients.append(sema_mcp)

    # Start MCP servers
    for mcp in mcp_clients:
        try:
            await mcp.start()
            print(f"Connected to {mcp.name}. Found {len(mcp.tools)} tools.")
        except Exception as e:
            print(f"Failed to connect to {mcp.name}: {e}")

    # Create agents from YAML
    agents = {}
    kickstart_agent = None
    kickstart_message = scenario.get("kickstart", "Begin.")

    for agent_def in scenario.get("agents", []):
        agent_name = agent_def["name"]
        agent = SwarmAgent(
            name=agent_name,
            system_instructions=agent_def["instructions"],
            bus=bus,
            client=client,
            model_name=model,
            allowed_to_submit=agent_def.get("can_submit", False),
            extra_tools=mcp_clients,
        )
        agents[agent_name] = agent

        if agent_def.get("can_submit", False) and kickstart_agent is None:
            kickstart_agent = agent

    if not agents:
        print("ERROR: No agents defined in scenario")
        sys.exit(1)

    # Kickstart
    if kickstart_agent:
        kickstart_agent.kickstart(kickstart_message)
    else:
        first = list(agents.values())[0]
        first.kickstart(kickstart_message)

    print(f"\n--- STARTING EXPERIMENT: {name} ---")
    print(f"Agents: {list(agents.keys())}")
    print(f"Max turns: {max_turns}")
    print(f"Trace: {trace_path}")

    try:
        await run_swarm(agents, bus, max_turns=max_turns)
    finally:
        for mcp in mcp_clients:
            await mcp.close()
        for agent in agents.values():
            agent.close()

    # Save final product if any
    if bus.final_product:
        product_path = os.path.join(run_dir, "final_product.txt")
        with open(product_path, "w") as f:
            f.write(bus.final_product)
        print(f"Final product saved to {product_path}")

    print(f"Experiment complete. Trace saved to {trace_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a swarm experiment from a YAML scenario")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    parser.add_argument("--turns", type=int, default=50, help="Max turns (default: 50)")
    parser.add_argument("--output-dir", help="Output directory for traces")
    parser.add_argument("--subdir-name", help="Subdirectory name within output-dir")
    args = parser.parse_args()

    asyncio.run(run(args.scenario, args.turns, args.output_dir, args.subdir_name))


if __name__ == "__main__":
    main()
