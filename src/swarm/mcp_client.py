import json
import os
import sys
import asyncio
from typing import Any, Dict, Optional


class MCPClient:
    """
    A lightweight AsyncIO JSON-RPC 2.0 client for Model Context Protocol (MCP) servers.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: str | None = None,
        name: str = "mcp_client",
        default_params: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command
        self.args = args
        self.cwd = cwd or os.getcwd()
        self.name = name
        self.default_params = default_params or {}
        self.env = env
        self.process = None
        self.request_id = 0
        self.pending_requests: Dict[int, asyncio.Future] = {}
        self.tools = []
        self._reader_task = None

    async def start(self):
        """Start the MCP server subprocess and initialize the connection."""
        try:
            cmd = [self.command] + self.args
            print(f"[{self.name}] Starting: {' '.join(cmd)} (cwd={self.cwd})")
            
            # Merge env with current environment
            proc_env = os.environ.copy()
            if self.env:
                proc_env.update(self.env)
                
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=sys.stderr,  # Forward stderr to see server logs
                cwd=self.cwd,
                env=proc_env,
                limit=10 * 1024 * 1024, # Increase buffer limit to 10MB for large JSON responses
            )

            # Start reader task
            self._reader_task = asyncio.create_task(self._read_loop())

            # Perform MCP Handshake
            await self._initialize()
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server '{self.name}': {e}") from e

    async def _read_loop(self):
        """Background task to read lines from the server's stdout."""
        while self.process:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                line = line.decode('utf-8').strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    pass
            except ValueError:
                continue
            except Exception as e:
                print(f"[{self.name}] Reader error: {e}", file=sys.stderr)
                break
        
        # Cleanup pending requests if process dies
        for fut in self.pending_requests.values():
            if not fut.done():
                fut.set_exception(RuntimeError("MCP Process died"))

    async def _handle_message(self, message):
        """Handle incoming JSON-RPC messages."""
        if "id" in message and message["id"] in self.pending_requests:
            # Response to a pending request
            future = self.pending_requests.pop(message["id"])
            if not future.done():
                if "error" in message:
                    future.set_exception(Exception(f"MCP Error: {message['error']}"))
                else:
                    future.set_result(message.get("result"))

    async def _send_request(self, method, params=None, timeout=30):
        """Send a JSON-RPC request and wait for the response."""
        self.request_id += 1
        rid = self.request_id
        payload = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params or {},
        }

        future = asyncio.get_running_loop().create_future()
        self.pending_requests[rid] = future

        json_str = json.dumps(payload)
        try:
            self.process.stdin.write((json_str + "\n").encode('utf-8'))
            await self.process.stdin.drain()
        except (BrokenPipeError, AttributeError):
            if rid in self.pending_requests:
                del self.pending_requests[rid]
            print(f"[{self.name}] Error: Pipe closed", file=sys.stderr)
            return None

        # Wait for response
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            if rid in self.pending_requests:
                del self.pending_requests[rid]
            print(f"[{self.name}] Timeout waiting for {method}", file=sys.stderr)
            return None

    async def _initialize(self):
        """Perform the MCP initialization handshake."""
        # 1. Initialize
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "SwarmEngine", "version": "1.0"},
        }
        await self._send_request("initialize", init_params)

        # 2. Initialized notification (no response expected)
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )
        self.process.stdin.write((msg + "\n").encode('utf-8'))
        await self.process.stdin.drain()

        # 3. List tools immediately
        await asyncio.sleep(0.5) # Allow server buffers to flush
        await self.refresh_tools()

    async def refresh_tools(self):
        """Query the server for available tools."""
        result = await self._send_request("tools/list")
        if result and "tools" in result:
            self.tools = result["tools"]

    def _sanitize_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively clean JSON schema for Gemini compatibility."""
        if not isinstance(schema, dict):
            return schema

        clean = {}
        for k, v in schema.items():
            # Remove forbidden fields
            if k in ["additionalProperties", "title", "$schema"]:
                continue

            # Recursively clean nested dicts or lists
            if isinstance(v, dict):
                clean[k] = self._sanitize_schema(v)
            elif isinstance(v, list):
                clean[k] = [
                    self._sanitize_schema(i) if isinstance(i, dict) else i for i in v
                ]
            else:
                clean[k] = v

        # Post-processing: Validate 'required' fields
        if (
            "properties" in clean
            and "required" in clean
            and isinstance(clean["required"], list)
        ):
            existing_props = set(clean["properties"].keys())
            clean["required"] = [r for r in clean["required"] if r in existing_props]
            if not clean["required"]:
                del clean["required"]

        return clean

    def get_gemini_tools(self) -> list[dict[str, Any]]:
        """Convert discovered MCP tools to Gemini function declaration format."""
        gemini_tools = []
        for tool in self.tools:
            # Gemini naming: regex [a-zA-Z0-9_]+ (no hyphens)
            safe_name = tool["name"].replace("-", "_")

            # Sanitize the input schema
            raw_schema = tool.get("inputSchema", {})
            clean_schema = self._sanitize_schema(raw_schema)

            # Ensure "type": "object" is present for parameters
            if "type" not in clean_schema:
                clean_schema["type"] = "object"

            g_tool = {
                "name": safe_name,
                "description": tool.get("description", ""),
                "parameters": clean_schema,
            }

            gemini_tools.append(g_tool)
        return gemini_tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool. tool_name should be the Gemini-safe name (underscores)."""
        # Map back to original MCP name (potentially with hyphens)
        mcp_name = tool_name
        for tool in self.tools:
            if tool["name"].replace("-", "_") == tool_name:
                mcp_name = tool["name"]
                break

        # --- SELF-HEALING PARAMETERS ---
        # Remap common agent mistakes to ensure tools don't fail silently
        # Convert legacy 'text' to 'title' (migration: text -> title is complete)
        if "doc_" in tool_name:
            if "text" in arguments and "title" not in arguments:
                arguments["title"] = arguments.pop("text")
        if "nodeId" in arguments and "node" not in arguments:
            arguments["node"] = arguments.pop("nodeId")
        if "id" in arguments and "node" not in arguments:
            arguments["node"] = arguments.pop("id")
        if "content" in arguments and "prose" not in arguments and "doc_" in tool_name:
            arguments["prose"] = arguments.pop("content")
        if "edgeType" in arguments and "type" not in arguments:
            arguments["type"] = arguments.pop("edgeType")
        if "nodeId" in arguments and "node" not in arguments and "graph_" in tool_name:
            arguments["node"] = arguments.pop("nodeId")
        if "targetId" in arguments and "target_id" not in arguments:
            arguments["target_id"] = arguments.pop("targetId")
        if "target_id" in arguments and "node" not in arguments and "graph_revise" in tool_name:
            arguments["node"] = arguments.pop("target_id")
        if "node" in arguments and "nodeId" not in arguments and "doc_revise" in tool_name:
            arguments["nodeId"] = arguments.pop("node")

        # Inject default parameters (e.g. active project) if not provided by agent
        full_args = self.default_params.copy()
        full_args.update(arguments)

        try:
            result = await self._send_request(
                "tools/call", {"name": mcp_name, "arguments": full_args}
            )

            if result and "content" in result:
                # Concatenate all text content
                text = ""
                for item in result["content"]:
                    if item["type"] == "text":
                        text += item["text"]
                return text
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def close(self):
        """Terminate the server process."""
        if self.process:
            try:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self.process.kill()
            except Exception:
                pass
            self.process = None
