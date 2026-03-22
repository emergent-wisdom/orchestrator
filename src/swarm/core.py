"""
Swarm Engine Core
-----------------
A generic, multi-agent engine using the "Focus + Recall" architecture.
Powered by Pydantic for validation and schema generation.
"""

import signal
import time
import threading
import inspect # Added missing import
import asyncio # Explicit asyncio usage
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

# Optional: Google GenAI import
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

# --- DATA STRUCTURES (Pydantic) ---


class TraceEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    actor: str
    event_type: str
    content: str
    metadata: dict[str, Any] | None = None


class Message(BaseModel):
    id: str
    sender: str
    recipient: str
    content: str
    timestamp: float = Field(default_factory=time.time)


# --- TOOL MODELS (Pydantic) ---


class SendMessageTool(BaseModel):
    """Send a message to a teammate."""

    recipient: str = Field(..., description="Name of the agent to send to")
    content: str = Field(..., description="The message text")


class ReadHistoryTool(BaseModel):
    """Read the full history of messages from a specific sender."""

    sender_name: str = Field(..., description="Name of the sender to view archive for")


class SubmitProductTool(BaseModel):
    """Submit final artifact to end the mission."""

    content: str = Field(..., description="The final artifact content")


def pydantic_to_gemini_tool(model: type[BaseModel]):
    """Convert a Pydantic model to a Gemini FunctionDeclaration."""
    schema = model.model_json_schema()

    # Clean up schema for Gemini (remove title, strictly type object)
    parameters = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }

    return {
        "name": schema.get("title") or model.__name__,  # Fallback
        "description": schema.get("description", ""),
        "parameters": parameters,
    }


# --- MESSAGE BUS ---


class MessageBus:
    def __init__(self, autosave_path: str | None = None, web_log_callback: Callable | None = None):
        self.mailboxes: dict[str, deque] = {}
        self.execution_queue: deque = deque()
        self.trace_log: list[TraceEvent] = []
        self.final_product: str | None = None
        self.finished = False
        self.system_messages: dict[str, list[str]] = {}
        self.msg_counter = 0
        self.autosave_file = None
        self.web_log_callback = web_log_callback

        if autosave_path:
            self.enable_autosave(autosave_path)

    def enable_autosave(self, filepath):
        self.autosave_file = open(filepath, "a", encoding="utf-8")  # noqa: SIM115

    def log_event(self, actor, event_type, content, metadata=None):
        event = TraceEvent(
            actor=actor, event_type=event_type, content=content, metadata=metadata
        )
        self.trace_log.append(event)
        if self.web_log_callback:
            self.web_log_callback(actor, event_type, content, metadata)
        if self.autosave_file:
            self.autosave_file.write(event.model_dump_json() + "\n")
            self.autosave_file.flush()

    def register(self, agent_name):
        if agent_name not in self.mailboxes:
            self.mailboxes[agent_name] = deque()
            self.system_messages[agent_name] = []

    def inject_system_message(self, agent_name: str, content: str):
        if agent_name in self.system_messages:
            self.system_messages[agent_name].append(content)
            self.log_event(
                "Supervisor", "INJECTION", f"To {agent_name}: {content}"
            )
            if agent_name not in self.execution_queue:
                self.execution_queue.append(agent_name)

    def pop_system_message(self, agent_name: str) -> str | None:
        if self.system_messages.get(agent_name):
            return self.system_messages[agent_name].pop(0)
        return None

    def send(self, sender, recipient, content):
        self.msg_counter += 1
        msg_id = f"msg_{self.msg_counter}"

        self.log_event(
            sender,
            "MESSAGE_SENT",
            f"To {recipient}: {content}",
            {"full_content": content, "id": msg_id},
        )

        if recipient not in self.mailboxes:
            return f"Error: Recipient '{recipient}' not found."

        msg = Message(id=msg_id, sender=sender, recipient=recipient, content=content)
        self.mailboxes[recipient].append(msg)

        if recipient not in self.execution_queue:
            self.execution_queue.append(recipient)

        return f"Message sent to {recipient}."

    def pop_all(self, agent_name) -> list[Message]:
        if not self.mailboxes.get(agent_name):
            return []

        messages = list(self.mailboxes[agent_name])
        self.mailboxes[agent_name].clear()

        for m in messages:
            self.log_event(
                agent_name,
                "MESSAGE_RECEIVED",
                f"From {m.sender}",
                {"content": m.content},
            )

        return messages

    def has_messages(self, agent_name):
        return bool(self.mailboxes.get(agent_name))

    def submit(self, content):
        self.final_product = content
        self.finished = True
        self.log_event("System", "SUBMISSION", "Final product received.")
        return "Submission received. Experiment ending."

    def close(self):
        if self.autosave_file:
            self.autosave_file.close()


# --- AGENT ---

import asyncio as _asyncio
import time as _time
import os as _os
import random as _random

# --- Configuration ---
# Load from swarm.config.yaml if present, env vars override everything

def _load_config():
    """Load config from YAML file, with env var overrides."""
    defaults = {
        "rate_limit_min": 1,
        "rate_limit_max": 2,
        "rate_limit_cooldown": 60,
        "tool_cooldown_min": 0.5,
        "tool_cooldown_max": 1.0,
    }
    # Try loading YAML config
    config_paths = [
        _os.path.join(_os.getcwd(), "swarm.config.yaml"),
        _os.path.join(_os.path.dirname(__file__), "..", "..", "swarm.config.yaml"),
    ]
    for path in config_paths:
        if _os.path.exists(path):
            try:
                import yaml
                with open(path) as f:
                    file_config = yaml.safe_load(f) or {}
                defaults.update({k: v for k, v in file_config.items() if k in defaults})
            except ImportError:
                pass  # yaml not available, use defaults
            break

    # Env vars override everything
    return {
        "rate_limit_min": float(_os.environ.get("RATE_LIMIT_MIN", defaults["rate_limit_min"])),
        "rate_limit_max": float(_os.environ.get("RATE_LIMIT_MAX", defaults["rate_limit_max"])),
        "rate_limit_cooldown": int(_os.environ.get("RATE_LIMIT_COOLDOWN", defaults["rate_limit_cooldown"])),
        "tool_cooldown_min": float(_os.environ.get("TOOL_COOLDOWN_MIN", defaults["tool_cooldown_min"])),
        "tool_cooldown_max": float(_os.environ.get("TOOL_COOLDOWN_MAX", defaults["tool_cooldown_max"])),
    }

_config = _load_config()

# Global rate limiter for all SwarmAgent instances
_global_last_request_time = 0.0
_global_rate_lock = _asyncio.Lock()
_RATE_LIMIT_MIN = _config["rate_limit_min"]
_RATE_LIMIT_MAX = _config["rate_limit_max"]
_RATE_LIMIT_COOLDOWN = _config["rate_limit_cooldown"]
_TOOL_COOLDOWN_MIN = _config["tool_cooldown_min"]
_TOOL_COOLDOWN_MAX = _config["tool_cooldown_max"]

def _get_rate_limit_interval():
    """Random jitter for API rate limiting."""
    return _random.uniform(_RATE_LIMIT_MIN, _RATE_LIMIT_MAX)

def _get_tool_cooldown():
    """Random jitter between tool calls to avoid rate limit patterns."""
    return _random.uniform(_TOOL_COOLDOWN_MIN, _TOOL_COOLDOWN_MAX)


async def _global_throttle():
    """Ensure minimum interval between API calls across ALL agents."""
    global _global_last_request_time
    async with _global_rate_lock:
        now = _time.time()
        elapsed = now - _global_last_request_time
        required_interval = _get_rate_limit_interval()
        if elapsed < required_interval:
            wait_time = required_interval - elapsed
            print(f"  [THROTTLE] Waiting {wait_time:.1f}s for rate limit...")
            await _asyncio.sleep(wait_time)
        _global_last_request_time = _time.time()


class SwarmAgent:
    def __init__(
        self,
        name: str,
        system_instructions: str,
        bus: MessageBus,
        client: Any,
        model_name: str,  # Required - set GEMINI_MODEL in .env
        allowed_to_submit: bool = False,
        extra_tools: list[Any] = None,
        tool_callbacks: dict[str, Callable] = None,
        turn_appendix: str | None = None,
        usage_callback: Callable[[str, int, int], None] = None,  # (model, input_tokens, output_tokens)
    ):
        self.name = name
        self.system_instructions = system_instructions
        self.bus = bus
        self.bus.register(self.name)
        self.client = client
        self.model_name = model_name
        self.turn_appendix = turn_appendix
        self.usage_callback = usage_callback

        self.tool_map = {}
        self.sdk_tools = []
        extra_tools = extra_tools or []
        tool_callbacks = tool_callbacks or {}

        # 1. Local Archive
        self.archive: dict[str, list[Message]] = defaultdict(list)

        # 2. Built-in Tools
        def send_message_wrapper(recipient: str, content: str):
            return self.bus.send(self.name, recipient, content)

        def read_history_wrapper(sender_name: str):
            if sender_name not in self.archive:
                return f"No history found for '{sender_name}'."
            text = f"--- HISTORY FOR {sender_name} ---\\n"
            for m in self.archive[sender_name]:
                text += f"[{m.timestamp:.0f}] {m.content[:200]}...\n"
            return text

        self.tool_map["send_message"] = send_message_wrapper
        self.tool_map["read_history"] = read_history_wrapper

        std_defs = [
            {"name": "send_message", "model": SendMessageTool},
            {"name": "read_history", "model": ReadHistoryTool},
        ]

        if allowed_to_submit:
            self.tool_map["submit_final_product"] = lambda content: self.bus.submit(
                content
            )
            std_defs.append(
                {"name": "submit_final_product", "model": SubmitProductTool}
            )

        if types:
            std_funcs = []
            for item in std_defs:
                spec = pydantic_to_gemini_tool(item["model"])
                spec["name"] = item["name"]
                std_funcs.append(types.FunctionDeclaration(**spec))

            self.sdk_tools.append(types.Tool(function_declarations=std_funcs))

        # 3. Extra Tools (MCP)
        for tool_provider in extra_tools:
            if hasattr(tool_provider, "get_gemini_tools"):
                gemini_defs = tool_provider.get_gemini_tools()

                if types and gemini_defs:
                    extra_funcs = [
                        types.FunctionDeclaration(
                            name=t["name"],
                            description=t["description"],
                            parameters=t["parameters"],
                        )
                        for t in gemini_defs
                    ]
                    self.sdk_tools.append(types.Tool(function_declarations=extra_funcs))

                # Support both sync and async call_tool
                if hasattr(tool_provider, "call_tool"):
                    def make_wrapper(provider, tool_name):
                        async def wrapper(**kwargs):
                            # CORRECTED LOGIC: Call then inspect result with isawaitable
                            print(f"[DEBUG_WRAPPER] Calling tool '{tool_name}' on provider {type(provider)}")
                            res = provider.call_tool(tool_name, kwargs)
                            
                            # Check if it's awaitable
                            if inspect.isawaitable(res):
                                print(f"[DEBUG_WRAPPER] Awaiting coroutine for '{tool_name}'")
                                return await res
                            
                            print(f"[DEBUG_WRAPPER] Returning sync result for '{tool_name}'")
                            return res
                        return wrapper

                    for t in gemini_defs:
                        self.tool_map[t["name"]] = make_wrapper(
                            tool_provider, t["name"]
                        )

        # 4. Custom Callbacks
        for name, callback in tool_callbacks.items():
            self.tool_map[name] = callback

        # Store config for creating fresh chats each turn (avoids history accumulation)
        # Use .aio if available, otherwise assume client is already async-compatible
        self.client_interface = getattr(self.client, "aio", self.client)
        # NOTE: Do NOT use system_instruction in config - it triggers stricter rate limits
        # on preview models. Instead, embed system prompt in first message.
        self.chat_config = types.GenerateContentConfig(
            tools=self.sdk_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        self.chat = None  # Created once on first step
        self._system_sent = False  # Track if we've sent system prompt

    def close(self):
        """Close chat session and cleanup resources."""
        if self.chat is not None:
            # Chat doesn't have explicit close, but we can clear it
            self.chat = None
            print(f"[{self.name}] Chat session closed")

    async def _ensure_chat(self):
        """Create chat session if not exists (once per run)."""
        if self.chat is None:
            print(f"[{self.name}] Creating chat session...")
            self.chat = self.client_interface.chats.create(
                model=self.model_name,
                config=self.chat_config,
            )
        return self.chat

    async def step(self):
        # 1. Inbox Processing
        system_injection = self.bus.pop_system_message(self.name)
        new_messages = self.bus.pop_all(self.name)

        for m in new_messages:
            self.archive[m.sender].append(m)

        if not system_injection and not new_messages:
            return

        latest_map = {}
        for m in new_messages:
            latest_map[m.sender] = m

        prompt_parts = []
        if system_injection:
            prompt_parts.append(f"[SYSTEM ALERT]: {system_injection}")

        if latest_map:
            prompt_parts.append(f"--- INBOX: {len(new_messages)} NEW MESSAGES ---")
            for sender, msg in latest_map.items():
                count = len([m for m in new_messages if m.sender == sender])
                prompt_parts.append(f"FROM: {sender} (Latest of {count})")
                prompt_parts.append(f"{msg.content}")
                if count > 1:
                    prompt_parts.append(f"   [+ {count - 1} older messages archived.]")

        if self.turn_appendix:
            prompt_parts.append(f"\n--- PERSISTENT PROTOCOL REMINDER ---\n{self.turn_appendix}")

        prompt = "\n\n".join(prompt_parts)

        # Prepend system instructions to first message (avoids GenerateContentConfig rate limits)
        if not self._system_sent:
            prompt = f"SYSTEM INSTRUCTIONS:\n{self.system_instructions}\n\n---\n\n{prompt}"
            self._system_sent = True

        print(f"[{self.name}] Active. Processing...")

        # Ensure chat exists (created once per run)
        await self._ensure_chat()

        response = None
        max_retries = 6
        base_delay = 2

        for i in range(max_retries):
            try:
                # Global rate limit across all agents
                await _global_throttle()

                print(f"[{self.name}] Sending to Gemini (Attempt {i+1}/{max_retries})...")
                if i > 0:
                     self.bus.log_event("System", "INFO", f"Retrying API (Attempt {i+1}/{max_retries})...")

                res = self.chat.send_message(prompt)
                if inspect.isawaitable(res):
                    response = await res
                else:
                    response = res
                # Track and log token usage
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    um = response.usage_metadata
                    print(f"[{self.name}] Response received. Tokens: {um.prompt_token_count} in, {um.candidates_token_count} out")
                    self.bus.log_event(self.name, "TOKENS", f"in={um.prompt_token_count} out={um.candidates_token_count}")
                    if self.usage_callback:
                        self.usage_callback(self.model_name, um.prompt_token_count, um.candidates_token_count)
                else:
                    print(f"[{self.name}] Gemini Response Received.")
                break
            except Exception as e:
                error_str = str(e)
                print(f"[{self.name}] API Error: {error_str}")
                self.bus.log_event("System", "ERROR", f"API Error: {error_str[:100]}...")
                
                if i == max_retries - 1:
                    print(f"[{self.name}] Max retries reached.")
                    self.bus.log_event("System", "CRITICAL", "Max retries reached. Giving up.")
                    break

                is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                sleep_time = _RATE_LIMIT_COOLDOWN if is_429 else (base_delay * (2 ** i))

                if is_429:
                    print(f"[{self.name}] Rate limit hit. Cooling down for {sleep_time // 60} minutes...")
                    self.bus.log_event("System", "WARNING", f"Rate Limit Hit. Cooling down for {sleep_time}s...")
                    # Penalty Box: Push global timer forward to stop other agents
                    async with _global_rate_lock:
                        _global_last_request_time = _time.time() + sleep_time
                
                await asyncio.sleep(sleep_time)

        if not response:
            print(f"[{self.name}] No response from Gemini.")
            self.bus.log_event("System", "ERROR", "No response from Gemini API after retries")
            # NEVER move on with incomplete state - raise so caller can handle
            raise RuntimeError(f"[{self.name}] No response from Gemini API after {max_retries} retries")

        if not response.parts:
             print(f"[{self.name}] Empty response (Safety?)")
             self.bus.log_event("System", "WARNING", "Gemini returned empty response (Safety filter?)")
             # Empty response could be safety filter - don't raise, just return (not a rate limit)
             return

        tool_call_count = 0

        while response.parts:
            function_calls = [
                part.function_call for part in response.parts if part.function_call
            ]
            text_parts = [p.text for p in response.parts if p.text]

            if text_parts:
                thought = "".join(text_parts)
                print(f"[{self.name}] Thought...")
                self.bus.log_event(self.name, "THOUGHT", thought)

            if not function_calls:
                break

            tool_outputs = []
            print(f"  [DEBUG] Processing {len(function_calls)} tool calls sequentially...")
            for call in function_calls:
                fn_name = call.name
                fn_args = {k: v for k, v in call.args.items()}
                tool_call_count += 1

                # Cooldown between tool calls (prevents rate limit bursts)
                if tool_call_count > 1:
                    cooldown = _get_tool_cooldown()
                    print(f"  [TOOL_COOLDOWN] Tool #{tool_call_count}: Waiting {cooldown:.1f}s before {fn_name}...")
                    await asyncio.sleep(cooldown)
                    print(f"  [TOOL_COOLDOWN] Done waiting, proceeding with {fn_name}")

                print(f"[{self.name}] Tool: {fn_name}...")
                self.bus.log_event(
                    self.name, "TOOL_CALL", fn_name, metadata={"args": fn_args}
                )

                result = "Error: Tool not found."
                if fn_name in self.tool_map:
                    try:
                        res = self.tool_map[fn_name](**fn_args)
                        if inspect.isawaitable(res):
                            result = await res
                        else:
                            result = res
                    except Exception as e:
                        result = f"Error executing {fn_name}: {e}"

                self.bus.log_event(
                    self.name,
                    "TOOL_RESULT",
                    "Result",
                    metadata={"result": str(result)},
                )

                tool_outputs.append(
                    types.Part.from_function_response(
                        name=fn_name, response={"result": str(result)}
                    )
                )

            # Retry loop for tool outputs (same as initial message)
            tool_response = None
            last_error = None
            for retry in range(6):
                try:
                    # Global rate limit across all agents
                    await _global_throttle()
                    res = self.chat.send_message(tool_outputs)
                    if inspect.isawaitable(res):
                        tool_response = await res
                    else:
                        tool_response = res
                    # Track and log token usage for tool responses
                    if hasattr(tool_response, 'usage_metadata') and tool_response.usage_metadata:
                        um = tool_response.usage_metadata
                        print(f"[{self.name}] Tool response. Tokens: {um.prompt_token_count} in, {um.candidates_token_count} out")
                        if self.usage_callback:
                            self.usage_callback(self.model_name, um.prompt_token_count, um.candidates_token_count)
                    break  # Success
                except Exception as e:
                    error_str = str(e)
                    last_error = e
                    is_429 = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                    if is_429 and retry < 5:
                        sleep_time = _RATE_LIMIT_COOLDOWN
                        print(f"[{self.name}] Tool output rate limit. Cooling down for {sleep_time // 60} minutes...")
                        # Penalty Box: Push global timer forward to stop other agents
                        async with _global_rate_lock:
                            _global_last_request_time = _time.time() + sleep_time
                        await asyncio.sleep(sleep_time)
                    else:
                        print(f"[{self.name}] Error sending tool output: {e}")
                        break

            if not tool_response:
                # NEVER move on with incomplete state - raise so caller can handle
                raise RuntimeError(f"[{self.name}] Failed to send tool output after retries: {last_error}")
            response = tool_response

    def kickstart(self, instruction):
        print(f"\n[{self.name}] KICKSTART: {instruction}")
        self.bus.send("System", self.name, f"MISSION START: {instruction}")


async def run_swarm(agents: dict[str, SwarmAgent], bus: MessageBus, max_turns: int = 50, turn_delay: float = 2.0):
    """Run the swarm until finished or max_turns.

    Args:
        agents: Dict of agent name -> SwarmAgent
        bus: MessageBus for communication
        max_turns: Maximum turns before stopping
        turn_delay: Seconds to wait between turns (helps with rate limits)
    """
    print("DEBUG: run_swarm started (Async)")
    turn = 0
    idle_strikes = 0
    MAX_IDLE_STRIKES = 5
    interrupted = False

    try:
        while not bus.finished and turn < max_turns and not interrupted:
            if not bus.execution_queue:
                stuck = [name for name in agents if bus.has_messages(name)]
                if stuck:
                    bus.execution_queue.extend(stuck)
                else:
                    idle_strikes += 1
                    if idle_strikes >= MAX_IDLE_STRIKES:
                        first_agent = list(agents.keys())[0]
                        print(f"[SUPERVISOR] System stalling. Nudging {first_agent}...")
                        # Actually inject a message so step() doesn't return early
                        bus.inject_system_message(first_agent, "NUDGE: Continue reading and creating nodes. Call source_read to get more content.")
                        idle_strikes = 0
                        continue

                    await asyncio.sleep(_get_tool_cooldown())  # Forced cooldown with jitter
                    continue

            idle_strikes = 0
            turn += 1
            agent_name = bus.execution_queue.popleft()

            print(
                f"\n--- Turn {turn} (Active: {agent_name}) [Queue: {len(bus.execution_queue)}] ---"
            )

            if agent_name in agents:
                try:
                    await agents[agent_name].step()
                    # Rate limit protection: pause between turns
                    if turn_delay > 0:
                        await asyncio.sleep(turn_delay)
                except Exception as e:
                    print(f"ERROR: Agent {agent_name} crashed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Error: Agent {agent_name} not found.")

    finally:
        bus.close()
        print(f"Stopped at Turn {turn}.")

    return turn


async def run_with_orchestrator(
    agents: dict[str, SwarmAgent],
    bus: MessageBus,
    readers: list[str],
    reviewers: list[str],
    get_progress = None,
    round_delay: float = 1.0
):
    """Run swarm with orchestrator deciding which agents run in parallel.

    Args:
        agents: Dict of agent name -> SwarmAgent
        bus: MessageBus for communication
        readers: Agent names that read and create nodes (run in parallel)
        reviewers: Agent names that review/coach (run after readers)
        get_progress: Optional async callback returning (percent, done) tuple
        round_delay: Seconds between rounds
    """
    print("=" * 60)
    print("ORCHESTRATOR MODE: Parallel agent execution")
    print(f"  Readers: {readers}")
    print(f"  Reviewers: {reviewers}")
    print("=" * 60)

    round_num = 0

    try:
        while not bus.finished:
            round_num += 1
            if get_progress:
                percent, done = await get_progress()
            else:
                percent, done = 0, False

            print(f"\n{'='*60}")
            print(f"ROUND {round_num} | Progress: {percent}%")
            print(f"{'='*60}")

            if done:
                print("Reading complete!")
                bus.finished = True
                break

            # Phase 1: Run all readers in parallel
            print(f"\n[ORCHESTRATOR] Phase 1: Readers working in parallel...")
            reader_agents = [agents[name] for name in readers if name in agents]
            if reader_agents:
                await asyncio.gather(*[agent.step() for agent in reader_agents])

            # Phase 2: Run reviewers in parallel
            print(f"\n[ORCHESTRATOR] Phase 2: Reviewers working...")
            reviewer_agents = [agents[name] for name in reviewers if name in agents]
            if reviewer_agents:
                await asyncio.gather(*[agent.step() for agent in reviewer_agents])

            # Brief delay between rounds
            if round_delay > 0:
                await asyncio.sleep(round_delay)

    except KeyboardInterrupt:
        print("\n[ORCHESTRATOR] Interrupted by user")
    finally:
        bus.close()
        print(f"\n[ORCHESTRATOR] Finished after {round_num} rounds")

    return round_num


from enum import Enum

class Phase(Enum):
    READ = "read"
    UNDERSTAND = "understand"
    THINK = "think"
    FLUID = "fluid"


async def run_four_phase_orchestrator(
    agents: dict[str, SwarmAgent],
    bus: MessageBus,
    get_progress = None,
    round_delay: float = 1.0,
    cooldown: float = 1.0,  # Delay between agent steps to avoid rate limits
):
    """Run swarm with strict three-phase sequence per chunk.

    Phases:
        1. READ + UNDERSTAND: Reader loads ONE chunk, then Connector, Skeptic, Belief Tracker
           create nodes in parallel, then Curator reviews
        2. THINK: Synthesizer creates thinking (if pivotal), Gatekeeper signs
        3. FLUID: Translator converts to natural language

    Expected agents:
        - reader: Loads ONE chunk, assesses density
        - connector, skeptic, belief_tracker: Create understanding nodes
        - curator: Reviews diversity
        - synthesizer: Creates thinking nodes
        - gatekeeper: Signs thinking
        - translator: Converts to fluid prose
    """
    print("=" * 60)
    print("THREE-PHASE ORCHESTRATOR")
    print("  Phase 1: READ + UNDERSTAND (reader → connector, skeptic, belief_tracker → curator)")
    print("  Phase 2: THINK (synthesizer → gatekeeper)")
    print("  Phase 3: FLUID (translator)")
    print("=" * 60)

    chunk_num = 0

    try:
        while not bus.finished:
            chunk_num += 1

            if get_progress:
                percent, done = await get_progress()
            else:
                percent, done = 0, False

            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_num} | Progress: {percent}%")
            print(f"{'='*60}")

            if done:
                print("Reading complete!")
                bus.finished = True
                break

            # =========== PHASE 1: READ + UNDERSTAND ===========
            print(f"\n[PHASE 1: READ + UNDERSTAND]")

            # Step 1a: Reader loads ONE chunk
            if "reader" in agents:
                print(f"  Reader loading chunk...")
                await agents["reader"].step()
                await asyncio.sleep(cooldown)
            else:
                print("  WARNING: No reader agent")

            # Step 1b: Understanding agents process sequentially (avoid rate limits)
            understand_agents = ["connector", "skeptic", "belief_tracker"]
            for agent_name in understand_agents:
                if agent_name in agents:
                    print(f"  {agent_name} processing...")
                    await agents[agent_name].step()
                    await asyncio.sleep(cooldown)

            # Step 1c: Curator reviews diversity
            if "curator" in agents:
                print(f"  Curator reviewing...")
                await agents["curator"].step()

            await asyncio.sleep(cooldown)

            # =========== PHASE 2: THINK ===========
            print(f"\n[PHASE 2: THINK]")
            if "synthesizer" in agents:
                print(f"  Synthesizer creating thinking...")
                await agents["synthesizer"].step()

                await asyncio.sleep(cooldown)
                # Gatekeeper must sign
                if "gatekeeper" in agents:
                    print(f"  Gatekeeper reviewing...")
                    await agents["gatekeeper"].step()
            else:
                print("  No synthesizer agent, skipping THINK phase")

            await asyncio.sleep(cooldown)

            # =========== PHASE 3: FLUID ===========
            print(f"\n[PHASE 3: FLUID]")
            if "translator" in agents:
                print(f"  Translator converting to fluid prose...")
                await agents["translator"].step()
                await asyncio.sleep(cooldown)
            else:
                print("  No translator agent, skipping FLUID phase")

            # Delay between chunks
            if round_delay > 0:
                await asyncio.sleep(round_delay)

    except KeyboardInterrupt:
        print("\n[ORCHESTRATOR] Interrupted by user")
    finally:
        bus.close()
        print(f"\n[ORCHESTRATOR] Finished after {chunk_num} chunks")

    return chunk_num
