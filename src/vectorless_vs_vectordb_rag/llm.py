"""
FILE:     llm.py
PURPOSE:  One helper, `ask_claude()`, for every Claude call in the project (answering, grading,
          judging). It uses `claude-agent-sdk`, which runs Claude Code on your Claude subscription,
          so no Anthropic API key is needed.

USED BY:  vectordb/rag.py (generate + grade), compare/evaluate.py (judge)

EXAMPLE:
    ask_claude("What is RAG?")                                   -> "RAG is ..."
    ask_claude("Is Paris in France?", schema={...json schema...}) -> {"yes": True, ...}
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from . import config


def ask_claude(prompt: str, system: str = "", schema: dict | None = None, model: str = config.CHAT_MODEL) -> Any:
    """Send one prompt to Claude and return its answer.

    prompt:  the question / task
    system:  instructions for how Claude should behave (system prompt)
    schema:  optional JSON schema -> Claude's answer comes back as a dict matching it
    model:   "sonnet", "haiku" or "opus"
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system,
        tools=[],              # Claude only writes text: no file access, no shell commands
        setting_sources=[],    # do not load your CLAUDE.md / settings into the answering model
        max_turns=1,           # one question, one answer
        # structured output: Claude must reply with JSON that matches the schema
        output_format={"type": "json_schema", "schema": schema} if schema else None,
    )

    async def run() -> Any:
        result = None
        # The SDK streams messages (start-up info, the answer, then a final ResultMessage).
        # Read the stream to the end, and keep the ResultMessage: it holds the final answer.
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, ResultMessage):
                result = message
        if result is None or result.is_error:
            raise RuntimeError(f"Claude call failed: {getattr(result, 'result', None)}")
        return result.structured_output if schema else result.result

    # The SDK is async. Running it in a worker thread with its own event loop makes this function
    # work everywhere: plain scripts, Jupyter and Streamlit (which already run their own event loop).
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, run()).result()
