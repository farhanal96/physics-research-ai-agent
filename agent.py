"""
agent.py — The core agent loop.

How it works (this is the key concept):
  1. We send the user's question to Claude along with a list of available tools.
  2. Claude decides: "I need to search arXiv" → returns a tool_use block.
  3. We actually run the tool and send the result back to Claude.
  4. Claude either calls another tool OR writes its final answer.
  5. Repeat until Claude is done.

This loop is called an "agentic loop" or "ReAct loop" in AI engineering.
"""

import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
from tools import search_arxiv, get_paper_details, TOOL_DEFINITIONS

load_dotenv()

SYSTEM_PROMPT = """You are an expert physics research assistant with deep knowledge across all areas of physics — quantum mechanics, condensed matter, astrophysics, biophysics, particle physics, and more.

Your job is to help researchers and students understand the latest findings from arXiv preprints.

When a user asks a research question:
1. ALWAYS use search_arxiv first to find relevant papers — never answer from memory alone.
2. If a paper looks highly relevant, use get_paper_details to read its full abstract.
3. Synthesize findings across multiple papers into a clear, structured answer.
4. Include specific paper titles and authors when citing findings.
5. If relevant, mention key equations, methods, or experimental techniques.
6. Point out open questions or debates in the field.
7. Use scientific language but explain clearly — your user may be a student or researcher.

Format your final answer with clear sections:
- **Key Findings** (bullet points of the most important results)
- **Notable Papers** (the papers you found most relevant)
- **Open Questions** (what the field is still working on)
- **Suggested Follow-up** (what else they might want to search)
"""


def run_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return the result as a JSON string."""
    if tool_name == "search_arxiv":
        result = search_arxiv(**tool_input)
    elif tool_name == "get_paper_details":
        result = get_paper_details(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, indent=2)


class PhysicsResearchAgent:
    """
    The agent that manages conversation state and the tool-use loop.

    Keeping this as a class means we can maintain conversation history
    across multiple questions in the same session — the agent remembers
    what papers it already found.
    """

    def __init__(self, verbose: bool = True):
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from environment
        self.verbose = verbose
        self.conversation_history = []  # full message history for multi-turn chat
        self.model = "claude-sonnet-4-6"

    def _log(self, message: str):
        """Print agent status messages (the agent 'thinking out loud')."""
        if self.verbose:
            print(f"\n\033[90m[Agent] {message}\033[0m")  # grey text

    def ask(self, user_question: str) -> str:
        """
        Send a question to the agent and get a research-backed answer.

        This is the agentic loop — it keeps running until Claude decides
        it has enough information to answer without using any more tools.
        """
        # Add the user's message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_question
        })

        iteration = 0
        max_iterations = 10  # safety limit — prevents infinite loops

        while iteration < max_iterations:
            iteration += 1
            self._log(f"Thinking... (step {iteration})")

            # Call Claude with the full conversation history and tool list
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.conversation_history,
            )

            # ── Case 1: Claude wants to use a tool ────────────────────────────
            if response.stop_reason == "tool_use":
                # There may be multiple tool calls in one response
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        self._log(f"Using tool: {tool_name}({json.dumps(tool_input)})")

                        # Actually run the tool
                        result_str = run_tool(tool_name, tool_input)

                        self._log(f"Tool returned {len(result_str)} characters of data")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                # Add Claude's response (containing tool_use blocks) to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Add the tool results back as a user message (this is the API format)
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results,
                })

            # ── Case 2: Claude is done — it has a final answer ────────────────
            elif response.stop_reason == "end_turn":
                # Extract the text from the response
                final_answer = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_answer += block.text

                # Add the final answer to conversation history (for follow-ups)
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_answer,
                })

                self._log(f"Done after {iteration} steps.")
                return final_answer

            else:
                # Unexpected stop reason
                return f"Agent stopped unexpectedly: {response.stop_reason}"

        return "Agent hit the iteration limit without finishing. Try a more specific question."

    def reset(self):
        """Clear conversation history to start a fresh research session."""
        self.conversation_history = []
        self._log("Conversation history cleared.")
