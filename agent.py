"""
agent.py — The core agent loop, powered by Groq (free tier).

Groq runs Llama 3.3 70B for free — no credit card, no expiry.
The tool-use logic (agentic loop) is identical regardless of which
AI provider we use. Only the API calls differ.

How it works:
  1. We send the user's question to the model with a list of tools.
  2. The model decides: "I need to search arXiv" → returns tool_calls.
  3. We actually run the tool and send the result back.
  4. The model either calls another tool OR writes its final answer.
  5. Repeat until done. This is the agentic loop.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv
from tools import search_arxiv, get_paper_details

load_dotenv()

SYSTEM_PROMPT = """You are an expert physics research assistant with deep knowledge across all areas of physics — quantum mechanics, condensed matter, astrophysics, biophysics, particle physics, and more.

Your job is to help researchers and students understand the latest findings from arXiv preprints.

When a user asks a research question:
1. ALWAYS use search_arxiv first to find relevant papers — never answer from memory alone.
2. If a paper looks highly relevant, use get_paper_details to read its full abstract.
3. Synthesise findings across multiple papers into a clear, structured answer.
4. Include specific paper titles and authors when citing findings.
5. If relevant, mention key equations, methods, or experimental techniques.
6. Point out open questions or debates in the field.

Format your final answer with clear sections:
- **Key Findings** (bullet points of the most important results)
- **Notable Papers** (the papers you found most relevant, with arXiv links)
- **Open Questions** (what the field is still working on)
- **Suggested Follow-up** (what else they might want to search)
"""

# ── Tool definitions (OpenAI / Groq format) ───────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": (
                "Search arXiv for physics research papers on any topic. "
                "Returns titles, authors, abstracts, and paper IDs. "
                "Use this first when the user asks about a physics topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — use specific physics terminology for best results",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of papers to return. Default 5, max 10.",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional arXiv category filter. Examples: 'quant-ph', "
                            "'cond-mat', 'hep-th', 'astro-ph', 'physics.bio-ph'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": (
                "Fetch the complete abstract and full metadata for a specific arXiv paper "
                "using its arXiv ID. Use this after search_arxiv to dig deeper into a relevant paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "The arXiv paper ID, e.g. '2401.12345' or '2401.12345v2'",
                    }
                },
                "required": ["arxiv_id"],
            },
        },
    },
]


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

    Keeping this as a class means we maintain conversation history
    across multiple questions — the agent remembers papers it already found.
    """

    def __init__(self, verbose: bool = True):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.verbose = verbose
        self.model = "llama3-groq-70b-8192-tool-use-preview"  # fine-tuned for tool use
        # Full conversation history — includes system, user, assistant, and tool messages
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _log(self, message: str):
        if self.verbose:
            print(f"\n\033[90m[Agent] {message}\033[0m")

    def ask(self, user_question: str) -> str:
        """
        Send a question to the agent and get a research-backed answer.
        Runs the agentic loop until the model stops calling tools.
        """
        self.messages.append({"role": "user", "content": user_question})

        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            self._log(f"Thinking... (step {iteration})")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            )

            message = response.choices[0].message

            # Add the assistant's response to history
            self.messages.append(message)

            # ── Case 1: model wants to use tools ──────────────────────────────
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    self._log(f"Using tool: {tool_name}({json.dumps(tool_args)})")

                    result_str = run_tool(tool_name, tool_args)
                    self._log(f"Tool returned {len(result_str)} characters")

                    # Send the tool result back as a "tool" role message
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

            # ── Case 2: model has a final answer ──────────────────────────────
            else:
                self._log(f"Done after {iteration} steps.")
                return message.content or "No answer generated."

        return "Agent hit the iteration limit. Try a more specific question."

    def reset(self):
        """Clear conversation history (keep system prompt) for a fresh session."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._log("Conversation history cleared.")
