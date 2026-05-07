"""
agent.py — The core agent loop, powered by Google Gemini (free tier).

How it works:
  1. We send the user's question to Gemini along with our two arXiv tools.
  2. Gemini decides: "I need to search arXiv" → returns a function_call.
  3. We actually run the tool and send the result back to Gemini.
  4. Gemini either calls another tool OR writes its final answer.
  5. Repeat until Gemini is done (this is called the agentic loop).
"""

import json
import os
from google import genai
from google.genai import types
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

# ── Tool definitions for Gemini ───────────────────────────────────────────────

ARXIV_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_arxiv",
                description=(
                    "Search arXiv for physics research papers on any topic. "
                    "Returns titles, authors, abstracts, and paper IDs. "
                    "Use this first when the user asks about a physics topic."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Search query — use specific physics terminology for best results",
                        ),
                        "max_results": types.Schema(
                            type=types.Type.INTEGER,
                            description="Number of papers to return. Default 5, max 10.",
                        ),
                        "category": types.Schema(
                            type=types.Type.STRING,
                            description=(
                                "Optional arXiv category filter. Examples: 'quant-ph', "
                                "'cond-mat', 'hep-th', 'astro-ph', 'physics.bio-ph'"
                            ),
                        ),
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="get_paper_details",
                description=(
                    "Fetch the complete abstract and full metadata for a specific arXiv paper "
                    "using its arXiv ID. Use this after search_arxiv to dig deeper into a paper."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "arxiv_id": types.Schema(
                            type=types.Type.STRING,
                            description="The arXiv paper ID, e.g. '2401.12345' or '2401.12345v2'",
                        ),
                    },
                    required=["arxiv_id"],
                ),
            ),
        ]
    )
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
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.verbose = verbose
        self.model = "gemini-2.0-flash"
        # Full conversation history stored as a list of Content objects
        self.history = []

    def _log(self, message: str):
        if self.verbose:
            print(f"\n\033[90m[Agent] {message}\033[0m")

    def ask(self, user_question: str) -> str:
        """
        Send a question to the agent and get a research-backed answer.
        Runs the agentic loop until Gemini stops calling tools.
        """
        # Add the user message to history
        self.history.append(
            types.Content(role="user", parts=[types.Part(text=user_question)])
        )

        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1
            self._log(f"Thinking... (step {iteration})")

            response = self.client.models.generate_content(
                model=self.model,
                contents=self.history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=ARXIV_TOOLS,
                ),
            )

            candidate = response.candidates[0]
            response_content = candidate.content

            # Add Gemini's response to history
            self.history.append(response_content)

            # Collect any function calls from the response
            function_calls = [
                part.function_call
                for part in response_content.parts
                if part.function_call is not None
            ]

            if not function_calls:
                # No tool calls — extract final text and return
                final_text = "".join(
                    part.text
                    for part in response_content.parts
                    if part.text is not None
                )
                self._log(f"Done after {iteration} steps.")
                return final_text or "No answer generated. Try rephrasing your question."

            # Run each tool and collect results
            tool_response_parts = []
            for fc in function_calls:
                tool_args = dict(fc.args) if fc.args else {}
                self._log(f"Using tool: {fc.name}({json.dumps(tool_args)})")

                result_str = run_tool(fc.name, tool_args)
                self._log(f"Tool returned {len(result_str)} characters")

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result_str},
                    )
                )

            # Send all tool results back as a user message
            self.history.append(
                types.Content(role="user", parts=tool_response_parts)
            )

        return "Agent hit the iteration limit. Try a more specific question."

    def reset(self):
        """Clear conversation history to start a fresh research session."""
        self.history = []
        self._log("Conversation history cleared.")
