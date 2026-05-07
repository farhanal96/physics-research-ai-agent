"""
main.py — Run the physics research agent.

Two modes:
  1. Interactive mode (default): a REPL where you type questions and get answers.
  2. Single question mode: pass a question as a command-line argument.

Usage:
  python main.py                          # interactive mode
  python main.py "What is quantum foam?"  # single question
"""

import sys
import os
from agent import PhysicsResearchAgent


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Physics Research AI Agent                          ║
║          Powered by Gemini + arXiv                          ║
╠══════════════════════════════════════════════════════════════╣
║  Ask any physics research question.                         ║
║  The agent will search arXiv and summarise real papers.     ║
║                                                             ║
║  Commands:                                                  ║
║    'reset'  — clear memory and start fresh                  ║
║    'quit'   — exit                                          ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLE_QUESTIONS = [
    "What are the latest findings in quantum biology and biophysics?",
    "Summarise recent progress in room-temperature superconductivity",
    "What do papers say about black hole information paradox in 2024?",
    "What is the current state of research on quantum error correction?",
]


def check_api_key():
    """Make sure the API key is set before we start."""
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_api_key_here":
        print("\n❌  GEMINI_API_KEY not set.")
        print("    1. Copy .env.example to .env")
        print("    2. Get your free key from https://aistudio.google.com/apikey")
        print("    3. Paste it into your .env file")
        print("    4. Run again\n")
        sys.exit(1)


def run_interactive():
    """Interactive REPL — keeps the agent alive between questions."""
    print(BANNER)
    print("Example questions to get you started:")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print()

    agent = PhysicsResearchAgent(verbose=True)

    while True:
        try:
            user_input = input("\n🔬 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("✅ Memory cleared. Starting fresh.")
            continue

        print("\n" + "─" * 64)
        try:
            answer = agent.ask(user_input)
            print("\n" + answer)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("\n❌  Rate limit hit. Two possible causes:")
                print("    1. Your API key quota is exhausted — wait a minute and retry.")
                print("    2. You shared your key publicly — regenerate it at aistudio.google.com/apikey")
            elif "API_KEY" in err or "401" in err or "403" in err:
                print("\n❌  Invalid API key. Check your .env file.")
            else:
                print(f"\n❌  Error: {err}")
        print("─" * 64)


def run_single(question: str):
    """Run one question and exit — useful for scripting."""
    agent = PhysicsResearchAgent(verbose=True)
    print(f"\nQuestion: {question}\n")
    print("─" * 64)
    answer = agent.ask(question)
    print("\n" + answer)


if __name__ == "__main__":
    check_api_key()

    if len(sys.argv) > 1:
        # Join all args in case the question wasn't quoted
        question = " ".join(sys.argv[1:])
        run_single(question)
    else:
        run_interactive()
