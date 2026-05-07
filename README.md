# Physics Research AI Agent

An autonomous AI agent that searches arXiv for physics papers, reads abstracts, extracts key findings, and answers research questions — powered by **Google Gemini** (free) and the **arXiv API**.

> Built as a portfolio project to demonstrate AI agent engineering in a physics domain.

## What It Does

You ask a research question like:
> *"What are the latest findings in quantum tunneling in biological systems?"*

The agent automatically:
1. Searches arXiv for the most relevant papers
2. Reads and ranks abstracts by relevance
3. Fetches full details on the most promising papers
4. Synthesises findings into a structured answer with citations

This is what makes it an **agent** (not a chatbot) — it decides on its own which papers to read and how many searches to run.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI Model | Gemini 1.5 Flash (Google — free tier) |
| Paper Database | arXiv via `arxiv` Python library |
| Language | Python 3.10+ |
| Tool Use | Gemini function calling API |

## Project Structure

```
physics-research-ai-agent/
├── tools/
│   ├── __init__.py
│   └── arxiv_tool.py      # arXiv search + paper detail tools
├── agent.py               # agentic loop — Gemini + tools
├── main.py                # CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/farhanal96/physics-research-ai-agent.git
cd physics-research-ai-agent
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get your free API key**
- Go to https://aistudio.google.com/apikey
- Sign in with your Google account
- Click **Create API key** — it's free, no credit card needed

**4. Set your API key**
```bash
copy .env.example .env
# Open .env and paste your key
```

**5. Run the agent**
```bash
python main.py
```

## Example Usage

```
🔬 Your question: What is the current state of quantum error correction?

[Agent] Thinking... (step 1)
[Agent] Using tool: search_arxiv({"query": "quantum error correction 2024", "max_results": 5})
[Agent] Using tool: get_paper_details({"arxiv_id": "2401.xxxxx"})
[Agent] Done after 3 steps.

**Key Findings**
- Surface codes remain the leading approach with recent threshold improvements...

**Notable Papers**
- "Logical qubit performance..." — Google Quantum AI (2024)
```

## How the Agent Works

The agent uses Gemini's **function calling API** in a loop:

```
User question
     ↓
Gemini decides: "I need to search arXiv"
     ↓
We run search_arxiv() and send results back to Gemini
     ↓
Gemini decides: "I want more detail on paper X"
     ↓
We run get_paper_details() and send results back
     ↓
Gemini has enough info → writes final answer
```

This loop pattern (often called a ReAct loop) is the foundation of all modern AI agents.

## License

MIT
