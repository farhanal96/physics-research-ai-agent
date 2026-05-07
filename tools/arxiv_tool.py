"""
arXiv tools — two actions the agent can take:
  1. search_arxiv   → find papers by keyword/topic
  2. get_paper_details → fetch full abstract + metadata for one paper by ID
"""

import arxiv


def search_arxiv(query: str, max_results: int = 5, category: str = "") -> dict:
    """
    Search arXiv for physics papers.

    Args:
        query:       What to search for, e.g. "quantum tunneling biology"
        max_results: How many papers to return (1–10)
        category:    Optional arXiv category filter, e.g. "quant-ph", "cond-mat"

    Returns:
        A dict with a list of papers or an error message.
    """
    try:
        # If a category is given, prepend it to the query in arXiv syntax
        full_query = f"cat:{category} AND {query}" if category else query

        search = arxiv.Search(
            query=full_query,
            max_results=min(max_results, 10),  # hard cap at 10
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in search.results():
            papers.append({
                "arxiv_id": result.entry_id.split("/")[-1],  # e.g. "2401.12345v1"
                "title": result.title,
                "authors": [a.name for a in result.authors[:4]],
                "abstract": result.summary[:600] + "..." if len(result.summary) > 600 else result.summary,
                "published": str(result.published.date()),
                "url": result.entry_id,
                "categories": result.categories,
            })

        if not papers:
            return {"error": "No papers found. Try broader search terms."}

        return {"papers": papers, "total_found": len(papers)}

    except Exception as e:
        return {"error": f"arXiv search failed: {str(e)}"}


def get_paper_details(arxiv_id: str) -> dict:
    """
    Fetch full details for a single paper by its arXiv ID.

    Args:
        arxiv_id: The arXiv paper ID, e.g. "2401.12345" or "2401.12345v1"

    Returns:
        Full paper metadata including the complete abstract.
    """
    try:
        # Strip version suffix if present so the search works
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

        search = arxiv.Search(id_list=[clean_id])
        result = next(search.results())

        return {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,  # full abstract, no truncation
            "published": str(result.published.date()),
            "updated": str(result.updated.date()),
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "categories": result.categories,
            "journal_ref": result.journal_ref or "Not published in journal yet",
            "doi": result.doi or "No DOI",
        }

    except StopIteration:
        return {"error": f"Paper '{arxiv_id}' not found on arXiv."}
    except Exception as e:
        return {"error": f"Failed to fetch paper details: {str(e)}"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions — this is what we pass to Claude so it knows what tools
# exist and what arguments each one accepts.
# ──────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "search_arxiv",
        "description": (
            "Search arXiv for physics research papers on any topic. "
            "Returns titles, authors, abstracts, and paper IDs. "
            "Use this first when the user asks about a physics topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — use specific physics terminology for best results",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of papers to return. Default 5, max 10.",
                    "default": 5,
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional arXiv category to narrow search. "
                        "Examples: 'quant-ph' (quantum physics), 'cond-mat' (condensed matter), "
                        "'hep-th' (high energy theory), 'astro-ph' (astrophysics), "
                        "'physics.bio-ph' (biological physics)"
                    ),
                    "default": "",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_paper_details",
        "description": (
            "Fetch the complete abstract and full metadata for a specific arXiv paper "
            "using its arXiv ID (e.g. '2401.12345'). "
            "Use this after search_arxiv to dig deeper into a paper that looks relevant."
        ),
        "input_schema": {
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
]
