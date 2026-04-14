#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langgraph>=0.2",
#   "langchain-openai>=0.1",
#   "langchain-core>=0.2",
# ]
# ///
#
# ─────────────────────────────────────────────────────────────────────────────
#  gg — AI-powered terminal assistant (LangGraph + Nemotron + Serper)
# ─────────────────────────────────────────────────────────────────────────────
#
#  REQUIREMENTS
#  ─────────────
#  - uv       : https://docs.astral.sh/uv/getting-started/installation/
#  - OpenRouter API key (free) : https://openrouter.ai/
#  - Serper API key (free tier): https://serper.dev/
#
#  SETUP (one-time)
#  ─────────────────
#  1. Install uv:
#       brew install uv
#     or:
#       curl -LsSf https://astral.sh/uv/install.sh | sh
#
#  2. Add your API keys to ~/.zshrc or ~/.bashrc:
#       export OPENROUTER_API_KEY="sk-or-..."
#       export SERPER_API_KEY="your-key-here"
#
#  3. Make the script executable:
#       chmod +x ~/scripts/gg.py
#
#  4. Add the alias to your shell config:
#       alias gg="uv run ~/scripts/gg.py"
#     then reload:
#       source ~/.zshrc
#
#  USAGE
#  ──────
#  gg "your question"     →  web search + AI answer
#  gg -n "your question"  →  AI answer without web search (faster)
#  gg -i                  →  interactive chat mode (web enabled)
#  gg -i -n               →  interactive chat mode (web disabled)
#
#  CACHE
#  ──────
#  uv caches all packages in ~/.cache/uv/ after the first run.
#  Subsequent calls are near-instant (~200-300ms).
#  To clear the cache manually: uv cache clean
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse
import urllib.request
from typing import Annotated, TypedDict
from operator import add

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, BaseMessage
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ─── Configuration ────────────────────────────────────────────────────────────

# Model used via OpenRouter — swap for any OpenRouter-compatible model
# Full list: https://openrouter.ai/models
MODEL    = "nvidia/nemotron-3-nano-30b-a3b:free"
BASE_URL = "https://openrouter.ai/api/v1"

# API keys — loaded from environment variables (never hardcode these)
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SERPER_KEY     = os.environ.get("SERPER_API_KEY", "")

# Number of Serper search results to fetch per query
SERPER_NUM_RESULTS = 5

# Base system prompt (no web search)
SYSTEM_BASE = (
    "You are a helpful and concise AI assistant running in a terminal. "
    "Answer directly and clearly. "
    "Avoid unnecessary markdown (no ** or ##) unless writing code. "
    "If you are unsure about something, say so explicitly."
)

# Extended system prompt injected when web search results are available
SYSTEM_WITH_SEARCH = (
    SYSTEM_BASE
    + " You are provided with Google search results as context below. "
    "Use them to answer with up-to-date information. "
    "Cite your sources when relevant (just the site name is enough)."
)

# ─── ANSI colors ──────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
YELLOW = "\033[93m"
GRAY   = "\033[90m"
GREEN  = "\033[92m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

# ─── LangGraph state ──────────────────────────────────────────────────────────
#
# GGState is the shared state passed between all graph nodes.
# Each node returns a partial dict that updates the state.
#
# Annotated[list, add] means messages are ACCUMULATED across nodes,
# not overwritten — new messages are appended to the existing list.

class GGState(TypedDict):
    messages:       Annotated[list[BaseMessage], add]  # full conversation history
    question:       str                                # current user question
    search_results: str                                # Serper results (empty if web off)
    web_enabled:    bool                               # web search on/off flag

# ─── Startup checks ───────────────────────────────────────────────────────────

def check_env() -> None:
    """Ensure required environment variables are set before doing anything."""
    missing = []
    if not OPENROUTER_KEY:
        missing.append("OPENROUTER_API_KEY")
    if not SERPER_KEY:
        missing.append("SERPER_API_KEY")

    if missing:
        print(f"\n{RED}✗  Missing environment variable(s): {', '.join(missing)}{RESET}")
        print(f"{GRAY}   Add them to your ~/.zshrc or ~/.bashrc:{RESET}")
        for var in missing:
            print(f"{GRAY}   export {var}=\"your-key-here\"{RESET}")
        print()
        sys.exit(1)

# ─── Node 1 : web search (Serper) ────────────────────────────────────────────
#
# Calls the Serper API (Google Search) and stores results in state["search_results"].
# Skipped entirely if web_enabled is False.

def search_node(state: GGState) -> dict:
    if not state["web_enabled"]:
        return {"search_results": ""}

    question = state["question"]
    print(f"{GRAY}⟳  searching: {question}{RESET}", flush=True)

    payload = json.dumps({"q": question, "num": SERPER_NUM_RESULTS}).encode()
    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={
            "X-API-KEY": SERPER_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        # Network failure: continue without results rather than crashing
        print(f"{YELLOW}⚠  Search failed: {e}{RESET}", flush=True)
        return {"search_results": f"[Search failed: {e}]"}

    parts = []

    # Answer box: direct Google answer when available (e.g. calculations, definitions)
    if ab := data.get("answerBox"):
        answer = ab.get("answer") or ab.get("snippet", "")
        if answer:
            parts.append(f"Direct answer: {answer}")

    # Organic results: title + snippet + URL
    for r in data.get("organic", [])[:SERPER_NUM_RESULTS]:
        title   = r.get("title", "")
        snippet = r.get("snippet", "")
        link    = r.get("link", "")
        parts.append(f"- {title} ({link})\n  {snippet}")

    results = "\n".join(parts) if parts else "[No results found]"
    return {"search_results": results}

# ─── Node 2 : LLM call (Nemotron via OpenRouter) ─────────────────────────────
#
# Builds the message list and streams the LLM response to the terminal.
#
# Message structure sent to the model:
#   1. SystemMessage  — base instructions + search results embedded (if any)
#   2. Conversation history (for interactive mode)
#   3. Current user question
#
# Search results are embedded directly in the SystemMessage rather than injected
# as fake user/assistant turns — this keeps the conversation history clean and
# prevents context confusion in multi-turn sessions.

def llm_node(state: GGState) -> dict:
    llm = ChatOpenAI(
        model=MODEL,
        api_key=OPENROUTER_KEY,
        base_url=BASE_URL,
        streaming=True,
    )

    # Embed search results into the system message if available.
    # This avoids polluting the conversation history with fake exchanges.
    if state.get("search_results"):
        system_content = (
            SYSTEM_WITH_SEARCH
            + f"\n\n[Google search results for this question]\n{state['search_results']}"
        )
    else:
        system_content = SYSTEM_BASE

    messages_to_send: list[BaseMessage] = [SystemMessage(content=system_content)]

    # Append real conversation history (previous HumanMessage/AIMessage pairs)
    messages_to_send.extend(state.get("messages", []))

    # Append the current question
    messages_to_send.append(HumanMessage(content=state["question"]))

    # Stream the response token by token
    print(f"\n{CYAN}{BOLD}gg ›{RESET} ", end="", flush=True)
    full_response: list[str] = []

    try:
        for chunk in llm.stream(messages_to_send):
            token = chunk.content
            if token:
                print(token, end="", flush=True)
                full_response.append(token)
    except Exception as e:
        print(f"\n{RED}API error: {e}{RESET}")
        sys.exit(1)

    print("\n")

    # Return the two new messages to be accumulated into state["messages"]
    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content="".join(full_response)),
        ]
    }

# ─── Conditional routing ──────────────────────────────────────────────────────
#
# Determines the first node to execute:
#   web on  → search → llm → END
#   web off →          llm → END

def should_search(state: GGState) -> str:
    return "search" if state["web_enabled"] else "llm"

# ─── Graph construction ───────────────────────────────────────────────────────
#
# Topology:
#
#   [entry]
#     ├─ web on  ──► [search] ──► [llm] ──► END
#     └─ web off ──────────────► [llm] ──► END

def build_graph():
    graph = StateGraph(GGState)

    graph.add_node("search", search_node)
    graph.add_node("llm",    llm_node)

    graph.set_conditional_entry_point(
        should_search,
        {"search": "search", "llm": "llm"},
    )

    graph.add_edge("search", "llm")
    graph.add_edge("llm", END)

    return graph.compile()

# ─── Single-turn runner ───────────────────────────────────────────────────────
#
# Invokes the graph for one question and returns the updated message history.
# In one-shot mode, history is None.
# In interactive mode, the accumulated history is passed on every turn.

def run_once(
    question: str,
    web: bool,
    history: list[BaseMessage] | None = None,
) -> list[BaseMessage]:
    app = build_graph()
    result = app.invoke({
        "question":       question,
        "messages":       history or [],
        "search_results": "",
        "web_enabled":    web,
    })
    # state["messages"] holds the full accumulated history (old + new)
    return result["messages"]

# ─── Interactive mode ─────────────────────────────────────────────────────────
#
# REPL loop: reads a question, invokes the graph, accumulates history.
# The graph is stateless by design — conversation memory is maintained
# by explicitly passing history into the state on every turn.

def interactive_mode(web: bool) -> None:
    history: list[BaseMessage] = []
    mode_label = f"{GRAY}(web off){RESET}" if not web else f"{GREEN}(web on){RESET}"
    print(
        f"\n{CYAN}{BOLD}gg — interactive mode{RESET}  {mode_label}  "
        f"{GRAY}Ctrl+C or 'exit' to quit{RESET}\n"
    )

    while True:
        try:
            user_input = input(f"{YELLOW}you › {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GRAY}Bye!{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print(f"{GRAY}Bye!{RESET}\n")
            break

        # run_once returns the full updated history (previous + 2 new messages)
        history = run_once(user_input, web=web, history=history)

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    check_env()

    parser = argparse.ArgumentParser(prog="gg", add_help=False)
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Start interactive chat mode")
    parser.add_argument("-n", "--no-web",      action="store_true",
                        help="Disable web search (faster)")
    parser.add_argument("-h", "--help",        action="store_true",
                        help="Show this help message")
    parser.add_argument("question", nargs="*",
                        help="Question to ask directly")
    args = parser.parse_args()

    if args.help:
        print(f"""
{CYAN}{BOLD}gg{RESET} — AI terminal assistant · LangGraph + Nemotron + Serper

{BOLD}Usage:{RESET}
  gg "your question"     Web search + AI answer
  gg -n "your question"  AI answer without web search (faster)
  gg -i                  Interactive chat mode (web enabled)
  gg -i -n               Interactive chat mode (web disabled)

{BOLD}Environment variables:{RESET}
  OPENROUTER_API_KEY     Your OpenRouter key · https://openrouter.ai/
  SERPER_API_KEY         Your Serper key    · https://serper.dev/

{BOLD}Model:{RESET}  {MODEL}
{BOLD}Graph:{RESET}  [search] → [llm]   (or just [llm] with -n)
{BOLD}Cache:{RESET}  ~/.cache/uv/  —  clear with: uv cache clean
""")
        sys.exit(0)

    web = not args.no_web

    if args.interactive:
        interactive_mode(web=web)
    elif args.question:
        run_once(" ".join(args.question), web=web)
    else:
        # No arguments → default to interactive mode
        interactive_mode(web=web)


if __name__ == "__main__":
    main()
