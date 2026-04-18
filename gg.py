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
#  - uv             : https://docs.astral.sh/uv/getting-started/installation/
#  - OpenRouter key : https://openrouter.ai/
#  - Serper key     : https://serper.dev/
#
#  SETUP (one-time)
#  ─────────────────
#  1. Install uv:
#       brew install uv
#     or: curl -LsSf https://astral.sh/uv/install.sh | sh
#
#  2. Add API keys to ~/.zshrc or ~/.bashrc:
#       export OPENROUTER_API_KEY="sk-or-..."
#       export SERPER_API_KEY="your-key-here"
#
#  3. Make executable + add alias:
#       chmod +x ~/scripts/gg.py
#       alias gg="uv run ~/scripts/gg.py"
#       source ~/.zshrc
#
#  USAGE
#  ──────
#  gg "question"      →  auto-décide si recherche web nécessaire
#  gg -n "question"   →  force sans recherche web (plus rapide)
#  gg -w "question"   →  force avec recherche web
#  gg -i              →  mode chat interactif (web auto)
#  gg -i -n           →  mode chat interactif (web désactivé)
#  gg -i -w           →  mode chat interactif (web toujours actif)
#
#  GRAPH
#  ──────
#  [router] → décide → [search] → [llm] → END
#                    ↘           ↗
#                      (skip)
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import argparse
import urllib.request
from typing import Annotated, TypedDict, Optional
from operator import add

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, BaseMessage
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL    = "nvidia/nemotron-3-nano-30b-a3b:free"
BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_KEY     = os.environ.get("OPENROUTER_API_KEY", "")
SERPER_KEY         = os.environ.get("SERPER_API_KEY", "")
SERPER_NUM_RESULTS = 5

SYSTEM_BASE = (
    "You are a helpful and concise AI assistant running in a terminal. "
    "Answer directly and clearly. "
    "Avoid unnecessary markdown (no ** or ##) unless writing code. "
    "If you are unsure about something, say so explicitly."
)

SYSTEM_WITH_SEARCH = (
    SYSTEM_BASE
    + " You are provided with Google search results as context below. "
    "Use them to answer with up-to-date information. "
    "Cite your sources when relevant (just the site name is enough)."
)

# Prompt for the router node — must return strict JSON, nothing else
ROUTER_SYSTEM = """\
You are a routing assistant. Given a user question (and optional recent conversation),
decide if a live web search is needed to answer accurately.

Search IS needed when the question:
- Asks about current events, news, prices, weather, sports scores
- References a person, company, or product whose status may have changed
- Contains time words: today, now, latest, recent, 2024, 2025, etc.
- Requires live or frequently-updated data

Search is NOT needed when the question:
- Is about programming, math, logic, or stable scientific concepts
- Asks for writing help, code generation, or text transformation
- Is a conversational follow-up with enough context already provided
- Is purely creative, hypothetical, or opinion-based

Respond ONLY with valid JSON — no markdown, no explanation, nothing else:
{"needs_search": true/false, "query": "<optimized English search query or empty string>", "reason": "<one short sentence>"}"""

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
# web_mode : "auto" | "force_on" | "force_off"
#   - "auto"      → router decides each turn
#   - "force_on"  → always search  (-w flag)
#   - "force_off" → never search   (-n flag)

class GGState(TypedDict):
    messages:       Annotated[list[BaseMessage], add]
    question:       str
    search_results: str
    web_mode:       str   # "auto" | "force_on" | "force_off"
    will_search:    bool  # final decision made by router_node
    search_query:   str   # refined query (may differ from raw question)
    search_reason:  str   # shown to user in terminal

# ─── Startup checks ───────────────────────────────────────────────────────────

def check_env() -> None:
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

# ─── Node 0 : router ─────────────────────────────────────────────────────────
#
# Three paths:
#   force_off → will_search=False, no LLM call
#   force_on  → will_search=True,  no LLM call, raw question as query
#   auto      → asks the LLM; also refines the search query if needed

def router_node(state: GGState) -> dict:
    mode = state.get("web_mode", "auto")

    # ── Forced off ──────────────────────────────────────────────────────────
    if mode == "force_off":
        return {
            "will_search":   False,
            "search_query":  "",
            "search_reason": "web désactivé (-n)",
        }

    # ── Forced on ───────────────────────────────────────────────────────────
    if mode == "force_on":
        return {
            "will_search":   True,
            "search_query":  state["question"],
            "search_reason": "web forcé (-w)",
        }

    # ── Auto : ask the LLM ───────────────────────────────────────────────────
    llm = ChatOpenAI(
        model=MODEL,
        api_key=OPENROUTER_KEY,
        base_url=BASE_URL,
        streaming=False,
        temperature=0,
    )

    # Include recent conversation so the router can detect follow-up questions
    # that don't need a fresh search
    recent_context = ""
    history = state.get("messages", [])
    if history:
        tail  = history[-2:] if len(history) >= 2 else history
        lines = []
        for m in tail:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            lines.append(f"{role}: {m.content[:300]}")
        recent_context = "\n\nRecent conversation:\n" + "\n".join(lines)

    prompt = f"Question: {state['question']}{recent_context}"

    try:
        resp = llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = resp.content.strip()

        # Strip accidental markdown fences the model might add
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        parsed       = json.loads(raw)
        needs_search = bool(parsed.get("needs_search", False))
        query        = parsed.get("query", state["question"]) if needs_search else ""
        reason       = parsed.get("reason", "")

    except Exception as e:
        # Routing failed → default to searching to avoid wrong answers
        needs_search = True
        query        = state["question"]
        reason       = f"erreur router ({e}), recherche par défaut"

    return {
        "will_search":   needs_search,
        "search_query":  query,
        "search_reason": reason,
    }

# ─── Conditional edge after router ───────────────────────────────────────────

def after_router(state: GGState) -> str:
    if state.get("will_search"):
        reason = state.get("search_reason", "")
        query  = state.get("search_query", "")
        print(f"{GRAY}⟳  recherche · {reason} · \"{query}\"{RESET}", flush=True)
        return "search"
    else:
        reason = state.get("search_reason", "")
        print(f"{GRAY}✦  pas de recherche · {reason}{RESET}", flush=True)
        return "llm"

# ─── Node 1 : web search (Serper) ────────────────────────────────────────────

def search_node(state: GGState) -> dict:
    query   = state.get("search_query") or state["question"]
    payload = json.dumps({"q": query, "num": SERPER_NUM_RESULTS}).encode()
    req     = urllib.request.Request(
        "https://google.serper.dev/search",
        data=payload,
        headers={
            "X-API-KEY":    SERPER_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"{YELLOW}⚠  Recherche échouée : {e}{RESET}", flush=True)
        return {"search_results": f"[Search failed: {e}]"}

    parts = []

    # Answer box: direct Google answer when available
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

# ─── Node 2 : LLM answer ─────────────────────────────────────────────────────

def llm_node(state: GGState) -> dict:
    llm = ChatOpenAI(
        model=MODEL,
        api_key=OPENROUTER_KEY,
        base_url=BASE_URL,
        streaming=True,
    )

    if state.get("search_results"):
        system_content = (
            SYSTEM_WITH_SEARCH
            + f"\n\n[Google search results]\n{state['search_results']}"
        )
    else:
        system_content = SYSTEM_BASE

    messages_to_send: list[BaseMessage] = [SystemMessage(content=system_content)]
    messages_to_send.extend(state.get("messages", []))
    messages_to_send.append(HumanMessage(content=state["question"]))

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

    return {
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content="".join(full_response)),
        ],
        # Reset per-turn search state so the next question starts clean
        "search_results": "",
        "will_search":    False,
        "search_query":   "",
        "search_reason":  "",
    }

# ─── Graph construction ───────────────────────────────────────────────────────
#
#  [router]
#     ├─ will_search=True  ──► [search] ──► [llm] ──► END
#     └─ will_search=False ──────────────► [llm] ──► END

def build_graph():
    graph = StateGraph(GGState)

    graph.add_node("router", router_node)
    graph.add_node("search", search_node)
    graph.add_node("llm",    llm_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        after_router,
        {"search": "search", "llm": "llm"},
    )

    graph.add_edge("search", "llm")
    graph.add_edge("llm",    END)

    return graph.compile()

# ─── Single-turn runner ───────────────────────────────────────────────────────

def run_once(
    question: str,
    web_mode: str = "auto",
    history:  Optional[list[BaseMessage]] = None,
) -> list[BaseMessage]:
    app    = build_graph()
    result = app.invoke({
        "question":       question,
        "messages":       history or [],
        "search_results": "",
        "web_mode":       web_mode,
        "will_search":    False,
        "search_query":   "",
        "search_reason":  "",
    })
    return result["messages"]

# ─── Interactive mode ─────────────────────────────────────────────────────────

def interactive_mode(web_mode: str) -> None:
    history: list[BaseMessage] = []

    if web_mode == "force_off":
        mode_label = f"{RED}(web off){RESET}"
    elif web_mode == "force_on":
        mode_label = f"{GREEN}(web toujours on){RESET}"
    else:
        mode_label = f"{CYAN}(web auto){RESET}"

    print(
        f"\n{CYAN}{BOLD}gg — mode interactif{RESET}  {mode_label}  "
        f"{GRAY}Ctrl+C ou 'exit' pour quitter{RESET}\n"
    )

    while True:
        try:
            user_input = input(f"{YELLOW}vous › {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{GRAY}À bientôt !{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print(f"{GRAY}À bientôt !{RESET}\n")
            break

        history = run_once(user_input, web_mode=web_mode, history=history)

# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    check_env()

    parser = argparse.ArgumentParser(prog="gg", add_help=False)
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Mode chat interactif")
    parser.add_argument("-n", "--no-web",      action="store_true",
                        help="Forcer sans recherche web")
    parser.add_argument("-w", "--web",         action="store_true",
                        help="Forcer avec recherche web")
    parser.add_argument("-h", "--help",        action="store_true",
                        help="Afficher l'aide")
    parser.add_argument("question", nargs="*",
                        help="Question directe")
    args = parser.parse_args()

    # -n prend le dessus sur -w si les deux sont passés
    if args.no_web:
        web_mode = "force_off"
    elif args.web:
        web_mode = "force_on"
    else:
        web_mode = "auto"

    if args.help:
        print(f"""
{CYAN}{BOLD}gg{RESET} — assistant terminal IA · LangGraph + Nemotron + Serper

{BOLD}Usage:{RESET}
  gg "question"      Décide automatiquement si une recherche est utile
  gg -n "question"   Forcer sans recherche web (plus rapide)
  gg -w "question"   Forcer avec recherche web
  gg -i              Mode chat interactif (web auto)
  gg -i -n           Mode chat interactif (web désactivé)
  gg -i -w           Mode chat interactif (web toujours actif)

{BOLD}Variables d'environnement:{RESET}
  OPENROUTER_API_KEY     Clé OpenRouter · https://openrouter.ai/
  SERPER_API_KEY         Clé Serper     · https://serper.dev/

{BOLD}Modèle:{RESET}  {MODEL}
{BOLD}Graph:{RESET}   [router] → [search?] → [llm]
{BOLD}Cache:{RESET}   ~/.cache/uv/  —  vider avec : uv cache clean
""")
        sys.exit(0)

    if args.interactive:
        interactive_mode(web_mode=web_mode)
    elif args.question:
        run_once(" ".join(args.question), web_mode=web_mode)
    else:
        interactive_mode(web_mode=web_mode)


if __name__ == "__main__":
    main()
