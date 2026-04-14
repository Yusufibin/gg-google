# gg — AI terminal assistant

A lightweight AI assistant that lives in your terminal. Ask questions, get answers with live Google search context, and hold multi-turn conversations — powered by [LangGraph](https://github.com/langchain-ai/langgraph), [Nemotron](https://openrouter.ai/nvidia/nemotron-3-nano-30b-a3b:free) via OpenRouter, and [Serper](https://serper.dev/).

No Python environment to manage. No dependencies to install manually. Just run it.

```
$ gg "what is the current price of bitcoin"

⟳  searching: what is the current price of bitcoin

gg › Bitcoin is currently trading at around $62,400 (CoinMarketCap).
    It's up about 2.3% in the last 24 hours.
```

---

## Features

- **Web-augmented answers** — searches Google in real time and feeds results to the model as context
- **Interactive chat** — multi-turn sessions with full conversation memory
- **Zero environment pollution** — powered by [uv](https://docs.astral.sh/uv/), packages are cached and never touch your system Python
- **Streamable** — responses stream token by token directly in your terminal
- **Fast** — first run downloads packages (~10s), every subsequent call is near-instant (~200ms)

---

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — modern Python package manager
- An [OpenRouter](https://openrouter.ai/) API key (free tier available)
- A [Serper](https://serper.dev/) API key (free tier: 2,500 searches/month)

---

## Installation

**1. Install uv**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with Homebrew
brew install uv
```

**2. Clone the repo**

```bash
git clone https://github.com/yusufibin/gg.git
cd gg
```

**3. Add your API keys**

Add these lines to your `~/.zshrc` or `~/.bashrc`:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export SERPER_API_KEY="your-key-here"
```

Then reload your shell:

```bash
source ~/.zshrc
```

**4. Set up the alias**

```bash
chmod +x gg.py

# Add to ~/.zshrc
echo 'alias gg="uv run /path/to/gg.py"' >> ~/.zshrc
source ~/.zshrc
```

---

## Usage

```bash
# One-shot question with web search
gg "who won the last FIFA World Cup"

# One-shot question without web search (faster)
gg -n "explain what a binary tree is"

# Interactive chat mode (web enabled)
gg -i

# Interactive chat mode (web disabled)
gg -i -n

# Help
gg -h
```

---

## How it works

gg is built as a [LangGraph](https://github.com/langchain-ai/langgraph) graph with two nodes and conditional routing:

```
[entry]
  ├─ web on  ──► [search] ──► [llm] ──► END
  └─ web off ──────────────► [llm] ──► END
```

- **search node** — calls the Serper API and stores results in the graph state
- **llm node** — builds the prompt (system + history + question), streams the response

Search results are injected into the `SystemMessage` rather than as fake conversation turns, which keeps the session history clean and multi-turn context intact.

---

## Configuration

You can change the model by editing the `MODEL` variable at the top of `gg.py`:

```python
MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
```

Any model available on [OpenRouter](https://openrouter.ai/models) works as a drop-in replacement.

---

## Cache

uv caches packages in `~/.cache/uv/` after the first run. No reinstallation happens on subsequent calls.

To clear the cache:

```bash
uv cache clean
```

---

## License

MIT
