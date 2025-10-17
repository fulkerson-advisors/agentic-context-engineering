# Agentic Context Engineering (ACE)

Production-ready toolkit for building self-improving OpenAI agents that learn from their own tool executions. This repository implements the workflow introduced in **Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models** (Zhang et al., Stanford &amp; SambaNova, Oct 2025) and packages it for practical use with the OpenAI Agents SDK.

---

## Why ACE?

The original ACE paper showed that treating prompts as evolving playbooks—rather than repeatedly compressing them—yields large gains on agent and finance benchmarks (+10 pp vs. strong baselines) while cutting adaptation cost and latency. Two chronic issues in previous prompt optimizers were called out:

- **Brevity bias** – iterative refiners drift toward terse, generic instructions that drop high-value tactics.
- **Context collapse** – monolithic rewrites can suddenly shrink a carefully curated context to a few lines, erasing institutional knowledge.

ACE solves this by splitting responsibility across three lightweight roles:

| Component  | Responsibility | Effect |
|-----------|----------------|--------|
| **Generator** | Execute the task with current context | surfaces success/failure traces |
| **Reflector** | Diagnose trajectories, extract concrete lessons | preserves detail, avoids collapse |
| **Curator** | Merge lessons as *delta* bullets, deduplicate semantically | keeps contexts structured and scalable |

Each insight is a bullet with metadata (usage counts, timestamps, origin tool). Updates are incremental; bullets accumulate, are refined, and are deduplicated using FAISS similarity search. This repository mirrors that architecture so you can reproduce the paper’s behaviour with OpenAI’s APIs.

---

## Repository Tour

- `ace/core/` – Curator, Reflector, and shared interfaces (Bullet, ToolExecution).
- `ace/agents/` – Integration with the OpenAI Agents SDK (`ACEAgent` wrapper, framework shim).
- `ace/storage/` – SQLite-backed bullet storage, FAISS similarity index, OpenAI embedder.
- `examples/` – Standalone demos:
  - `simple_test.py` exercises each ACE component in isolation.
  - `weather_agent.py` shows ACE wrapped around an OpenAI Agent with reactive tool use.
- `scripts/manage_storage.py` – CLI for setting up or tearing down the example SQLite/FAISS artefacts.

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) (recommended) or plain `pip`
- OpenAI API key with access to your chosen models

### Installation

1. **Clone and enter the repo**
   ```bash
   git clone https://github.com/fulkerson-advisors/agentic-context-engineering
   cd ace
   ```
2. **Sync dependencies**
   ```bash
   uv sync
   ```
3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # edit .env with your OpenAI API key and models
   ```
4. **(Optional) Activate the environment**
   ```bash
   source .venv/bin/activate
   ```
   or prefix commands with `uv run`.

---

## Storage Management

ACE persists two artefact types:

- **SQLite (`*.db`)** – canonical bullet metadata: content, category, tool name, stats.
- **FAISS (`*.faiss`, `*.faiss.meta`)** – semantic index used for deduplication and retrieval.

Use the helper script to manage the example files:

```bash
# Create the default example databases and FAISS indices
uv run python scripts/manage_storage.py setup

# Remove them again
uv run python scripts/manage_storage.py teardown
```

Custom paths are supported:

```bash
uv run python scripts/manage_storage.py setup \
  --db tmp/my_agent.db \
  --faiss tmp/my_agent.faiss \
  --dimension 3072 \
  --overwrite
```

> Note: embeddings now live only inside the FAISS index. If you delete the `.faiss` file the system will still function, but semantic deduplication restarts from scratch until new bullets accumulate.

To inspect what’s stored:

- SQLite: `sqlite3 examples/weather_agent.db` → `.tables`, `.schema bullets`, `SELECT * FROM bullets;`
- FAISS: in Python:
  ```python
  from ace.storage.faiss_index import FAISSVectorIndex
  index = FAISSVectorIndex(dimension=1536, index_path="examples/weather_agent.faiss")
  print(index.index.ntotal)
  ```

---

## Running the Examples

Ensure storage artefacts exist (`manage_storage.py setup`) and your `.env` contains a valid `OPENAI_API_KEY`.

1. **Core component smoke test**
   ```bash
   uv run python examples/simple_test.py
   ```
   Demonstrates reflective learning and FAISS deduplication without the Agents SDK.

2. **Weather agent with OpenAI Agents SDK**
   ```bash
   uv run python examples/weather_agent.py
   ```
   Shows the full Generator→Reflector→Curator loop as the agent encounters erroneous tool calls, learns ACE bullets, and improves on subsequent queries.

---

## Configuration Reference

`.env.example` documents the supported variables:

| Variable | Purpose | Default behaviour |
|----------|---------|-------------------|
| `OPENAI_API_KEY` | Required for all OpenAI calls | – |
| `OPENAI_MODEL` | Default generation/reflection model | Pass-through unless specialised overrides are set |
| `OPENAI_EMBEDDING_MODEL` | Embedding endpoint | Falls back to `text-embedding-3-small` if unset or non-embedding |
| `OPENAI_REFLECTOR_MODEL` | Reflector override | Falls back to `OPENAI_MODEL` or `gpt-4.1-mini` |

Override per-instance by passing `model=` when creating `OpenAIEmbedder` or `OpenAIReflector`.

---

## Extensibility

ACE’s components are intentionally decoupled so you can swap pieces without rewriting the core loop:

- **Agent frameworks** – `ACEAgent` wraps the OpenAI Agents SDK, but the `AgentFramework` interface lets you add bindings for LangGraph, DSPy, or custom orchestrators.
- **Vector stores** – `FAISSVectorIndex` implements `VectorIndex`; drop in Milvus, Pinecone, Chroma, or pgvector by conforming to the same interface.
- **Storage backends** – `SQLiteBulletStorage` is the default, yet you can back the curator with Postgres, DynamoDB, RedisJSON, etc. by subclassing `BulletStorage`.

This modularity keeps ACE adaptable as your stack evolves.

## Using ACE After Installation

1. **Install and configure credentials**
   ```bash
   uv pip install agentic-context-engineering
   uv pip install --upgrade \"openai>=1.109.1\"
   uv pip install --upgrade \"openai-agents>=0.3.3\"
   export OPENAI_API_KEY=sk-...
   # optionally set OPENAI_MODEL / OPENAI_EMBEDDING_MODEL / OPENAI_REFLECTOR_MODEL
   ```
2. **Create the core components**
   ```python
   from ace import (
       Curator,
       OpenAIReflector,
       SQLiteBulletStorage,
       FAISSVectorIndex,
       OpenAIEmbedder,
       ACEAgent,
   )
   
   storage = SQLiteBulletStorage("my_agent.db")
   embedder = OpenAIEmbedder()
   vector_index = FAISSVectorIndex(embedder.dimension(), "my_agent.faiss")
   curator = Curator(storage, vector_index, embedder)
   reflector = OpenAIReflector()
   ```
3. **Wire an OpenAI Agent (optional but recommended)**
   ```python
   from agents import Agent
   
   agent = Agent(
       model="gpt-4.1-mini",
       instructions="Handle user questions using the available tools.",
       tools=[...],  # your tool definitions here
   )
   
   ace_agent = ACEAgent(agent=agent, curator=curator, reflector=reflector)
   result = await ace_agent.run("Plan tomorrow's meetings.")
   ```
4. **Manual integration (custom frameworks)**
   - Call `OpenAIReflector.reflect(...)` with a `ToolExecution` to generate insights.
   - Feed the returned bullets into `Curator.add_bullets(...)`.
   - Retrieve the playbook with `Curator.get_playbook(...)` and format it via `Curator.format_bullets_for_prompt(...)`.

### End-to-End Meeting Planner Example

Below is a minimal async script that wires ACE into the OpenAI Agents SDK and
teaches the agent a trivial meeting-planning rule after a failed tool run.

```python
import asyncio
import datetime

from dotenv import load_dotenv
from agents import Agent, function_tool

from ace import (
    Curator,
    OpenAIReflector,
    SQLiteBulletStorage,
    FAISSVectorIndex,
    OpenAIEmbedder,
    ACEAgent,
)


load_dotenv()  # pull OPENAI_API_KEY / model hints from your .env


@function_tool
def plan_meeting(date: datetime.date):
    """Plan a meeting for a specific date."""

    print(f"planning meeting for {date}")

    if date == datetime.date(2025, 10, 18):
        return "Actually, today is not October 17, 2025 but October 18, 2025."
    if date == datetime.date(2025, 10, 19):
        return "Meeting confirmed for October 19, 2025. Jay-Z will attend."
    return "Date not recognised."


async def main():
    storage = SQLiteBulletStorage("my_agent.db")
    embedder = OpenAIEmbedder()
    vector_index = FAISSVectorIndex(embedder.dimension(), "my_agent.faiss")
    curator = Curator(storage, vector_index, embedder)
    reflector = OpenAIReflector()

    agent = Agent(
        name="alphonse",
        model="gpt-4.1-mini",
        instructions="Handle scheduling questions using available tools.",
        tools=[plan_meeting],
    )

    ace_agent = ACEAgent(agent=agent, curator=curator, reflector=reflector)

    result = await ace_agent.run(
        "Plan tomorrow's meetings. Today is Oct 17, 2025. Try planning again with the new date."
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

Example output:

```
planning meeting for 2025-10-18
planning meeting for 2025-10-19
The meeting has been planned for tomorrow, October 19, 2025, and Jay-Z will be attending. If you need any more details or want to schedule additional meetings, please let me know!
```

---

## Project Status & Roadmap

- ✅ OpenAI Agents SDK integration mirroring ACE’s architecture
- ✅ Structured reflector output via Pydantic parsing
- ✅ Semantic deduplication with FAISS
- ✅ Storage management CLI & documentation
- 🟡 Possible future enhancements:
  - FAISS rebuild utility using stored bullets
  - Automated tests for multi-tool extraction and structured category handling
  - Pluggable vector backends

Issues and PRs are welcome—focus on shipping high-signal insights rather than sweeping rewrites.

---

## License

MIT © 2025 ACE contributors. See `LICENSE` for details.
