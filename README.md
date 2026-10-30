# KORZ Library Bot (Sania) 📚🤖

An autonomous Telegram bot for book-grounded dialogue, conceptual research, and semantic exploration over scientific and philosophical corpora (Alfred Korzybski, Jacque Fresco, Robert Sapolsky).

Built with local hybrid retrieval (**FTS5 lexical search + FastEmbed multilingual-e5-base dense embeddings**) and an anti-hallucination **Source Fidelity Protocol (v3.1)**.

---

## Key Features

- **Hybrid Retrieval (RRF@60):** Combines SQLite FTS5 BM25 keyword matching with dense vector search (`multilingual-e5-base`) via Reciprocal Rank Fusion, running entirely locally without external vector databases.
- **Chunk Expansion:** Automatically reconstructs surrounding paragraphs across page and section boundaries so the model never loses narrative context.
- **Dialogue Anaphora Resolution:** Resolves pronouns and contextual queries ("what about his second experiment?", "why did he disagree?") into structured semantic search queries.
- **Source Fidelity Protocol:** Enforces strict boundary verification (`[1]`, `[2]` page citations). If information is missing from the corpus, the bot states so explicitly rather than fabricating claims.
- **Zero-Downtime Local SQLite Store:** All document chunks, dense embeddings, and dialogue histories reside in self-contained SQLite databases.

---

## Architecture Overview

```
User Query (Telegram)
       │
       ▼
[LLM Dialogue Resolver] ──► Extracts core concepts & resolves context
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 [SQLite FTS5 (Lexical)]     [FastEmbed E5 (Dense Vectors)]
       │                                 │
       └────────────────┬────────────────┘
                        ▼
               [RRF Fusion Ranker]
                        ▼
             [Reading Chunk Expansion]
                        ▼
         [Answer Generator + Fidelity Check]
                        ▼
             Telegram Response with Citations
```

---

## Quick Start Guide (Zero to Running Bot)

Follow these steps to set up the repository, ingest your book library, compute vector embeddings, and start the Telegram bot.

### 1. Prerequisites

- **Python 3.10+** (64-bit recommended)
- **Telegram Bot Token** from [@BotFather](https://t.me/BotFather)
- **Gemini / OpenAI-compatible API key** for LLM answer generation

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/diredix/korz-library-bot.git
cd korz-library-bot
pip install -r requirements.txt
```

> **Core dependencies:** `fastembed`, `aiogram`, `aiohttp`, `pydantic`, `numpy`, `python-dotenv`.

### 3. Environment Configuration

Copy the example configuration file:

```bash
cp .env.example .env
```

Open `.env` in any text editor and configure:

```env
# Required Telegram Settings
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
ALLOWED_TELEGRAM_USERS="your_telegram_numeric_id"

# LLM Generation Provider
# By default uses Gemini API or an OpenAI-compatible gateway (e.g. Reformboss)
GEMINI_API_KEY="your_api_key_here"
KORZ_GEMINI_MODEL="gemini-3.8-flash-high"
KORZ_ANSWER_BACKEND="gemini"

# Pipeline Feature Flags
KORZ_ANSWER_PROGRESS_ENABLED=1
KORZ_LLM_RESOLVER_ENABLED=1
KORZ_SOURCE_FIDELITY_ENABLED=1
```

---

## Adding Books to the Library (Corpus Ingestion & Vectorization)

The bot requires books to be indexed in the local SQLite database (`data/korz_library.sqlite`). You can add any book in **PDF**, **TXT**, or **Markdown directory** format.

### Supported File Formats:
- `.pdf` — extracts pages directly using built-in PDF text extraction.
- `.txt` — plain text files (page separators like `=== Page 1 ===` or markdown page headers are supported).
- `.md` or a folder of markdown files — ideal for clean OCR scans where each file represents a page (e.g. `page_001.md`, `page_002.md`).

### Ingestion Script

You can import books and build vector embeddings in a few lines of Python:

```python
from korz_core import KorzLibrary

# 1. Initialize Library (opens/creates data/korz_library.sqlite)
lib = KorzLibrary()

# 2. Ingest Book
# Supported arguments: file path, book title, author, edition
print("Ingesting book...")
result = lib.add_book(
    path_input="path/to/your_book.pdf",  # or .txt, or folder of markdown pages
    title="A Primate's Memoir",
    author="Robert Sapolsky"
)
print("Ingestion result:", result)

# 3. Compute Dense Vector Embeddings (FastEmbed E5)
# This will vectorize all paragraphs/chunks that don't have embeddings yet
print("Building semantic index...")
embed_result = lib.build_semantic_index()
print("Vectorization complete:", embed_result)

# 4. Verify Library Status
status = lib.status()
print(f"Library ready: {status['documents_count']} books, {status['paragraphs_count']} paragraphs.")

lib.close()
```

### Verification CLI One-Liner

To quickly verify that the database is loaded and indexed:

```bash
python -c "from korz_core import KorzLibrary; lib = KorzLibrary(); print(lib.list_books()); print(lib.get_semantic_status()); lib.close()"
```

---

## Running the Bot

Once the environment is configured and at least one book is ingested:

```bash
python bot.py
```

Upon launch, the bot:
1. Validates the local SQLite library and runs schema migrations if needed.
2. Initializes the `korz_dialogue.sqlite` session database.
3. Warms up the `FastEmbed` dense embedding model (~1-3 seconds).
4. Begins long-polling Telegram for incoming queries from authorized users.

---

## How It Works in Conversation

1. **User asks:** *"Чому низький соціальний статус викликає стрес за Сапольскі?"*
2. **Dialogue Resolver:** Identifies author `Robert Sapolsky`, resolves pronouns, and builds an optimized semantic query.
3. **Hybrid Engine:**
   - FTS5 finds exact term matches in milliseconds.
   - FastEmbed finds semantically related paragraphs using cosine similarity over `multilingual-e5-base`.
   - RRF combines candidates into a unified rank.
4. **Chunk Expansion:** Fetches neighboring paragraphs to preserve full narrative context across pages.
5. **Source Fidelity:** Formulates the final response, strictly attaching exact page citations (`[1] p. 172`, `[2] p. 228`).

---

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
