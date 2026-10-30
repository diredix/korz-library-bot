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
       ├───────────────────────────────┐
       ▼                               ▼
 [SQLite FTS5 (Lexical)]     [FastEmbed E5 (Dense Vectors)]
       │                               │
       └───────────────┬───────────────┘
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

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Gemini / OpenAI-compatible API key for answer generation

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/diredix/korz-library-bot.git
cd korz-library-bot
pip install -r requirements.txt  # or install: fastembed aiohttp aiogram pydantic
```

### 3. Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
TELEGRAM_BOT_TOKEN="your_bot_token_from_botfather"
ALLOWED_TELEGRAM_USERS="your_telegram_numeric_id"
KORZ_GEMINI_MODEL=gemini-3.8-flash-high
KORZ_ANSWER_PROGRESS_ENABLED=1
KORZ_LLM_RESOLVER_ENABLED=1
KORZ_SOURCE_FIDELITY_ENABLED=1
```

### 4. Running the Bot

```bash
python bot.py
```

---

## Adding Your Own Books / Corpora

1. Convert your books (PDF/EPUB/TXT) into structured paragraphs with page metadata.
2. Initialize and index the documents into a local SQLite database (`korz_library.sqlite`) using `korz_corpus.py`.
3. Dense embeddings are generated automatically using FastEmbed on first run.

---

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
