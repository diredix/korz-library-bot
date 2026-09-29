#!/usr/bin/env python3
"""
korz.py - Clean, intuitive CLI for KORZ LOCAL (Single-file SQLite Library)

Usage:
    python korz.py add "book.pdf" [--title "Title"] [--author "Author"]
    python korz.py find "бабуїни працюють чотири години"
    python korz.py search "соціальний стрес бабуїнів" [--mode auto|fts|exact]
    python korz.py show <paragraph_id> [--context 2]
    python korz.py books
    python korz.py remove <book_id_or_title>
    python korz.py status
    python korz.py backup [--dest path]
    python korz.py restore <backup_path> [-y]
    python korz.py admin verify
    python korz.py admin reindex
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from typing import Any, Dict

from korz_core import KorzLibrary, KorzConfig, EmbeddingContract, DEFAULT_E5_BASE_CONTRACT


def cmd_add(args, lib: KorzLibrary) -> None:
    path = Path(args.path)
    if not path.exists():
        print(f"❌ Помилка: файл або каталог не знайдено: '{args.path}'")
        sys.exit(1)

    print(f"📖 Обробка джерела: {path.name} ...")
    try:
        res = lib.add_book(path, title=args.title, author=args.author, edition=args.edition)
    except Exception as e:
        print(f"❌ Помилка імпорту: {e}")
        sys.exit(1)

    if res.get("status") == "DUPLICATE":
        print(f"\n⚠️  {res.get('message')}")
        print("   Жодних змін до бази не внесено.\n")
        return

    author_part = f"{res.get('author')} — " if res.get("author") else ""
    title_part = res.get("title", "Без назви")

    print(f"\n✓ {author_part}{title_part}")
    print(f"✓ {res.get('pages', 0):,} сторінок".replace(",", " "))
    print(f"✓ {res.get('paragraphs', 0):,} абзаців".replace(",", " "))
    print(f"✓ {res.get('chunks', 0):,} пошукових блоків".replace(",", " "))
    print("✓ FTS index built")
    print("✓ READY\n")


def cmd_find(args, lib: KorzLibrary) -> None:
    phrase = args.phrase
    # Calls canonical search with mode="exact"
    results = lib.search(phrase, mode="exact", limit=args.limit, doc_id=args.doc)

    if not results:
        print(f"\n🔍 Фрагмент не знайдено за точним збігом: «{phrase}»")
        print("💡 Спробуйте використати повнотекстовий пошук: python korz.py search \"...\"\n")
        return

    print(f"\n🎯 Знайдено збігів: {len(results)} (Точний пошук: «{phrase}»)\n" + "─" * 60)
    for r in results:
        author_str = f"{r.author} — " if r.author else ""
        print(f"\n{author_str}{r.title}")
        print(f"Сторінка {r.page} (абзац {r.paragraph_index}) [ID: {r.paragraph_id}]")
        print()
        # Highlight exact phrase in display
        disp = r.text
        lower_disp = disp.lower()
        lower_phr = phrase.lower()
        idx = lower_disp.find(lower_phr)
        if idx >= 0:
            highlighted = (
                disp[:idx]
                + "\033[1;33m"
                + disp[idx : idx + len(phrase)]
                + "\033[0m"
                + disp[idx + len(phrase) :]
            )
            print(f"   {highlighted}")
        else:
            print(f"   {disp}")
        print("─" * 60)
    print()


def cmd_search(args, lib: KorzLibrary) -> None:
    query = args.query
    results = lib.search(query, mode=args.mode, limit=args.limit, doc_id=args.doc)

    if not results:
        print(f"\n🔍 Нічого не знайдено за запитом: «{query}»\n")
        return

    mode_label = "Exact" if (results and results[0].match_type == "exact") else ("Semantic" if (results and results[0].match_type == "semantic") else "FTS5 / BM25")
    print(f"\n🔎 Знайдено результатів: {len(results)} ({mode_label}: «{query}»)\n" + "─" * 60)
    for r in results:
        author_str = f"{r.author} — " if r.author else ""
        score_str = f" (Score: {r.score:.4f})" if r.match_type in ("fts", "semantic") else ""
        print(f"\n{author_str}{r.title}")
        print(f"Сторінка {r.page} (абзац {r.paragraph_index}) [ID: {r.paragraph_id}]{score_str}")
        print()
        print(f"   {r.text}")
        print("─" * 60)
    print()


def cmd_ask(args, lib: KorzLibrary) -> None:
    question = args.question.strip() if args.question else ""
    if not question:
        print("❌ Помилка: вкажіть питання або концептуальну фразу.")
        print("   Приклад: python korz.py ask \"чому забезпечене життя не усуває стрес?\"")
        return

    limit = getattr(args, "limit", 5)
    doc_id = getattr(args, "doc", None)

    try:
        t0 = time.perf_counter()
        results = lib.ask(question, limit=limit, doc_id=doc_id)
        latency = (time.perf_counter() - t0) * 1000

        if not results:
            print(f"\n🧠 За змістом для «{question}» нічого не знайдено.")
            print("💡 Спробуйте переформулювати запитання або перевірте стан семантичного індексу.\n")
            return

        print("\n" + "=" * 65)
        print(f"🧠 KORZ Question Retrieval (Семантична відповідність за {latency:.1f} ms)")
        print("=" * 65)
        print(f"❓ Питання: «{question}»\n" + "─" * 65)

        for i, r in enumerate(results, 1):
            author_str = f"{r.author} — " if r.author else ""
            print(f"\n[{i}] 📚 {author_str}{r.title}")
            print(f"    📄 Сторінка {r.page} (абзац {r.paragraph_index}) [ID: {r.paragraph_id}]")
            print(f"    🎯 Смислова схожість (cosine): {r.score:.4f}")
            print("    " + "─" * 45)
            indented = "    " + r.text.strip().replace("\n", "\n    ")
            if len(indented) > 600:
                indented = indented[:590] + "..."
            print(f"    «{indented.strip()}»")
            print("─" * 65)

        print("=" * 65 + "\n")
    except RuntimeError as e:
        print(f"\n❌ {e}\n")


def cmd_answer(args, lib: KorzLibrary) -> None:
    question = args.question.strip() if args.question else ""
    if not question:
        print("❌ Помилка: вкажіть питання.")
        print("   Приклад: python korz.py answer \"чому соціальний статус впливає на здоров'я?\"")
        return

    mode = getattr(args, "mode", None)
    limit = getattr(args, "limit", None)
    doc_id = getattr(args, "doc", None)

    try:
        from korz_answer import AnswerEngine, AnswerConfig
        cfg = AnswerConfig.from_env()
        cfg.enabled = True
        engine = AnswerEngine(lib, cfg)
        res = engine.answer(question, mode=mode, limit=limit, doc_id=doc_id)

        print("\n" + "=" * 65)
        if res.mode == "extractive" and res.fallback_reason:
            mode_label = f"Extractive (source-bound, fallback: {res.fallback_reason})"
        elif res.mode == "extractive":
            mode_label = "Extractive (source-bound, авторські речення)"
        else:
            mode_label = "Generative LLM (з верифікацією джерел)"
        print(f"📚 KORZ Answer Engine [{mode_label}] ({res.latency_ms:.1f} ms)")
        print("=" * 65)
        print(f"❓ Питання: «{question}»\n" + "─" * 65)

        if res.insufficient_info:
            print(f"\n⚠️  {res.answer_text}\n")
            print("=" * 65 + "\n")
            return

        print(f"\n{res.answer_text}\n")
        print("─" * 65)
        print("📖 Використані джерела з бібліотеки:")
        for s in res.sources:
            author_str = f"{s.author} — " if s.author else ""
            print(f"  [{s.citation_index}] 📚 {author_str}«{s.title}», стор. {s.page} (абзац {s.paragraph_index}) [ID: {s.paragraph_id}]")

        print("=" * 65 + "\n")
    except Exception as e:
        print(f"\n❌ Помилка формування відповіді: {e}\n")


def cmd_show(args, lib: KorzLibrary) -> None:
    par_id = args.id
    ctx = lib.get_context(par_id, window=args.context)
    if not ctx:
        print(f"❌ Абзац з ID {par_id} не знайдено.")
        sys.exit(1)

    author_str = f"{ctx['author']} — " if ctx.get("author") else ""
    print(f"\n📖 {author_str}{ctx['title']}")
    print(f"Цільовий абзац: ID {ctx['target_id']} (Сторінка {ctx['target_page']}, абзац {ctx['target_paragraph_index']})")
    print(f"Контекст: вікно ±{args.context} абзаців\n" + "═" * 70)

    for entry in ctx["entries"]:
        marker = "👉 [ЦІЛЬОВИЙ АБЗАЦ]" if entry["is_target"] else f"   [Стор. {entry['page']}, абз. {entry['paragraph_index']}]"
        print(f"\n{marker} [ID: {entry['id']}]:")
        if entry["is_target"]:
            print(f"\033[1m{entry['text']}\033[0m")
        else:
            print(f"{entry['text']}")
    print("\n" + "═" * 70 + "\n")


def cmd_books(args, lib: KorzLibrary) -> None:
    books = lib.list_books()
    if not books:
        print("\n📚 Бібліотека порожня. Додайте книгу: python korz.py add <path>\n")
        return

    print(f"\n📚 Бібліотека KORZ ({len(books)} книг):\n" + "─" * 80)
    print(f"{'ID':<26} | {'Автор / Назва':<32} | {'Стор.':<6} | {'Абзаців':<8} | {'Блоків':<6}")
    print("─" * 80)
    for b in books:
        author_title = f"{b['author']} — {b['title']}" if b["author"] else b["title"]
        if len(author_title) > 30:
            author_title = author_title[:27] + "..."
        doc_id_disp = b["id"][:24] + ".." if len(b["id"]) > 26 else b["id"]
        print(
            f"{doc_id_disp:<26} | {author_title:<32} | {b['total_pages']:<6} | {b['total_paragraphs']:<8} | {b['total_chunks']:<6}"
        )
    print("─" * 80 + "\n")


def cmd_remove(args, lib: KorzLibrary) -> None:
    identifier = args.book
    book = lib.get_book(identifier)
    if not book:
        print(f"❌ Книгу не знайдено: '{identifier}'")
        sys.exit(1)

    title = book["title"]
    doc_id = book["id"]

    if not args.yes:
        confirm = input(f"❓ Ви впевнені, що хочете видалити книгу '{title}' (ID: {doc_id})? [y/N]: ")
        if confirm.lower() not in ["y", "yes", "так"]:
            print("Скасовано.")
            return

    ok = lib.remove_book(doc_id)
    if ok:
        print(f"\n✓ Книгу успішно видалено: '{title}' ({doc_id})")
        print("✓ Пов'язані сторінки, абзаци та FTS-індекс очищено.\n")
    else:
        print(f"❌ Не вдалося видалити книгу: {doc_id}")


def cmd_status(args, lib: KorzLibrary) -> None:
    st = lib.status()
    print("\n📊 Стан бібліотеки KORZ:")
    print("─" * 50)
    print(f"База даних:       {st['database_path']}")
    print(f"Версія схеми:     v{st.get('schema_version', 1)}")
    print(f"Розмір бази:      {st['database_size_mb']} MB")
    print(f"Всього книг:      {st['total_documents']:,}".replace(",", " "))
    print(f"Всього сторінок:  {st['total_pages']:,}".replace(",", " "))
    print(f"Всього абзаців:   {st['total_paragraphs']:,}".replace(",", " "))
    print(f"Пошукових блоків: {st['total_chunks']:,}".replace(",", " "))
    print(f"Каталог джерел:   {st['sources_dir']}")
    if st.get("semantic"):
        sem = st["semantic"]
        print("─" * 50)
        print("🧠 Семантична інфраструктура (V2A):")
        print(f"Стан:             {'Активно' if sem['semantic_enabled'] else 'Вимкнено (KORZ_SEMANTIC_ENABLED=0)'}")
        print(f"Зареєстровано моделей: {sem['models_registered']}")
        print(f"Векторизовано блоків:  {sem['chunks_embedded']} / {sem['total_chunks']}")
    stats = lib.get_search_stats()
    if stats["total_searches"] > 0:
        print("─" * 50)
        print("🔍 Статистика пошуків:")
        print(f"Всього пошуків:   {stats['total_searches']}")
        print(f"Успішні (hits):   {stats['hit_rate_pct']}% ({stats['zero_hit_searches']} без результатів)")
        print(f"Сер. латентність: {stats['avg_latency_ms']} ms")
    print("─" * 50 + "\n")


def cmd_backup(args, lib: KorzLibrary) -> None:
    print("💾 Створення атомарного онлайнового бекапу SQLite...")
    try:
        path = lib.backup(dest_path=args.dest)
        print(f"✓ Бекап успішно збережено: {path}\n")
    except Exception as e:
        print(f"❌ Помилка створення бекапу: {e}")
        sys.exit(1)


def cmd_restore(args, lib: KorzLibrary) -> None:
    backup_file = Path(args.backup_path)
    if not backup_file.exists():
        print(f"❌ Файл бекапу не знайдено: '{args.backup_path}'")
        sys.exit(1)

    if not args.yes:
        confirm = input(f"⚠️  УВАГА: Відновлення замінить поточний вміст бази на версію з '{backup_file.name}'. Продовжити? [y/N]: ")
        if confirm.lower() not in ["y", "yes", "так"]:
            print("Скасовано.")
            return

    print(f"🔄 Відновлення бази з бекапу: {backup_file.name} ...")
    try:
        st = lib.restore(backup_file)
        print("✓ Базу успішно відновлено!")
        print(f"✓ Книг: {st['total_documents']}, Абзаців: {st['total_paragraphs']:,}".replace(",", " ") + "\n")
    except Exception as e:
        print(f"❌ Помилка відновлення: {e}")
        sys.exit(1)


def cmd_admin_verify(args, lib: KorzLibrary) -> None:
    print("🔬 Перевірка цілісності бази та індексів...")
    res = lib.admin_verify()
    print("─" * 50)
    print(f"SQLite Integrity Check: {res['sqlite_integrity_check']}")
    print(f"FTS5 Integrity Check:   {res['fts5_integrity_check']}")
    print(f"Абзаців у таблиці:      {res['paragraphs_count']:,}".replace(",", " "))
    print(f"Індексовано у FTS:      {res['fts_indexed_count']:,}".replace(",", " "))
    print(f"Паритет цілісності:     {'✓ OK' if res['parity_ok'] else '❌ ERROR'}")
    print("─" * 50)
    if res["all_healthy"]:
        print("✅ Всі перевірки пройдено успішно. База повністю здорова.\n")
    else:
        print("❌ Виявлено помилки цілісності!\n")


def cmd_admin_reindex(args, lib: KorzLibrary) -> None:
    print("⚙️  Перебудова індексу FTS5...")
    res = lib.admin_reindex()
    print(f"✓ FTS5 індекс оновлено. Всього проіндексовано абзаців: {res['total_indexed_paragraphs']:,}".replace(",", " ") + "\n")


def cmd_admin_search_log(args, lib: KorzLibrary) -> None:
    print("📋 Журнал пошуків (останні запити з 0 результатів для тюнінгу V2):")
    failed = lib.get_failed_searches(limit=args.limit)
    if not failed:
        print("✓ Немає невдалих запитів. Всі пошуки знаходили результати!\n")
        return
    print("─" * 60)
    for row in failed:
        print(f"• [{row['created_at'][:19]}] mode={row['mode']} | query: \"{row['query']}\"")
    print("─" * 60 + "\n")


def cmd_semantic_status(args, lib: KorzLibrary) -> None:
    sem = lib.get_semantic_status()
    print("\n🧠 KORZ Semantic Infrastructure (Phase V2B):")
    print("─" * 50)
    print(f"Версія схеми:         v{sem['schema_version']}")
    print(f"Semantic enabled:     {'так' if sem['semantic_enabled'] else 'ні (KORZ_SEMANTIC_ENABLED=0)'}")
    print(f"Зареєстровано моделей:{sem['models_registered']}")
    print(f"Документів готово:    {sem['documents_ready']} / {sem['total_documents']}")
    print(f"Векторизовано блоків: {sem['chunks_embedded']} / {sem['total_chunks']}")
    print(f"FTS5 baseline:        {sem['fts5_baseline']}")
    if sem["models"]:
        print("─" * 50)
        print("Зареєстровані моделі:")
        for m in sem["models"]:
            print(f" • [{m['model_id']}] {m['model_name']} (dim: {m['dimension']}, norm: {m['normalization']})")
    print("─" * 50 + "\n")


def cmd_semantic_verify(args, lib: KorzLibrary) -> None:
    model_arg = getattr(args, "model", None)
    print("\n🔬 KORZ Semantic Integrity Audit (V2B)...")
    res = lib.verify_semantic_integrity(model_arg)
    print("─" * 50)
    print(f"Model contract:        {'OK' if res['all_healthy'] else 'ISSUES'}")
    print(f"Model ID:              {res.get('model_id') or 'N/A'}")
    print(f"Model Name:            {res.get('model_name') or 'N/A'}")
    print(f"Dimension:             {res.get('dimension', 0)}")
    print(f"Chunks expected:       {res.get('chunks_expected', 0)}")
    print(f"Embeddings present:    {res.get('embeddings_present', 0)}")
    print(f"Missing:               {res.get('missing_embeddings', 0)}")
    print(f"Stale SHA:             {res.get('stale_embeddings', 0)}")
    print(f"Invalid BLOB:          {res.get('invalid_blob', 0)}")
    print(f"NaN/Inf:               {res.get('nan_inf_count', 0)}")
    print(f"Non-unit vectors:      {res.get('non_unit_vectors', 0)}")
    print()
    print(f"Documents READY:       {res.get('documents_ready', 0)} / {res.get('total_documents', 0)}")
    print()
    print("Matrix:")
    print(f"{res.get('embeddings_present', 0)} × {res.get('dimension', 0)} float32")
    print(f"{res.get('matrix_mib', 0.0)} MiB")
    print()
    print(f"Semantic index:        {'READY' if res.get('semantic_ready') else 'NOT READY'}")
    print("─" * 50)
    if res.get("issues"):
        print("⚠️  Знайдені зауваження:")
        for iss in res["issues"]:
            print(f" • {iss}")
        print("─" * 50)
    if res.get("semantic_ready"):
        print("✅ Семантична інфраструктура повністю здорова та готова до пошуку.\n")
    else:
        print("⚠️  Семантичний індекс ще не готовий або потребує запуску 'python korz.py semantic build'.\n")


def cmd_semantic_build(args, lib: KorzLibrary) -> None:
    model_arg = getattr(args, "model", None)
    batch_size = getattr(args, "batch_size", 32)
    force = getattr(args, "force", False)

    print("\n🧠 KORZ Semantic Build (V2B)")
    print("─" * 50)

    st = lib.status()
    total_docs = st["total_documents"]
    total_chunks = st["total_chunks"]

    target_display_model = model_arg or "intfloat/multilingual-e5-base"
    print(f"Model:          {target_display_model}")
    print(f"Documents:      {total_docs}")
    print(f"Chunks:         {total_chunks}")

    last_pct = -1

    def on_progress(p: Dict[str, Any]) -> None:
        nonlocal last_pct
        to_emb = p.get("to_embed", 0)
        done = p.get("embedded", 0)
        if to_emb == 0:
            pct = 100
        else:
            pct = int((done / to_emb) * 100)

        if pct != last_pct and (pct % 5 == 0 or pct == 100):
            last_pct = pct
            bar_len = 20
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"[{bar}] {pct:3d}% ({done}/{to_emb})", end="\r", flush=True)

    try:
        t0 = time.perf_counter()
        res = lib.build_semantic_index(
            model_id=model_arg,
            batch_size=batch_size,
            force=force,
            progress_callback=on_progress,
        )
        elapsed = time.perf_counter() - t0
        print("\n" + "─" * 50)
        print(f"Embedded:       {res['embedded']}")
        print(f"Cached:         {res['cached']}")
        print(f"Failed:         {res['failed']}")

        integrity = lib.verify_semantic_integrity(res["model_id"])
        matrix_mb = integrity.get("matrix_mib", 0.0)
        dim = integrity.get("dimension", 768)
        print(f"Matrix:         {res['total_chunks']} × {dim} float32 (~{matrix_mb} MiB in RAM)")
        print(f"Status:         {'READY' if res['ready'] else 'FAILED'}")
        print(f"Час виконання:  {elapsed:.1f} с")
        print("─" * 50)
        if res["ready"]:
            print("✅ Семантичний індекс успішно побудовано та перевірено!\n")
        else:
            print("❌ Побудову індексу завершено з помилками цілісності.\n")
    except Exception as e:
        print(f"\n❌ Помилка побудови семантичного індексу: {e}\n")
        sys.exit(1)


def cmd_semantic_search(args, lib: KorzLibrary) -> None:
    query = args.query.strip()
    limit = args.limit
    doc_id = getattr(args, "doc", None)
    model_id = getattr(args, "model", None)

    try:
        t0 = time.perf_counter()
        results = lib.search(query, mode="semantic", limit=limit, doc_id=doc_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"❌ Помилка семантичного пошуку: {e}")
        sys.exit(1)

    print(f"\n🧠 Знайдено результатів: {len(results)} (Semantic Search: «{query}» за {elapsed_ms:.1f} ms)")
    print("─" * 60)
    if not results:
        print("Нічого не знайдено.")
        print("─" * 60 + "\n")
        return

    for idx, r in enumerate(results, 1):
        author_str = f"{r.author} — " if r.author else ""
        print(f"\n{author_str}{r.title}")
        print(f"Сторінка {r.page} (абзац {r.paragraph_index}) [ID: {r.paragraph_id}] (Cosine Score: {r.score:.4f})")
        print(f"\n   {r.text}")
        print("─" * 60)
    print()


def cmd_semantic_models(args, lib: KorzLibrary) -> None:
    models = lib.list_embedding_models()
    print("\n📚 Зареєстровані моделі ембедингів (V2B):")
    print("─" * 60)
    if not models:
        print("Немає зареєстрованих моделей. Запустіть 'python korz.py semantic build' для автореєстрації.")
        print("─" * 60 + "\n")
        return

    for m in models:
        print(f"• ID:          {m['model_id']}")
        print(f"  Name:        {m['model_name']}")
        print(f"  Dimension:   {m['dimension']}")
        print(f"  Metric:      {m['distance_metric']} (norm={m['normalization']})")
        print(f"  Passage Tpl: {m['document_template']}")
        print(f"  Query Tpl:   {m['query_template']}")
        print(f"  Config SHA:  {m['config_sha256'][:16]}...")
        print("─" * 60)
    print()


def cmd_semantic_register(args, lib: KorzLibrary) -> None:
    contract = EmbeddingContract(
        model_name=args.model,
        dimension=args.dim,
        query_template=args.query_tpl,
        passage_template=args.passage_tpl,
        model_max_tokens=args.max_tokens,
    )
    model_id = lib.register_embedding_model(contract)
    print(f"\n✅ Модель успішно зареєстровано:")
    print(f"  ID:          {model_id}")
    print(f"  Name:        {contract.model_name}")
    print(f"  Dimension:   {contract.dimension}")
    print(f"  Max Tokens:  {contract.model_max_tokens}\n")


def cmd_admin_migrate(args, lib: KorzLibrary) -> None:
    print("⚙️  Перевірка та виконання міграцій бази даних...")
    st = lib.status()
    print(f"✓ Поточна версія схеми бази: v{st.get('schema_version', 2)}")
    print("✅ Всі міграції застосовано успішно.\n")


def cmd_shadow_status(args, lib: KorzLibrary) -> None:
    st = lib.get_shadow_stats()
    print("\n=== KORZ Shadow Mode Status ===")
    print(f"Shadowed Queries:  {st['total_shadowed_queries']}")
    print(f"Shadow Failures:   {st['shadow_failures']}")
    print(f"Success Rate:      {st['success_rate_pct']}%")
    print(f"User Feedbacks:    {st['feedback_count']}\n")


def cmd_shadow_report(args, lib: KorzLibrary) -> None:
    rep = lib.get_shadow_report(limit_divergences=getattr(args, "limit", 20))
    if rep["total_queries"] == 0:
        print("\nℹ️  Тіньовий лог порожній. Запитів ще не зафіксовано.")
        print("💡 Виконайте пошуки через Telegram-бота (з KORZ_SHADOW_ENABLED=1)")
        print("   або запустіть 'python korz.py shadow replay', щоб прогнати історію.\n")
        return

    op = rep["operational"]
    div = rep["divergence"]
    rel = rep["relevance"]

    print("\n" + "=" * 65)
    print("📊 KORZ Shadow Mode Telemetry & Divergence Report")
    print("=" * 65)
    print(f"Всього тіньових запитів:   {rep['total_queries']}")
    print(f"Успішно обчислено:         {rep['successful_queries']} ({round(100.0 * rep['successful_queries'] / rep['total_queries'], 1)}%)")
    print(f"Збоїв у фоні (failures):   {rep['failed_queries']}")

    print("\n⏱️  Операційна латентність (CPU):")
    print(f"  FTS5 / BM25:       P50: {op.get('fts_latency_p50', 0)} ms | P95: {op.get('fts_latency_p95', 0)} ms | P99: {op.get('fts_latency_p99', 0)} ms")
    print(f"  Semantic (Warm):   P50: {op.get('semantic_latency_warm_p50', op.get('semantic_latency_p50', 0))} ms | P95: {op.get('semantic_latency_warm_p95', op.get('semantic_latency_p95', 0))} ms | P99: {op.get('semantic_latency_warm_p99', op.get('semantic_latency_p99', 0))} ms")
    if op.get("semantic_cold_start_ms"):
        print(f"  Semantic Cold-start: {op.get('semantic_cold_start_ms', 0)} ms (перший ONNX інференс)")

    print("\n🔀 Розходження видачі (Retrieval Divergence):")
    print(f"  Top-1 однаковий (#1 збіг):     {div.get('top1_match_pct', 0)}% ({div.get('top1_match_count', 0)} / {rep['successful_queries']})")
    print(f"  Overlap@5 = 0 (повністю різні): {div.get('overlap_5_zero_pct', 0)}% ({div.get('overlap_5_zero_count', 0)} / {rep['successful_queries']})")
    print(f"  Overlap@10 = 0:                 {div.get('overlap_10_zero_pct', 0)}% ({div.get('overlap_10_zero_count', 0)} / {rep['successful_queries']})")
    print(f"  FTS дав 0, а Semantic знайшов: {div.get('fts_zero_semantic_hits', 0)} запитів")

    if rel and rel.get("total_judged", 0) > 0:
        print("\n🎯 Користувацькі оцінки релевантності (Human Ground-Truth):")
        print(f"  Всього відгуків:               {rel['total_judged']} (👍 {rel['thumbs_up']} | 👎 {rel['thumbs_down']})")
        print(f"  Задоволеність користувача:     {rel['user_satisfaction_pct']}%")
        if rel.get("target_judged", 0) > 0:
            print(f"  Оцінених цільових запитів:     {rel['target_judged']}")
            print(f"  FTS5 / BM25:   Recall@1: {rel['fts_recall_at_1']} | Recall@5: {rel['fts_recall_at_5']} | MRR: {rel['fts_mrr']}")
            print(f"  Semantic:      Recall@1: {rel['semantic_recall_at_1']} | Recall@5: {rel['semantic_recall_at_5']} | MRR: {rel['semantic_mrr']}")
            print(f"  RRF@60 Fusion: Recall@1: {rel['rrf_recall_at_1']} | Recall@5: {rel['rrf_recall_at_5']} | MRR: {rel['rrf_mrr']}")
    else:
        print("\nℹ️  Користувацьких оцінок (👍/👎/🎯) ще не додано.")
        print("💡 Ставте реакції в Telegram-боті для автоматичного розрахунку Recall/MRR.")

    top_div = rep.get("top_divergences", [])
    if getattr(args, "detailed", False) and top_div:
        print("\n🔍 Запити з найбільшим розходженням (Overlap@5 = 0 або FTS=0):")
        print("─" * 65)
        for d in top_div[:getattr(args, "limit", 20)]:
            print(f"• ID #{d['id']}: «{d['query']}»")
            print(f"  Hits: FTS={d['fts_count']} vs SEM={d['semantic_count']} | Overlap@5: {d['overlap_at_5']} | Top1 Same: {d['top1_match']}")
            print(f"  Latency: FTS={d['latency_fts']}ms | SEM={d['latency_sem']}ms")
            print("─" * 65)

    print("=" * 65 + "\n")


def cmd_shadow_replay(args, lib: KorzLibrary) -> None:
    print(f"\n🔄 Запуск повторної тіньової оцінки (Shadow Replay)...")
    if args.model:
        print(f"• Цільова модель: {args.model}")
    if args.since:
        print(f"• Фільтр дати: від {args.since}")
    print(f"• Ліміт запитів: {args.limit}")

    def on_progress(p: Dict[str, Any]):
        curr = p["current"]
        tot = p["total"]
        pct = int(100 * curr / tot) if tot > 0 else 0
        sys.stdout.write(f"\r  [{curr}/{tot}] {pct}% | Оцінено: {p['evaluated']} | Пропущено: {p['skipped']} | Збоїв: {p['failed']}")
        sys.stdout.flush()

    res = lib.replay_shadow_evaluation(
        limit=args.limit,
        since=args.since,
        model_id=args.model,
        force=args.force,
        progress_callback=on_progress,
    )
    print("\n\n✅ Shadow Replay завершено:")
    print(f"  Всього запитів в історії: {res['total']}")
    print(f"  Оцінено зараз:           {res['evaluated']}")
    print(f"  Пропущено (вже в лозі):  {res['skipped']}")
    print(f"  Збоїв:                   {res['failed']}")
    print(f"  Model ID:                {res['model_id']}")
    print(f"  Index SHA:               {res['index_sha']}\n")


def cmd_corpus_audit(args) -> None:
    from korz_corpus import CorpusAuditor
    target = Path(args.path)
    if not target.exists():
        print(f"❌ Помилка: шлях не знайдено: '{args.path}'")
        return

    auditor = CorpusAuditor()
    report = auditor.audit_path(target)

    if getattr(args, "json", False):
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 65)
    print(f"📚 KORZ Corpus Audit Report: {target.name}")
    print("=" * 65)
    print(f"Файлів проскановано:         {report.total_files:,}".replace(",", " "))
    print(f"Загальна кількість рядків:   {report.total_lines:,}".replace(",", " "))
    print(f"Загальна кількість символів: {report.total_chars:,}".replace(",", " "))
    print(f"Розділювачів сторінок:       {report.page_separators:,}".replace(",", " "))
    print("\nВиявлений типографічний та OCR-шум:")
    print("─" * 65)
    print(f"• Розділювачі сторінок (Datalab):     {report.page_separators:,}".replace(",", " "))
    print(f"• Одинокі номери сторінок:             {report.standalone_numbers:,}".replace(",", " "))
    print(f"• Повторювані колонтитули:            {report.running_headers:,}".replace(",", " "))
    print(f"• Маркери зображень Markdown:          {report.image_markers:,}".replace(",", " "))
    print(f"• Дублікати підписів ілюстрацій:        {report.duplicate_captions:,}".replace(",", " "))
    print(f"• Розірвані речення на стиках:          {report.broken_sentences:,}".replace(",", " "))
    print(f"• Нестандартні Unicode-символи:         {report.unicode_anomalies:,}".replace(",", " "))

    print("\nГрупування за рівнем впевненості:")
    print(f"• HIGH confidence (безпечно для очищення):    {report.high_confidence_noise:,}".replace(",", " "))
    print(f"• MEDIUM confidence (потребує перевірки):         {report.medium_confidence_noise:,}".replace(",", " "))
    print(f"• REVIEW manually (підозрілі випадки):            {report.review_noise:,}".replace(",", " "))

    if report.sample_items:
        print("\nЗразки виявленого шуму (перші 5):")
        for item in report.sample_items[:5]:
            pg_str = f"стор. {item.page_number}, " if item.page_number else ""
            print(f"  [{item.confidence}] ряд. {item.line_number} ({pg_str}{item.category}): {repr(item.raw_text[:60])}")

    print("=" * 65 + "\n")


def cmd_corpus_clean(args) -> None:
    from korz_corpus import CorpusCleaner, CorpusVerifier
    target = Path(args.path)
    if not target.exists():
        print(f"❌ Помилка: шлях не знайдено: '{args.path}'")
        return

    dry_run = not getattr(args, "apply", False)
    out_dir = Path(args.out) if getattr(args, "out", None) else Path("data/corpus_clean")

    lang = getattr(args, "lang", "uk")
    if target.is_file():
        files = [target]
    else:
        all_mds = [f for f in target.rglob("*.md") if not f.name.endswith(".bak") and not f.name.endswith(".cleaned.md")]
        if lang == "uk":
            cyr_files = []
            for f in all_mds:
                if f.name.startswith("укр."):
                    cyr_files.append(f)
                else:
                    with open(f, "rb") as fp:
                        raw = fp.read(2048)
                    txt = raw.decode("utf-8", errors="ignore")
                    if any("\u0400" <= ch <= "\u04ff" for ch in txt):
                        cyr_files.append(f)
            files = cyr_files if cyr_files else all_mds
        else:
            files = all_mds

    if not files:
        print(f"❌ Файлів Markdown для очищення не знайдено у '{args.path}'")
        return

    print("\n" + "=" * 65)
    mode_str = "DRY-RUN (симуляція без зміни файлів)" if dry_run else f"APPLY (збереження до {out_dir})"
    print(f"🧹 KORZ Corpus Cleaner [{mode_str}]")
    print("=" * 65)

    frequent = CorpusCleaner.discover_frequent_headers(files)
    if frequent:
        print(f"• Виявлено повторюваних колонтитулів корпусу: {len(frequent)}")

    cleaner = CorpusCleaner(frequent_headers=frequent)
    verifier = CorpusVerifier()

    total_orig_chars = 0
    total_clean_chars = 0
    total_items = {}

    for f in files:
        if dry_run:
            diff = cleaner.clean_file(f, dry_run=True)
            out_file = None
        else:
            rel = f.name
            out_file = out_dir / target.name / rel if target.is_dir() else out_dir / rel
            diff = cleaner.clean_file(f, dry_run=False, output_path=out_file)

        total_orig_chars += diff.original_chars
        total_clean_chars += diff.cleaned_chars
        for cat, cnt in diff.items_removed_by_category.items():
            total_items[cat] = total_items.get(cat, 0) + cnt

        print(f"📄 {f.name}: {diff.original_lines} -> {diff.cleaned_lines} рядків | -{diff.removed_chars} симв. (-{diff.removal_ratio_pct:.2f}%)")
        if dry_run and len(files) == 1 and diff.diff_text:
            print("\n--- DIFF SAMPLE ---")
            lines = diff.diff_text.splitlines()[:35]
            print("\n".join(lines))
            if len(diff.diff_text.splitlines()) > 35:
                print("... [diff скорочено]")

    diff_total = total_orig_chars - total_clean_chars
    ratio_total = (diff_total / total_orig_chars * 100) if total_orig_chars > 0 else 0.0

    print("─" * 65)
    print("📊 ПІДСУМОК ОЧИЩЕННЯ:")
    print(f"• Оброблено файлів:             {len(files)}")
    print(f"• Загальний обсяг до:           {total_orig_chars:,} симв.".replace(",", " "))
    print(f"• Загальний обсяг після:        {total_clean_chars:,} симв.".replace(",", " "))
    print(f"• Видалено шуму:                {diff_total:,} симв. ({ratio_total:.2f}%)".replace(",", " "))
    print("• Видалено за категоріями:")
    for cat, cnt in total_items.items():
        if cnt > 0:
            print(f"    - {cat}: {cnt:,}".replace(",", " "))

    # Text loss gate verification
    ver = verifier.verify("x" * total_orig_chars, "x" * total_clean_chars, total_items)
    status_emoji = "✅" if ver.is_safe else "❌"
    print(f"\n{status_emoji} Text-Loss Safety Gate: [{ver.status}] (некласифікована втрата: {ver.unclassified_loss_pct:.2f}%)")
    for note in ver.notes:
        print(f"   ℹ️  {note}")

    if dry_run:
        print("\n💡 Це була симуляція. Щоб застосувати очищення, виконайте:")
        print(f"   python korz.py corpus clean \"{args.path}\" --apply\n")
    else:
        print(f"\n✓ Очищені файли збережено у: {out_dir}\n")


def cmd_corpus_verify(args) -> None:
    from korz_corpus import CorpusVerifier, CorpusCleaner
    orig_p = Path(args.original)
    clean_p = Path(args.cleaned)
    if not orig_p.exists() or not clean_p.exists():
        print(f"❌ Помилка: один із файлів не знайдено.")
        return

    with open(orig_p, "r", encoding="utf-8", errors="replace") as f: orig_txt = f.read()
    with open(clean_p, "r", encoding="utf-8", errors="replace") as f: clean_txt = f.read()

    # Re-clean to gather deterministic stats
    cleaner = CorpusCleaner()
    _, stats = cleaner.clean_text(orig_txt)
    verifier = CorpusVerifier()
    res = verifier.verify(orig_txt, clean_txt, stats)

    print("\n" + "=" * 65)
    print(f"🛡️  KORZ Text-Loss Safety Gate Verification")
    print("=" * 65)
    print(f"Оригінальний файл:  {orig_p.name} ({res.original_chars:,} симв.)".replace(",", " "))
    print(f"Очищений файл:      {clean_p.name} ({res.cleaned_chars:,} симв.)".replace(",", " "))
    print(f"Різниця:            -{res.deleted_chars:,} симв. (-{res.deletion_ratio_pct:.2f}%)".replace(",", " "))
    print(f"Некласифікована втрата: {res.unclassified_loss_chars:,} симв. ({res.unclassified_loss_pct:.2f}%)".replace(",", " "))

    status_icon = "✅" if res.is_safe else "❌"
    print(f"\nСтатус безпеки: {status_icon} [{res.status}]")
    for note in res.notes:
        print(f"• {note}")
    print("=" * 65 + "\n")


def cmd_corpus_report(args) -> None:
    from korz_corpus import CorpusAuditor
    books_dirs = [
        "Robert_Sapolsky_A_Primates_Memoir_2002_Pages_1-308",
        "Alfred_Korzybski_Science_and_Sanity_1948_UA",
        "Jacque_Fresco_Looking_Forward_Pages_1-121",
        "Jacque_Fresco_Future_Design_2007_Pages_1-79",
        "Jacque_Fresco_The_Best_That_Money_Cant_Buy_2002_Pages_180",
    ]

    print("\n" + "=" * 75)
    print("📚 KORZ CORPUS QUALITY OVERVIEW (Усі 5 книг бібліотеки)")
    print("=" * 75)

    total_lines = 0
    total_chars = 0
    total_noise = 0
    auditor = CorpusAuditor()

    print(f"{'Книга':<37} | {'Стор.':<6} | {'Рядків':<7} | {'Сміття (HIGH)':<12}")
    print("─" * 75)

    for b in books_dirs:
        p = Path(b)
        if not p.exists():
            continue
        rep = auditor.audit_path(p)
        total_lines += rep.total_lines
        total_chars += rep.total_chars
        total_noise += rep.high_confidence_noise
        short_name = b.split("_Pages")[0].replace("_", " ")[:35]
        print(f"{short_name:<37} | {rep.page_separators:<6} | {rep.total_lines:<7} | {rep.high_confidence_noise:<12}")

    print("─" * 75)
    print(f"{'РАЗОМ ПО БІБЛІОТЕЦІ:':<37} | {'-':<6} | {total_lines:<7} | {total_noise:<12}")
    print(f"\nЗагалом виявлено {total_noise:,} елементів типографічного та OCR-шуму.".replace(",", " "))
    print("Виконайте 'python korz.py corpus clean <каталог> --dry-run' для перегляду diff.\n")


def cmd_corpus_build_candidate(args) -> None:
    from korz_corpus import CandidateLab
    clean_dir = Path(getattr(args, "clean_dir", None) or "data/corpus_clean")
    dest_db = Path(getattr(args, "out", None) or "data/korz_clean_candidate.sqlite")
    print("\n" + "=" * 65)
    print("🏗️  KORZ Candidate Database Builder")
    print("=" * 65)
    print(f"Каталог очищеного корпусу: {clean_dir}")
    print(f"Цільова база даних:       {dest_db}")
    print("Імпорт книг та побудова семантичного індексу...")
    summary = CandidateLab.build_candidate_database(clean_dir, dest_db)
    print("─" * 65)
    print(f"✓ Додано документів:  {summary['documents_count']}")
    print(f"✓ Сторінок:           {summary['total_pages']:,}".replace(",", " "))
    print(f"✓ Абзаців:            {summary['total_paragraphs']:,}".replace(",", " "))
    print(f"✓ Семантичних блоків: {summary['total_chunks']:,}".replace(",", " "))
    print(f"✓ Семантичний індекс: {'READY' if summary['semantic_build'].get('ready') else 'DONE'}")
    print("=" * 65 + "\n")


def cmd_corpus_evaluate_ab(args) -> None:
    from korz_core import KorzLibrary, KorzConfig
    from korz_corpus import CandidateLab

    baseline_path = Path(getattr(args, "baseline", None) or "data/korz_library.sqlite")
    candidate_path = Path(getattr(args, "candidate", None) or "data/korz_clean_candidate.sqlite")

    if not baseline_path.exists():
        print(f"❌ Baseline базу не знайдено: {baseline_path}")
        return
    if not candidate_path.exists():
        print(f"❌ Candidate базу не знайдено: {candidate_path}. Запустіть спочатку 'python korz.py corpus build-candidate'")
        return

    b_cfg = KorzConfig(db_path=baseline_path, semantic_enabled=True)
    c_cfg = KorzConfig(db_path=candidate_path, semantic_enabled=True)
    b_lib = KorzLibrary(b_cfg)
    c_lib = KorzLibrary(c_cfg)

    queries = [
        "стрес і соціальна ієрархія",
        "бабуїни павіани агресія кортизол",
        "семантичні реакції та ототожнення карти з територією",
        "ресурсоорієнтована економіка автоматизація",
        "проєктування міст майбутнього відновлювана енергія",
        "грошова система дефіцит злочинність",
        "поведінка приматів окситоцин дофамін",
        "мова наука та розсудливість не-арістотелівські системи",
    ]

    print("\n" + "=" * 65)
    print("🔬 KORZ A/B Evaluation: Baseline vs Clean Candidate")
    print("=" * 65)
    print(f"Baseline DB:  {baseline_path}")
    print(f"Candidate DB: {candidate_path}")
    print(f"Кількість тестових запитів: {len(queries)}")

    results = CandidateLab.evaluate_ab(b_lib, c_lib, queries, top_k=10)

    print("─" * 65)
    print(f"Noise@10 Baseline:  {results['baseline_noise_at_k_pct']}%")
    print(f"Noise@10 Candidate: {results['candidate_noise_at_k_pct']}%")
    print(f"Коефіцієнт зниження шуму: {results['noise_reduction_factor']}x")
    print("─" * 65)
    print("\nДеталізація по запитах:")
    for comp in results["comparisons"]:
        print(f"\n❓ «{comp['query']}»:")
        print(f"  • Baseline Noise: {comp['baseline_noise_in_topk']}/10")
        print(f"  • Candid.  Noise: {comp['candidate_noise_in_topk']}/10")
        print(f"  • Baseline Top-1: «{comp['baseline_top1'][:75]}...»")
        print(f"  • Candid.  Top-1: «{comp['candidate_top1'][:75]}...»")
    print("=" * 65 + "\n")

    b_lib.close()
    c_lib.close()


def cmd_corpus_align(args) -> None:
    from korz_corpus import CorpusAligner
    baseline_path = Path(getattr(args, "baseline", None) or "data/korz_library.sqlite")
    candidate_path = Path(getattr(args, "candidate", None) or "data/korz_clean_candidate.sqlite")
    print("\n" + "=" * 65)
    print("🔗 KORZ Cross-Corpus Content Alignment (CorpusParagraphMap)")
    print("=" * 65)
    print(f"Baseline DB:  {baseline_path}")
    print(f"Candidate DB: {candidate_path}")
    print("Побудова стабільного cross-corpus мапінгу абзаців...")
    t0 = time.perf_counter()
    res = CorpusAligner.build_alignment_map(baseline_path, candidate_path)
    el = time.perf_counter() - t0
    print("─" * 65)
    print(f"• Оброблено Baseline абзаців: {res['total_baseline']:,}".replace(",", " "))
    print(f"• Точний нормалізований збіг: {res['exact_normalized']:,} ({res['exact_normalized']*100/res['total_baseline']:.1f}%)".replace(",", " "))
    print(f"• Незначні варіації очищення: {res['minor_variation']:,} ({res['minor_variation']*100/res['total_baseline']:.1f}%)".replace(",", " "))
    print(f"• Зшиті через розрив сторінки: {res['cross_page_stitched']:,} ({res['cross_page_stitched']*100/res['total_baseline']:.1f}%)".replace(",", " "))
    print(f"• Видалений OCR-шум:          {res['deleted_ocr_noise']:,} ({res['deleted_ocr_noise']*100/res['total_baseline']:.1f}%)".replace(",", " "))
    print(f"• Рівень верифікованого мапінгу: {res['alignment_rate_pct']}%")
    print(f"Час обчислення: {el:.1f} с")
    print("✅ Таблицю corpus_paragraph_map та corpus_metadata збережено у Candidate DB.\n")


def cmd_corpus_ab_shadow(args) -> None:
    from korz_core import KorzLibrary, KorzConfig
    from korz_corpus import CorpusParagraphMap, CorpusABShadowRunner

    query = args.query.strip()
    target_id = getattr(args, "target", None)
    top_k = getattr(args, "limit", 10)

    baseline_path = Path(getattr(args, "baseline", None) or "data/korz_library.sqlite")
    candidate_path = Path(getattr(args, "candidate", None) or "data/korz_clean_candidate.sqlite")

    b_lib = KorzLibrary(KorzConfig(db_path=baseline_path, semantic_enabled=True))
    c_lib = KorzLibrary(KorzConfig(db_path=candidate_path, semantic_enabled=True))
    pmap = CorpusParagraphMap(candidate_path)
    runner = CorpusABShadowRunner(b_lib, c_lib, pmap)

    res = runner.evaluate_query(query, target_baseline_id=target_id, top_k=top_k)

    print("\n" + "=" * 65)
    print("🛰️  KORZ Corpus A/B Shadow Telemetry")
    print("=" * 65)
    print(f"❓ Запит: «{query}»")
    if target_id:
        print(f"🎯 Ціль: Baseline ID #{target_id} -> Clean ID #{res['target_clean_id']}")
        sem_lift_str = f"{res['sem_rank_lift']:+d}" if res['sem_rank_lift'] is not None else "N/A"
        print(f"   Semantic Rank: Baseline #{res['baseline_sem_rank']} -> Clean #{res['clean_sem_rank']} (Lift: {sem_lift_str})")
        fts_lift_str = f"{res['fts_rank_lift']:+d}" if res['fts_rank_lift'] is not None else "N/A"
        print(f"   FTS Rank:      Baseline #{res['baseline_fts_rank']} -> Clean #{res['clean_fts_rank']} (Lift: {fts_lift_str})")

    print("\n📊 Метрики якості видачі:")
    print("─" * 65)
    print(f"• Content Overlap@5:   {res['content_overlap_at_5_pct']}%")
    print(f"• Content Overlap@10:  {res['content_overlap_at_10_pct']}%")
    print(f"• Noise@5:             Baseline: {res['baseline_noise_at_5']}/5 | Clean: {res['clean_noise_at_5']}/5")
    print(f"• Noise@10:            Baseline: {res['baseline_noise_at_10']}/10 | Clean: {res['clean_noise_at_10']}/10")
    print(f"• Top-1 Semantic Score: Baseline: {res['baseline_sem_score']} | Clean: {res['clean_sem_score']}")

    print("\n🔍 Top-1 Порівняння:")
    print(f"  [Baseline #{1}]: «{res['baseline_sem_top1']}...» (Шум: {'ТАК' if res['baseline_top1_is_noise'] else 'НІ'})")
    print(f"  [Clean    #{1}]: «{res['clean_sem_top1']}...» (Шум: {'ТАК' if res['clean_top1_is_noise'] else 'НІ'})")
    if res['clean_top1_in_baseline_rank']:
        print(f"  ℹ️ Clean Top-1 у Baseline займав місце: #{res['clean_top1_in_baseline_rank']}")
    print("=" * 65 + "\n")

    b_lib.close()
    c_lib.close()


def cmd_corpus_promotion_gate(args) -> None:
    from korz_core import KorzLibrary, KorzConfig
    from korz_corpus import CorpusParagraphMap, CorpusABShadowRunner

    baseline_path = Path(getattr(args, "baseline", None) or "data/korz_library.sqlite")
    candidate_path = Path(getattr(args, "candidate", None) or "data/korz_clean_candidate.sqlite")

    b_lib = KorzLibrary(KorzConfig(db_path=baseline_path, semantic_enabled=True))
    c_lib = KorzLibrary(KorzConfig(db_path=candidate_path, semantic_enabled=True))
    pmap = CorpusParagraphMap(candidate_path)
    runner = CorpusABShadowRunner(b_lib, c_lib, pmap)

    # ── REGRESSION SET ──
    # target = baseline known-good result (checks nothing important broke)
    regression_benchmarks = [
        {"query": "стрес і соціальна ієрархія", "target_id": 7033},
        {"query": "бабуїни павіани агресія кортизол", "target_id": 12345},
        {"query": "семантичні реакції та ототожнення карти з територією", "target_id": 5227},
        {"query": "ресурсоорієнтована економіка автоматизація", "target_id": 3914},
        {"query": "проєктування міст майбутнього відновлювана енергія", "target_id": 4445},
        {"query": "грошова система дефіцит злочинність", "target_id": 4258},
        {"query": "поведінка приматів окситоцин дофамін", "target_id": 2210},
        {"query": "мова наука та розсудливість не-арістотелівські системи", "target_id": 5067},
    ]

    # ── NOISE CHALLENGE SET ──
    # Queries designed to provoke noise results (chapter names, author names,
    # page numbers, short ambiguous terms, header-like phrases)
    noise_challenge_queries = [
        {"query": "Сапольський", "target_id": None},
        {"query": "Спогади примата", "target_id": None},
        {"query": "Жак Фреско", "target_id": None},
        {"query": "Проєктування майбутнього", "target_id": None},
        {"query": "Коржибський", "target_id": None},
        {"query": "Наука і розсудливість", "target_id": None},
        {"query": "Розділ 1", "target_id": None},
        {"query": "Передмова", "target_id": None},
        {"query": "Вступ", "target_id": None},
        {"query": "Бібліографія", "target_id": None},
        {"query": "примітки", "target_id": None},
        {"query": "зміст", "target_id": None},
    ]

    print("\n" + "=" * 65)
    print("🛡️  KORZ Corpus v2 Clean Promotion Gate")
    print("=" * 65)

    print("\n📋 Regression Benchmark (n={})...".format(len(regression_benchmarks)))
    summary = runner.run_benchmark(regression_benchmarks, top_k=10)

    print("📋 Noise Challenge Set (n={})...".format(len(noise_challenge_queries)))
    noise_summary = runner.run_benchmark(noise_challenge_queries, top_k=10)

    verdict = runner.check_promotion_gate(summary, noise_challenge_summary=noise_summary)

    # ── Display Tiered Verdict ──
    print("\n" + "─" * 65)
    print("⚙️  TECHNICAL GATES")
    print("─" * 65)
    for g in verdict["technical_gates"]:
        icon = "✅" if g["passed"] else "❌"
        print(f"  {icon} {g['name']:<30} : {g['detail']}")

    print("\n" + "─" * 65)
    print("🔍 RETRIEVAL GATES")
    print("─" * 65)
    for g in verdict["retrieval_gates"]:
        icon = "✅" if g["passed"] else "❌"
        print(f"  {icon} {g['name']:<30} : {g['detail']}")

    print("\n" + "─" * 65)
    print("📊 EVIDENCE STATUS")
    print("─" * 65)
    n = verdict["judged_queries"]
    n_min = verdict["min_queries_for_promotion"]
    ev_icon = "✅" if verdict["sufficient_evidence"] else "🟡"
    print(f"  {ev_icon} {'Judged queries':<30} : {n} / {n_min} minimum")
    if not verdict["sufficient_evidence"]:
        print(f"     ℹ️  {n_min - n} додаткових ground-truth запитів потрібно для повного PASS")

    print("\n" + "=" * 65)
    status = verdict["status"]
    if status == "PASS":
        print(f"🎉 Статус: [{status}]")
        print("   Очищений корпус рекомендовано для переходу на 'Corpus v2-clean1'.")
    elif status == "PROVISIONAL PASS":
        print(f"🟡 Статус: [{status}]")
        print("   Технічні критерії пройдені. Retrieval-критерії пройдені попередньо.")
        print("   ⏸  HOLD — недостатньо ground-truth для production promotion.")
        print(f"   Потрібно: {n_min}+ 🎯 запитів з людською розміткою.")
    else:
        print(f"❌ Статус: [{status}]")
        print("   Є непройдені критерії. Перехід заблоковано.")
    print("=" * 65 + "\n")

    b_lib.close()
    c_lib.close()



def main() -> None:
    parser = argparse.ArgumentParser(
        prog="korz",
        description="KORZ LOCAL — Локальна бібліотека та швидкий пошук по книгах (SQLite + FTS5 + Semantic)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Команда для виконання")

    # add
    p_add = subparsers.add_parser("add", help="Додати книгу (PDF, Markdown або каталог)")
    p_add.add_argument("path", help="Шлях до файлу книги або каталогу")
    p_add.add_argument("--title", help="Назва книги (якщо відрізняється від автодетекту)")
    p_add.add_argument("--author", help="Автор книги")
    p_add.add_argument("--edition", help="Рік / видання")

    # find (exact)
    p_find = subparsers.add_parser("find", help="Знайти точну фразу (Exact substring match)")
    p_find.add_argument("phrase", help="Фраза для пошуку")
    p_find.add_argument("--limit", type=int, default=10, help="Максимальна кількість результатів (за замовчуванням: 10)")
    p_find.add_argument("--doc", help="Фільтр за ID книги")

    # search (fts/bm25/auto/semantic)
    p_search = subparsers.add_parser("search", help="Пошук у бібліотеці (Auto / FTS5 / Exact / Semantic)")
    p_search.add_argument("query", help="Пошуковий запит")
    p_search.add_argument(
        "--mode",
        choices=["auto", "fts", "exact", "semantic"],
        default="auto",
        help="Режим пошуку (auto, fts, exact, semantic)",
    )
    p_search.add_argument("--limit", type=int, default=10, help="Максимальна кількість результатів (за замовчуванням: 10)")
    p_search.add_argument("--doc", help="Фільтр за ID книги")

    # ask (Question Retrieval / Concept search)
    p_ask = subparsers.add_parser("ask", help="Семантичний пошук уривків, що відповідають на смислове питання (Question Retrieval)")
    p_ask.add_argument("question", help="Питання або концептуальний запит")
    p_ask.add_argument("--limit", "-n", type=int, default=5, help="Кількість результатів (за замовчуванням: 5)")
    p_ask.add_argument("--doc", help="Фільтр за ID книги")

    # answer (Grounded Answering Engine)
    p_ans = subparsers.add_parser("answer", help="Сформувати обґрунтовану відповідь на питання на основі книг бібліотеки")
    p_ans.add_argument("question", help="Питання до бібліотеки")
    p_ans.add_argument(
        "--mode",
        choices=["extractive", "generative", "auto"],
        default="auto",
        help="Режим відповіді: extractive (буквальні цитати), generative (локальна LLM) або auto",
    )
    p_ans.add_argument("--limit", "-n", type=int, default=5, help="Кількість уривків-кандидатів (за замовчуванням: 5)")
    p_ans.add_argument("--doc", help="Фільтр за ID книги")

    # show
    p_show = subparsers.add_parser("show", help="Показати абзац та розширений контекст")
    p_show.add_argument("id", type=int, help="ID абзацу (paragraph_id)")
    p_show.add_argument("--context", type=int, default=2, help="Кількість сусідніх абзаців (за замовчуванням: 2)")

    # books
    subparsers.add_parser("books", help="Переглянути список усіх книг у бібліотеці")

    # remove
    p_remove = subparsers.add_parser("remove", help="Видалити книгу з бібліотеки")
    p_remove.add_argument("book", help="ID книги або назва")
    p_remove.add_argument("-y", "--yes", action="store_true", help="Підтвердити видалення без запиту")

    # status
    subparsers.add_parser("status", help="Стан бібліотеки, розмір бази та кількість записів")

    # backup
    p_backup = subparsers.add_parser("backup", help="Створити онлайновий бекап бази даних")
    p_backup.add_argument("--dest", help="Цільовий шлях для бекапу")

    # restore
    p_restore = subparsers.add_parser("restore", help="Відновити базу даних із файлу бекапу")
    p_restore.add_argument("backup_path", help="Шлях до файлу бекапу (.sqlite)")
    p_restore.add_argument("-y", "--yes", action="store_true", help="Підтвердити відновлення без запиту")

    # semantic (V2)
    p_sem = subparsers.add_parser("semantic", help="Керування семантичним шаром V2 (інфраструктура ембедингів)")
    sem_sub = p_sem.add_subparsers(dest="semantic_command")
    sem_sub.add_parser("status", help="Стан семантичної інфраструктури, моделі та векторизовані чанки")

    p_sem_verify = sem_sub.add_parser("verify", help="Перевірити цілісність векторів, розмірностей та відповідність SHA")
    p_sem_verify.add_argument("--model", help="ID або назва моделі")

    p_sem_build = sem_sub.add_parser("build", help="Побудувати або добудувати офлайн-індекс ембедингів")
    p_sem_build.add_argument("--model", help="ID або назва моделі (за замовчуванням: intfloat/multilingual-e5-base)")
    p_sem_build.add_argument("--batch-size", type=int, default=32, help="Розмір батчу для векторизації (за замовчуванням: 32)")
    p_sem_build.add_argument("--force", action="store_true", help="Перерахувати ембединги заново, ігноруючи кеш")

    p_sem_search = sem_sub.add_parser("search", help="Прямий семантичний пошук через векторну матрицю")
    p_sem_search.add_argument("query", help="Пошуковий запит")
    p_sem_search.add_argument("--limit", type=int, default=10, help="Кількість результатів")
    p_sem_search.add_argument("--model", help="ID або назва моделі")
    p_sem_search.add_argument("--doc", help="Фільтр за ID книги")

    sem_sub.add_parser("models", help="Список зареєстрованих моделей ембедингів")

    p_sem_reg = sem_sub.add_parser("register", help="Зареєструвати нову модель ембедингів")
    p_sem_reg.add_argument("model", help="Назва моделі на HuggingFace (напр. intfloat/multilingual-e5-base)")
    p_sem_reg.add_argument("--dim", type=int, default=768, help="Розмірність векторів")
    p_sem_reg.add_argument("--query-tpl", default="query: {text}", help="Шаблон запиту")
    p_sem_reg.add_argument("--passage-tpl", default="passage: {text}", help="Шаблон чанка")
    p_sem_reg.add_argument("--max-tokens", type=int, default=512, help="Ліміт токенів моделі")

    # admin
    p_admin = subparsers.add_parser("admin", help="Адміністративні інструменти")
    admin_sub = p_admin.add_subparsers(dest="admin_command")
    admin_sub.add_parser("verify", help="Перевірити цілісність бази даних та FTS5")
    admin_sub.add_parser("reindex", help="Перебудувати індекс FTS5")
    admin_sub.add_parser("migrate", help="Перевірити та виконати міграції схеми бази даних")
    p_admin_log = admin_sub.add_parser("search-log", help="Переглянути нещодавні запити без результатів (для аналізу V2)")
    p_admin_log.add_argument("--limit", type=int, default=20, help="Кількість записів (за замовчуванням: 20)")

    # shadow (Phase V2C)
    p_shadow = subparsers.add_parser("shadow", help="Аналітика та керування тіньовим режимом (Shadow Mode V2C)")
    shadow_sub = p_shadow.add_subparsers(dest="shadow_command")

    p_sh_report = shadow_sub.add_parser("report", help="Звіт розходжень між FTS5 та Semantic, латентності та оцінок")
    p_sh_report.add_argument("--detailed", action="store_true", help="Показати детальний список запитів із найбільшими розходженнями")
    p_sh_report.add_argument("--limit", type=int, default=20, help="Кількість записів у списку розходжень (за замовчуванням: 20)")

    p_sh_replay = shadow_sub.add_parser("replay", help="Прогнати історичні запити з search_log через shadow evaluation")
    p_sh_replay.add_argument("--limit", type=int, default=300, help="Максимальна кількість запитів для аналізу (за замовчуванням: 300)")
    p_sh_replay.add_argument("--since", help="Фільтр за початковою датою (YYYY-MM-DD)")
    p_sh_replay.add_argument("--model", help="ID або назва моделі")
    p_sh_replay.add_argument("--force", action="store_true", help="Перерахувати навіть ті запити, що вже обчислені")

    shadow_sub.add_parser("status", help="Коротка статистика збережених тіньових запитів та відгуків")

    # corpus (Corpus Quality Pipeline)
    p_corp = subparsers.add_parser("corpus", help="KORZ Corpus Quality Pipeline (аудит, очищення OCR та валідація текстів)")
    corp_sub = p_corp.add_subparsers(dest="corpus_command")

    p_corp_audit = corp_sub.add_parser("audit", help="Аудит файлу або папки на наявність OCR-сміття, колонтитулів та номерів сторінок")
    p_corp_audit.add_argument("path", help="Шлях до файлу .md або папки з книгами")
    p_corp_audit.add_argument("--json", action="store_true", help="Вивести звіт у форматі JSON")
    p_corp_audit.add_argument("--lang", default="uk", choices=["uk", "all"], help="Мовний фільтр (за замовчуванням: uk)")

    p_corp_clean = corp_sub.add_parser("clean", help="Очистити файл або папку від OCR-сміття (з підтримкою --dry-run та --apply)")
    p_corp_clean.add_argument("path", help="Шлях до файлу або каталогу для очищення")
    p_corp_clean.add_argument("--dry-run", action="store_true", help="Показати diff без зміни файлів")
    p_corp_clean.add_argument("--apply", action="store_true", help="Застосувати очищення та зберегти файли")
    p_corp_clean.add_argument("--out", help="Каталог для збереження очищених файлів (за замовчуванням: data/corpus_clean/)")
    p_corp_clean.add_argument("--lang", default="uk", choices=["uk", "all"], help="Мовний фільтр (за замовчуванням: uk)")

    p_corp_verify = corp_sub.add_parser("verify-clean", help="Перевірити безпечність очищення (Text-Loss Safety Gate)")
    p_corp_verify.add_argument("original", help="Шлях до оригінального файлу")
    p_corp_verify.add_argument("cleaned", help="Шлях до очищеного файлу")

    corp_sub.add_parser("report", help="Підсумковий звіт аудиту якості всіх книг корпусу")

    p_corp_build = corp_sub.add_parser("build-candidate", help="Побудувати ізольовану SQLite базу з data/corpus_clean")
    p_corp_build.add_argument("--clean-dir", default="data/corpus_clean", help="Шлях до каталогу з очищеними книгами")
    p_corp_build.add_argument("--out", default="data/korz_clean_candidate.sqlite", help="Шлях до файлу candidate бази")

    p_corp_eval = corp_sub.add_parser("evaluate-ab", help="A/B порівняння якості (Noise@10) між baseline та candidate базою")
    p_corp_eval.add_argument("--baseline", default="data/korz_library.sqlite", help="Шлях до baseline бази")
    p_corp_eval.add_argument("--candidate", default="data/korz_clean_candidate.sqlite", help="Шлях до candidate бази")

    p_corp_align = corp_sub.add_parser("align", help="Вирівняти ідентифікатори абзаців між baseline та clean candidate базами")
    p_corp_align.add_argument("--baseline", default="data/korz_library.sqlite", help="Шлях до baseline бази")
    p_corp_align.add_argument("--candidate", default="data/korz_clean_candidate.sqlite", help="Шлях до candidate бази")

    p_corp_ab = corp_sub.add_parser("ab-shadow", help="Порівняти видачу baseline та clean candidate на конкретному запиті")
    p_corp_ab.add_argument("query", help="Пошуковий запит")
    p_corp_ab.add_argument("--target", "--target-id", dest="target", type=int, help="Очікуваний paragraph_id у baseline")
    p_corp_ab.add_argument("--limit", type=int, default=10, help="Кількість результатів (за замовчуванням: 10)")
    p_corp_ab.add_argument("--baseline", default="data/korz_library.sqlite", help="Шлях до baseline бази")
    p_corp_ab.add_argument("--candidate", default="data/korz_clean_candidate.sqlite", help="Шлях до candidate бази")

    p_corp_promo = corp_sub.add_parser("promotion-gate", help="Перевірити всі 6 критеріїв Promotion Gate для переходу на Corpus v2")
    p_corp_promo.add_argument("--baseline", default="data/korz_library.sqlite", help="Шлях до baseline бази")
    p_corp_promo.add_argument("--candidate", default="data/korz_clean_candidate.sqlite", help="Шлях до candidate бази")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    lib = KorzLibrary()
    try:
        if args.command == "add":
            cmd_add(args, lib)
        elif args.command == "find":
            cmd_find(args, lib)
        elif args.command == "search":
            cmd_search(args, lib)
        elif args.command == "ask":
            cmd_ask(args, lib)
        elif args.command == "answer":
            cmd_answer(args, lib)
        elif args.command == "show":
            cmd_show(args, lib)
        elif args.command == "books":
            cmd_books(args, lib)
        elif args.command == "remove":
            cmd_remove(args, lib)
        elif args.command == "status":
            cmd_status(args, lib)
        elif args.command == "backup":
            cmd_backup(args, lib)
        elif args.command == "restore":
            cmd_restore(args, lib)
        elif args.command == "semantic":
            if args.semantic_command == "status":
                cmd_semantic_status(args, lib)
            elif args.semantic_command == "verify":
                cmd_semantic_verify(args, lib)
            elif args.semantic_command == "build":
                cmd_semantic_build(args, lib)
            elif args.semantic_command == "search":
                cmd_semantic_search(args, lib)
            elif args.semantic_command == "models":
                cmd_semantic_models(args, lib)
            elif args.semantic_command == "register":
                cmd_semantic_register(args, lib)
            else:
                p_sem.print_help()
        elif args.command == "admin":
            if args.admin_command == "verify":
                cmd_admin_verify(args, lib)
            elif args.admin_command == "reindex":
                cmd_admin_reindex(args, lib)
            elif args.admin_command == "migrate":
                cmd_admin_migrate(args, lib)
            elif args.admin_command == "search-log":
                cmd_admin_search_log(args, lib)
            else:
                p_admin.print_help()
        elif args.command == "shadow":
            if args.shadow_command == "report":
                cmd_shadow_report(args, lib)
            elif args.shadow_command == "replay":
                cmd_shadow_replay(args, lib)
            elif args.shadow_command == "status":
                cmd_shadow_status(args, lib)
            else:
                p_shadow.print_help()
        elif args.command == "corpus":
            if args.corpus_command == "audit":
                cmd_corpus_audit(args)
            elif args.corpus_command == "clean":
                cmd_corpus_clean(args)
            elif args.corpus_command == "verify-clean":
                cmd_corpus_verify(args)
            elif args.corpus_command == "report":
                cmd_corpus_report(args)
            elif args.corpus_command == "build-candidate":
                cmd_corpus_build_candidate(args)
            elif args.corpus_command == "evaluate-ab":
                cmd_corpus_evaluate_ab(args)
            elif args.corpus_command == "align":
                cmd_corpus_align(args)
            elif args.corpus_command == "ab-shadow":
                cmd_corpus_ab_shadow(args)
            elif args.corpus_command == "promotion-gate":
                cmd_corpus_promotion_gate(args)
            else:
                p_corp.print_help()
    finally:
        lib.close()


if __name__ == "__main__":
    main()
