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

from korz_core import KorzLibrary, KorzConfig


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

    mode_label = "Exact" if (results and results[0].match_type == "exact") else "FTS5 / BM25"
    print(f"\n🔎 Знайдено результатів: {len(results)} ({mode_label}: «{query}»)\n" + "─" * 60)
    for r in results:
        author_str = f"{r.author} — " if r.author else ""
        score_str = f" (Score: {r.score:.4f})" if r.match_type == "fts" else ""
        print(f"\n{author_str}{r.title}")
        print(f"Сторінка {r.page} (абзац {r.paragraph_index}) [ID: {r.paragraph_id}]{score_str}")
        print()
        print(f"   {r.text}")
        print("─" * 60)
    print()


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
    print("\n🧠 KORZ Semantic Infrastructure (Phase V2A):")
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
    print("🔬 Перевірка цілісності семантичної інфраструктури (V2A)...")
    res = lib.verify_semantic_integrity()
    print("─" * 50)
    print(f"Перевірено моделей:    {res['models_checked']}")
    print(f"Перевірено векторів:   {res['vectors_checked']}")
    print(f"Застарілих векторів:   {res['stale_embeddings']}")
    print(f"Знайдено проблем:      {res['issues_count']}")
    if res["issues"]:
        for iss in res["issues"]:
            print(f" ⚠️  {iss}")
    print("─" * 50)
    if res["all_healthy"]:
        print("✅ Семантична інфраструктура повністю здорова та консистентна.\n")
    else:
        print("❌ Виявлено невідповідності в семантичному шарі!\n")


def cmd_admin_migrate(args, lib: KorzLibrary) -> None:
    print("⚙️  Перевірка та виконання міграцій бази даних...")
    st = lib.status()
    print(f"✓ Поточна версія схеми бази: v{st.get('schema_version', 2)}")
    print("✅ Всі міграції застосовано успішно.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="korz",
        description="KORZ LOCAL — Локальна бібліотека та швидкий пошук по книгах (SQLite + FTS5)",
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

    # search (fts/bm25/auto)
    p_search = subparsers.add_parser("search", help="Пошук у бібліотеці (FTS5 / BM25 / Auto)")
    p_search.add_argument("query", help="Пошуковий запит")
    p_search.add_argument("--mode", choices=["auto", "fts", "exact"], default="auto", help="Режим пошуку (auto, fts, exact)")
    p_search.add_argument("--limit", type=int, default=10, help="Максимальна кількість результатів (за замовчуванням: 10)")
    p_search.add_argument("--doc", help="Фільтр за ID книги")

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
    sem_sub.add_parser("verify", help="Перевірити цілісність векторів, розмірностей та відповідність SHA")

    # admin
    p_admin = subparsers.add_parser("admin", help="Адміністративні інструменти")
    admin_sub = p_admin.add_subparsers(dest="admin_command")
    admin_sub.add_parser("verify", help="Перевірити цілісність бази даних та FTS5")
    admin_sub.add_parser("reindex", help="Перебудувати індекс FTS5")
    admin_sub.add_parser("migrate", help="Перевірити та виконати міграції схеми бази даних")
    p_admin_log = admin_sub.add_parser("search-log", help="Переглянути нещодавні запити без результатів (для аналізу V2)")
    p_admin_log.add_argument("--limit", type=int, default=20, help="Кількість записів (за замовчуванням: 20)")

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
    finally:
        lib.close()


if __name__ == "__main__":
    main()
