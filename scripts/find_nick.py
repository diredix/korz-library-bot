import sqlite3
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("data/korz_library.sqlite")
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT p.id, p.page_number, p.paragraph_index, p.display_text
    FROM paragraphs p
    JOIN documents d ON d.id = p.document_id
    WHERE (p.display_text LIKE '%Нік %' OR p.display_text LIKE '%Нік,%' OR p.display_text LIKE '%Ніка%' OR p.display_text LIKE '%Ніком%' OR p.display_text LIKE '%Ніку%')
      AND d.title LIKE '%Primate%'
    ORDER BY p.id
""").fetchall()

print(f"Total occurrences of Нік: {len(rows)}")
for r in rows:
    print(f"PID {r['id']} (p.{r['page_number']} #{r['paragraph_index']}): {r['display_text']}\n---")
