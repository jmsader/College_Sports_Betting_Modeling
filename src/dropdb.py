from GLOBS import DB_SLUG, confirm_intent
import sqlite3


confirm_intent()

with sqlite3.connect(DB_SLUG) as conn:
    conn.execute("DROP TABLE IF EXISTS games;")
    conn.execute("DROP TABLE IF EXISTS teams;")
print("Dropped existing database tables.")
