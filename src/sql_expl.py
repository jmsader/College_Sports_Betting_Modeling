# scratchpad for DB exploration
from GLOBS import DB_SLUG
import sqlite3


with sqlite3.connect(DB_SLUG) as conn:
    cursor = conn.cursor()
