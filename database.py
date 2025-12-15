import sqlite3
import datetime
import os

DB_NAME = 'database.db'

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fake_probability REAL NOT NULL,
                    real_probability REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            print("✅ Database initialized successfully.")
        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
        finally:
            conn.close()

def add_scan(filename, prediction, confidence, fake_prob, real_prob):
    conn = get_db_connection()
    if conn:
        try:
            conn.execute('''
                INSERT INTO history (filename, prediction, confidence, fake_probability, real_probability)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, prediction, confidence, fake_prob, real_prob))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error adding scan: {e}")
            return False
        finally:
            conn.close()
    return False

def get_history():
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.execute('SELECT * FROM history ORDER BY timestamp DESC')
            history = [dict(row) for row in cursor.fetchall()]
            return history
        except sqlite3.Error as e:
            print(f"Error retrieving history: {e}")
            return []
        finally:
            conn.close()
    return []

def clear_history():
    conn = get_db_connection()
    if conn:
        try:
            conn.execute('DELETE FROM history')
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error clearing history: {e}")
            return False
        finally:
            conn.close()
    return False

# Initialize DB on module load if not exists
if not os.path.exists(DB_NAME):
    init_db()
