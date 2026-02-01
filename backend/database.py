import sqlite3
import datetime
import os

DB_NAME = os.path.join(os.path.dirname(__file__), 'database.db')

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
        
        # Migration: Add image_path, notes, tags if not exists
        try:
            conn.execute('ALTER TABLE history ADD COLUMN image_path TEXT')
            print("✅ Added image_path column.")
        except sqlite3.Error:
            pass # Column likely exists

        try:
            conn.execute('ALTER TABLE history ADD COLUMN notes TEXT')
            print("✅ Added notes column.")
        except sqlite3.Error:
            pass

        try:
            conn.execute('ALTER TABLE history ADD COLUMN tags TEXT')
            print("✅ Added tags column.")
        except sqlite3.Error:
            pass
            
        finally:
            conn.close()

def add_scan(filename, prediction, confidence, fake_prob, real_prob, image_path=""):
    conn = get_db_connection()
    if conn:
        try:
            conn.execute('''
                INSERT INTO history (filename, prediction, confidence, fake_probability, real_probability, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, prediction, confidence, fake_prob, real_prob, image_path))
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

def delete_scan(scan_id):
    conn = get_db_connection()
    if conn:
        try:
            conn.execute('DELETE FROM history WHERE id = ?', (scan_id,))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error deleting scan: {e}")
            return False
        finally:
            conn.close()
    return False

def update_scan(scan_id, data):
    conn = get_db_connection()
    if conn:
        try:
            fields = []
            values = []
            if 'notes' in data:
                fields.append("notes = ?")
                values.append(data['notes'])
            if 'tags' in data:
                fields.append("tags = ?")
                values.append(data['tags'])
            
            if not fields:
                return True
                
            values.append(scan_id)
            query = f"UPDATE history SET {', '.join(fields)} WHERE id = ?"
            conn.execute(query, tuple(values))
            conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating scan: {e}")
            return False
        finally:
            conn.close()
    return False

# Initialize DB on module load
init_db()
