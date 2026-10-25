
import sqlite3

DATABASE = "hospital.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_checkins_table():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            requested_at TEXT NOT NULL,
            responded_at TEXT,
            response_confidence REAL,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)

    connection.commit()
    connection.close()


def create_database():
    connection = get_connection()
    cursor = connection.cursor()

    # Patient information
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            room_number TEXT,
            medical_information TEXT
        )
    """)

    # Live requests
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            gesture TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'New',
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)

    # Add request_date to existing databases if needed
    cursor.execute("PRAGMA table_info(requests)")
    columns = [row["name"] for row in cursor.fetchall()]

    if "request_date" not in columns:
        cursor.execute(
            "ALTER TABLE requests ADD COLUMN request_date TEXT"
        )

    # Permanent archive: records survive dashboard resets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS archived_requests (
            archive_id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_request_id INTEGER,
            patient_id TEXT,
            patient_name TEXT,
            gesture TEXT NOT NULL,
            confidence REAL DEFAULT 0,
            request_date TEXT,
            time TEXT,
            status TEXT,
            archived_at TEXT NOT NULL
        )
    """)

    # Insert the initial patient if not already present
    cursor.execute("""
        INSERT OR IGNORE INTO patients
        (id, name, age, room_number, medical_information)
        VALUES (?, ?, ?, ?, ?)
    """, (
        "Patient 01",
        "Patient 01",
        65,
        "204",
        "Post-surgery observation"
    ))

    connection.commit()
    connection.close()

    print("Database and archive created successfully!")


if __name__ == "__main__":
    create_database()
