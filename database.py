import sqlite3

DATABASE = "hospital.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_database():
    connection = get_connection()
    cursor = connection.cursor()

    # Patients table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            room_number TEXT,
            medical_information TEXT
        )
    """)

    # Gesture requests table
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

    # Add our initial patient
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

    print("Database created successfully!")


if __name__ == "__main__":
    create_database()