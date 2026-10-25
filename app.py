
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime
import sqlite3
from database import get_connection, create_database

app = Flask(__name__)
CORS(app)

# Initialize database and archive table
create_database()


# ==========================================
# Patient information
# ==========================================

current_patient = {
    "id": "Patient 01",
    "name": "Patient 01"
}


# ==========================================
# Dashboard
# ==========================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ==========================================
# Get current patient
# ==========================================

@app.route("/api/patient", methods=["GET"])
def get_patient():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, name, age, room_number, medical_information
            FROM patients
            LIMIT 1
        """)

        patient = cursor.fetchone()

        if patient:
            return jsonify(dict(patient))

        return jsonify({"message": "No patient found"}), 404

    finally:
        connection.close()


# ==========================================
# Get current request
# ==========================================

@app.route("/api/current-request", methods=["GET"])
def get_current_request():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                id,
                patient_id,
                gesture,
                confidence,
                request_date,
                time,
                status
            FROM requests
            ORDER BY id DESC
            LIMIT 1
        """)

        result = cursor.fetchone()

        if result:
            return jsonify(dict(result))

        return jsonify({
            "gesture": "No request",
            "confidence": 0,
            "time": "--",
            "status": "Waiting"
        })

    finally:
        connection.close()


# ==========================================
# Get live request history
# ==========================================

@app.route("/api/history", methods=["GET"])
def get_history():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                r.id,
                r.patient_id,
                p.name AS patient,
                r.gesture,
                r.confidence,
                r.request_date,
                r.time,
                r.status
            FROM requests r
            LEFT JOIN patients p
                ON r.patient_id = p.id
            ORDER BY r.id DESC
            LIMIT 50
        """)

        history = [dict(row) for row in cursor.fetchall()]
        return jsonify(history)

    finally:
        connection.close()


# ==========================================
# Receive a new gesture
# ==========================================

@app.route("/api/gesture", methods=["POST"])
def receive_gesture():
    data = request.get_json(silent=True) or {}

    gesture = data.get("gesture", "Unknown")
    confidence = data.get("confidence", 0)

    # Validate confidence
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Invalid confidence value"
        }), 400

    now = datetime.now()

    time_string = now.strftime("%I:%M:%S %p")
    date_string = now.strftime("%Y-%m-%d")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Get current patient
        cursor.execute("""
            SELECT id
            FROM patients
            LIMIT 1
        """)

        patient = cursor.fetchone()

        if not patient:
            return jsonify({
                "success": False,
                "message": "No patient found"
            }), 404

        patient_id = patient["id"]

        # Save gesture request with date and time
        cursor.execute("""
            INSERT INTO requests (
                patient_id,
                gesture,
                confidence,
                time,
                status,
                request_date
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            gesture,
            confidence,
            time_string,
            "New",
            date_string
        ))

        connection.commit()

        print(
            f"[{date_string} {time_string}] "
            f"{patient_id} -> {gesture} "
            f"({confidence:.1f}%)"
        )

        return jsonify({
            "success": True,
            "message": "Gesture received",
            "patient_id": patient_id,
            "gesture": gesture,
            "confidence": confidence,
            "request_date": date_string,
            "time": time_string
        })

    except sqlite3.Error:
        connection.rollback()
        app.logger.exception("Failed to save gesture")

        return jsonify({
            "success": False,
            "message": "Could not save gesture"
        }), 500

    finally:
        connection.close()


# ==========================================
# Mark request as handled
# ==========================================

@app.route("/api/handle-request", methods=["POST"])
def handle_request():
    data = request.get_json(silent=True) or {}
    request_id = data.get("request_id")

    connection = get_connection()

    try:
        cursor = connection.cursor()

        if request_id is not None:
            cursor.execute("""
                UPDATE requests
                SET status = 'Handled'
                WHERE id = ?
            """, (request_id,))
        else:
            cursor.execute("""
                UPDATE requests
                SET status = 'Handled'
                WHERE id = (
                    SELECT id
                    FROM requests
                    ORDER BY id DESC
                    LIMIT 1
                )
            """)

        connection.commit()

        if cursor.rowcount == 0:
            return jsonify({
                "success": False,
                "message": "Request not found"
            }), 404

        return jsonify({
            "success": True,
            "message": "Request marked as handled"
        })

    except sqlite3.Error:
        connection.rollback()
        app.logger.exception("Failed to handle request")

        return jsonify({
            "success": False,
            "message": "Could not update request"
        }), 500

    finally:
        connection.close()


# ==========================================
# Get permanently archived history
# ==========================================

@app.route("/api/archived-history", methods=["GET"])
def get_archived_history():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                archive_id,
                original_request_id,
                patient_id,
                patient_name AS patient,
                gesture,
                confidence,
                request_date,
                time,
                status,
                archived_at
            FROM archived_requests
            ORDER BY archived_at DESC, archive_id DESC
        """)

        history = [dict(row) for row in cursor.fetchall()]
        return jsonify(history)

    finally:
        connection.close()


# ==========================================
# Reset live dashboard and archive all requests
# ==========================================

@app.route("/api/reset-dashboard", methods=["POST"])
def reset_dashboard():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Save current requests in the archive before clearing them
        cursor.execute("""
            INSERT INTO archived_requests (
                original_request_id,
                patient_id,
                patient_name,
                gesture,
                confidence,
                request_date,
                time,
                status,
                archived_at
            )
            SELECT
                r.id,
                r.patient_id,
                COALESCE(p.name, r.patient_id),
                r.gesture,
                r.confidence,
                r.request_date,
                r.time,
                r.status,
                CURRENT_TIMESTAMP
            FROM requests r
            LEFT JOIN patients p ON p.id = r.patient_id
        """)

        archived_count = cursor.rowcount

        # Clear only the live requests
        cursor.execute("DELETE FROM requests")

        connection.commit()

        return jsonify({
            "success": True,
            "archived_count": archived_count,
            "current_request": {
                "gesture": "No request",
                "confidence": 0,
                "time": "--",
                "status": "Waiting"
            },
            "request_count": 0
        })

    except Exception as error:
        connection.rollback()
        return jsonify({
            "success": False,
            "message": str(error)
        }), 500

    finally:
        connection.close()

# ==========================================
# Run server
# ==========================================

if __name__ == "__main__":
    print("\n======================================")
    print(" SMART PATIENT ASSISTANCE SERVER")
    print("======================================")
    print("Dashboard: http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
