from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime
from database import get_connection

app = Flask(__name__)
CORS(app)


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
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, name, age, room_number, medical_information
        FROM patients
        LIMIT 1
    """)

    patient = cursor.fetchone()

    connection.close()

    if patient:
        return jsonify(dict(patient))

    return jsonify({
        "message": "No patient found"
    }), 404


# ==========================================
# Get current request
# ==========================================

@app.route("/api/current-request", methods=["GET"])
def get_current_request():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            patient_id,
            gesture,
            confidence,
            time,
            status
        FROM requests
        ORDER BY id DESC
        LIMIT 1
    """)

    result = cursor.fetchone()

    connection.close()

    if result:
        return jsonify(dict(result))

    return jsonify({
        "gesture": "No request",
        "confidence": 0,
        "time": "--",
        "status": "Waiting"
    })


# ==========================================
# Get request history
# ==========================================

@app.route("/api/history", methods=["GET"])
def get_history():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            r.id,
            r.patient_id,
            p.name AS patient,
            r.gesture,
            r.confidence,
            r.time,
            r.status
        FROM requests r
        JOIN patients p
        ON r.patient_id = p.id
        ORDER BY r.id DESC
        LIMIT 50
    """)

    history = [dict(row) for row in cursor.fetchall()]

    connection.close()

    return jsonify(history)


# ==========================================
# Receive a new gesture
# ==========================================

@app.route("/api/gesture", methods=["POST"])
def receive_gesture():

    data = request.get_json()

    gesture = data.get("gesture", "Unknown")
    confidence = data.get("confidence", 0)

    now = datetime.now()

    time_string = now.strftime("%I:%M:%S %p")

    connection = get_connection()
    cursor = connection.cursor()

    # Get current patient
    cursor.execute("""
        SELECT id
        FROM patients
        LIMIT 1
    """)

    patient = cursor.fetchone()

    if not patient:
        connection.close()

        return jsonify({
            "success": False,
            "message": "No patient found"
        }), 404

    patient_id = patient["id"]

    # Save gesture request
    cursor.execute("""
        INSERT INTO requests
        (patient_id, gesture, confidence, time, status)
        VALUES (?, ?, ?, ?, ?)
    """, (
        patient_id,
        gesture,
        confidence,
        time_string,
        "New"
    ))

    connection.commit()
    connection.close()

    print(
        f"[{time_string}] "
        f"{patient_id} → "
        f"{gesture} "
        f"({confidence:.1f}%)"
    )

    return jsonify({
        "success": True,
        "message": "Gesture received"
    })


# ==========================================
# Mark request as handled
# ==========================================

@app.route("/api/handle-request", methods=["POST"])
def handle_request():

    data = request.get_json(silent=True) or {}

    request_id = data.get("request_id")

    connection = get_connection()
    cursor = connection.cursor()

    if request_id:

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

    connection.close()

    return jsonify({
        "success": True
    })


# ==========================================
# Run server
# ==========================================

if __name__ == "__main__":

    print("\n======================================")
    print(" SMART PATIENT ASSISTANCE SERVER")
    print("======================================")
    print("Dashboard:")
    print("http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )