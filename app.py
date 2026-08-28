from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==========================================
# Patient information
# ==========================================

current_patient = {
    "id": "Patient 01",
    "name": "Patient 01"
}


# ==========================================
# Current request
# ==========================================

current_request = {
    "gesture": "No request",
    "confidence": 0,
    "time": "--",
    "status": "Waiting"
}


# ==========================================
# Request history
# ==========================================

request_history = []


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

    return jsonify(current_patient)


# ==========================================
# Get current request
# ==========================================

@app.route("/api/current-request", methods=["GET"])
def get_current_request():

    return jsonify(current_request)


# ==========================================
# Get request history
# ==========================================

@app.route("/api/history", methods=["GET"])
def get_history():

    return jsonify(request_history)


# ==========================================
# Receive a new gesture
# ==========================================

@app.route("/api/gesture", methods=["POST"])
def receive_gesture():

    global current_request

    data = request.get_json()

    gesture = data.get(
        "gesture",
        "Unknown"
    )

    confidence = data.get(
        "confidence",
        0
    )

    now = datetime.now()

    time_string = now.strftime(
        "%I:%M:%S %p"
    )

    # Create request
    current_request = {
        "gesture": gesture,
        "confidence": confidence,
        "time": time_string,
        "status": "New"
    }

    # Add to history
    request_history.insert(
        0,
        {
            "patient": current_patient["id"],
            "gesture": gesture,
            "confidence": confidence,
            "time": time_string,
            "status": "New"
        }
    )

    # Keep only latest 50 requests
    if len(request_history) > 50:

        request_history.pop()

    print(
        f"[{time_string}] "
        f"{current_patient['id']} → "
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

    global current_request

    current_request["status"] = "Handled"

    if request_history:

        request_history[0]["status"] = "Handled"

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