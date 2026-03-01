#!/usr/bin/env python3
"""
Heart Disease Prediction Web Application
==========================================
Author : Sandeep Kumar
Stack  : Python, Flask, scikit-learn, SQLite, Docker
"""

import os
import json
import sqlite3
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, g

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_disease_model.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")
DB_PATH = os.path.join(BASE_DIR, "data", "predictions.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "heart-disease-predictor-2026")

# ---------------------------------------------------------------------------
# Load ML artefacts
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

FEATURE_NAMES = metadata["feature_names"]

# Detailed info for each clinical parameter
FEATURE_INFO = {
    "age": {
        "label": "Age",
        "type": "number",
        "min": 1, "max": 120, "step": 1,
        "help": "Patient's age in years.",
        "detail": "Heart disease risk increases with age. In this dataset, patients range from 29 to 77 years old. Risk rises notably after 45 for men and 55 for women.",
        "example": "For example: 52",
        "normal": "Dataset range: 29–77 years",
        "placeholder": "e.g. 52",
    },
    "sex": {
        "label": "Biological Sex",
        "type": "select",
        "options": [("1", "Male"), ("0", "Female")],
        "help": "Biological sex of the patient.",
        "detail": "Men generally face a higher risk of heart disease than pre-menopausal women. This is one of the demographic factors the model considers.",
        "example": "Select Male or Female",
        "normal": "Both values are common in the dataset",
    },
    "cp": {
        "label": "Chest Pain Type",
        "type": "select",
        "options": [
            ("0", "Typical Angina"),
            ("1", "Atypical Angina"),
            ("2", "Non-anginal Pain"),
            ("3", "Asymptomatic"),
        ],
        "help": "What kind of chest pain does the patient experience?",
        "detail": "Typical angina is chest pain triggered by exertion and relieved by rest. Atypical angina has some but not all features of typical angina. Non-anginal pain is chest discomfort unlikely related to the heart. Asymptomatic means no chest pain at all.",
        "example": "If the patient feels pressure during exercise that goes away with rest, choose 'Typical Angina'",
        "normal": "Asymptomatic patients can still have heart disease",
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "type": "number",
        "min": 50, "max": 300, "step": 1,
        "help": "Blood pressure measured at rest (mm Hg).",
        "detail": "This is the systolic blood pressure (top number) measured when the patient is admitted. High resting blood pressure is a known risk factor for heart disease.",
        "example": "For example: 130",
        "normal": "Normal: below 120 mm Hg. Elevated: 120–129. High: 130+",
        "placeholder": "e.g. 130",
    },
    "chol": {
        "label": "Serum Cholesterol",
        "type": "number",
        "min": 50, "max": 600, "step": 1,
        "help": "Total cholesterol level in mg/dl.",
        "detail": "Serum cholesterol is the total cholesterol measured from a blood sample. Higher levels contribute to plaque buildup in arteries, increasing heart disease risk.",
        "example": "For example: 220",
        "normal": "Desirable: below 200 mg/dl. Borderline high: 200–239. High: 240+",
        "placeholder": "e.g. 220",
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "type": "select",
        "options": [("0", "No (≤ 120 mg/dl)"), ("1", "Yes (> 120 mg/dl)")],
        "help": "Is fasting blood sugar above 120 mg/dl?",
        "detail": "Fasting blood sugar above 120 mg/dl may indicate diabetes, which is a significant risk factor for cardiovascular disease. This is measured after an 8-hour fast.",
        "example": "Select 'Yes' if the patient's fasting glucose is above 120 mg/dl",
        "normal": "Normal fasting blood sugar: 70–100 mg/dl",
    },
    "restecg": {
        "label": "Resting ECG Results",
        "type": "select",
        "options": [
            ("0", "Normal"),
            ("1", "ST-T wave abnormality"),
            ("2", "Left ventricular hypertrophy"),
        ],
        "help": "Results of the resting electrocardiogram.",
        "detail": "An ECG records the electrical activity of the heart. ST-T wave abnormality can indicate reduced blood flow. Left ventricular hypertrophy means the heart's main pumping chamber has thickened walls, often due to high blood pressure.",
        "example": "Choose based on the ECG report. If no ECG is available, select 'Normal'",
        "normal": "'Normal' indicates no concerning findings on the ECG",
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "type": "number",
        "min": 50, "max": 250, "step": 1,
        "help": "Highest heart rate during a stress test.",
        "detail": "This is typically measured during a treadmill exercise stress test. A rough estimate of your max heart rate is 220 minus your age. Failure to reach at least 85% of the predicted max is considered abnormal.",
        "example": "For a 50-year-old, max is roughly 170 bpm. Enter the actual measured value.",
        "normal": "Typical range: 100–200 bpm during exercise",
        "placeholder": "e.g. 150",
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "type": "select",
        "options": [("0", "No"), ("1", "Yes")],
        "help": "Does the patient get chest pain during exercise?",
        "detail": "Exercise-induced angina is chest pain or discomfort occurring during physical activity. It happens when the heart muscle doesn't get enough blood during exertion and is a strong indicator of coronary artery disease.",
        "example": "Select 'Yes' if the patient reports chest tightness or pressure during the stress test",
        "normal": "'No' is the healthy response",
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "type": "number",
        "min": 0, "max": 10, "step": 0.1,
        "help": "How much the ST segment drops during exercise vs. rest.",
        "detail": "ST depression on an ECG during exercise can indicate the heart is not getting enough oxygen. It is measured in millimeters. Higher values suggest more significant ischemia (reduced blood flow to the heart).",
        "example": "For example: 1.5 (common values range from 0 to 4)",
        "normal": "0 means no ST depression. Values above 2 are concerning.",
        "placeholder": "e.g. 1.0",
    },
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "type": "select",
        "options": [
            ("0", "Upsloping"),
            ("1", "Flat"),
            ("2", "Downsloping"),
        ],
        "help": "Shape of the ST segment at peak exercise on ECG.",
        "detail": "Upsloping is generally normal. A flat slope is somewhat concerning. Downsloping is most strongly associated with heart disease. This comes from the stress test ECG reading.",
        "example": "Refer to the exercise ECG report for this value",
        "normal": "Upsloping is considered the most normal pattern",
    },
    "ca": {
        "label": "Number of Major Vessels (Fluoroscopy)",
        "type": "select",
        "options": [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")],
        "help": "How many of the 3 main coronary arteries show narrowing?",
        "detail": "During cardiac catheterization (fluoroscopy), a dye is injected to visualize the coronary arteries. This counts how many of the major vessels show significant narrowing or blockage. More blocked vessels means higher risk.",
        "example": "If the report says 'double-vessel disease', select 2",
        "normal": "0 means no vessels are blocked (healthiest)",
    },
    "thal": {
        "label": "Thalassemia",
        "type": "select",
        "options": [
            ("0", "Normal"),
            ("1", "Fixed Defect"),
            ("2", "Reversible Defect"),
        ],
        "help": "Result of the thallium stress test.",
        "detail": "A thallium stress test uses a radioactive tracer to show blood flow to the heart muscle. 'Normal' means blood flow is fine. 'Fixed defect' means a permanent area of reduced flow (possibly from a past heart attack). 'Reversible defect' means reduced flow during stress that returns to normal at rest — this suggests ischemia.",
        "example": "Choose based on the nuclear stress test report",
        "normal": "'Normal' indicates healthy blood flow to all areas",
    },
}

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def get_db():
    """Get a database connection for the current request."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def init_db():
    """Create the predictions table if it doesn't exist."""
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            age         REAL, sex REAL, cp REAL, trestbps REAL,
            chol        REAL, fbs REAL, restecg REAL, thalach REAL,
            exang       REAL, oldpeak REAL, slope REAL, ca REAL, thal REAL,
            prediction  INTEGER NOT NULL,
            probability REAL    NOT NULL,
            risk_level  TEXT    NOT NULL
        )
    """)
    db.commit()
    db.close()


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# Initialise DB on import
init_db()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Home / About page."""
    return render_template("index.html", metadata=metadata)


@app.route("/model-info")
def model_info():
    """Model metrics and methodology page."""
    return render_template("model_info.html", metadata=metadata)


@app.route("/parameters")
def parameters():
    """Detailed parameter explanations page."""
    return render_template(
        "parameters.html",
        feature_names=FEATURE_NAMES,
        feature_info=FEATURE_INFO,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """GET: show form. POST: compute prediction and show result."""
    if request.method == "GET":
        return render_template(
            "predict.html",
            feature_names=FEATURE_NAMES,
            feature_info=FEATURE_INFO,
            metadata=metadata,
        )

    # POST — compute prediction
    try:
        is_json = request.is_json
        values = []
        for feat in FEATURE_NAMES:
            if is_json:
                val = request.json.get(feat)
            else:
                val = request.form.get(feat)
            if val is None or val == "":
                error_msg = f"Missing input: {feat}"
                if is_json:
                    return jsonify({"error": error_msg}), 400
                return render_template(
                    "result.html", error=error_msg, metadata=metadata
                ), 400
            values.append(float(val))

        features = np.array(values).reshape(1, -1)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        confidence = probability if prediction == 1 else 1 - probability

        risk_level = (
            "High Risk" if probability > 0.7
            else "Moderate Risk" if probability > 0.4
            else "Low Risk"
        )

        result = {
            "prediction": prediction,
            "label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected",
            "probability": round(probability * 100, 2),
            "confidence": round(confidence * 100, 2),
            "risk_level": risk_level,
            "inputs": {FEATURE_INFO[f]["label"]: values[i] for i, f in enumerate(FEATURE_NAMES)},
        }

        # Save to SQLite
        db = get_db()
        db.execute(
            """INSERT INTO predictions
               (timestamp, age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal,
                prediction, probability, risk_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                datetime.now().isoformat(timespec="seconds"),
                *values,
                prediction,
                round(probability * 100, 2),
                risk_level,
            ],
        )
        db.commit()

        if is_json:
            return jsonify(result)

        return render_template("result.html", result=result, metadata=metadata)

    except Exception as exc:
        error_msg = f"Prediction error: {str(exc)}"
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        return render_template("result.html", error=error_msg, metadata=metadata), 500


@app.route("/history")
def history():
    """Show past predictions from the database."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 100"
    ).fetchall()
    return render_template("history.html", predictions=rows, feature_info=FEATURE_INFO)


@app.route("/api/health")
def health():
    """Health-check endpoint for Docker / monitoring."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
