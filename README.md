# ❤️ CardioPredict — Heart Disease Prediction System

> **A Machine Learning–Based Web Application for Heart Disease Risk Assessment**

**Author:** Sandeep Kumar  
**Technology Stack:** Python · Flask · scikit-learn · Docker · Git  
**Dataset:** Cleveland Heart Disease — UCI Machine Learning Repository  
**License:** MIT

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Dataset Description](#dataset-description)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Project Structure](#project-structure)
8. [Installation & Setup](#installation--setup)
9. [Running with Docker](#running-with-docker)
10. [Running Locally (Without Docker)](#running-locally-without-docker)
11. [Testing](#testing)
12. [API Documentation](#api-documentation)
13. [Model Performance](#model-performance)
14. [Screenshots](#screenshots)
15. [Future Enhancements](#future-enhancements)
16. [References](#references)
17. [Acknowledgements](#acknowledgements)

---

## 📖 Project Overview

Heart disease is one of the leading causes of mortality worldwide, accounting for approximately **17.9 million deaths** each year (World Health Organisation). Early detection is critical for effective treatment and improved patient outcomes.

**CardioPredict** is an intelligent web application that leverages machine learning to predict the likelihood of heart disease based on 13 clinical parameters. Built as a final-year academic project, this system demonstrates the practical application of data science and software engineering methodologies in healthcare.

### Objectives

- Develop a robust ML classification model for heart disease prediction
- Compare and evaluate 7 different classification algorithms using stratified cross-validation
- Deploy the best model as an accessible, production-ready web application
- Containerize the application using Docker for reproducibility and portability
- Maintain comprehensive version control using Git
- Implement thorough end-to-end testing for quality assurance

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **ML Prediction Engine** | Predicts heart disease risk using a trained Random Forest classifier |
| **Interactive Web Interface** | Clean, professional light-themed UI with form-based input |
| **Interactive Help Panels** | Expandable info (ⓘ) panels on every form field with clinical explanations, examples, and normal ranges |
| **Probability Scoring** | Returns confidence percentage and risk-level categorisation (Low / Moderate / High) |
| **Model Transparency** | Dedicated model info page with metrics, visualisations, and feature descriptions |
| **REST API** | JSON endpoint for programmatic access (`POST /predict`) |
| **Health Check** | Monitoring endpoint (`GET /api/health`) |
| **Print Reports** | Browser-optimised print stylesheet for result pages |
| **Medical Disclaimer** | Proper disclaimers on all prediction outputs |
| **Docker Support** | Full containerisation with Docker and Docker Compose |
| **Security** | Non-root container user, secret key configuration, input validation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                     Client (Browser)                 │
│   ┌──────────────────────────────────────────────┐  │
│   │  Responsive Web UI (HTML/CSS/JS)             │  │
│   │  - Prediction Form (13 clinical features)    │  │
│   │  - Result Display (probability gauge)        │  │
│   │  - Model Info & About Pages                  │  │
│   └──────────────────────┬───────────────────────┘  │
└──────────────────────────┼──────────────────────────┘
                           │ HTTP (Form / JSON)
┌──────────────────────────┼──────────────────────────┐
│                   Docker Container                   │
│   ┌──────────────────────┴───────────────────────┐  │
│   │          Gunicorn WSGI Server                 │  │
│   │  ┌──────────────────────────────────────────┐│  │
│   │  │          Flask Application                ││  │
│   │  │  ┌────────────────┐  ┌─────────────────┐ ││  │
│   │  │  │  Route Handler │  │  Input Validator │ ││  │
│   │  │  └───────┬────────┘  └─────────────────┘ ││  │
│   │  │          │                                ││  │
│   │  │  ┌───────▼────────┐  ┌─────────────────┐ ││  │
│   │  │  │  ML Pipeline   │──│  StandardScaler  │ ││  │
│   │  │  │  (joblib .pkl) │  │  + RandomForest  │ ││  │
│   │  │  └────────────────┘  └─────────────────┘ ││  │
│   │  └──────────────────────────────────────────┘│  │
│   └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.9+ | Core programming language |
| **Web Framework** | Flask | 3.0.3 | HTTP request handling and routing |
| **ML Library** | scikit-learn | 1.5.2 | Model training and inference |
| **Data Processing** | Pandas | 2.1.4 | Dataset loading and manipulation |
| **Numerical** | NumPy | 1.26.4 | Array operations |
| **Visualisation** | Matplotlib + Seaborn | 3.8.4 / 0.13.2 | Charts and heatmaps |
| **Model Persistence** | Joblib | 1.4.2 | Model serialisation |
| **WSGI Server** | Gunicorn | 22.0.0 | Production HTTP server |
| **Containerisation** | Docker | Latest | Application packaging |
| **Orchestration** | Docker Compose | Latest | Container management |
| **Version Control** | Git | Latest | Source code versioning |
| **Testing** | pytest | 8.3.4 | Unit and integration tests |

---

## 📊 Dataset Description

### Cleveland Heart Disease Dataset (UCI ML Repository)

| Property | Value |
|----------|-------|
| **Source** | UCI Machine Learning Repository |
| **Original Creators** | Hungarian Institute of Cardiology; University Hospital, Zurich; University Hospital, Basel; V.A. Medical Center, Long Beach |
| **Samples** | 303 patient records |
| **Features** | 13 clinical attributes |
| **Target** | Binary (1 = Heart Disease, 0 = No Heart Disease) |
| **Class Distribution** | 165 positive (54.5%) / 138 negative (45.5%) |

### Feature Descriptions

| # | Feature | Description | Type | Range |
|---|---------|-------------|------|-------|
| 1 | `age` | Patient age in years | Continuous | 29–77 |
| 2 | `sex` | Biological sex (1 = male, 0 = female) | Binary | 0–1 |
| 3 | `cp` | Chest pain type | Categorical | 0–3 |
| 4 | `trestbps` | Resting blood pressure (mm Hg) | Continuous | 94–200 |
| 5 | `chol` | Serum cholesterol (mg/dl) | Continuous | 126–564 |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 0–1 |
| 7 | `restecg` | Resting ECG results | Categorical | 0–2 |
| 8 | `thalach` | Maximum heart rate achieved | Continuous | 71–202 |
| 9 | `exang` | Exercise-induced angina | Binary | 0–1 |
| 10 | `oldpeak` | ST depression (exercise vs. rest) | Continuous | 0–6.2 |
| 11 | `slope` | Slope of peak exercise ST segment | Categorical | 0–2 |
| 12 | `ca` | Major vessels coloured by fluoroscopy | Discrete | 0–3 |
| 13 | `thal` | Thalassemia type | Categorical | 0–2 |

### Chest Pain Types (cp)
- **0** — Typical Angina
- **1** — Atypical Angina
- **2** — Non-anginal Pain
- **3** — Asymptomatic

---

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- No missing values in the dataset (verified programmatically)
- Feature standardisation using `StandardScaler` (integrated into the pipeline)
- 80/20 stratified train-test split (random_state=42 for reproducibility)

### 2. Model Selection
Seven classification algorithms were evaluated using **10-fold Stratified Cross-Validation**:

| Model | Mean CV Accuracy | Std Deviation |
|-------|:---------------:|:-------------:|
| **Random Forest** | **84.32%** | ±7.69% |
| Logistic Regression | 82.67% | ±10.51% |
| K-Nearest Neighbours | 80.58% | ±7.13% |
| Support Vector Machine | 80.15% | ±8.43% |
| AdaBoost | 78.97% | ±6.87% |
| Gradient Boosting | 78.58% | ±10.04% |
| Decision Tree | 73.95% | ±7.26% |

### 3. Hyperparameter Tuning
The best model (**Random Forest**) was fine-tuned using `GridSearchCV`:

| Parameter | Optimised Value |
|-----------|:---------------:|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `min_samples_split` | 2 |

### 4. Test Set Performance

| Metric | Score |
|--------|:-----:|
| **Accuracy** | 81.97% |
| **Precision** | 76.19% |
| **Recall** | 96.97% |
| **F1 Score** | 85.33% |
| **ROC AUC** | 91.56% |

> **Note:** The model achieves high recall (96.97%), meaning it correctly identifies almost all patients with heart disease — a critical metric in medical diagnostics where false negatives can be life-threatening.

---

## 📁 Project Structure

```
HeartDiseaseProject/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies (pinned versions)
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose orchestration
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker build context exclusions
├── README.md                   # Project documentation (this file)
│
├── data/
│   └── heart.csv               # Cleveland Heart Disease dataset
│
├── models/
│   ├── heart_disease_model.pkl # Trained ML pipeline (StandardScaler + RandomForest)
│   └── model_metadata.json     # Training metadata, metrics, and parameters
│
├── scripts/
│   └── train_model.py          # Model training and evaluation pipeline
│
├── static/
│   ├── css/
│   │   └── style.css           # Application stylesheet (clean light theme)
│   ├── js/
│   │   └── main.js             # Client-side JavaScript (form validation, animations)
│   ├── confusion_matrix.png    # Confusion matrix visualisation
│   ├── model_comparison.png    # Model comparison bar chart
│   └── correlation_heatmap.png # Feature correlation heatmap
│
├── templates/
│   ├── base.html               # Base template (nav, footer, meta tags)
│   ├── index.html              # Home page with prediction form
│   ├── result.html             # Prediction result display
│   ├── about.html              # About project page
│   └── model_info.html         # Model metrics and visualisations
│
└── tests/
    ├── __init__.py
    └── test_app.py             # Comprehensive test suite (23 tests)
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.9+** (tested with 3.9.6 and 3.11)
- **Docker** and **Docker Compose** (for containerised deployment)
- **Git** (for version control)

---

## 🐳 Running with Docker

This is the **recommended** method for running the application.

### 1. Clone the Repository

```bash
git clone https://github.com/sandeeep-kr/HeartDiseaseProject.git
cd HeartDiseaseProject
```

### 2. Build and Run with Docker Compose

```bash
docker compose up --build
```

### 3. Access the Application

Open your browser and navigate to: **http://localhost:5000**

### 4. Stop the Application

```bash
docker compose down
```

### Docker Build Details
- **Base Image:** `python:3.11-slim-bookworm`
- **Security:** Runs as non-root user (`appuser`)
- **Health Check:** Automated health monitoring every 30 seconds
- **Server:** Gunicorn with 2 workers and 2 threads

---

## 💻 Running Locally (Without Docker)

### 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python scripts/train_model.py
```

### 4. Run the Application

```bash
python app.py
```

Or with a custom port:

```bash
PORT=8080 python app.py
```

### 5. Access the Application

Open your browser: **http://localhost:5000** (or the port you specified)

---

## 🧪 Testing

The project includes a comprehensive test suite with **23 test cases** across 5 categories:

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Test Categories

| Category | Tests | Description |
|----------|:-----:|-------------|
| **TestModel** | 6 | Model loading, prediction shape, probability ranges, batch processing |
| **TestRoutes** | 9 | All Flask routes (GET/POST), form submission, JSON API, error handling |
| **TestContent** | 3 | Author attribution, medical disclaimer, form field completeness |
| **TestEdgeCases** | 3 | Extreme values, zero inputs, 404 handling |
| **TestMetadata** | 2 | Metadata file existence and structure validation |

### Test Output

```
tests/test_app.py::TestModel::test_model_loads                    PASSED
tests/test_app.py::TestModel::test_model_has_predict              PASSED
tests/test_app.py::TestModel::test_model_predicts_binary          PASSED
tests/test_app.py::TestModel::test_model_probability_range        PASSED
tests/test_app.py::TestModel::test_model_accepts_13_features      PASSED
tests/test_app.py::TestModel::test_model_batch_prediction         PASSED
tests/test_app.py::TestRoutes::test_index_page                    PASSED
tests/test_app.py::TestRoutes::test_about_page                    PASSED
tests/test_app.py::TestRoutes::test_model_info_page               PASSED
tests/test_app.py::TestRoutes::test_health_endpoint               PASSED
tests/test_app.py::TestRoutes::test_predict_form_submission       PASSED
tests/test_app.py::TestRoutes::test_predict_at_risk               PASSED
tests/test_app.py::TestRoutes::test_predict_missing_field         PASSED
tests/test_app.py::TestRoutes::test_predict_json_api              PASSED
tests/test_app.py::TestRoutes::test_predict_json_missing_field    PASSED
tests/test_app.py::TestContent::test_author_in_footer             PASSED
tests/test_app.py::TestContent::test_disclaimer_on_result         PASSED
tests/test_app.py::TestContent::test_form_has_all_fields          PASSED
tests/test_app.py::TestEdgeCases::test_extreme_age                PASSED
tests/test_app.py::TestEdgeCases::test_zero_values                PASSED
tests/test_app.py::TestEdgeCases::test_404_page                   PASSED
tests/test_app.py::TestMetadata::test_metadata_exists             PASSED
tests/test_app.py::TestMetadata::test_metadata_structure          PASSED

============================== 23 passed ==============================
```

---

## 🔌 API Documentation

### Health Check

```
GET /api/health
```

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

### Predict (JSON)

```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
    "age": "55",
    "sex": "1",
    "cp": "3",
    "trestbps": "140",
    "chol": "260",
    "fbs": "0",
    "restecg": "1",
    "thalach": "130",
    "exang": "1",
    "oldpeak": "2.5",
    "slope": "1",
    "ca": "2",
    "thal": "1"
}
```

**Response:**
```json
{
    "prediction": 0,
    "label": "No Heart Disease Detected",
    "probability": 30.37,
    "confidence": 69.63,
    "risk_level": "Low Risk",
    "inputs": { ... }
}
```

### Predict (Form)

```
POST /predict
Content-Type: application/x-www-form-urlencoded
```

Returns an HTML result page with the prediction visualisation.

---

## 📈 Model Performance

### Confusion Matrix

The confusion matrix on the hold-out test set (61 samples) shows:

|  | Predicted: No Disease | Predicted: Disease |
|--|:---------------------:|:------------------:|
| **Actual: No Disease** | 18 (TN) | 10 (FP) |
| **Actual: Disease** | 1 (FN) | 32 (TP) |

- **True Positive Rate (Recall):** 96.97% — The model correctly identifies 32 out of 33 patients with heart disease
- **False Negative Rate:** 3.03% — Only 1 patient with heart disease was missed
- **Specificity:** 64.29% — The model correctly identifies 18 out of 28 healthy patients

### Model Comparison

All 7 models were evaluated using 10-fold stratified cross-validation. Random Forest achieved the highest mean accuracy (84.32%) with reasonable variance.

### Feature Correlation

The correlation heatmap reveals key relationships:
- **cp** (chest pain type) and **thalach** (max heart rate) show strong positive correlation with the target
- **exang** (exercise-induced angina) and **oldpeak** (ST depression) show negative correlation with the target
- Features are not highly multi-collinear, supporting the use of all 13 features

---

## 🖼️ Screenshots

The application features a clean, professional light-themed design:
- **Prediction page** with key model statistics and a two-column form layout
- **Interactive info panels** — click ⓘ on any field for clinical explanations, examples, and normal ranges
- **Result page** with probability bar, risk-level badge, parameters table, and medical disclaimer
- **Model info page** with performance metrics, comparison charts, and confusion matrix
- **About page** with project description, dataset info, technology stack, and references

---

## 🔮 Future Enhancements

1. **Deep Learning Models** — Implement neural network classifiers (e.g., TensorFlow/Keras) for comparison
2. **Additional Datasets** — Integrate larger datasets (e.g., Hungarian, Swiss, VA Long Beach) for improved generalisability
3. **Feature Engineering** — Derive new features from existing ones (e.g., BMI, pulse pressure)
4. **Model Explainability** — Integrate SHAP or LIME for individual prediction explanations
5. **User Authentication** — Add login system for storing patient histories
6. **Database Integration** — PostgreSQL for persistent storage of predictions
7. **CI/CD Pipeline** — Automated testing and deployment with GitHub Actions
8. **Mobile Application** — React Native or Flutter frontend

---

## 📚 References

1. Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., ... & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.

2. UCI Machine Learning Repository — Heart Disease Data Set. https://archive.ics.uci.edu/ml/datasets/heart+disease

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

5. World Health Organization. (2021). Cardiovascular Diseases (CVDs) Fact Sheet. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

---

## 🙏 Acknowledgements

- **UCI Machine Learning Repository** for providing the Cleveland Heart Disease dataset
- **scikit-learn** community for the machine learning library
- **Flask** community for the web framework
- **Docker** for containerisation tools

---

<div align="center">

**Created by Sandeep Kumar** · © 2026

*This project is intended for educational and research purposes only. It does not constitute medical advice.*

</div>
