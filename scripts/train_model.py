#!/usr/bin/env python3
"""
Heart Disease Prediction — Model Training Pipeline
====================================================
Author: Sandeep Kumar
Description: Trains and evaluates multiple ML classifiers on the Cleveland
             Heart Disease dataset (UCI ML Repository). The best model is
             serialised for deployment with the Flask web application.

Dataset Features:
    age      — Age in years
    sex      — Sex (1 = male; 0 = female)
    cp       — Chest pain type (0–3)
    trestbps — Resting blood pressure (mm Hg)
    chol     — Serum cholesterol (mg/dl)
    fbs      — Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    restecg  — Resting ECG results (0–2)
    thalach  — Maximum heart rate achieved
    exang    — Exercise-induced angina (1 = yes; 0 = no)
    oldpeak  — ST depression induced by exercise relative to rest
    slope    — Slope of the peak exercise ST segment (0–2)
    ca       — Number of major vessels coloured by fluoroscopy (0–3)
    thal     — Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)
    target   — Diagnosis (1 = heart disease; 0 = no heart disease)
"""

import os
import sys
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# ---------------------------------------------------------------------------
# 1. Load & Explore
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Heart Disease Prediction — Model Training Pipeline")
print("  Author: Sandeep Kumar")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n📂 Dataset loaded: {DATA_PATH}")
print(f"   Shape: {df.shape}")
print(f"   Missing values:\n{df.isnull().sum().to_string()}\n")

X = df[FEATURE_NAMES].values
y = df["target"].values

print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ---------------------------------------------------------------------------
# 2. Train / Test Split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y,
)
print(f"\n   Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 3. Define Candidate Models (Pipeline with StandardScaler)
# ---------------------------------------------------------------------------
candidates = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42)),
    ]),
    "K-Nearest Neighbours": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(random_state=42)),
    ]),
    "AdaBoost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", AdaBoostClassifier(n_estimators=200, random_state=42, algorithm="SAMME")),
    ]),
}

# ---------------------------------------------------------------------------
# 4. Cross-Validation Evaluation
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = {}

print("\n" + "-" * 60)
print("  10-Fold Stratified Cross-Validation Results")
print("-" * 60)
print(f"  {'Model':<28} {'Accuracy':>10} {'Std':>8}")
print("-" * 60)

for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    results[name] = {
        "mean_cv_accuracy": round(float(scores.mean()), 4),
        "std_cv_accuracy": round(float(scores.std()), 4),
    }
    print(f"  {name:<28} {scores.mean():.4f}     ±{scores.std():.4f}")

# ---------------------------------------------------------------------------
# 5. Select Best & Hyperparameter Tuning
# ---------------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["mean_cv_accuracy"])
print(f"\n✅ Best cross-validation model: {best_name}")

# Fine-tune the best model with GridSearchCV
param_grids = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__solver": ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 10, 15],
        "clf__min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
    },
    "Support Vector Machine": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    },
    "K-Nearest Neighbours": {
        "clf__n_neighbors": [3, 5, 7, 9, 11],
        "clf__weights": ["uniform", "distance"],
    },
    "Decision Tree": {
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_split": [2, 5, 10],
    },
    "AdaBoost": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    },
}

print(f"\n🔧 Hyperparameter tuning ({best_name}) with GridSearchCV …")
grid = GridSearchCV(
    candidates[best_name],
    param_grids[best_name],
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0,
)
grid.fit(X_train, y_train)
best_pipe = grid.best_estimator_
print(f"   Best params: {grid.best_params_}")
print(f"   Best CV accuracy: {grid.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 6. Evaluate on Hold-Out Test Set
# ---------------------------------------------------------------------------
y_pred = best_pipe.predict(X_test)
y_prob = best_pipe.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
}

print("\n" + "-" * 60)
print("  Hold-Out Test Set Evaluation")
print("-" * 60)
for k, v in test_metrics.items():
    print(f"  {k:<15} {v:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))

# ---------------------------------------------------------------------------
# 7. Generate Visualisations
# ---------------------------------------------------------------------------
# — Confusion Matrix —
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
cm_path = os.path.join(STATIC_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"\n📊 Confusion matrix saved: {cm_path}")

# — Model Comparison Bar Chart —
model_names = list(results.keys())
accuracies = [results[n]["mean_cv_accuracy"] for n in model_names]

fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.barh(model_names, accuracies, color=sns.color_palette("viridis", len(model_names)))
ax2.set_xlabel("Mean CV Accuracy")
ax2.set_title("Model Comparison — 10-Fold Cross-Validation")
ax2.set_xlim(0.6, 1.0)
for bar, acc in zip(bars, accuracies):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{acc:.4f}", va="center", fontsize=10)
plt.tight_layout()
comp_path = os.path.join(STATIC_DIR, "model_comparison.png")
fig2.savefig(comp_path, dpi=150)
plt.close(fig2)
print(f"📊 Model comparison chart saved: {comp_path}")

# — Feature Correlation Heatmap —
fig3, ax3 = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax3)
ax3.set_title("Feature Correlation Heatmap")
plt.tight_layout()
corr_path = os.path.join(STATIC_DIR, "correlation_heatmap.png")
fig3.savefig(corr_path, dpi=150)
plt.close(fig3)
print(f"📊 Correlation heatmap saved: {corr_path}")

# ---------------------------------------------------------------------------
# 8. Save Model Artefacts
# ---------------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "heart_disease_model.pkl")
joblib.dump(best_pipe, model_path)
print(f"\n💾 Model saved: {model_path}")

# Save metadata
metadata = {
    "project": "Heart Disease Prediction System",
    "author": "Sandeep Kumar",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "dataset": "Cleveland Heart Disease — UCI ML Repository",
    "dataset_shape": list(df.shape),
    "best_model": best_name,
    "best_params": grid.best_params_,
    "cross_validation": results,
    "test_metrics": test_metrics,
    "feature_names": FEATURE_NAMES,
    "sklearn_version": __import__("sklearn").__version__,
    "python_version": sys.version,
}

meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"📄 Metadata saved: {meta_path}")

print("\n" + "=" * 60)
print("  Training pipeline completed successfully! ✅")
print("=" * 60)
