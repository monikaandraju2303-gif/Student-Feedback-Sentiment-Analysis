"""
Student Feedback Sentiment Analysis - Model Training Script
=============================================================
This script loads the labelled feedback dataset, preprocesses the text,
trains a machine learning pipeline (TF-IDF + Logistic Regression),
evaluates it on a held-out test set, and saves the trained pipeline and
actual evaluation metrics for the Streamlit web application.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Configuration constants
DATASET_PATH = "dataset.csv"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_pipeline.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "evaluation_metrics.json")
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load dataset and perform data cleaning and validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found!")

    df = pd.read_csv(filepath)

    print("=" * 65)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 65)
    print(f"Total raw rows loaded: {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    required_columns = {"feedback", "sentiment"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Check and handle missing / null values
    initial_count = len(df)
    df = df.dropna(subset=["feedback", "sentiment"]).copy()
    df["feedback"] = df["feedback"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    # Filter out empty feedback strings
    df = df[df["feedback"].str.len() > 0]
    cleaned_count = len(df)

    if cleaned_count < initial_count:
        print(f"Dropped {initial_count - cleaned_count} rows with missing or empty values.")

    print(f"Valid clean samples for training: {cleaned_count}")
    print("\nClass Distribution:")
    class_counts = df["sentiment"].value_counts().to_dict()
    for sentiment, count in class_counts.items():
        print(f"  - {sentiment}: {count} ({count / cleaned_count * 100:.1f}%)")

    valid_classes = {"Positive", "Negative", "Neutral"}
    if not set(class_counts.keys()).issubset(valid_classes):
        print(f"Warning: Found unexpected sentiment classes: {set(class_counts.keys()) - valid_classes}")

    return df


def build_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline with:
    1. TF-IDF Vectorizer (n-grams 1 to 2, negation-aware stop words, sublinear TF)
    2. Logistic Regression (multinomial with balanced class weights)
    """
    # In sentiment analysis, standard stop words like 'not', 'no', 'never', 'without'
    # are critical polarity markers (e.g. "not helpful" vs "helpful").
    # We retain these key polarity words while filtering standard English stop words.
    polarity_words = {"not", "no", "never", "without", "neither", "too", "very", "zero", "least", "less"}
    custom_stop_words = list(ENGLISH_STOP_WORDS.difference(polarity_words))

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=custom_stop_words,
            ngram_range=(1, 2),  # unigrams and bigrams
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])
    return pipeline


def train_and_evaluate():
    """Main function to train and evaluate the sentiment model."""
    # 1. Load data
    df = load_and_validate_data(DATASET_PATH)
    X = df["feedback"]
    y = df["sentiment"]

    # 2. Train-Test Split (Reproducible & Stratified)
    print("\n" + "=" * 65)
    print("STEP 2: STRATIFIED TRAIN-TEST SPLIT")
    print("=" * 65)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Training set size: {len(X_train)} samples ({100 - TEST_SIZE * 100:.0f}%)")
    print(f"Testing set size:  {len(X_test)} samples ({TEST_SIZE * 100:.0f}%)")
    print(f"Random seed used:  {RANDOM_STATE} (reproducible)")

    # 3. Build & Train Pipeline
    print("\n" + "=" * 65)
    print("STEP 3: MODEL TRAINING (TF-IDF + LOGISTIC REGRESSION)")
    print("=" * 65)
    pipeline = build_pipeline()
    print("Training pipeline exclusively on the training set...")
    pipeline.fit(X_train, y_train)
    print("Training completed successfully! (No test data was seen by the model)")

    # 4. Evaluate on Held-out Test Data
    print("\n" + "=" * 65)
    print("STEP 4: MODEL EVALUATION ON HELD-OUT TEST DATA")
    print("=" * 65)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classes = sorted(list(y.unique()))

    report_str = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(report_str)

    print("Confusion Matrix:")
    print(f"Classes (Rows = Actual, Columns = Predicted): {classes}")
    print(conf_matrix)

    # 5. Save Artifacts for Streamlit Web App
    print("\n" + "=" * 65)
    print("STEP 5: SAVING MODEL & METRICS ARTIFACTS")
    print("=" * 65)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the complete pipeline
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved trained pipeline to: {MODEL_FILE}")

    # Prepare metrics dictionary to display in the Streamlit UI
    metrics_data = {
        "model_name": "TF-IDF + Logistic Regression",
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": RANDOM_STATE,
        "test_size_ratio": TEST_SIZE,
        "dataset_stats": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": df["sentiment"].value_counts().to_dict()
        },
        "evaluation": {
            "accuracy": round(float(accuracy), 4),
            "classes": classes,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": report_dict
        },
        "parameters": {
            "tfidf_ngram_range": [1, 2],
            "tfidf_sublinear_tf": True,
            "classifier_type": "LogisticRegression",
            "classifier_c": 1.0,
            "classifier_class_weight": "balanced"
        }
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved actual evaluation metrics to: {METRICS_FILE}")

    print("\n" + "=" * 65)
    print("IMPORTANT METHODOLOGY NOTE:")
    print(f"This accuracy is evaluated on the held-out test split of {len(X_test)} student")
    print("feedback samples. The model's real-world accuracy depends heavily on")
    print("the size, diversity, and linguistic richness of feedback collected.")
    print("=" * 65)

    return pipeline, metrics_data


if __name__ == "__main__":
    train_and_evaluate()
