"""
Unit and Integration Tests for Student Feedback Sentiment Analysis
===================================================================
Run with:
    pytest tests/ -v
"""

import os
import json
import pytest
import pandas as pd
import joblib


DATASET_PATH = "dataset.csv"
MODEL_PATH = os.path.join("model", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join("model", "evaluation_metrics.json")


def validate_input_text(text: str) -> tuple[bool, str]:
    """Helper validation function used by the application."""
    if text is None:
        return False, "Input cannot be None."
    cleaned = text.strip()
    if len(cleaned) == 0:
        return False, "Feedback text cannot be empty or only spaces."
    if len(cleaned) < 3:
        return False, "Feedback text is too short to analyze (minimum 3 characters required)."
    return True, cleaned


class TestDataset:
    """Test suite for dataset integrity and formatting."""

    def test_dataset_file_exists(self):
        assert os.path.exists(DATASET_PATH), f"Dataset file '{DATASET_PATH}' does not exist!"

    def test_dataset_columns_and_non_empty(self):
        df = pd.read_csv(DATASET_PATH)
        assert "feedback" in df.columns, "Column 'feedback' missing from dataset"
        assert "sentiment" in df.columns, "Column 'sentiment' missing from dataset"
        assert len(df) >= 100, f"Expected at least 100 rows, found {len(df)}"

    def test_dataset_no_null_values(self):
        df = pd.read_csv(DATASET_PATH)
        assert df["feedback"].isnull().sum() == 0, "Dataset contains null values in feedback column"
        assert df["sentiment"].isnull().sum() == 0, "Dataset contains null values in sentiment column"

    def test_dataset_classes(self):
        df = pd.read_csv(DATASET_PATH)
        unique_classes = set(df["sentiment"].str.strip().unique())
        expected_classes = {"Positive", "Negative", "Neutral"}
        assert unique_classes == expected_classes, f"Unexpected classes: {unique_classes}"

    def test_dataset_class_balance(self):
        df = pd.read_csv(DATASET_PATH)
        counts = df["sentiment"].value_counts()
        # Each class should have at least 40 samples
        for label, count in counts.items():
            assert count >= 40, f"Class {label} has too few samples ({count})"


class TestModelPipeline:
    """Test suite for trained model pipeline and inference."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        assert os.path.exists(MODEL_PATH), (
            f"Trained model '{MODEL_PATH}' not found. Run 'python train_model.py' first!"
        )
        return joblib.load(MODEL_PATH)

    def test_pipeline_classes(self, pipeline):
        expected_classes = {"Positive", "Negative", "Neutral"}
        assert set(pipeline.classes_) == expected_classes

    def test_predict_positive_sentiment(self, pipeline):
        text = "The professor is wonderful and explains every concept with exceptional clarity and patience!"
        pred = pipeline.predict([text])[0]
        assert pred == "Positive", f"Expected 'Positive', got '{pred}'"

    def test_predict_negative_sentiment(self, pipeline):
        text = "The lecturer was extremely rude, disorganized, and the exams were completely unfair."
        pred = pipeline.predict([text])[0]
        assert pred == "Negative", f"Expected 'Negative', got '{pred}'"

    def test_predict_neutral_sentiment(self, pipeline):
        text = "The class meets three times a week on Monday, Wednesday, and Friday according to the timetable."
        pred = pipeline.predict([text])[0]
        assert pred == "Neutral", f"Expected 'Neutral', got '{pred}'"

    def test_prediction_probabilities(self, pipeline):
        text = "Great hands-on coding exercises in Python really helped me understand data structures."
        probs = pipeline.predict_proba([text])[0]
        # Check that probabilities sum to 1.0 (with slight float margin)
        assert abs(sum(probs) - 1.0) < 1e-4
        assert len(probs) == 3
        # Ensure all probabilities are between 0 and 1
        for p in probs:
            assert 0.0 <= p <= 1.0


class TestInputValidation:
    """Test suite for input sanitization and error prevention."""

    def test_empty_string(self):
        is_valid, msg = validate_input_text("")
        assert not is_valid
        assert "empty" in msg.lower()

    def test_whitespace_string(self):
        is_valid, msg = validate_input_text("    \n\t  ")
        assert not is_valid
        assert "empty" in msg.lower()

    def test_too_short_string(self):
        is_valid, msg = validate_input_text("ok")
        assert not is_valid
        assert "too short" in msg.lower()

    def test_valid_input(self):
        is_valid, cleaned = validate_input_text("  The lectures were good.  ")
        assert is_valid
        assert cleaned == "The lectures were good."


class TestEvaluationMetrics:
    """Test suite for saved metrics file."""

    def test_metrics_file_exists_and_valid(self):
        assert os.path.exists(METRICS_PATH), f"Metrics file '{METRICS_PATH}' not found."
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "evaluation" in data
        assert "accuracy" in data["evaluation"]
        assert 0.5 <= data["evaluation"]["accuracy"] <= 1.0
        assert "confusion_matrix" in data["evaluation"]
        assert "classification_report" in data["evaluation"]
