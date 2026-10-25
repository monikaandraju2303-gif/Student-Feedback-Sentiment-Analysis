"""
Student Feedback Sentiment Analysis - Streamlit Web Application
================================================================
A beginner-friendly, interactive web application that classifies student
feedback into Positive, Negative, or Neutral sentiment using a locally
trained scikit-learn machine learning pipeline (TF-IDF + Logistic Regression).
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Configuration
st.set_page_config(
    page_title="Student Feedback Sentiment Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration Paths
MODEL_PATH = os.path.join("model", "sentiment_pipeline.joblib")
METRICS_PATH = os.path.join("model", "evaluation_metrics.json")
DATASET_PATH = "dataset.csv"

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Global Font & Header adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-blue { background-color: rgba(99, 102, 241, 0.2); color: #818CF8; border: 1px solid #6366F1; }
    .badge-green { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .badge-purple { background-color: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid #A855F7; }

    /* Prediction Cards */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .card-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border: 1px solid #10B981;
    }
    .card-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
        border: 1px solid #EF4444;
    }
    .card-neutral {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.05));
        border: 1px solid #3B82F6;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background-color: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #F59E0B;
        padding: 0.85rem 1.2rem;
        border-radius: 0.5rem;
        font-size: 0.88rem;
        color: #CBD5E1;
        margin-bottom: 1.5rem;
    }

    /* Metric cards */
    .metric-container {
        background: #1E293B;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the trained machine learning pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_metrics():
    """Load the actual evaluation metrics saved during model training."""
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None


# Load artifacts
pipeline = load_pipeline()
metrics_data = load_metrics()

# Sidebar
with st.sidebar:
    st.markdown("### 🎓 Project Overview")
    st.write(
        "**Student Feedback Sentiment Analysis** is a B.Tech CSE AI mini-capstone project "
        "designed to categorize student feedback into **Positive**, **Negative**, or **Neutral** "
        "using classical Machine Learning and Natural Language Processing."
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if pipeline is not None:
        st.success("✅ ML Model Loaded: `sentiment_pipeline.joblib`")
    else:
        st.error("❌ Model Missing! Please run `python train_model.py`")

    if metrics_data is not None:
        eval_acc = metrics_data["evaluation"]["accuracy"] * 100
        st.info(f"📊 Evaluated Test Accuracy: **{eval_acc:.1f}%**")
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **Language:** Python 3.9+
    - **Interface:** Streamlit
    - **Data Processing:** pandas
    - **Feature Extraction:** TF-IDF Vectorizer
    - **Classifier:** Logistic Regression
    - **Serialization:** joblib
    - **Testing:** pytest
    """)

    st.markdown("---")
    st.markdown("### 👥 4-Member Team Roles")
    st.caption("1. **Data Engineer:** Dataset collection & cleaning")
    st.caption("2. **ML Engineer:** TF-IDF pipeline & model training")
    st.caption("3. **App Developer:** Streamlit frontend & UI/UX")
    st.caption("4. **QA & DevOps:** Unit tests, Git & deployment")

# Top Header
st.markdown('<div class="main-title">🎓 Student Feedback Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Beginner-Friendly Machine Learning Mini-Capstone '
    'built with TF-IDF, Logistic Regression, and Streamlit</div>',
    unsafe_allow_html=True
)

# Badges
st.markdown("""
<div>
    <span class="badge-pill badge-blue">Natural Language Processing</span>
    <span class="badge-pill badge-purple">TF-IDF Feature Extraction</span>
    <span class="badge-pill badge-green">Logistic Regression Classifier</span>
    <span class="badge-pill badge-blue">Streamlit Web Interface</span>
</div>
""", unsafe_allow_html=True)

# Machine Learning Notice Banner
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Important Machine Learning Notice:</strong> 
    Predictions and probabilities shown below are calculated by a locally trained mathematical model 
    (TF-IDF + Logistic Regression). This is a statistical estimation based on patterns in the academic training data, 
    <strong>not a guaranteed human judgment</strong>. Human review is always recommended for official evaluations.
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_metrics, tab_theory, tab_limitations = st.tabs([
    "🔮 Predict Sentiment",
    "📊 Model Performance & Dataset",
    "💡 How TF-IDF & Logistic Regression Work",
    "⚠️ Limitations & Ethics"
])

# ==========================================
# TAB 1: PREDICT SENTIMENT
# ==========================================
with tab_predict:
    if pipeline is None:
        st.error(
            "⚠️ Trained model file `model/sentiment_pipeline.joblib` was not found!\n\n"
            "**How to fix:** Open your terminal and run:\n"
            "```bash\n"
            "python train_model.py\n"
            "```\n"
            "Then refresh this webpage."
        )
    else:
        st.markdown("### Enter Student Feedback to Analyze")
        st.write("Type or paste student feedback below, or click any sample example to test the model.")

        # Sample Feedback Quick Chips
        st.markdown("**Sample Feedback Examples:**")
        sample_cols = st.columns(3)
        
        sample_positive = "The professor explains difficult concepts step by step with great examples and patience."
        sample_negative = "The lab computers were constantly crashing and the instructor was rude and unhelpful."
        sample_neutral = "The class meets three times a week on Monday, Wednesday, and Friday as per the schedule."

        if "feedback_area" not in st.session_state:
            st.session_state["feedback_area"] = ""

        if sample_cols[0].button("🟢 Try Positive Sample", use_container_width=True):
            st.session_state["feedback_area"] = sample_positive
            st.rerun()

        if sample_cols[1].button("🔴 Try Negative Sample", use_container_width=True):
            st.session_state["feedback_area"] = sample_negative
            st.rerun()

        if sample_cols[2].button("🔵 Try Neutral Sample", use_container_width=True):
            st.session_state["feedback_area"] = sample_neutral
            st.rerun()

        # User Text Input
        user_feedback = st.text_area(
            label="Feedback Text",
            height=120,
            placeholder="e.g., 'The course assignments were very challenging but helped me understand Python programming deeply...'",
            key="feedback_area"
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        predict_clicked = col_btn1.button("🔍 Predict Sentiment", type="primary", use_container_width=True)
        clear_clicked = col_btn2.button("🗑️ Clear Input", use_container_width=False)

        if clear_clicked:
            st.session_state["feedback_area"] = ""
            st.rerun()

        # Handle Prediction
        if predict_clicked:
            cleaned_input = user_feedback.strip()

            if not cleaned_input:
                st.warning("⚠️ Please enter some student feedback text before clicking Predict.")
            elif len(cleaned_input) < 3:
                st.warning("⚠️ The input text is too short to analyze. Please enter a complete phrase or sentence.")
            else:
                # Real locally trained model prediction
                predicted_class = pipeline.predict([cleaned_input])[0]
                probabilities = pipeline.predict_proba([cleaned_input])[0]
                classes = list(pipeline.classes_)

                # Map class probabilities
                prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
                confidence = prob_dict[predicted_class] * 100

                # Visual Card Styling based on sentiment
                if predicted_class == "Positive":
                    card_class = "card-positive"
                    emoji = "😊"
                    color_hex = "#10B981"
                    description = "The feedback expresses satisfaction, appreciation, or positive encouragement."
                elif predicted_class == "Negative":
                    card_class = "card-negative"
                    emoji = "🙁"
                    color_hex = "#EF4444"
                    description = "The feedback expresses dissatisfaction, frustration, or areas requiring urgent improvement."
                else:
                    card_class = "card-neutral"
                    emoji = "😐"
                    color_hex = "#3B82F6"
                    description = "The feedback is factual, descriptive, or balanced without strong emotional polarity."

                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="font-size: 2.2rem; margin-right: 0.75rem;">{emoji}</span>
                            <span style="font-size: 1.8rem; font-weight: 700; color: {color_hex};">
                                {predicted_class} Sentiment
                            </span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; color: #94A3B8;">Model Confidence</div>
                            <div style="font-size: 1.6rem; font-weight: 700; color: {color_hex};">{confidence:.1f}%</div>
                        </div>
                    </div>
                    <p style="margin-top: 0.75rem; color: #CBD5E1; margin-bottom: 0;">{description}</p>
                </div>
                """, unsafe_allow_html=True)

                # Class Probabilities Breakdown
                st.markdown("#### 📊 Model Probability Distribution")
                st.write("The Logistic Regression algorithm assigns a probability score to all three classes:")

                prob_cols = st.columns(3)
                for i, cls in enumerate(classes):
                    score = prob_dict[cls] * 100
                    if cls == "Positive":
                        prob_cols[i].metric(label="🟢 Positive Probability", value=f"{score:.1f}%")
                        prob_cols[i].progress(min(score / 100.0, 1.0))
                    elif cls == "Negative":
                        prob_cols[i].metric(label="🔴 Negative Probability", value=f"{score:.1f}%")
                        prob_cols[i].progress(min(score / 100.0, 1.0))
                    else:
                        prob_cols[i].metric(label="🔵 Neutral Probability", value=f"{score:.1f}%")
                        prob_cols[i].progress(min(score / 100.0, 1.0))

                # Feature Inspection (NLP transparency)
                st.markdown("---")
                st.markdown("#### 🔍 Explainability: Which Words Influenced This Decision?")
                
                try:
                    tfidf_step = pipeline.named_steps["tfidf"]
                    transformed_vec = tfidf_step.transform([cleaned_input])
                    feature_names = np.array(tfidf_step.get_feature_names_out())
                    non_zero_indices = transformed_vec.nonzero()[1]

                    if len(non_zero_indices) > 0:
                        extracted_words = feature_names[non_zero_indices]
                        tfidf_scores = transformed_vec.data
                        word_df = pd.DataFrame({
                            "Recognized Word / N-gram": extracted_words,
                            "TF-IDF Importance Weight": [round(s, 4) for s in tfidf_scores]
                        }).sort_values(by="TF-IDF Importance Weight", ascending=False)

                        st.write("The following words from your text were matched in the model's TF-IDF vocabulary:")
                        st.dataframe(word_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("ℹ️ The words in your feedback were not found in the model's learned training vocabulary. The prediction is based on baseline class probabilities.")
                except Exception as ex:
                    st.caption(f"Feature breakdown unavailable: {ex}")

# ==========================================
# TAB 2: MODEL PERFORMANCE & DATASET
# ==========================================
with tab_metrics:
    st.markdown("### 📊 Actual Model Evaluation & Dataset Statistics")
    st.write(
        "These metrics represent the **actual performance** evaluated on a held-out test set "
        "(20% of the dataset) that the model never saw during training. No hardcoded or fake numbers are used."
    )

    if metrics_data is not None:
        stats = metrics_data["dataset_stats"]
        eval_info = metrics_data["evaluation"]

        # Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['total_samples']}</div>
            <div class="metric-label">Total Feedback Rows</div>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['train_samples']}</div>
            <div class="metric-label">Training Split (80%)</div>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{stats['test_samples']}</div>
            <div class="metric-label">Testing Split (20%)</div>
        </div>
        """, unsafe_allow_html=True)

        acc_pct = eval_info['accuracy'] * 100
        m4.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #34D399;">{acc_pct:.2f}%</div>
            <div class="metric-label">Held-out Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Classification Report Table
        st.markdown("#### 📋 Detailed Classification Report (Per Class)")
        st.write(f"Precision, Recall, and F1-score evaluated on the {stats['test_samples']} held-out test samples:")

        report_dict = eval_info["classification_report"]
        report_rows = []
        for label in eval_info["classes"]:
            if label in report_dict:
                row = report_dict[label]
                report_rows.append({
                    "Sentiment Class": label,
                    "Precision": f"{row['precision'] * 100:.1f}%",
                    "Recall": f"{row['recall'] * 100:.1f}%",
                    "F1-Score": f"{row['f1-score'] * 100:.1f}%",
                    "Test Samples (Support)": int(row['support'])
                })

        report_df = pd.DataFrame(report_rows)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Confusion Matrix
        col_cm, col_dist = st.columns(2)

        with col_cm:
            st.markdown("#### 🎯 Confusion Matrix")
            st.write("Rows represent the **actual sentiment**, columns represent the **predicted sentiment**:")
            cm_array = np.array(eval_info["confusion_matrix"])
            cm_df = pd.DataFrame(
                cm_array,
                index=[f"Actual {c}" for c in eval_info["classes"]],
                columns=[f"Pred {c}" for c in eval_info["classes"]]
            )
            st.dataframe(cm_df, use_container_width=True)

        with col_dist:
            st.markdown("#### ⚖️ Dataset Class Balance")
            st.write("Distribution of classes across the full dataset:")
            dist_data = []
            for k, v in stats["class_distribution"].items():
                dist_data.append({"Sentiment": k, "Sample Count": v, "Percentage": f"{v / stats['total_samples'] * 100:.1f}%"})
            st.dataframe(pd.DataFrame(dist_data), use_container_width=True, hide_index=True)

        st.caption(f"Last Model Training Date & Time: {metrics_data.get('training_timestamp', 'N/A')}")
    else:
        st.warning("Metrics file `model/evaluation_metrics.json` not found. Please run `python train_model.py`.")

# ==========================================
# TAB 3: HOW IT WORKS (THEORY)
# ==========================================
with tab_theory:
    st.markdown("### 💡 How the Machine Learning System Works")
    st.write(
        "This project converts human English feedback into numerical math vectors using **TF-IDF**, "
        "and then classifies the vector into a sentiment class using **Logistic Regression**."
    )

    st.markdown("""
    #### 1. Text Preprocessing & Cleaning
    Computers cannot understand raw words directly. Before running any machine learning algorithm, the text is preprocessed:
    - **Lowercasing:** Converting all letters to lowercase so that `"Great"` and `"great"` are treated identically.
    - **Punctuation & Space Normalization:** Removing extraneous spaces and formatting.
    - **Negation-Aware Stop-Word Filtering:** Common filler words (like `"is"`, `"the"`, `"at"`) are ignored, but crucial sentiment modifiers like `"not"`, `"no"`, `"never"`, and `"without"` are intentionally preserved!
    """)

    st.markdown("""
    #### 2. TF-IDF (Term Frequency - Inverse Document Frequency)
    TF-IDF measures how important a word is to a particular feedback sentence compared to the whole dataset.
    
    It is computed by multiplying two factors:
    1. **Term Frequency (TF):** How often a word appears in the current feedback sentence.
       $$\\text{TF}(w, d) = \\frac{\\text{Number of times word } w \\text{ appears in sentence } d}{\\text{Total words in sentence } d}$$
    2. **Inverse Document Frequency (IDF):** Measures how rare or informative a word is across all sentences. Words appearing in every sentence receive lower weight.
       $$\\text{IDF}(w) = \\log\\left(\\frac{1 + N}{1 + \\text{count}(w)}\\right) + 1$$
    
    *Example:* The word `"excellent"` has high TF-IDF in a positive review because it is informative and specific, while `"lecture"` appears everywhere and gets lower weight.
    """)

    st.markdown("""
    #### 3. Logistic Regression Algorithm
    Logistic Regression is a foundational classification algorithm. Despite its name, it is used for **classification**, not regression!
    - It assigns a numerical weight $w_i$ to every TF-IDF feature based on its association with Positive, Negative, or Neutral sentiments.
    - It computes a weighted linear combination:
      $$z = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b$$
    - It applies the **Softmax function** to convert the linear scores into normalized probabilities summing to 100%:
      $$P(y = c \\mid x) = \\frac{e^{z_c}}{\\sum_{k} e^{z_k}}$$
    - The class with the highest probability is chosen as the predicted sentiment.
    """)

    st.markdown("""
    #### 4. End-to-End System Workflow
    ```
    [ Student Feedback Text ]
              │
              ▼
    [ Preprocessing & Lowercasing ]
              │
              ▼
    [ TF-IDF Vectorizer (Unigrams + Bigrams) ]
              │
              ▼
    [ Feature Vector (e.g. 5,000 dimensions) ]
              │
              ▼
    [ Logistic Regression Classifier (Softmax) ]
              │
              ▼
    [ Sentiment Class & Probabilities: Positive / Negative / Neutral ]
    ```
    """)

# ==========================================
# TAB 4: LIMITATIONS & ETHICS
# ==========================================
with tab_limitations:
    st.markdown("### ⚠️ Methodology, Limitations & Ethics")
    st.write("Every responsible AI project must acknowledge its boundaries and edge cases.")

    col_l1, col_l2 = st.columns(2)

    with col_l1:
        st.markdown("#### 1. Why Neutral Feedback is Difficult to Classify")
        st.write(
            "In student feedback sentiment analysis, **Neutral** sentiment is notoriously hard for ML models. "
            "Neutral sentences (such as *'The class meets on Tuesdays at 9 AM'*) contain academic nouns "
            "like *class*, *course*, and *instructor* without emotive adjectives. "
            "Because those same nouns also appear in positive and negative sentences, the model may occasionally "
            "lean slightly toward Positive or Negative."
        )

        st.markdown("#### 2. Sarcasm and Subtle Negation")
        st.write(
            "TF-IDF models analyze word frequencies and small local n-grams (1 to 2 words). "
            "They do not understand sarcasm or double meanings, such as: "
            "*'Oh wonderful, another three hour exam with broken keyboards!'* "
            "Because the word *'wonderful'* is present, a simple model might mistake the tone."
        )

    with col_l2:
        st.markdown("#### 3. Dataset Size and Generalization")
        st.write(
            "Our model is trained on a curated benchmark dataset of 210 student feedback rows. "
            "While this is ideal for demonstrating machine learning fundamentals in a first-year mini-capstone, "
            "it does not capture every regional accent, slang term, or domain-specific college abbreviation. "
            "Real-world institutional deployment requires tens of thousands of varied reviews."
        )

        st.markdown("#### 4. Ethical AI & Student Well-being")
        st.write(
            "Automated sentiment tools should **never** be used as the sole basis for faculty evaluations, "
            "promotions, or grading decisions. They are designed to help department heads quickly spot aggregate trends "
            "and address systemic issues (such as broken lab hardware or unclear deadlines) constructively."
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "Student Feedback Sentiment Analysis • B.Tech CSE Mini-Capstone • Built with Python, Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
