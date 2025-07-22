import streamlit as st
import pickle
import numpy as np
import spacy

# =======================
# ✅ Load Model & Preprocessors
# =======================
with open("my_model.h5", "rb") as f:
    model = pickle.load(f)  # single multiclass XGBoost / sklearn model

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)
    class_labels = ohe.categories_[0]  # ['low', 'medium', 'high']

# Load SpaCy model for text cleaning
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.write("Downloading SpaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# =======================
# ✅ Text Cleaning
# =======================
def lemmatize_text(text: str) -> str:
    """Lowercase, remove stopwords & lemmatize text."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# =======================
# ✅ Prediction Function
# =======================
def predict_priority(title: str, body: str, labels: str) -> str:
    """Combine title, body, labels → preprocess → predict priority"""

    # Combine all inputs into one text
    combined_text = f"{title} {body} {labels}"

    # Clean the text
    clean_text = lemmatize_text(combined_text)

    # Convert to TF-IDF features
    vec = vectorizer.transform([clean_text])

    # Predict class index (0/1/2)
    pred_class_idx = model.predict(vec)[0]

    # Map index → actual class label
    pred_label = class_labels[pred_class_idx]

    return pred_label

# =======================
# ✅ Streamlit UI
# =======================
st.set_page_config(page_title="GitHub Issue Priority Predictor", layout="centered")

st.title("🚀 GitHub Issue Priority Classifier")
st.write("Predict **issue priority** (low / medium / high) based on title, body & labels")

# User Inputs
title = st.text_input("📝 Issue Title")
body = st.text_area("📄 Issue Body")
labels = st.text_input("🏷️ Labels (comma separated)")

# Prediction Button
if st.button("🔮 Predict Priority"):
    if not title.strip() and not body.strip():
        st.warning("⚠️ Please enter at least a title or body!")
    else:
        with st.spinner("🔄 Analyzing..."):
            prediction = predict_priority(title, body, labels)
        st.success(f"✅ Predicted Priority: **{prediction.upper()}**")
