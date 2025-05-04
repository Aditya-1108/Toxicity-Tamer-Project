import streamlit as st
import pandas as pd
from textblob import TextBlob
import spacy
import re
import subprocess
import sys

# Ensure spaCy model is downloaded
def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

download_spacy_model()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Clean tweet text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
    return text

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# UI
st.title('💬 Toxicity Tamer: Tweet Sentiment Analysis')
st.info("📌 Upload a CSV file that contains a 'text' column for analysis.")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'text' not in df.columns:
        st.error("❌ CSV must have a 'text' column.")
    else:
        # Clean and analyze
        df['cleaned_text'] = df['text'].apply(clean_text)
        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

        # Show results
        st.subheader("✅ Processed Data")
        st.write(df[['text', 'cleaned_text', 'sentiment']])
