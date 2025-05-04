import streamlit as st
import pandas as pd
from textblob import TextBlob
import spacy
import re
import subprocess
import sys

# Function to ensure that spaCy's model is downloaded
def download_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Download spaCy model if not available
download_spacy_model()

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Process tweet for tokens and entities
def process_tweet(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return tokens, entities

# Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

# Streamlit UI
st.title('Toxicity Tamer: Tweet Sentiment Analysis')
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV Data:", df.head())

    if 'text' in df.columns:
        df['cleaned_text'] = df['text'].apply(clean_text)
        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)
        st.write("Processed Data:")
        st.write(df[['text', 'cleaned_text', 'sentiment']])
    else:
        st.error("CSV file must have a 'text' column.")
