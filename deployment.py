import streamlit as st
import pandas as pd
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json


# Load the trained LSTM model
model = load_model("sentiment_lstm_model.h5")

# Load tokenizer (we saved it as JSON)
with open("tokenizer.json", "r") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Set max_len same as used in training
max_len = 20

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text


st.title("Sentiment Analysis (LSTM)")

st.write("Upload an Excel/CSV file with a column named 'review'.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "review" not in df.columns:
        st.error("Column 'review' not found! Please make sure your file has a 'review' column.")
    else:
        st.success(f"File uploaded successfully! {df.shape[0]} reviews found.")

        reviews_clean = df['review'].apply(clean_text)
        sequences = tokenizer.texts_to_sequences(reviews_clean)
        padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

        pred_probs = model.predict(padded)
        pred_labels = (pred_probs > 0.5).astype(int)
        df['predicted_sentiment'] = np.where(pred_labels==1, "Positive", "Negative")
        df['probability'] = pred_probs

        if st.checkbox("Show predictions in app"):
            st.write(df[['review', 'predicted_sentiment', 'probability']])

        if st.checkbox("Download predictions as CSV"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
