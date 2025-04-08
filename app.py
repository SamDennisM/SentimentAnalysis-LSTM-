import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Page config MUST be first Streamlit command
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Load model and tokenizer
@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

# Load data
@st.cache_data
def load_data():
    file_path = "training.1600000.processed.noemoticon (1).csv"
    df = pd.read_csv(file_path, encoding="latin-1", header=None, skiprows=1, low_memory=False)
    df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df = df[['polarity', 'text']]
    df['polarity'] = df['polarity'].replace({0: 0, 4: 1})
    df['polarity'] = pd.to_numeric(df['polarity'], errors='coerce').fillna(0).astype(int)
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    return df

df = load_data()

# Prediction function
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)[0][0]
    return "Positive 😊" if pred > 0.5 else "Negative 😞", pred

# Streamlit Layout
st.title("🧠 Sentiment Analysis with Bi-LSTM")
st.markdown("Analyze Twitter sentiment with Bi-LSTM and explore training stats and class distributions.")

# Tabs for UI sections
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Visualizations", "📈 Model Accuracy"])

# Tab 1: Prediction
with tab1:
    st.header("Predict Tweet Sentiment")
    user_input = st.text_area("Enter tweet text here", height=150)

    if st.button("Predict Sentiment"):
        if user_input.strip():
            sentiment, prob = predict_sentiment(user_input)
            st.success(f"**Sentiment:** {sentiment}")
            st.write(f"Confidence Score: `{prob:.2f}`")
        else:
            st.warning("Please enter a tweet!")

# Tab 2: Visualizations
with tab2:
    st.header("Tweet Distribution Visualizations")

    # Word Count Histogram
    st.subheader("Word Count Distribution in Tweets")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['text_length'], bins=30, kde=True, ax=ax1)
    ax1.set_xlabel("Number of Words")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Word Count Distribution")
    st.pyplot(fig1)

    # Sentiment Class Distribution
    st.subheader("Sentiment Class Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(x=df['polarity'], palette=['red', 'green'], ax=ax2)
    ax2.set_title("Sentiment Distribution")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Negative", "Positive"])
    ax2.set_xlabel("Sentiment")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

# Tab 3: Accuracy Chart (you need to save accuracy history)
with tab3:
    st.header("RNN vs Bi-LSTM Accuracy ")
    
    # Simulated dummy data for illustration (replace with real data)
    rnn_acc = [0.70, 0.75, 0.77, 0.78, 0.79]
    lstm_acc = [0.72, 0.78, 0.81, 0.83, 0.85]

    fig3, ax3 = plt.subplots()
    ax3.plot(rnn_acc, label='RNN Accuracy', marker='o')
    ax3.plot(lstm_acc, label='LSTM Accuracy', marker='o')
    ax3.set_title("Training Accuracy Over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    st.pyplot(fig3)
