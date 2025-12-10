import gradio as gr
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Load model + tokenizer + encoder
model = tf.keras.models.load_model("emotion_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 50
LOG_FILE = "mood_log.csv"

# Emotion → Emojis map
EMOJI_MAP = {
    "anger": "😡",
    "fear": "😨",
    "disgust": "🤢",
    "sadness": "😢",
    "surprise": "😲",
    "joy": "😄",
    "happy": "😄",
    "love": "❤️",
    "neutral": "🙂"
}

# Keyword dictionaries for rule-based correction
LOVE_WORDS = ["love", "romantic", "crush", "babe", "baby", "sweetheart"]
HAPPY_WORDS = ["happy", "enjoy", "fun", "good", "smile", "awesome"]
SAD_WORDS = ["sad", "low", "depressed", "cry", "upset"]
FEAR_WORDS = ["stress", "exam", "tension", "worry", "scared", "fear"]
ANGER_WORDS = ["angry", "hate", "fight", "rude", "annoyed"]

def correct_emotion(text, emotion, confidence):
    t = text.lower()

    if any(w in t for w in LOVE_WORDS):
        return "love"
    if any(w in t for w in HAPPY_WORDS):
        return "joy"
    if any(w in t for w in SAD_WORDS):
        return "sadness"
    if any(w in t for w in FEAR_WORDS):
        return "fear"
    if any(w in t for w in ANGER_WORDS):
        return "anger"

    # If model very confident, keep AI prediction
    if confidence > 90:
        return emotion
    
    return emotion  # Default fallback

def analyze_and_log(text):
    if not text.strip():
        return "Please enter text", ""

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len)
    probs = model.predict(pad)[0]

    idx = int(np.argmax(probs))
    emotion = label_encoder.classes_[idx]
    confidence = float(np.max(probs)) * 100.0

    # Rule-based correction
    corrected_emotion = correct_emotion(text, emotion, confidence)

    # Select correct emoji
    emoji = EMOJI_MAP.get(corrected_emotion.lower(), "🙂")

    # Logging
    ts = datetime.now()
    df = pd.DataFrame([[ts, text, corrected_emotion]], columns=["timestamp", "text", "emotion"])
    exists = os.path.exists(LOG_FILE)
    df.to_csv(LOG_FILE, mode="a", header=not exists, index=False)

    result = f"Emotion: {corrected_emotion} (confidence: {confidence:.2f}%)"
    return result, emoji

def mood_stats():
    if not os.path.exists(LOG_FILE):
        return "No entries yet!", None, None

    df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])

    fig1 = plt.figure()
    df["emotion"].value_counts().plot(kind="bar", title="Emotion Distribution")

    df["date"] = df["timestamp"].dt.date
    fig2 = plt.figure()
    df.groupby("date").size().plot(marker="o", title="Daily Mood Logs")

    summary = (
        f"Total Entries: {len(df)}\n"
        f"Days Tracked: {df['date'].nunique()}\n"
        f"Most Common Emotion: {df['emotion'].value_counts().idxmax()}"
    )
    return summary, fig1, fig2

# UI
with gr.Blocks() as demo:
    gr.Markdown("# EMO-TRACK: Live Deep Learning Mood Analytics")

    with gr.Tab("Log Mood"):
        text_input = gr.Textbox(label="How do you feel?", lines=2)
        analyze_btn = gr.Button("Analyze & Save")
        out_pred = gr.Textbox(label="Prediction")
        out_emoji = gr.Textbox(label="Emoji")
        analyze_btn.click(analyze_and_log, text_input, [out_pred, out_emoji])

    with gr.Tab("Analytics"):
        stats_btn = gr.Button("Show Graphs")
        stats_out = gr.Textbox(label="Summary")
        graph1 = gr.Plot()
        graph2 = gr.Plot()
        stats_btn.click(mood_stats, None, [stats_out, graph1, graph2])

demo.launch()
