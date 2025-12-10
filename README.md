# 🚀 EMO-TRACK: Deep Learning Emotion & Mood Analytics App

**EMO-TRACK** is an AI-powered system that predicts emotion from text using a **BiLSTM Deep Learning model** and performs **mood analytics** using daily user input.

🔗 **Live App (24/7):** https://pranavedula-emo-track.hf.space  
📌 **Tech:** Python · TensorFlow · NLP · Gradio · HuggingFace Spaces

---

## 🧠 What this App Does

✔ Detects emotions from text  
✔ Displays matching **emoji**  
✔ Stores mood entries in a log database  
✔ Shows **graphs** of:
- Emotion distribution
- Daily mood tracking  
✔ Works online **forever** through Hugging Face Spaces  
✔ Simple UI for anyone to use

---

## 🎯 Real-world Use Cases

| Area | Usage |
|------|------|
| Mental Health | Mood monitoring & stress awareness |
| Social Media | Comments / feedback emotion analysis |
| Chatbots | Understand user feelings more accurately |
| Smart Assistants | Empathetic communication |

---

## 🏗️ System Architecture

```mermaid
flowchart LR
User --> UI[Gradio Web UI]
UI --> Preprocess[NLP Preprocessing]
Preprocess --> DL[BiLSTM Deep Learning Model]
DL --> Emotion[Emotion Output + Emoji]
Emotion --> Log[(Mood Log CSV)]
Log --> Analytics[Graphs & Stats]
Analytics --> UI
⚙️ Technologies Used

Python (Deep Learning + NLP)

TensorFlow / Keras (BiLSTM model)

Gradio (Web interface)

Hugging Face Spaces (Deployment)

Matplotlib / Pandas (Analytics & Graphs)

📂 Project Structure
emo-track/
│
├── app.py                # Gradio Web App
├── requirements.txt      # Dependencies
├── emotion_model.h5      # Trained BiLSTM model
├── tokenizer.pkl         # Tokenizer for text sequences
├── label_encoder.pkl     # Label mapping
└── final_year_project.ipynb  # Model training code

🚀 How to Run Locally
git clone https://github.com/Pranavreddyedula/emo-track.git
cd emo-track

pip install -r requirements.txt
python app.py


⏳ Wait few seconds → App will open in a browser window automatically.

📈 Screenshots (Example)

Replace these placeholders with real screenshots

Log Mood Screen	Analytics Screen

	
🧩 Emotion Labels & Emojis
Emotion	Emoji
Love	❤️
Joy / Happy	😄
Fear / Stress	😨
Sadness	😢
Anger	😡
Surprise	😲
Neutral	🙂
🔍 How it Works (Short Explanation)
1️⃣ Model Training

Kaggle Emotion in Text dataset

Text → Tokenization → Padding

BiLSTM learns context from both sides of sentence

Softmax classifier predicts final emotion

2️⃣ Hybrid AI System

✔ AI model prediction
+
✔ Smart keyword rules (romantic, stress, etc.)
→ Gives more human-like predictions

📊 Output Example

Input: I feel romantic today ❤️
Output: Emotion: love ❤️

Input: Exam tension killing me
Output: Emotion: fear 😨

🧪 Performance
Metric	Score
Accuracy	~85–90% depending on dataset
Classes	joy, sadness, anger, fear, love, surprise

(Exact scores shown in project report)

🧩 Future Improvements

🔹 Hindi/Telugu language support
🔹 Speech-to-emotion recognition
🔹 Weekly mood analytics report download
🔹 Cloud database instead of CSV
🔹 Transformer model (BERT) for higher accuracy

👨‍💻 Developer:
Edula Sai Pranav Reddy — Final Year CSE
