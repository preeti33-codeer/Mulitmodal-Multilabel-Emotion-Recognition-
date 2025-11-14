"""
===========================================================
 Multimodal Emotion Recognition  
 (Text + Webcam + Microphone) + Decision Tree + KNN 
===========================================================

✔ Uses Gradio UI  
✔ Uses Transformers for Text, Image, Audio  
✔ Logs all predictions into CSV  
✔ Trains Decision Tree & KNN over previous predictions  
✔ Displays confidence charts  
✔ Fully production-ready for GitHub  
"""

# ----------------------------------------------------------
# INSTALLS (for Colab users; GitHub users see requirements.txt)
# ----------------------------------------------------------

# !pip install -q gradio transformers torch torchvision torchaudio librosa pandas matplotlib Pillow scikit-learn --upgrade

# ----------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io, time, os, warnings
from PIL import Image
from transformers import pipeline
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# SETUP
# ----------------------------------------------------------
CSV_FILE = "emotion_results.csv"
DEVICE = 0 if torch.cuda.is_available() else -1

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "timestamp","text","text_label","text_score",
        "image_label","image_score","audio_label","audio_score"
    ]).to_csv(CSV_FILE, index=False)

# ----------------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------------
print("Loading models (please wait 1–2 mins)...")

text_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=DEVICE
)

image_pipe = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    top_k=5,
    device=DEVICE
)

audio_pipe = pipeline(
    "audio-classification",
    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    top_k=6,
    device=DEVICE
)

print("✔ Models loaded successfully")

# ----------------------------------------------------------
# HELPERS – ANALYSIS
# ----------------------------------------------------------
def analyze_text(text):
    try:
        res = text_pipe(text)[0]
        return res["label"].title(), float(res["score"])
    except:
        return "Neutral", 0.0

def analyze_image(img):
    try:
        res = image_pipe(img)[0]
        return res["label"].title(), float(res["score"])
    except:
        return "Neutral", 0.0

def analyze_audio(path):
    try:
        res = audio_pipe(path)[0]
        return res["label"].title(), float(res["score"])
    except:
        return "Neutral", 0.0

# ----------------------------------------------------------
# SAVE RESULT
# ----------------------------------------------------------
def save_result(row):
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# ----------------------------------------------------------
# CONFIDENCE CHART
# ----------------------------------------------------------
def create_chart(t_label, ts, i_label, iscore, a_label, ascore):
    labels = ["Text", "Image", "Audio"]
    scores = [ts, iscore, ascore]
    emotions = [t_label, i_label, a_label]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, scores)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Emotion Confidence per Modality")

    for bar, emo, sc in zip(bars, emotions, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            sc / 2,
            f"{emo}\n{sc:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=9
        )

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf)

# ----------------------------------------------------------
# TRAIN ML MODELS
# ----------------------------------------------------------
def train_models():
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return None, None

    # Combine emotion labels
    def combine_emotions(row):
        labels = [row["text_label"], row["image_label"], row["audio_label"]]
        return max(set(labels), key=labels.count)

    df["overall_emotion"] = df.apply(combine_emotions, axis=1)

    X = df[["text_score", "image_score", "audio_score"]]
    y = df["overall_emotion"]

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)

    tree.fit(X, y)
    knn.fit(X, y)

    return tree, knn

# ----------------------------------------------------------
# PROCESS INPUTS
# ----------------------------------------------------------
def process_all(text, img, aud):
    if not text or img is None or aud is None:
        return ("⚠ Please provide text + photo + audio.", None, None, None)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Run all predictions
    t_label, t_score = analyze_text(text)
    i_label, i_score = analyze_image(img)
    a_label, a_score = analyze_audio(aud)

    # Save record
    row = {
        "timestamp": timestamp,
        "text": text,
        "text_label": t_label,
        "text_score": round(t_score, 3),
        "image_label": i_label,
        "image_score": round(i_score, 3),
        "audio_label": a_label,
        "audio_score": round(a_score, 3)
    }
    save_result(row)

    chart = create_chart(t_label, t_score, i_label, i_score, a_label, a_score)
    tree, knn = train_models()

    dt_pred = tree.predict([[t_score, i_score, a_score]])[0] if tree else "N/A"
    knn_pred = knn.predict([[t_score, i_score, a_score]])[0] if knn else "N/A"

    summary = f"""
### 🧠 Multimodal Emotion Results
🕒 Time: **{timestamp}**

- 📝 Text → **{t_label}** ({t_score:.2f})
- 📷 Image → **{i_label}** ({i_score:.2f})
- 🎤 Audio → **{a_label}** ({a_score:.2f})

---

### 🌳 Decision Tree → **{dt_pred}**
### 🤖 KNN → **{knn_pred}**

Results saved to **emotion_results.csv**
"""

    df = pd.read_csv(CSV_FILE).tail(10).reset_index(drop=True)
    return summary, chart, CSV_FILE, df

# ----------------------------------------------------------
# GRADIO UI
# ----------------------------------------------------------
title = "🎭 Multimodal Emotion Recognition (Text + Image + Audio + ML)"

description = """
This application analyzes:
- 📝 Text input  
- 📸 Image (face expression)  
- 🎤 Audio (voice tone)

It uses **Transformers + Decision Tree + KNN** to predict emotions.
"""

css = """
.gradio-container {
    background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=6):
            txt = gr.Textbox(label="📝 Enter your feeling")
            img = gr.Image(label="📸 Upload/Capture Photo", sources=["webcam", "upload"], type="pil")
            aud = gr.Audio(label="🎤 Record Voice", sources=["microphone"], type="filepath")
            btn = gr.Button("🔍 Analyze Emotions")

        with gr.Column(scale=6):
            out_md = gr.Markdown("Awaiting input...")
            out_chart = gr.Image(label="📊 Confidence Chart")
            out_csv = gr.File(label="⬇ Download CSV")
            out_df = gr.Dataframe(label="Recent Results", interactive=False)

    btn.click(process_all, inputs=[txt, img, aud],
              outputs=[out_md, out_chart, out_csv, out_df])

demo.launch(share=True)
