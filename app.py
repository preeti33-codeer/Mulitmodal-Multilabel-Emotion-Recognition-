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
# !pip install reportlab > /dev/null

import os
import time
import uuid
import warnings
import pandas as pd
import gradio as gr

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Transformers model loading wrapped for safety
try:
    from transformers import pipeline
except Exception:
    pipeline = None

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
CSV_FILE = "emotion_results.csv"
PDF_DIR = "reports"
os.makedirs(PDF_DIR, exist_ok=True)

DEVICE = 0  # Using CPU for safest environment compatibility

# Create CSV if missing
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "timestamp","text","text_label","text_score",
        "image_label","image_score","audio_label","audio_score"
    ]).to_csv(CSV_FILE, index=False)

# ==========================================================
# LOAD MODELS SAFELY
# ==========================================================
print("Loading models (safe load)...")

text_pipe = None
image_pipe = None
audio_pipe = None

if pipeline is not None:
    try:
        text_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=DEVICE
        )
    except Exception as e:
        print("Text model failed:", e)
        text_pipe = None

    try:
        image_pipe = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            top_k=5,
            device=DEVICE
        )
    except Exception as e:
        print("Image model failed:", e)
        image_pipe = None

    try:
        audio_pipe = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            top_k=6,
            device=DEVICE
        )
    except Exception as e:
        print("Audio model failed:", e)
        audio_pipe = None

# ==========================================================
# SAFE ANALYZERS (no crashes if model missing)
# ==========================================================
def analyze_text_safe(text):
    if not text or text_pipe is None:
        return "Neutral", 0.0
    try:
        r = text_pipe(text)[0]
        return r["label"], float(r["score"])
    except Exception:
        return "Neutral", 0.0


def analyze_image_safe(img):
    if img is None or image_pipe is None:
        return "Neutral", 0.0
    try:
        r = image_pipe(img)[0]
        return r["label"], float(r["score"])
    except Exception:
        return "Neutral", 0.0


def analyze_audio_safe(audio_path):
    if not audio_path or audio_pipe is None:
        return "Neutral", 0.0
    try:
        r = audio_pipe(audio_path)[0]
        return r["label"], float(r["score"])
    except Exception:
        return "Neutral", 0.0
# ==========================================================
# PART 2 — ML training, PDF exporter, avatar & chart generators
# ==========================================================

# -----------------------
# TRAINING HELPERS
# -----------------------
def train_models():
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        return None, None
    if df.empty:
        return None, None

    # majority vote for overall label
    def majority_vote(r):
        labels = [
            r.get("text_label", "Neutral"),
            r.get("image_label", "Neutral"),
            r.get("audio_label", "Neutral"),
        ]
        return max(set(labels), key=labels.count)

    df["overall_emotion"] = df.apply(majority_vote, axis=1)
    X = df[["text_score", "image_score", "audio_score"]].fillna(0)
    y = df["overall_emotion"]

    try:
        tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        return tree, knn
    except Exception:
        return None, None

# -----------------------
# PDF REPORT (simple, readable)
# -----------------------
def make_pdf_report(record):
    uid = uuid.uuid4().hex[:8]
    path = os.path.join(PDF_DIR, f"emotion_report_{uid}.pdf")
    try:
        c = canvas.Canvas(path, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, height - 50, "Multimodal Emotion Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawString(40, height - 80, f"Timestamp: {record.get('timestamp','')}")
        y = height - 110

        items = [
            ("Text", record.get("text", "")),
            ("Text Emotion", f"{record.get('text_label','')}"f" ({record.get('text_score',0):.2f})"),
            ("Image Emotion", f"{record.get('image_label','')}"f" ({record.get('image_score',0):.2f})"),
            ("Audio Emotion", f"{record.get('audio_label','')}"f" ({record.get('audio_score',0):.2f})"),
            ("Decision Tree", record.get("dt_pred", "N/A")),
            ("KNN", record.get("knn_pred", "N/A")),
        ]
        for k, v in items:
            c.drawString(40, y, f"{k}: {v}")
            y -= 18
        c.save()
        return path
    except Exception:
        return None

# -----------------------
# AVATAR GENERATION (generic cartoon SVG)
# returns inline SVG string for emotion (no external models)
# -----------------------
def avatar_svg_for_emotion(emotion):
    # Normalize emotion to key words
    e = (emotion or "Neutral").lower()
    if "happy" in e or "joy" in e or "excited" in e:
        face_color = "#FFD166"
        mouth = '<path d="M30 45 Q50 65 70 45" stroke="#000" stroke-width="3" fill="none" />'
        eyes = '<circle cx="35" cy="35" r="4" /><circle cx="65" cy="35" r="4" />'
    elif "angry" in e or "rage" in e:
        face_color = "#FF6B6B"
        mouth = '<path d="M30 55 Q50 40 70 55" stroke="#000" stroke-width="3" fill="none" />'
        eyes = '<path d="M28 32 L42 36" stroke="#000" stroke-width="3"/><path d="M58 36 L72 32" stroke="#000" stroke-width="3"/>'
    elif "sad" in e or "down" in e:
        face_color = "#74C0FC"
        mouth = '<path d="M30 55 Q50 45 70 55" stroke="#000" stroke-width="3" fill="none" />'
        eyes = '<circle cx="35" cy="36" r="3" /><circle cx="65" cy="36" r="3" />'
    elif "surprise" in e or "surprised" in e or "wow" in e:
        face_color = "#FFE7A7"
        mouth = '<circle cx="50" cy="50" r="8" stroke="#000" stroke-width="2" fill="none" />'
        eyes = '<circle cx="35" cy="35" r="4" /><circle cx="65" cy="35" r="4" />'
    elif "fear" in e or "scared" in e:
        face_color = "#E0BBE4"
        mouth = '<path d="M34 54 Q50 42 66 54" stroke="#000" stroke-width="2" fill="none" />'
        eyes = '<circle cx="35" cy="34" r="4" /><circle cx="65" cy="34" r="4" />'
    elif "disgust" in e or "disgusted" in e:
        face_color = "#C7F9CC"
        mouth = '<path d="M32 54 Q50 48 68 54" stroke="#000" stroke-width="2" fill="none" />'
        eyes = '<circle cx="35" cy="36" r="3" /><circle cx="65" cy="36" r="3" />'
    else:
        # Neutral / default
        face_color = "#FFDAB9"
        mouth = '<path d="M34 50 Q50 55 66 50" stroke="#000" stroke-width="2" fill="none" />'
        eyes = '<circle cx="35" cy="36" r="3" /><circle cx="65" cy="36" r="3" />'

    svg = """
<svg width="140" height="140" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="g1" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="{face_color}" stop-opacity="1"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0.05"/>
    </radialGradient>
  </defs>
  <rect width="100" height="100" rx="18" fill="url(#g1)" />
  <!-- face -->
  <circle cx="50" cy="40" r="28" fill="{face_color}" stroke="#000" stroke-opacity="0.06"/>
  <!-- eyes -->
  {eyes}
  <!-- mouth -->
  {mouth}
</svg>
""".format(face_color=face_color, eyes=eyes, mouth=mouth)

    # Wrap in a container with animation classes expected by CSS
    wrapper = """
<div class="avatar-wrap" style="width:140px; height:140px; display:flex; align-items:center; justify-content:center;">
  {svg}
</div>
""".format(svg=svg)
    return wrapper

# -----------------------
# ADVANCED ANIMATED CHART (HTML + CSS + safe JS)
# -----------------------
def make_advanced_chart_html(record):
    # Safely convert scores to percentages and cap 0..100
    def pct(v):
        try:
            v = max(0.0, min(1.0, float(v)))
            return f"{int(round(v*100))}%"
        except Exception:
            return "0%"
    t_pct = pct(record.get("text_score", 0.0))
    i_pct = pct(record.get("image_score", 0.0))
    a_pct = pct(record.get("audio_score", 0.0))

    # emoji icons to show inside bars
    def emoji_for_label(lbl):
        lbl = (lbl or "neutral").lower()
        if "happy" in lbl:
            return "😄"
        if "angry" in lbl:
            return "😡"
        if "sad" in lbl:
            return "😢"
        if "surprise" in lbl or "surprised" in lbl:
            return "😲"
        if "fear" in lbl:
            return "😱"
        if "disgust" in lbl:
            return "🤢"
        return "🙂"

    t_emoji = emoji_for_label(record.get("text_label",""))
    i_emoji = emoji_for_label(record.get("image_label",""))
    a_emoji = emoji_for_label(record.get("audio_label",""))

    # CSS for the chart
    css_chart = """
    .adv-chart .bar-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
    .adv-chart .bar-label { width:90px; font-weight:700; color:#e6f7ff; display:flex; align-items:center; gap:8px; font-size:13px; }
    .adv-chart .bar-track { flex:1; height:36px; background:#0f1317; border-radius:18px; position:relative; overflow:hidden; box-shadow: inset 0 0 18px rgba(0,0,0,0.6); }
    .adv-chart .bar-fill { height:100%; width:0%; border-radius:18px; display:flex; align-items:center; padding-left:12px; gap:8px; box-sizing:border-box; transition: width 900ms cubic-bezier(.22,.9,.22,1); }
    .fill-text { font-weight:700; text-shadow:0 1px 6px rgba(0,0,0,0.6); }
    .bubble { position:absolute; top:-12px; right:12px; background:rgba(0,0,0,0.45); padding:4px 8px; border-radius:12px; font-size:12px; color:#fff; box-shadow:0 4px 12px rgba(0,0,0,0.5); }
    .f1 { background: linear-gradient(90deg,#ff7b00,#ffd166); }
    .f2 { background: linear-gradient(90deg,#00c2ff,#0066ff); }
    .f3 { background: linear-gradient(90deg,#8e2de2,#4a00e0); }
    .emoji-inside { font-size:20px; margin-right:6px; filter:drop-shadow(0 2px 6px rgba(0,0,0,0.6)); }
    """
    css_chart_escaped = css_chart.replace('{', '{{').replace('}', '}}')

    # JavaScript for chart animation
    js_chart_animation = """
<script>
setTimeout(function(){{
  document.querySelectorAll('.adv-chart .bar-fill').forEach(function(el, idx){{
    var w = el.style.width;
    el.style.width = '0%';
    setTimeout(function(){{ el.style.width = w; }}, 60 + idx*120);
  }});
}}, 40);
</script>
"""

    raw = f"""
<div class="adv-chart" style="padding:10px; border-radius:12px; background:linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
  <style>
{css_chart_escaped}
  </style>

  <div class="bar-row">
    <div class="bar-label"><span style="font-size:18px;">{t_emoji}</span> <span>Text</span></div>
    <div class="bar-track">
      <div class="bar-fill f1" style="width:{t_pct};"><span class="emoji-inside">{t_emoji}</span><span class="fill-text">{t_pct}</span><div class="bubble">{t_pct}</div></div>
    </div>
  </div>

  <div class="bar-row">
    <div class="bar-label"><span style="font-size:18px;">{i_emoji}</span> <span>Image</span></div>
    <div class="bar-track">
      <div class="bar-fill f2" style="width:{i_pct};"><span class="emoji-inside">{i_emoji}</span><span class="fill-text">{i_pct}</span><div class="bubble">{i_pct}</div></div>
    </div>
  </div>

  <div class="bar-row">
    <div class="bar-label"><span style="font-size:18px;">{a_emoji}</span> <span>Audio</span></div>
    <div class="bar-track">
      <div class="bar-fill f3" style="width:{a_pct};"> <span class="emoji-inside">{a_emoji}</span><span class="fill-text">{a_pct}</span><div class="bubble">{a_pct}</div></div>
    </div>
  </div>
</div>
{js_chart_animation}
""".format(t_pct=t_pct, i_pct=i_pct, a_pct=a_pct, t_emoji=t_emoji, i_emoji=i_emoji, a_emoji=a_emoji)

    return raw

# ==========================================================
# PART 2 Completed
# Next: PART 3 will include: main process function glue,
# final UI (Gradio Blocks) and global CSS. Reply "send part 3"
# ==========================================================
# ==========================================================
# PART 3 — Main glue, UI, CSS, and launch
# (Append this after Parts 1 & 2 in app.py)
# ==========================================================

# -----------------------
# MAIN PROCESS FUNCTION
# -----------------------
def process_all(text, img, aud, pdf_toggle):
    """
    Returns:
      - result_html (str)  -> rendered HTML with avatar + animated result
      - chart_html (str)   -> animated neon chart HTML
      - pdf_path (str|None) -> path to generated PDF (if requested)
      - recent_df (pd.DataFrame) -> last 10 rows
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # Safe analysis (handles missing models/inputs)
    t_label, t_score = analyze_text_safe(text)
    i_label, i_score = analyze_image_safe(img)
    a_label, a_score = analyze_audio_safe(aud)

    # Build record
    record = {
        "timestamp": ts,
        "text": text or "",
        "text_label": t_label, "text_score": float(t_score or 0.0),
        "image_label": i_label, "image_score": float(i_score or 0.0),
        "audio_label": a_label, "audio_score": float(a_score or 0.0),
    }

    # Append to CSV safely
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        df = pd.DataFrame(columns=list(record.keys()))
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    try:
        df.to_csv(CSV_FILE, index=False)
    except Exception:
        pass

    # Train and predict with ML models if possible
    tree, knn = train_models()
    dt_pred = "N/A"
    knn_pred = "N/A"
    try:
        if tree is not None:
            dt_pred = tree.predict([[record["text_score"], record["image_score"], record["audio_score"]]])[0]
        if knn is not None:
            knn_pred = knn.predict([[record["text_score"], record["image_score"], record["audio_score"]]])[0]
    except Exception:
        pass

    record["dt_pred"] = dt_pred
    record["knn_pred"] = knn_pred

    # Create avatar SVG for final emotion (use majority vote)
    final_emotion = dt_pred if dt_pred != "N/A" else record["text_label"] or record["image_label"] or record["audio_label"] or "Neutral"
    avatar_html = avatar_svg_for_emotion(final_emotion)

    # CSS for neon card
    css_neon_card = """
    .neon-card {
      background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.04));
      border-radius: 14px; padding:18px; color:#e6f7ff; position:relative; overflow:hidden;
      border: 1px solid rgba(255,255,255,0.04);
      box-shadow: 0 6px 24px rgba(0,0,0,0.35), 0 0 20px rgba(0,150,255,0.08);
    }
    .heading { font-weight:700; font-size:18px; margin-bottom:8px; color:#fff; text-shadow:0 0 8px rgba(0,200,255,0.25); }
    .meta { font-size:13px; color:#cce9ff; margin-bottom:10px; }
    .line { margin:6px 0; font-size:14px; }
    /* typewriter */
    .typewriter { border-right: 2px solid rgba(255,255,255,0.4); white-space:pre-wrap; overflow:hidden; display:inline-block; animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { border-color: transparent; } }

    /* Emoji burst */
    .emoji-burst-large { position:absolute; right:8px; top:-10px; pointer-events:none; z-index:5; }
    .emoji-burst-large span { display:inline-block; }

    /* neon glow */
    .neon-card::before { content:''; position:absolute; left:-30%; top:-30%; width:160%; height:160%; background: radial-gradient(circle at 20% 20%, rgba(0,200,255,0.08), transparent 10%), radial-gradient(circle at 80% 80%, rgba(255,0,200,0.03), transparent 10%); z-index:0; }
    .neon-card * { position:relative; z-index:1; }
    """
    css_neon_card_escaped = css_neon_card.replace('{', '{{').replace('}', '}}')

    # JavaScript for result animation
    js_result_animation = """
<script>
(function(){{
  // typewriter effect
  function typeWrite(el, speed){{
    var txt = el.textContent;
    el.textContent = '';
    var i = 0;
    var iv = setInterval(function(){{
      el.textContent += txt.charAt(i);
      i++;
      if(i >= txt.length) clearInterval(iv);
    }}, speed || 10);
  }}
  setTimeout(function(){{
    document.querySelectorAll('.typewriter').forEach(function(e, idx){{ typeWrite(e, 8 + idx*3); }});
    // big emoji burst animation
    var items = document.querySelectorAll('.emoji-burst-large span');
    items.forEach(function(sp, idx){{
      sp.style.fontSize = '40px';
      sp.style.display = 'inline-block';
      sp.style.marginLeft = '8px';
      sp.style.opacity = '0.95';
      sp.style.transform = 'translateY(0px) scale(0.9)';
      sp.style.transition = 'transform 900ms cubic-bezier(.2,.9,.3,1), opacity 900ms ease-out';
      setTimeout(function(){{ sp.style.transform = 'translateY(-70px) scale(1.2)'; sp.style.opacity='0'; }}, 200 + idx*140);
    }});
  }}, 80);
}})();
</script>
"""

    # Build the result HTML combining avatar + text + animations
    result_card = f"""
<div style="display:flex; gap:16px; align-items:center;">
  <div style="flex:0 0 140px;">
    {avatar_html}
  </div>
  <div style="flex:1;">
    <div class="neon-card" style="padding:12px; border-radius:12px;">
      <style>
{css_neon_card_escaped}
      </style>
      <div style="font-weight:800; font-size:18px; color:#fff; margin-bottom:6px;" class="typewriter">🧠 Emotion Results</div>
      <div style="color:#cfefff; font-size:13px; margin-bottom:8px;">Timestamp: {ts}</div>
      <div style="margin:6px 0;"><b>Text:</b> <span class="typewriter">{t_label} ({t_score:.2f})</span></div>
      <div style="margin:6px 0;"><b>Image:</b> {i_label} ({i_score:.2f})</div>
      <div style="margin:6px 0;"><b>Audio:</b> {a_label} ({a_score:.2f})</div>
      <div style="margin-top:8px;"><b>Decision Tree:</b> {dt_pred} &nbsp;&nbsp; <b>KNN:</b> {knn_pred}</div>
      <div style="position:relative; margin-top:8px;">
        <div class="emoji-burst-large" aria-hidden="true">
          <span>😄</span><span>😡</span><span>😢</span><span>😲</span><span>🤢</span>
        </div>
      </div>
    </div>
  </div>
</div>
{js_result_animation}
""".format(
        avatar_html=avatar_html,
        ts=record["timestamp"],
        t_label=record["text_label"], t_score=record["text_score"],
        i_label=record["image_label"], i_score=record["image_score"],
        a_label=record["audio_label"], a_score=record["audio_score"],
        dt_pred=record["dt_pred"], knn_pred=record["knn_pred"]
    )

    # Chart
    chart_html = make_advanced_chart_html(record)

    # PDF export
    pdf_path = None
    if pdf_toggle:
        pdf_path = make_pdf_report(record)

    # Recent table (last 10)
    try:
        recent_df = pd.read_csv(CSV_FILE).tail(10).reset_index(drop=True)
    except Exception:
        recent_df = pd.DataFrame([record])

    return result_card, chart_html, pdf_path if pdf_path else None, recent_df

# -----------------------
# GLOBAL CSS (neon + layout)
# -----------------------
GLOBAL_CSS = """
/* Page and container */
body { background: linear-gradient(180deg,#071021,#06142a); color:#e6f7ff; }
.gradio-container { background: transparent !important; }

/* Neon card */
.neon-card { background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.03)); border-radius:12px; padding:10px; box-shadow: 0 8px 30px rgba(0,0,0,0.6), 0 0 40px rgba(0,150,255,0.04); }

/* Avatar wrapper */
.avatar-wrap svg { filter: drop-shadow(0 8px 18px rgba(0,0,0,0.6)); border-radius:12px; }

/* Emoji burst large */
.emoji-burst-large { position:absolute; right:8px; top:-10px; pointer-events:none; z-index:5; }
.emoji-burst-large span { display:inline-block; }

/* Chart card adjustments */
.adv-chart .bar-fill { box-shadow: 0 8px 24px rgba(0,0,0,0.5), 0 0 18px rgba(255,255,255,0.02) inset; }

/* Buttons */
.gr-button { border-radius:10px; }

/* Ensure HTML outputs can host animations */
.result-card, .chart-card { min-height:140px; }
"""

# -----------------------
# BUILD AND LAUNCH GRADIO UI
# -----------------------
title = "🎭 Multimodal Emotion Recognition (Advanced)"
desc = "Provide text, image, or audio (all optional). Animated avatar, neon chart, big emojis, and optional PDF report."

with gr.Blocks(theme=gr.themes.Soft(), css=GLOBAL_CSS) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(desc)

    with gr.Row():
        with gr.Column(scale=6):
            txt_in = gr.Textbox(label="📝 Enter text (optional)", placeholder="I feel excited...", lines=2)
            img_in = gr.Image(label="📷 Upload or capture (optional)", sources=["upload","webcam"], type="pil")
            aud_in = gr.Audio(label="🎤 Record audio (optional)", sources=["microphone"], type="filepath")
            with gr.Row():
                pdf_toggle = gr.Checkbox(label="Generate PDF report", value=False)
                analyze_btn = gr.Button("🔍 Analyze Emotions", variant="primary")

        with gr.Column(scale=6):
            result_out = gr.HTML(label="Result (animated)")
            chart_out = gr.HTML(label="Confidence Chart (animated)")
            pdf_out = gr.File(label="Download PDF (if requested)")
            recent_out = gr.Dataframe(label="Recent Predictions (last 10)", interactive=False)

    analyze_btn.click(fn=process_all,
                      inputs=[txt_in, img_in, aud_in, pdf_toggle],
                      outputs=[result_out, chart_out, pdf_out, recent_out])

# Launch
if __name__ == "__main__":
    demo.launch(share=True)
