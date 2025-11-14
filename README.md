<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gradio-4.x-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Multimodal-Emotion%20AI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge" />
</p>
Multimodal Emotion Recognition • Text + Image + Audio • Animated UI

# 🎭 Multimodal Multilabel Emotion Recognition
This project is an interactive multimodal emotion recognition system that detects human emotions from text, facial expressions (webcam), and voice (microphone). It combines deep learning with traditional ML algorithms and provides a real-time, user-friendly interface built using Gradio.

### *(Text + Image + Audio) with Decision Tree & KNN Integration*
This project is a **Multimodal Emotion Recognition System** that takes **text**, **image**, and **audio** from a user and predicts emotions using:

- 🧠 **Transformer Models**  
- 🌳 **Decision Tree Classifier**  
- 🤖 **K-Nearest Neighbors (KNN)**  
- 📊 **Confidence Visualization**  
- 📝 Automatic CSV logging & dataset building  

Built with an elegant **Gradio UI**, this app is perfect for demos, college projects, research, and AI-based emotion analysis tools.

## 🚀 **Features**
✔ Multimodal Input  
&nbsp;&nbsp;&nbsp;• Text (feelings typed by the user)  
&nbsp;&nbsp;&nbsp;• Image (facial expression via upload/webcam)  
&nbsp;&nbsp;&nbsp;• Audio (voice emotion via microphone)  

✔ Deep Learning using Transformers  
✔ Confidence Score Bar Chart  
✔ Recent Predictions Table  
✔ CSV Logging for Training Dataset  
✔ Decision Tree & KNN trained from previous user interactions  
✔ Live Gradio Web App (shareable link)  

## 🧠 **Models Used**

### 🔹 **Text Emotion**
`j-hartmann/emotion-english-distilroberta-base`

### 🔹 **Image Emotion**
`trpakov/vit-face-expression`

### 🔹 **Audio Emotion**
`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

### 🔹 **Machine Learning**
- **DecisionTreeClassifier**
- **KNeighborsClassifier**

Both models learn from your **saved CSV dataset** and improve predictions with usage.

---

# ✅ **5. GitHub Social Preview Text (for repository preview card)**

**"Analyze emotions from text, images, and audio using advanced Transformers + CSS animated UI + dynamic avatars. Beautiful, fast, and safe."**

---

# ✅ **6. Preview GIF Creation Instructions**
(So your GitHub README can include an app demo)

I can generate a preview if you upload a screen recording.  
Here is the code snippet for README after you record it:

```markdown
## 🎥 Demo

<p align="center">
  <img src="demo.gif" width="700px"/>
</p>

2️⃣ Install Dependencies

pip install -r requirements.txt
## 📂 **Project Structure**

3️⃣ Run the Application

python app.py

4️⃣ Launch Gradio Interface

The terminal will display a link like:
Running on http://127.0.0.1:7860
Click to open the app in your browser.

🖥️ How It Works

1.User provides text, photo, and audio.
2.Transformers evaluate each input and produce:
   a.Emotion label
   b.Confidence score
3.A bar chart is generated comparing text/image/audio confidence.
4.Data is saved into emotion_results.csv.
5.Decision Tree & KNN get retrained automatically.
6.Both ML models predict the final emotion.
7.Output is shown in a clean UI + table.

📊 CSV Logging

Every prediction is stored in:
=> emotion_results.csv

Columns include:

| timestamp | text | text_label | text_score | image_label | audio_label | ... |

This acts as a growing dataset for ML training.

📸 Screenshots
(Add your own screenshots here)

🔹 User Interface
🔹 Confidence Chart

(Create an /images folder and add images if you want)

🔒 Requirements

See requirements.txt:
gradio
transformers
torch
torchvision
torchaudio
librosa
pandas
matplotlib
Pillow
scikit-learn

🧑‍💻 Author

Preeti Katti
AI | ML | Deep Learning | Python
Feel free to contribute or open issues!

📜 License

This project is licensed under the MIT License — free to use and modify.

⭐ Support

If you like this project:

✔ Star the repository ⭐
✔ Share it
✔ Contribute improvements
