import streamlit as st
import os
import shutil
import json
import torch
import whisper
import imageio_ffmpeg 
import subprocess
import re
from datetime import datetime
from transformers import pipeline
from PIL import Image

# --- KONFIGURASI SISTEM ---
# Memastikan binary FFmpeg tersedia untuk lingkungan Windows
try:
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(os.getcwd(), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        shutil.copy(ffmpeg_src, ffmpeg_dst)
except Exception:
    pass 

st.set_page_config(page_title="Interview Assessment System", layout="wide")

# --- DATA: POSISI PEKERJAAN & PERTANYAAN ---

# Pemetaan Posisi Pekerjaan ke ID Pertanyaan
JOB_ROLES = {
    "Machine Learning Engineer": [1, 2, 3, 4],
    "Data Scientist": [5, 6, 7, 8],
    "Computer Vision Engineer": [9, 10, 11, 12],
    "NLP Engineer": [13, 14, 15, 16],
    "AI Architect": [17, 18, 19, 20]
}

# Database Pertanyaan (Total 20)
QUESTION_DB = {
    # ML Engineer
    1: "Describe a complex ML model you built and how you ensured its accuracy.",
    2: "Explain the concept of Overfitting and techniques to prevent it.",
    3: "How do you handle imbalanced datasets in a classification problem?",
    4: "Explain the difference between Batch, Stochastic, and Mini-batch Gradient Descent.",
    
    # Data Scientist
    5: "Walk us through your process of Data Cleaning and Preprocessing.",
    6: "Explain the difference between Supervised and Unsupervised Learning.",
    7: "How do you select the most important features for your model?",
    8: "Explain the concept of Bias-Variance Tradeoff.",
    
    # Computer Vision Engineer
    9: "Describe the architecture of a Convolutional Neural Network (CNN).",
    10: "How does Max Pooling work and why is it used?",
    11: "Explain the concept of Transfer Learning in Image Classification.",
    12: "How do you handle Object Detection tasks (e.g., YOLO, R-CNN)?",
    
    # NLP Engineer
    13: "Explain the mechanism of Tokenization and Word Embeddings.",
    14: "Describe the Transformer architecture and the Attention mechanism.",
    15: "How do you handle sequence data using RNNs or LSTMs?",
    16: "Explain the concept of Named Entity Recognition (NER).",
    
    # AI Architect
    17: "How do you design a scalable MLOps pipeline?",
    18: "What factors do you consider when choosing between Cloud vs Edge deployment?",
    19: "How do you ensure data privacy and security in AI systems?",
    20: "Describe a high-level architecture for a real-time recommendation system."
}

# Database Kata Kunci untuk Logika Penilaian (Diperluas)
# Semakin banyak kata kunci yang disebut, semakin tinggi skor kandidat.
TOPIC_KEYWORDS = {
    1: ["model", "accuracy", "tuning", "optimization", "hyperparameter", "loss", "metric", "validation", "auc", "roc", "precision", "recall", "deploy", "pipeline"],
    2: ["overfitting", "regularization", "dropout", "l1", "l2", "early", "stopping", "data", "augmentation", "complexity", "generalization", "noise", "pruning"],
    3: ["imbalanced", "smote", "resampling", "oversampling", "undersampling", "class", "weight", "f1", "precision", "minority", "majority", "stratified", "synthetic"],
    4: ["gradient", "descent", "batch", "stochastic", "update", "weight", "convergence", "learning", "rate", "optimizer", "momentum", "adam", "cost", "function"],
    
    5: ["clean", "missing", "value", "imputation", "outlier", "normalize", "standardize", "pandas", "null", "duplicate", "scaling", "encoding", "categorical", "transform"],
    6: ["supervised", "unsupervised", "label", "clustering", "classification", "regression", "k-means", "target", "prediction", "discovery", "dimension", "pca"],
    7: ["feature", "selection", "importance", "pca", "correlation", "dimensional", "reduction", "recursive", "lasso", "ridge", "filter", "wrapper", "information"],
    8: ["bias", "variance", "tradeoff", "underfitting", "overfitting", "error", "complexity", "model", "flexible", "irreducible", "total", "generalize"],
    
    9: ["cnn", "convolution", "layer", "filter", "kernel", "feature", "map", "relu", "activation", "stride", "padding", "flatten", "dense", "softmax"],
    10: ["pooling", "max", "average", "dimension", "reduction", "spatial", "downsampling", "parameter", "invariance", "translation", "computation", "summary"],
    11: ["transfer", "learning", "pretrained", "imagenet", "vgg", "resnet", "finetuning", "weights", "freeze", "domain", "adaptation", "source", "target"],
    12: ["object", "detection", "yolo", "rcnn", "bounding", "box", "anchor", "intersection", "iou", "confidence", "suppression", "region", "proposal", "grid"],
    
    13: ["token", "embedding", "vector", "word2vec", "glove", "representation", "vocabulary", "corpus", "semantic", "similarity", "cosine", "context", "n-gram", "stemming"],
    14: ["transformer", "attention", "mechanism", "encoder", "decoder", "bert", "gpt", "self-attention", "multi-head", "positional", "encoding", "scale", "dot-product"],
    15: ["rnn", "lstm", "gru", "sequence", "temporal", "memory", "vanishing", "gradient", "gate", "forget", "dependency", "long-term", "short-term", "state"],
    16: ["ner", "entity", "recognition", "tagging", "extraction", "information", "text", "label", "person", "organization", "location", "chunking", "bio"],
    
    17: ["mlops", "pipeline", "ci/cd", "deployment", "monitoring", "docker", "kubernetes", "model", "registry", "versioning", "drift", "serving", "api", "automated"],
    18: ["cloud", "edge", "latency", "bandwidth", "cost", "privacy", "device", "resource", "server", "connectivity", "real-time", "power", "consumption", "security"],
    19: ["privacy", "security", "encryption", "gdpr", "compliance", "anonymization", "access", "control", "federated", "learning", "differential", "secure", "multiparty"],
    20: ["recommendation", "system", "real-time", "collaborative", "filtering", "latency", "streaming", "kafka", "matrix", "factorization", "content-based", "hybrid", "user", "item"]
}

# --- INISIALISASI MODEL ---
@st.cache_resource
def load_models():
    """
    Menginisialisasi model Whisper (STT) dan Flan-T5 (LLM).
    Menggunakan caching agar model tidak dimuat ulang setiap kali aplikasi direfresh.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Whisper untuk Transkripsi Suara
    whisper_model = whisper.load_model("base", device=device)
    
    # Flan-T5 untuk Generasi Penalaran (Reasoning)
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base", 
        device=0 if device == "cuda" else -1,
        max_length=512
    )
    return whisper_model, llm_pipeline

try:
    with st.spinner("Sedang memuat sistem AI (Whisper & LLM)..."):
        model_stt, model_llm = load_models()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- FUNGSI UTILITAS ---

def convert_video_to_audio(video_path, audio_path):
    """
    Mengekstrak audio dari file video menggunakan subprocess FFmpeg.
    Menghasilkan file .wav 16kHz mono.
    """
    try:
        ffmpeg_cmd = "ffmpeg.exe" if os.path.exists("ffmpeg.exe") else "ffmpeg"
        command = [
            ffmpeg_cmd, "-i", video_path, "-vn", 
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            audio_path, "-y"
        ]
        
        # Menyembunyikan jendela terminal pada Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=startupinfo)
        return True
    except Exception:
        return False

def transcribe(audio_path):
    """
    Melakukan transkripsi file audio menjadi teks menggunakan Whisper.
    Menggunakan prompt awal untuk meningkatkan akurasi istilah teknis.
    """
    technical_prompt = "Transcribe English technical interview. Keywords: Machine Learning, Data Science, AI, Engineering."
    result = model_stt.transcribe(audio_path, fp16=False, language="en", initial_prompt=technical_prompt)
    return result["text"].strip()

def assess_answer(text, question_id):
    """
    Menilai jawaban berdasarkan kepadatan kata kunci (Skoring) dan LLM (Penalaran).
    Logika Penilaian:
    - 0: Jawaban tidak valid / terlalu pendek
    - 1: Tidak Relevan (0 kata kunci ditemukan)
    - 2: Dasar (1-2 kata kunci ditemukan)
    - 3: Baik (3-5 kata kunci ditemukan)
    - 4: Sangat Baik (>5 kata kunci ditemukan)
    """
    # 1. Validasi Panjang Teks
    if len(text) < 10: 
        return 0, "No audible response detected or response too short."
    
    # 2. Analisis Kata Kunci
    target_keywords = TOPIC_KEYWORDS.get(question_id, [])
    text_lower = text.lower()
    
    # Mencari irisan antara kata kunci target dan teks transkrip
    found_keywords = [k for k in target_keywords if k in text_lower]
    hit_count = len(set(found_keywords)) # Menghitung kata kunci unik
    
    # 3. Penentuan Skor
    if hit_count == 0:
        score = 1
        reason_base = "The answer does not contain relevant technical keywords for this topic."
    elif hit_count <= 2:
        score = 2
        reason_base = f"Basic understanding. Mentioned keywords: {', '.join(found_keywords)}."
    elif hit_count <= 5:
        score = 3
        reason_base = f"Solid technical explanation. Key concepts covered: {', '.join(found_keywords)}."
    else:
        score = 4
        reason_base = f"Comprehensive and detailed response. Extensive vocabulary used: {', '.join(found_keywords)}."

    # 4. Memperhalus Alasan menggunakan LLM
    prompt = f"""
    Refine this assessment reason to sound professional.
    Original Reason: "{reason_base}"
    Score: {score}/4
    Output: A single professional sentence.
    """
    
    try:
        reason_ai = model_llm(prompt, max_length=128, do_sample=False)[0]['generated_text']
    except:
        reason_ai = reason_base

    return score, reason_ai

# --- ANTARMUKA PENGGUNA (UI) ---

# Sidebar: Profil Kandidat
with st.sidebar:
    st.header("Candidate Profile")
    cand_name = st.text_input("Full Name", "Hafiz Putra Mahesta")
    cand_email = st.text_input("Email", "hafiz@dicoding.com")
    
    # Pilihan Posisi Pekerjaan
    selected_role = st.selectbox("Position Applied", list(JOB_ROLES.keys()))
    
    uploaded_photo = st.file_uploader("Profile Picture", type=['jpg','png'])
    photo_url = "https://path/to/default.png"
    
    if uploaded_photo:
        image = Image.open(uploaded_photo)
        st.image(image, width=150)
        photo_url = f"uploads/{uploaded_photo.name}"
    
    st.divider()
    st.caption("System Version: 1.0.0 (Localhost)")

# Dashboard Utama
st.title("AI Interview Assessment System")
st.markdown(f"**Target Position:** {selected_role}")

col_vid, col_q = st.columns([1, 1])

with col_q:
    # Filter pertanyaan berdasarkan peran yang dipilih
    role_questions = JOB_ROLES[selected_role]
    selected_q_id = st.selectbox(
        "Select Interview Question:", 
        role_questions, 
        format_func=lambda x: f"Q{x}: {QUESTION_DB[x]}"
    )

with col_vid:
    uploaded_video = st.file_uploader("Upload Response Video (.mp4)", type=["mp4", "mov", "avi", "webm"])

if uploaded_video is not None:
    st.video(uploaded_video)
    
    if st.button("Analyze Response", type="primary"):
        # Setup Direktori Sementara
        os.makedirs("temp", exist_ok=True)
        video_path = os.path.join("temp", uploaded_video.name)
        audio_path = video_path.replace(".mp4", ".wav").replace(".webm", ".wav")
        
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
            
        # Pipeline Pemrosesan
        with st.status("Processing...", expanded=True):
            st.write("Extracting audio stream...")
            if convert_video_to_audio(video_path, audio_path):
                
                st.write("Transcribing audio content...")
                transcript_text = transcribe(audio_path)
                
                st.write("Evaluating technical relevance...")
                score, reason = assess_answer(transcript_text, selected_q_id)
                
                st.divider()
                
                # Menampilkan Hasil
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.info(f"**Transcription:**\n\n{transcript_text}")
                with c2:
                    if score == 4: st.success(f"**Score: {score}/4** (Excellent)")
                    elif score == 3: st.info(f"**Score: {score}/4** (Good)")
                    elif score == 2: st.warning(f"**Score: {score}/4** (Basic)")
                    else: st.error(f"**Score: {score}/4** (Irrelevant)")
                    st.write(f"**Assessment:**\n{reason}")
                
                # Pembuatan Payload JSON
                final_json = {
                    "success": True,
                    "data": {
                        "candidate": {
                            "name": cand_name, 
                            "email": cand_email,
                            "role": selected_role,
                            "photoUrl": photo_url
                        },
                        "assessorProfile": {"id": 47, "name": "AI System"},
                        "reviewedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "decision": "PASSED" if score >= 3 else "REVIEW NEEDED",
                        "scoresOverview": {"total": (score/4)*100},
                        "reviewChecklistResult": {
                            "project": [],
                            "interviews": {
                                "scores": [{
                                    "id": selected_q_id, 
                                    "score": score, 
                                    "reason": reason, 
                                    "transcript_preview": transcript_text
                                }]
                            }
                        }
                    }
                }
                
                st.download_button(
                    label="Export JSON Report",
                    data=json.dumps(final_json, indent=2),
                    file_name=f"Assessment_{cand_name.replace(' ', '_')}.json",
                    mime="application/json"
                )
            else:
                st.error("Error processing video file. Please check the format.")
