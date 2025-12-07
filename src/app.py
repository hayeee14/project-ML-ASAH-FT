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

@st.cache_data
def load_data():
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            return json.load(f)
    return None

data_ai = load_data()

# Tampilan Utama Aplikasi
if not data_ai:
    # Tampilkan pesan error di layar jika file JSON hilang
    st.error(f"File '{JSON_FILE}' tidak ditemukan. Harap masukkan file JSON hasil Colab ke folder ini.")
else:
    # Sidebar untuk Input Data Kandidat secara Manual
    with st.sidebar:
        st.header("👤 Profil Kandidat")
        st.info("Silakan lengkapi data kandidat di bawah ini untuk laporan final.")
        
        # Mengambil data default dari JSON AI
        candidate_data = data_ai['data']['candidate']
        default_name = candidate_data.get('name', '')
        default_email = candidate_data.get('email', '')
        
        # Input form untuk pengguna
        input_name = st.text_input("Nama Lengkap", value=default_name)
        input_email = st.text_input("Email Kandidat", value=default_email)
        
        # Fitur upload foto profil
        uploaded_file = st.file_uploader("Upload Foto Profil", type=['jpg', 'png', 'jpeg'])
        
        # Logika penanganan foto
        photo_url_for_json = candidate_data.get('photoUrl', '')
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Preview Foto", width=150)
            # Simpan nama file untuk referensi di JSON
            photo_url_for_json = f"uploads/{uploaded_file.name}"
        elif 'http' in photo_url_for_json:
            st.image(photo_url_for_json, width=100)

        st.divider()
        st.caption("Dinilai Oleh: Sistem AI (Google Flan-T5)")

    # Mengambil data review terbaru dari list pastReviews
    if 'pastReviews' in data_ai['data'] and data_ai['data']['pastReviews']:
        review_data = data_ai['data']['pastReviews'][-1]
    else:
        st.error("Format JSON tidak valid (Data Review Kosong).")
        st.stop()

    scores = review_data['scoresOverview']
    decision = review_data['decision']
    results = review_data['reviewChecklistResult']['interviews']['scores']

    # Header Dashboard
    st.title("📝 Laporan Penilaian Interview AI")
    st.markdown(f"**Kandidat:** {input_name} | **Status:** {decision}")

    # Kartu Nilai (Score Cards)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Skor", f"{scores['total']}/100")
    col2.metric("Skor Interview", f"{scores['interview']}/100")
    col3.metric("Skor Project", f"{scores['project']}/100")
    
    with col4:
        if "PASS" in decision.upper():
            st.success(f"## {decision}")
        else:
            st.error(f"## {decision}")

    st.divider()

    # Bagian Detail Jawaban
    st.subheader("🔍 Analisis Jawaban Per Soal")
    
    # Filter tampilan berdasarkan nilai
    filter_score = st.multiselect("Filter Nilai:", [1, 2, 3, 4], default=[1, 2, 3, 4])
    
    for item in results:
        if item['score'] in filter_score:
            # Memberikan ikon warna berdasarkan skor
            if item['score'] == 4:
                score_icon = "🟢"
            elif item['score'] == 3:
                score_icon = "🔵"
            else:
                score_icon = "🟠"
            
            with st.expander(f"{score_icon} Soal ID {item['id']} - Nilai AI: {item['score']}/4"):
                c1, c2 = st.columns([2, 1])
                # Kolom Kiri: Transkrip
                with c1:
                    st.markdown("**Transkripsi Suara (Whisper):**")
                    st.info(f"\"{item['transcript_preview']}\"")
                # Kolom Kanan: Alasan Penilaian
                with c2:
                    st.markdown("**Alasan Penilaian (LLM):**")
                    st.write(item['reason'])

    # Bagian Export Data
    st.divider()
    st.subheader("📂 Export Data")
    
    # Update data JSON di memori dengan inputan baru dari sidebar
    payload = data_ai.copy()
    payload['data']['candidate']['name'] = input_name
    payload['data']['candidate']['email'] = input_email
    payload['data']['candidate']['photoUrl'] = photo_url_for_json
    
    # Konversi ke format string JSON untuk didownload
    json_string = json.dumps(payload, indent=2)
    
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.info("File ini berisi gabungan data manual kandidat dan hasil analisis AI sesuai format payload.")
    with col_d2:
        st.download_button(
            label="📥 Download JSON",
            data=json_string,
            file_name=f"Assessment_{input_name.replace(' ', '_')}.json",
            mime="application/json"
        )
