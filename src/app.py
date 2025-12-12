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
# Memastikan library FFmpeg tersedia 
try:
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dst = os.path.join(os.getcwd(), "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        shutil.copy(ffmpeg_src, ffmpeg_dst)
except Exception:
    pass 

st.set_page_config(page_title="Sistem Penilaian Interview AI", layout="wide")

# --- DATA: POSISI PEKERJAAN & PERTANYAAN ---

# Pemetaan Posisi Pekerjaan ke ID Pertanyaan
JOB_ROLES = {
    "Machine Learning Engineer": [1, 2, 3, 4],
    "Data Scientist": [5, 6, 7, 8],
    "Computer Vision Engineer": [9, 10, 11, 12],
    "NLP Engineer": [13, 14, 15, 16],
    "AI Architect": [17, 18, 19, 20]
}

# Database Pertanyaan 
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

# --- DATABASE KATA KUNCI (DIPERLUAS & DIPERBANYAK) ---
# Semakin banyak kata kunci teknis yang disebut kandidat, semakin tinggi skornya.
TOPIC_KEYWORDS = {
    1: ["model", "accuracy", "tuning", "optimization", "hyperparameter", "loss", "metric", "validation", "auc", "roc", "precision", "recall", "deploy", "pipeline", "baseline", "evaluation", "confusion matrix", "cross-validation", "testing", "monitoring"],
    2: ["overfitting", "regularization", "dropout", "l1", "l2", "early", "stopping", "data", "augmentation", "complexity", "generalization", "noise", "pruning", "variance", "training data", "bias", "ensemble", "simplify", "penalty"],
    3: ["imbalanced", "smote", "resampling", "oversampling", "undersampling", "class", "weight", "f1", "precision", "minority", "majority", "stratified", "synthetic", "adayan", "cost-sensitive", "recall", "anomaly", "skewed", "distribution"],
    4: ["gradient", "descent", "batch", "stochastic", "update", "weight", "convergence", "learning", "rate", "optimizer", "momentum", "adam", "cost", "function", "epoch", "iteration", "minima", "backpropagation", "parameter"],
    
    5: ["clean", "missing", "value", "imputation", "outlier", "normalize", "standardize", "pandas", "null", "duplicate", "scaling", "encoding", "categorical", "transform", "preprocessing", "quality", "formatting", "binning", "wrangling", "etl"],
    6: ["supervised", "unsupervised", "label", "clustering", "classification", "regression", "k-means", "target", "prediction", "discovery", "dimension", "pca", "labeled", "unlabeled", "training", "association", "pattern", "input", "output"],
    7: ["feature", "selection", "importance", "pca", "correlation", "dimensional", "reduction", "recursive", "lasso", "ridge", "filter", "wrapper", "information", "gain", "mutual", "variance", "heatmap", "redundant", "engineering"],
    8: ["bias", "variance", "tradeoff", "underfitting", "overfitting", "error", "complexity", "model", "flexible", "irreducible", "total", "generalize", "training", "test", "balance", "sweet spot", "predictive", "performance"],
    
    9: ["cnn", "convolution", "layer", "filter", "kernel", "feature", "map", "relu", "activation", "stride", "padding", "flatten", "dense", "softmax", "visual", "image", "pixel", "architecture", "deep", "learning"],
    10: ["pooling", "max", "average", "dimension", "reduction", "spatial", "downsampling", "parameter", "invariance", "translation", "computation", "summary", "feature", "map", "size", "compress", "information", "receptive", "field"],
    11: ["transfer", "learning", "pretrained", "imagenet", "vgg", "resnet", "finetuning", "weights", "freeze", "domain", "adaptation", "source", "target", "efficientnet", "mobilenet", "feature", "extraction", "saving", "time"],
    12: ["object", "detection", "yolo", "rcnn", "bounding", "box", "anchor", "intersection", "iou", "confidence", "suppression", "region", "proposal", "grid", "localization", "classification", "fast-rcnn", "mask", "ssd"],
    
    13: ["token", "embedding", "vector", "word2vec", "glove", "representation", "vocabulary", "corpus", "semantic", "similarity", "cosine", "context", "n-gram", "stemming", "lemmatization", "split", "preprocessing", "dense", "dimension", "text"],
    14: ["transformer", "attention", "mechanism", "encoder", "decoder", "bert", "gpt", "self-attention", "multi-head", "positional", "encoding", "scale", "dot-product", "sequence", "parallel", "training", "context", "nlp", "model"],
    15: ["rnn", "lstm", "gru", "sequence", "temporal", "memory", "vanishing", "gradient", "gate", "forget", "dependency", "long-term", "short-term", "state", "recurrent", "network", "input", "output", "hidden"],
    16: ["ner", "entity", "recognition", "tagging", "extraction", "information", "text", "label", "person", "organization", "location", "chunking", "bio", "token", "sequence", "model", "spacy", "nltk", "crf"],
    
    17: ["mlops", "pipeline", "ci/cd", "deployment", "monitoring", "docker", "kubernetes", "model", "registry", "versioning", "drift", "serving", "api", "automated", "workflow", "production", "scale", "lifecycle", "infrastructure"],
    18: ["cloud", "edge", "latency", "bandwidth", "cost", "privacy", "device", "resource", "server", "connectivity", "real-time", "power", "consumption", "security", "iot", "processing", "centralized", "decentralized"],
    19: ["privacy", "security", "encryption", "gdpr", "compliance", "anonymization", "access", "control", "federated", "learning", "differential", "secure", "multiparty", "data", "protection", "leakage", "attack", "robustness"],
    20: ["recommendation", "system", "real-time", "collaborative", "filtering", "latency", "streaming", "kafka", "matrix", "factorization", "content-based", "hybrid", "user", "item", "ranking", "personalization", "cold-start", "feedback"]
}

# --- INISIALISASI MODEL ---
@st.cache_resource
def load_models():
    """
    Memuat model Whisper (STT) dan Flan-T5 (LLM).
    Menggunakan caching untuk performa.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Whisper untuk Transkripsi
    whisper_model = whisper.load_model("small", device=device)
    
    # Flan-T5 untuk Generasi Alasan Penilaian
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
    Ekstraksi audio dari video menggunakan subprocess FFmpeg.
    """
    try:
        ffmpeg_cmd = "ffmpeg.exe" if os.path.exists("ffmpeg.exe") else "ffmpeg"
        command = [
            ffmpeg_cmd, "-i", video_path, "-vn", 
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            audio_path, "-y"
        ]
        
        # Sembunyikan window terminal di Windows
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
    Transkripsi audio ke teks menggunakan Whisper.
    """
    technical_prompt = "Transcribe English technical interview. Keywords: Machine Learning, Data Science, AI, Engineering."
    result = model_stt.transcribe(audio_path, fp16=False, language="en", initial_prompt=technical_prompt)
    return result["text"].strip()

def assess_answer(text, question_id):
    """
    Menilai jawaban berdasarkan jumlah keyword (Skor) dan LLM (Alasan).
    Logika Skor:
    - 0: Audio rusak / teks < 10 karakter
    - 1: 0 keyword (Salah Topik)
    - 2: 1-2 keyword (Dasar)
    - 3: 3-5 keyword (Baik)
    - 4: >5 keyword (Sangat Baik)
    """
    # Validasi
    if len(text) < 10: 
        return 0, "No audible response detected or response too short."
    
    # Hitung Keyword
    target_keywords = TOPIC_KEYWORDS.get(question_id, [])
    text_lower = text.lower()
    
    found_keywords = [k for k in target_keywords if k in text_lower]
    hit_count = len(set(found_keywords)) 
    
    # Tentukan Skor
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

    # Perhalus reason dengan LLM
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

# --- INTERFACE PENGGUNA (UI) ---
with st.sidebar:
    st.header("Profil Kandidat")
    cand_name = st.text_input("Nama Lengkap", " ")
    cand_email = st.text_input("Email", " ")
    
    selected_role = st.selectbox("Posisi yang Dilamar", list(JOB_ROLES.keys()))
    
    uploaded_photo = st.file_uploader("Foto Profil", type=['jpg','png'])
    photo_url = "https://path/to/default.png"
    if uploaded_photo:
        image = Image.open(uploaded_photo)
        st.image(image, width=150)
        photo_url = f"{uploaded_photo.name}"
    
    st.divider()
    st.caption("Versi Sistem: 1.0.0 (Localhost)")

st.title("Sistem Penilaian Interview AI")
st.markdown(f"**Target Posisi:** {selected_role}")

col_vid, col_q = st.columns([1, 1])

with col_q:
    role_questions = JOB_ROLES[selected_role]
    selected_q_id = st.selectbox(
        "Pilih Pertanyaan Interview:", 
        role_questions, 
        format_func=lambda x: f"Q{x}: {QUESTION_DB[x]}"
    )

with col_vid:
    uploaded_video = st.file_uploader("Unggah Video Jawaban (.mp4)", type=["mp4", "mov", "avi", "webm"])

if uploaded_video is not None:
    st.video(uploaded_video)
    
    if st.button("Analisis Jawaban", type="primary"):
        # Setup Folder Temp
        os.makedirs("temp", exist_ok=True)
        video_path = os.path.join("temp", uploaded_video.name)
        audio_path = video_path.replace(".mp4", ".wav").replace(".webm", ".wav")
        
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
            
        with st.status("Sedang Memproses...", expanded=True):
            st.write("Mengekstrak audio...")
            if convert_video_to_audio(video_path, audio_path):
                
                st.write("Transkripsi suara...")
                transcript_text = transcribe(audio_path)
                
                st.write("Melakukan penilaian teknis...")
                score, reason = assess_answer(transcript_text, selected_q_id)
                
                st.divider()
                
                # Menampilkan Hasil di Layar
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.info(f"**Transkripsi:**\n\n{transcript_text}")
                with c2:
                    if score == 4: st.success(f"**Skor: {score}/4** (Sangat Baik)")
                    elif score == 3: st.info(f"**Skor: {score}/4** (Baik)")
                    elif score == 2: st.warning(f"**Skor: {score}/4** (Cukup)")
                    else: st.error(f"**Skor: {score}/4** (Tidak Relevan)")
                    st.write(f"**Analisis:**\n{reason}")
                
                # --- MEMBUAT OUTPUT JSON ---
                # Menghitung nilai proporsional untuk simulasi
                interview_percentage = (score / 4) * 100
                total_final_score = (100 + interview_percentage) / 2 # Asumsi Project Score 100
                
                final_json = {
                    "success": True,
                    "data": {
                        "id": 131, # ID Mockup
                        "candidate": {
                            "name": cand_name, 
                            "email": cand_email,
                            "photoUrl": photo_url
                        },
                        "certification": {
                            "abbreviatedType": "DCML",
                            "normalType": "DEV_CERTIFICATION_MACHINE_LEARNING",
                            "submittedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "FINISHED",
                            "projectType": "dcml_package_1",
                            "interviewQuestionSets": 1,
                            "examScore": total_final_score,
                            "autoGraderProjectScore": 100,
                            "downloadProjectUrl": "https://github.com/hayeee14/project-ML-ASAH-FT",
                            "isReviewedByMe": False,
                            "isAlreadyReviewedByMe": False,
                            "assess": {
                                "project": False,
                                "interviews": True
                            }
                        },
                        "reviewChecklists": {
                            "project": [],
                            "interviews": [
                                {
                                    "positionId": selected_q_id,
                                    "question": QUESTION_DB[selected_q_id],
                                    "isVideoExist": True,
                                    "recordedVideoUrl": "https://drive.google.com/file/d/mock-url-video/view" #hanya link dummy
                                }
                            ]
                        },
                        "pastReviews": [
                            {
                                "assessorProfile": {
                                    "id": 47, 
                                    "name": "AI Assessor System", 
                                    "photoUrl": "xxx"
                                },
                                "decision": "PASSED" if score >= 3 else "REVIEW NEEDED",
                                "reviewedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "scoresOverview": {
                                    "project": 100,
                                    "interview": round(interview_percentage, 1),
                                    "total": round(total_final_score, 1)
                                },
                                "reviewChecklistResult": {
                                    "project": [],
                                    "interviews": {
                                        "minScore": 0,
                                        "maxScore": 4,
                                        "scores": [
                                            {
                                                "id": selected_q_id,
                                                "score": score
                                            }
                                        ]
                                    }
                                },
                                "notes": reason # Menggunakan analisis AI 
                            }
                        ]
                    }
                }
                
                st.download_button(
                    label="Unduh Laporan JSON",
                    data=json.dumps(final_json, indent=2),
                    file_name=f"Assessment_{cand_name.replace(' ', '_')}.json",
                    mime="application/json"
                )
            else:
                st.error("Gagal memproses file video. Cek formatnya.")
