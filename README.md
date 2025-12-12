# AI-Powered Interview Assessment System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![AI Model](https://img.shields.io/badge/Model-Whisper%20%7C%20Flan--T5-green)

A comprehensive AI solution designed to automate the technical interview assessment process. This system utilizes **OpenAI Whisper** for high-accuracy Speech-to-Text (STT) and **Google Flan-T5** for intelligent scoring and reasoning generation.

> **Capstone Project Team A25-CS358**

## Key Features

* **Hybrid Scoring Engine:** Combines strict keyword density analysis (Python Logic) with LLM reasoning (Generative AI) for robust and hallucination-free assessment.
* **Multi-Model Pipeline:** Seamless integration of Speech-to-Text and NLP Reasoning models.
* **Dynamic JSON Reporting:** Generates backend-ready JSON reports including candidate profile, scores, detailed analysis, and dynamic video links.
* **Local Privacy First:** No 3rd-party APIs used. All processing happens locally on your machine (On-Premise), ensuring data privacy and zero cost.
* **User-Friendly Interface:** Interactive dashboard built with Streamlit.

---

## Tech Stack

* **Interface:** Streamlit
* **Speech-to-Text:** OpenAI Whisper (`base` model)
* **LLM (Reasoning):** Google Flan-T5 (`base` model) via Hugging Face Transformers
* **Video Processing:** FFmpeg (via `imageio-ffmpeg`)
* **Validation:** Word Error Rate (WER) Analysis (available in Notebook)

---

## Installation Guide

Follow these steps to set up the project on your local machine.

### 1. Prerequisite
Ensure you have **Python 3.10** or newer installed.
* **Recommended Hardware:** 8GB RAM minimum (16GB preferred for faster processing).
* **GPU:** Optional (Nvidia CUDA), but highly recommended for faster transcription.

### 2. Clone Repository
```
git clone https://github.com/hayeee14/project-ML-ASAH-A25-CS358.git
cd project-ML-ASAH-A25-CS358
```

### 3. Create Virtual Environment 
```
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
This will install all required libraries including PyTorch, Whisper, and Transformers.
```
pip install -r requirements.txt
```
(Note: ffmpeg binary is handled automatically by the application logic, no manual installation required).

## System Requirements

This application is optimized for standard laptops using **Base Models**. However, performance varies based on hardware specifications.

| Component | Minimum (Default 'base' Models) | Recommended (For 'small/Large' Models) |
| :--- | :--- | :--- |
| **OS** | Windows 10/11, macOS, Linux | Windows 10/11, Linux (Ubuntu) |
| **RAM** | 8 GB | **16 GB or higher** |
| **CPU** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / AMD Ryzen 7 |
| **GPU** | Optional (Runs on CPU) | **NVIDIA GPU (6GB+ VRAM)** |
| **Storage** | 2 GB free space | 10 GB free space |

> **Note on Performance:**
> By default, the `app.py` is configured to use **`whisper-small`** and **`flan-t5-base`** to ensure smooth performance on most laptops (CPU-only).
>
> If you wish to use **Large Models** for higher accuracy:
> 1. Ensure you have a dedicated NVIDIA GPU (CUDA enabled).
> 2. Open `app.py` and change `model="google/flan-t5-base"` to `model="google/flan-t5-large"`.
> 3. Change `whisper.load_model("base")` to `whisper.load_model("small")`.

## AI Model Information

This project utilizes state-of-the-art pre-trained models. The model weights are **downloaded automatically** via the Python libraries upon the first run.

**1. Speech-to-Text Model:**
* **Model:** OpenAI Whisper (`base` or `small`)
* **Source Repository:** [https://github.com/openai/whisper](https://github.com/openai/whisper)
* **Method:** Auto-downloaded via `pip install git+https://github.com/openai/whisper.git`

**2. Reasoning/LLM Model:**
* **Model:** Google Flan-T5 (`base` or `large`)
* **Source Repository:** [https://huggingface.co/google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
* **Method:** Auto-downloaded via `transformers` library.

> **Note to Reviewer:**
> As per the library's standard behavior, the model weights are fetched dynamically from the official repositories above. Therefore, there is no separate Google Drive link for manual model download. Please ensure internet access is available during the first execution.

# How to Run
1. Start the application using Streamlit:
```
streamlit run src/app.py
```
2. Your browser will automatically open http://localhost:8501.
3. Steps to use:
* Select the Job Position (e.g., Machine Learning Engineer).
* Select the Interview Question ID.
* Upload your Video Answer (.mp4, .mov).
* Click "Analyze Response".
4. Wait for the AI to process (Extraction -> Transcription -> Scoring).
5. Download the JSON Report for the result.

> **First Run Notice:**
> When running the application for the **very first time**, the system will automatically download the Whisper and Flan-T5 models (approx. 2-3 GB).
> Please allow **10-15 minutes** for this initial setup depending on your internet connection. Subsequent runs will be instant as the models are cached locally.

# Project Structure
```
Capstone-Interview-AI/
├── notebook/
│   └── notebook.ipynb          # Research Notebook (Accuracy/WER Validation)
├── src/
│   └── app.py                  # Main Application Logic (Streamlit)
├── requirements.txt            # Python Dependencies
├── packages.txt                # System Dependencies (FFmpeg)
├── final_assessment_result.json # Sample JSON Output
└── README.md                   # Project Documentation
```
# Assessment Logic (How it Works)
1. The system uses a Hybrid Evaluation Method to ensure fairness and accuracy:
Relevance Guardrail (Python): The system extracts technical keywords relevant to the specific question chosen.
* 0 Keywords: Score 1 (Irrelevant).
* 1-2 Keywords: Score 2 (Basic).
* 3-5 Keywords: Score 3 (Good).
* 6+ Keywords: Score 4 (Excellent).
2. Reasoning Generation (LLM): The score is fed into Google Flan-T5, which generates a professional, human-like justification for the score given.

# Team Members
* **Muhammad Rayhan**, M262D5Y1357, sebagai PIC Model & Training (Streamlit/Interface)
* **Hafiz Putra Mahesta**, M262D5Y0714, sebagai PIC Integrasi, Model STT, & Fitur (Confidence Score)
* **Fahri Rasyidin**, M262D5Y0566, sebagai PIC Data & Evaluasi (Dataset, Kunci Jawaban, WER)

Note: This project is designed for Localhost Deployment as a Proof of Concept (PoC). Video URLs in the JSON report are simulated for demonstration purposes.
