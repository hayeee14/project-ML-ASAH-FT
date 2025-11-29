import streamlit as st
import json
import pandas as pd
import os
from PIL import Image

# Konfigurasi halaman website
st.set_page_config(
    page_title="Sistem Penilaian Interview AI",
    page_icon="🤖",
    layout="wide"
)

# Lokasi file JSON hasil olahan AI dari Google Colab
# Pastikan file ini satu folder dengan app.py
JSON_FILE = "final_assessment_result.json"

# Fungsi untuk memuat data agar tidak berat (cache)
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
    st.subheader("📂 Export Data Final")
    
    # Update data JSON di memori dengan inputan baru dari sidebar
    final_payload = data_ai.copy()
    final_payload['data']['candidate']['name'] = input_name
    final_payload['data']['candidate']['email'] = input_email
    final_payload['data']['candidate']['photoUrl'] = photo_url_for_json
    
    # Konversi ke format string JSON untuk didownload
    json_string = json.dumps(final_payload, indent=2)
    
    col_d1, col_d2 = st.columns([3, 1])
    with col_d1:
        st.info("File ini berisi gabungan data manual kandidat dan hasil analisis AI sesuai format payload.")
    with col_d2:
        st.download_button(
            label="📥 Download JSON Final",
            data=json_string,
            file_name=f"Assessment_{input_name.replace(' ', '_')}.json",
            mime="application/json"
        )
