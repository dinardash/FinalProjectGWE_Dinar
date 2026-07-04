import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

st.set_page_config(
    page_title="Final Project - Dinar",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
    <style>
    /* ---------- General ---------- */
    .stApp {
        background-color: #F7F8FA;
    }
    #MainMenu, footer {visibility: hidden;}

    /* ---------- Hero Header ---------- */
    .hero {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #3B82F6 100%);
        padding: 36px 40px;
        border-radius: 16px;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(30, 58, 138, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background-color: rgba(255,255,255,0.15);
        color: #E0E7FF;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 8px;
        line-height: 1.15;
    }
    .sub-title {
        font-size: 16px;
        color: #DBEAFE;
        max-width: 700px;
        line-height: 1.5;
    }

    /* ---------- Section Titles ---------- */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .section-subtitle {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 18px;
    }

    /* ---------- Cards ---------- */
    .card {
        background-color: #FFFFFF;
        padding: 22px 24px;
        border-radius: 14px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    .card-title {
        font-size: 15px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-text {
        font-size: 14.5px;
        color: #374151;
        line-height: 1.6;
    }

    /* ---------- Pills / tags ---------- */
    .pill {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }

    /* ---------- Result boxes (prediction tab) ---------- */
    .result-box {
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.04);
    }
    .result-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        opacity: 0.75;
        display: block;
        margin-bottom: 6px;
    }

    /* ---------- Metric cards override ---------- */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 14px 16px 8px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #6B7280;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #EEF2FF;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        padding: 0 18px;
        font-weight: 600;
        color: #4B5563;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #1E3A8A !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* ---------- Footer note ---------- */
    .footer-note {
        text-align: center;
        color: #9CA3AF;
        font-size: 12.5px;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Tentang Proyek")
    st.markdown(
        "Platform ini menganalisis **sentimen** dan **topik** dari ulasan "
        "aplikasi ojeg online (Grab) di Google Play Store"
    )
    st.divider()
    st.markdown("**Model yang digunakan**")
    st.markdown("- IndoBERT (fine-tuned)")
    st.divider()
    st.markdown("**Ringkasan Data**")
    st.markdown(
        "- Dataset: Grab App Reviews - Indonesia (Google Play Store)\n"
        "- 100.000 baris data\n"
        "- Sumber: Kaggle\n"
    )
    st.divider()

# =========================================================
# HERO / TITLE SECTION
# =========================================================
st.markdown("""
    <div class="hero">
        <div class="hero-badge">Final Project of GWE</div>
        <div class="main-title">Analisis Sentimen &amp; Topik Ulasan Aplikasi Ojeg Online</div>
        <div class="sub-title">
            Platform analisis sentimen dan topik pada ulasan aplikasi
            ojeg online (Grab) di Google Play Store
        </div>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# TAB LAYOUT
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Beranda",
    "Informasi Dataset",
    "Prediksi Real-time",
    "Metodologi & Dokumentasi"
])

# --- TAB 1: BERANDA ---
with tab1:
    st.markdown('<div class="section-title">Selayang Pandang</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Latar belakang dan tujuan dari proyek analisis ini.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        st.markdown("""
            <div class="card">
                <div class="card-title">📌 Latar Belakang</div>
                <div class="card-text">
                    Berdasarkan riset Statista (2024), jumlah pengguna aktif ojek online 
                    di Indonesia telah mencapai lebih dari 36 juta orang per bulan, 
                    yang menunjukkan bahwa layanan ini telah menjadi bagian dari mobilitas masyarakat. 
                    Namun, seiring berkembangnya skala aplikasi dan jumlah pengguna, 
                    tantangan dalam memonitor kualitas layanan secara juga semakin besar. 
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
            <div class="card">
                <div class="card-title">🎯 Tujuan Proyek</div>
                <div class="card-text">
                    &bull; Mengetahui tanggapan pengguna mengenai ojek online.<br><br>
                    &bull; Mengetahui topik ojek online yang menjadi perhatian utama pengguna.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("""
        <div class="card">
            <div class="card-title">⚙️ Deskripsi Sistem</div>
            <div class="card-text">
                Proyek ini dibangun sebagai alat bantu berbasis Kecerdasan Buatan (AI) untuk melakukan penambangan
                opini pengguna melalui pemrosesan bahasa alami (NLP), menghasilkan klasifikasi sentimen yang
                presisi serta kluster topik yang terstruktur.
            </div>
            <div style="margin-top:14px;">
                <span class="pill">NLP</span>
                <span class="pill">IndoBERT</span>
                <span class="pill">BERTopic</span>
                <span class="pill">Analisis Sentimen</span>
                <span class="pill">Klasifikasi Topik</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- TAB 2: INFORMASI DATASET ---
with tab2:
    st.markdown('<div class="section-title">Eksplorasi Data (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ringkasan dan hasil eksplorasi terhadap dataset opini pendidikan.</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(label="Total Baris Data", value="100,000")
    m2.metric(label="Total Kolom", value="6")
    m3.metric(label="Missing Values", value="13,770")
    m4.metric(label="Data Duplikat", value="0")

    st.caption("Sumber data: [Kaggle Dataset — Grab App Reviews - Indonesia (Google Play Store)](https://www.kaggle.com/datasets/pandaa12/grab-app-reviews-indonesia-google-play-store)")
    st.divider()

    # --- Wordcloud & Text Analysis ---
    st.markdown("#### 1️⃣ Analisis Frekuensi Kata (Wordcloud)")
    col_img1, col_txt1 = st.columns([1.2, 1], gap="medium")
    with col_img1:
        st.image("gambar/wordcloud.png", use_container_width=True)
    with col_txt1:
        st.markdown("""
            <div class="card">
                <div class="card-title">🔍 Temuan Kata Kunci Ulasan Aplikasi</div>
                <div class="card-text">
                    <b>Pelayanan Driver: </b>Dominasi kata <i>‘driver’</i>, <i>‘pelayanan’</i>, <i>‘nunggu’</i> dan <i>‘bagus’</i>
                    menunjukkan bahwa bahwa kinerja dan perilaku pengemudi merupakan aspek yang paling 
                    disoroti oleh pengguna.
                    <b>Kualitas Aplikasi: </b>Kemunculan kata <i>‘aplikasi’</i>, <i>‘grabfood’</i>, dan <i>‘iklan’</i> menunjukan 
                    perhatian masyarakat terhadap fungsionalitas aplikasi Grab beserta fitur-fitur layanan di dalamnya.
                    <b>Tarif Layanan: </b> Kemunculan kata <i>‘promo’</i>, <i>‘mahal’</i>, dan <i>‘harga’</i>  menunjukan bahwa pengguna cukup kritis 
                    terhadap kebijakan tarif yang diterapkan oleh Grab.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- Topic Analysis ---
    st.markdown("#### 2️⃣ Analisis Distribusi Topik")
    col_txt2, col_img2 = st.columns([1, 1.2], gap="medium")
    with col_txt2:
        st.markdown("""
            <div class="card">
                <div class="card-title">🗂️ Daftar Topik Utama yang Teridentifikasi</div>
                <div class="card-text">
                    <b>Topik 1</b> — Oulier (Tidak dapat dikelompokan secara spesifik)<br><br>
                    <b>Topik 2</b> — Kualitas pelayanan Grab<br><br>
                    <b>Topik 3</b> — Kinerja aplikasi dan sistem Grab<br><br>
                    <b>Topik 4</b> — Kualitas pelayanan driver<br><br>
                    <b>Topik 5</b> — Akurasi lokasi dan Navigasi
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col_img2:
        st.image("gambar/Distribusi Topik.png", use_container_width=True)

    st.divider()

    # --- Sentiment Analysis ---
    st.markdown("#### 3️⃣ Analisis Sentimen Makro & Mikro")
    col_img3, col_img4 = st.columns(2, gap="medium")
    with col_img3:
        st.image("gambar/Distribusi Sentimen.png", use_container_width=True)
        st.markdown(
            '<div class="card-text" style="font-size:13.5px; color:#6B7280; margin-top:6px;">'
            'Sekitar <i>60%</i> ulasan dalam dataset diklasifikasikan sebagai sentimen positif.'
            'Dominasi sentimen positif ini menunjukan bahwa secara umum tingkat kepuasan pengguna' 
            'terhadap performa aplikasi dan pelayanan driver cenderung baik.</div>',
            unsafe_allow_html=True
        )
    with col_img4:
        st.image("gambar/Distribusi Sentimen Berdasarkan Topik.png", use_container_width=True)
        st.markdown(
            '<div class="card-text" style="font-size:13.5px; color:#6B7280; margin-top:6px;">'
            "Topik 'Kinerja aplikasi dan sistem Grab' dan 'Kualitas pelayanan driver' didominasi oleh sentimen negatif. "
            'Hal ini menegaskan bahwa meskipun kepuasan umum berada di angka 60%, kedua aspek operasional tersebut merupakan titik kritis (pain points) utama yang paling sering dikeluhkan oleh pengguna.</div>',
            unsafe_allow_html=True
        )

    # --- Sentiment Analysis ---
    st.markdown("#### 4️⃣ Analisis pengaruh ulasan terhadap citra aplikasi")
    col_img5, col_img6 = st.columns(2, gap="medium")
    with col_img5:
        st.image("gambar/Rata-Rata Rating Aplikasi.png", use_container_width=True)
        st.markdown(
            '<div class="card-text" style="font-size:13.5px; color:#6B7280; margin-top:6px;">'
            'Terdapat korelasi kuat antara jenis sentimen dengan rating yang diberikan oleh pengguna' 
            'Hal ini menunjukan  bahwa kepuasan emosional pengguna berdampak langsung pada rating penilaian aplikasi.</div>',
            unsafe_allow_html=True
        )
    with col_img6:
        st.image("gambar/Rata-Rata Like Ulasan.png", use_container_width=True)
        st.markdown(
            '<div class="card-text" style="font-size:13.5px; color:#6B7280; margin-top:6px;">'
            'Tingginya likes pada sentimen negatif menunjukan adanya kecenderungan di mana pengguna memberikan dukungan terhadap keluhan serupa agar lebih terlihat oleh pihak pengembang.</div>',
            unsafe_allow_html=True
        )
    
    col_img7, col_img8 = st.columns(2, gap="medium")
    with col_img7:
        st.image("gambar/Rata-Rata Jumlah Kata.png", use_container_width=True)
        st.markdown(
            '<div class="card-text" style="font-size:13.5px; color:#6B7280; margin-top:6px;">'
            'Ulasan bersentimen negatif memiliki rata-rata jumlah kata terpanjan.' 
            'Hal ini menunjukan bahwa pengguna yang mengalami ketidakpuasan cenderung menulis secara deskriptif ketidaknyamanan yang mereka rasakan.</div>',
            unsafe_allow_html=True
        )
        
# --- TAB 3: PREDIKSI ---
with tab3:
    st.markdown('<div class="section-title">Uji Coba Model Klasifikasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Masukkan teks opini atau tweet di bawah ini untuk melihat bagaimana '
        'kecerdasan buatan mengklasifikasikan sentimen dan topik secara langsung.</div>',
        unsafe_allow_html=True
    )

    @st.cache_resource
    def load_models():
        tokenizer = AutoTokenizer.from_pretrained("tokenizer_saved")
        model_sentimen = AutoModelForSequenceClassification.from_pretrained("model_sentimen_saved")
        model_topik = AutoModelForSequenceClassification.from_pretrained("model_topik_saved")
        return tokenizer, model_sentimen, model_topik

    model_loaded = False
    try:
        tokenizer, model_sentimen, model_topik = load_models()
        model_loaded = True
    except Exception as e:
        st.error(f"Sistem gagal memuat model terarsip di lokal. Pastikan folder model tersedia. (Error: {e})")

    def predict_text(text, model, tokenizer):
        inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        return np.argmax(logits.numpy(), axis=-1)[0]

    with st.container(border=True):
        user_input = st.text_area(
            "Input Ulasan",
            placeholder="Tulis opini mengenai pendidikan di sini...",
            height=120
        )
        run = st.button("Jalankan Analisis", type="primary", use_container_width=True)

    if run:
        if user_input.strip() == "":
            st.warning("⚠️ Mohon masukkan teks terlebih dahulu sebelum menekan tombol analisis.")
        elif not model_loaded:
            st.error("⛔ Proses dihentikan karena model machine learning belum siap.")
        else:
            with st.spinner("Model sedang mengevaluasi konten teks..."):
                pred_sentimen = predict_text(user_input, model_sentimen, tokenizer)
                pred_topik = predict_text(user_input, model_topik, tokenizer)

                sentiment_labels = {
                    0: "NEGATIF",
                    1: "NETRAL",
                    2: "POSITIF"
                }
                
                sentiment_colors = {
                    0: ("#FEE2E2", "#991B1B"),  # merah
                    1: ("#FEF3C7", "#92400E"),  # kuning
                    2: ("#D1FAE5", "#065F46")   # hijau
                }
                
                sentiment_label = sentiment_labels[pred_sentimen]
                bg_color, text_color = sentiment_colors[pred_sentimen]

                topik_list = [
                    "Oulier (Tidak dapat dikelompokan secara spesifik)",
                    "Kualitas pelayanan Grab",
                    "Kinerja aplikasi dan sistem Grab",
                    "Kualitas pelayanan driver",
                    "Akurasi lokasi dan Navigasi"
                ]

                st.write("")
                st.success("✅ Analisis Komputasi Selesai!")

                col_res1, col_res2 = st.columns(2, gap="medium")
                with col_res1:
                    st.markdown(
                        f'<div class="result-box" style="background-color:{bg_color}; color:{text_color};">'
                        f'<span class="result-label">Sentimen Terdeteksi</span>{sentiment_label}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with col_res2:
                    st.markdown(
                        f'<div class="result-box" style="background-color:#E0F2FE; color:#0369A1;">'
                        f'<span class="result-label">Kategori Topik</span>Topik {pred_topik + 1}: {topik_list[pred_topik]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# --- TAB 4: METODOLOGI & DOKUMENTASI ---
with tab4:
    st.markdown('<div class="section-title">Metodologi Arsitektur Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Pendekatan dan teknik yang digunakan dalam membangun model.</div>', unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown("""
            <div class="card">
                <div class="card-title">🧠 IndoBERT (Pre-trained Model)</div>
                <div class="card-text">
                    Model NLP berbasis transformer yang dilatih menggunakan korpus berbahasa Indonesia,
                    sehingga mampu memahami konteks bahasa Indonesia dengan baik. Model ini juga
                    efektif mengenali bahasa informal, singkatan, slang, serta gaya penulisan
                    yang umum digunakan pada ulasan pengguna aplikasi.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-title">Performa &amp; Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Perbandingan performa model klasifikasi topik dan sentimen.</div>', unsafe_allow_html=True)
 
    col_topik, col_sentimen = st.columns(2, gap="medium")
 
    # --- KOLOM 1: KLASIFIKASI TOPIK ---
    with col_topik:
        with st.container(border=True):
            st.markdown('<div class="card-title">Performa Klasifikasi Topik</div>', unsafe_allow_html=True)
 
            r1c1, r1c2 = st.columns(2)
            r1c1.metric(label="Accuracy", value="80%")
            r1c2.metric(label="Precision", value="78.5%")
            r2c1, r2c2 = st.columns(2)
            r2c1.metric(label="Recall", value="80%")
            r2c2.metric(label="F1-Score", value="78.84%")
 
            st.image("gambar/CM topik.png", caption="Confusion Matrix Model Topik Dominan", use_container_width=True)
 
    # --- KOLOM 2: KLASIFIKASI SENTIMEN ---
    with col_sentimen:
        with st.container(border=True):
            st.markdown('<div class="card-title">Performa Klasifikasi Sentimen</div>', unsafe_allow_html=True)
 
            r3c1, r3c2 = st.columns(2)
            r3c1.metric(label="Accuracy", value="93.75%")
            r3c2.metric(label="Precision", value="93.7%")
            r4c1, r4c2 = st.columns(2)
            r4c1.metric(label="Recall", value="93.75%")
            r4c2.metric(label="F1-Score", value="93.45%")
 
            st.image("gambar/CM Sentimen.png", caption="Confusion Matrix Model Sentimen", use_container_width=True)
 
    st.divider()

    # Developer Profile
    st.markdown("### 👤 Tim Pengembang")
    col_dev_img, col_dev_txt = st.columns([1, 4])
    with col_dev_img:
        st.image("gambar/dinar.png", use_container_width=True)
    with col_dev_txt:
        st.markdown("""
            <div style="padding-top:10px;">
                <div style="font-size:17px; font-weight:700; color:#111827;">Dinar Fadlilatunnisa</div>
                <div style="font-size:14px; color:#6B7280;">Information System '24</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer-note">© 2026 · Analisis Sentimen &amp; Topik Pendidikan Indonesia · Dibangun dengan Streamlit &amp; IndoBERTweet</div>', unsafe_allow_html=True)
