import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import json
import time
from datetime import datetime
import io

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoScan — Deteksi Sampah Cerdas",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --hijau-tua: #1a4731;
    --hijau-mid: #2d7a4f;
    --hijau-terang: #4caf7d;
    --krem: #f5f0e8;
    --krem-gelap: #ede8df;
    --teks-utama: #1a1a1a;
    --teks-abu: #6b7280;
    --kuning-aksen: #e8c547;
    --merah-muda: #ff6b6b;
    --biru-info: #4a90d9;
    --radius: 16px;
    --shadow: 0 4px 24px rgba(0,0,0,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--krem) !important;
    color: var(--teks-utama);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--hijau-tua) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #e8f5ed !important;
}
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 4px 0;
    display: block;
    cursor: pointer;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.15);
}
section[data-testid="stSidebar"] .stSlider > div { color: #e8f5ed !important; }

/* HEADER */
.eco-header {
    background: linear-gradient(135deg, var(--hijau-tua) 0%, var(--hijau-mid) 100%);
    border-radius: var(--radius);
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.eco-header::before {
    content: "♻";
    position: absolute;
    right: 30px;
    top: 10px;
    font-size: 120px;
    opacity: 0.07;
    line-height: 1;
}
.eco-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.eco-header p {
    color: #a8d5bc;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}

/* CARDS */
.eco-card {
    background: #ffffff;
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow);
    margin-bottom: 20px;
    border: 1px solid rgba(0,0,0,0.04);
}
.eco-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--hijau-tua);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* DETECTION BADGE */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}
.detect-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 100px;
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* INFO PANEL */
.info-panel {
    border-radius: var(--radius);
    padding: 20px 24px;
    margin: 12px 0;
    border-left: 4px solid;
}
.info-daur { background: #e8f5ed; border-color: var(--hijau-terang); }
.info-tidak { background: #fff3f3; border-color: var(--merah-muda); }
.info-tips { background: #fff8e7; border-color: var(--kuning-aksen); }

.info-panel h4 {
    margin: 0 0 8px 0;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
}
.info-panel p, .info-panel li {
    margin: 4px 0;
    font-size: 0.875rem;
    color: var(--teks-abu);
    line-height: 1.6;
}
.info-panel ul { padding-left: 18px; }

/* HISTORY */
.history-item {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border: 1px solid var(--krem-gelap);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.history-time {
    font-size: 0.75rem;
    color: var(--teks-abu);
}
.history-label {
    font-size: 0.875rem;
    font-weight: 500;
}

/* METRIC PILL */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.metric-pill {
    background: var(--krem-gelap);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--teks-utama);
}
.metric-pill span {
    font-weight: 700;
    color: var(--hijau-mid);
}

/* CONFIDENCE BAR */
.conf-bar-wrap { margin: 6px 0; }
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--teks-abu);
    margin-bottom: 4px;
}
.conf-bar-bg {
    background: var(--krem-gelap);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--hijau-terang), var(--hijau-mid));
    transition: width 0.5s ease;
}

/* EMPTY STATE */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--teks-abu);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state p { font-size: 0.95rem; }

/* BUTTONS */
.stButton > button {
    background: var(--hijau-mid) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--hijau-tua) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(45,122,79,0.3) !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: var(--krem-gelap);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 500 !important;
    color: var(--teks-abu) !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--hijau-tua) !important;
    font-weight: 600 !important;
}

/* DIVIDER */
hr { border-color: var(--krem-gelap) !important; }

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: var(--radius);
    border: 2px dashed var(--krem-gelap) !important;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA KELAS ─────────────────────────────────────────────────────────────────
KELAS_INFO = {
    "Kardus": {
        "emoji": "📦",
        "warna": "#8B5E3C",
        "warna_bg": "#f5ede6",
        "daur_ulang": True,
        "deskripsi": "Kardus dan karton bekas kemasan adalah bahan yang sangat mudah didaur ulang menjadi produk kertas baru.",
        "cara_daur": [
            "Ratakan kardus sebelum dibuang agar tidak memakan banyak tempat",
            "Pisahkan dari sampah basah — kardus yang basah sulit didaur ulang",
            "Lepaskan selotip, staples, atau plastik yang menempel",
            "Kumpulkan dan jual ke pengepul atau bawa ke bank sampah"
        ],
        "nilai_ekonomi": "Rp 800 – 1.500 / kg",
        "tips": "Kardus bekas bisa digunakan kembali sebagai bahan kerajinan, lapisan pelindung, atau media tanam.",
        "bahaya": None
    },
    "Kertas": {
        "emoji": "📄",
        "warna": "#4a7fc1",
        "warna_bg": "#e8f0fb",
        "daur_ulang": True,
        "deskripsi": "Kertas merupakan salah satu bahan yang paling umum didaur ulang. Satu ton kertas daur ulang menghemat 17 pohon.",
        "cara_daur": [
            "Pisahkan kertas bersih dari kertas yang terkontaminasi makanan/minyak",
            "Kertas karbon, kertas foto, dan tisu tidak bisa didaur ulang",
            "Sobek atau hancurkan kertas berisi data pribadi sebelum dibuang",
            "Setorkan ke bank sampah atau pengepul kertas"
        ],
        "nilai_ekonomi": "Rp 500 – 1.200 / kg",
        "tips": "Kertas yang masih satu sisi bisa dimanfaatkan sebagai kertas catatan sebelum didaur ulang.",
        "bahaya": None
    },
    "Metal": {
        "emoji": "🥫",
        "warna": "#6b7280",
        "warna_bg": "#f0f1f3",
        "daur_ulang": True,
        "deskripsi": "Logam seperti aluminium dan besi adalah bahan yang bisa didaur ulang berulang kali tanpa kehilangan kualitas.",
        "cara_daur": [
            "Bilas kaleng bekas agar tidak berbau dan tidak menarik hama",
            "Pisahkan jenis logam: aluminium (kaleng minuman) dan besi (kaleng makanan)",
            "Hancurkan atau pipihkan agar lebih ringkas",
            "Jual ke pengepul logam — nilainya cukup tinggi"
        ],
        "nilai_ekonomi": "Aluminium: Rp 8.000–15.000/kg | Besi: Rp 2.000–4.000/kg",
        "tips": "Kaleng bekas bisa dijadikan pot tanaman, tempat alat tulis, atau lampu hias kreatif.",
        "bahaya": "Hati-hati tepi kaleng yang tajam saat menghancurkan."
    },
    "Organik": {
        "emoji": "🥬",
        "warna": "#2d7a4f",
        "warna_bg": "#e8f5ed",
        "daur_ulang": False,
        "deskripsi": "Sampah organik seperti sisa makanan, kulit buah, dan tulang bisa diolah menjadi kompos yang bermanfaat untuk tanah.",
        "cara_daur": [
            "Pisahkan sampah organik dari plastik dan non-organik lainnya",
            "Buat lubang kompos atau gunakan komposter rumah tangga",
            "Campur sampah hijau (sisa sayur) dengan sampah coklat (daun kering)",
            "Kompos siap digunakan dalam 4–8 minggu"
        ],
        "nilai_ekonomi": "Kompos hasil: Rp 500–2.000/kg",
        "tips": "Sampah organik juga bisa diolah menjadi biogas dengan metode anaerob untuk menghasilkan energi.",
        "bahaya": None
    },
    "Plastik": {
        "emoji": "🧴",
        "warna": "#e67e22",
        "warna_bg": "#fef3e8",
        "daur_ulang": True,
        "deskripsi": "Plastik membutuhkan ratusan tahun untuk terurai. Daur ulang plastik sangat penting untuk mengurangi pencemaran lingkungan.",
        "cara_daur": [
            "Cek kode daur ulang (segitiga angka) di bawah produk — kode 1, 2, 4, 5 paling mudah didaur ulang",
            "Bilas wadah plastik dari sisa makanan atau minuman",
            "Pisahkan berdasarkan jenis plastik jika memungkinkan",
            "Bawa ke bank sampah atau pengepul plastik"
        ],
        "nilai_ekonomi": "Rp 1.000 – 5.000 / kg (tergantung jenis)",
        "tips": "Kurangi penggunaan plastik sekali pakai. Pilih produk dengan kemasan plastik yang dapat didaur ulang (kode 1 atau 2).",
        "bahaya": "Plastik jenis 3 (PVC), 6 (PS/styrofoam), dan 7 (lainnya) lebih sulit didaur ulang dan sebaiknya diminimalisir."
    }
}

WARNA_BBOX = {
    "Kardus": (139, 94, 60),
    "Kertas": (74, 127, 193),
    "Metal": (107, 114, 128),
    "Organik": (45, 122, 79),
    "Plastik": (230, 126, 34),
}

# ─── SESSION STATE ───────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "model" not in st.session_state:
    st.session_state.model = None

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────
def get_warna_hex(nama_kelas):
    warna_map = {
        "Kardus": "#8B5E3C",
        "Kertas": "#4a7fc1",
        "Metal": "#6b7280",
        "Organik": "#2d7a4f",
        "Plastik": "#e67e22",
    }
    return warna_map.get(nama_kelas, "#888888")

def get_warna_bg(nama_kelas):
    return KELAS_INFO.get(nama_kelas, {}).get("warna_bg", "#f0f0f0")

def detect_image(model, img_array, conf_threshold):
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    return results[0]

def draw_boxes(img_array, result, class_names):
    img = img_array.copy()
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_id] if cls_id < len(class_names) else f"Kelas {cls_id}"
        warna = WARNA_BBOX.get(label, (100, 100, 100))
        cv2.rectangle(img, (x1, y1), (x2, y2), warna, 3)
        label_text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), warna, -1)
        cv2.putText(img, label_text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def parse_detections(result, class_names):
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_id] if cls_id < len(class_names) else f"Kelas {cls_id}"
        detections.append({"kelas": label, "confidence": conf})
    return detections

def tambah_history(sumber, detections):
    kelas_unik = list(set([d["kelas"] for d in detections]))
    st.session_state.history.insert(0, {
        "waktu": datetime.now().strftime("%H:%M:%S"),
        "tanggal": datetime.now().strftime("%d/%m/%Y"),
        "sumber": sumber,
        "jumlah": len(detections),
        "kelas": kelas_unik,
    })
    if len(st.session_state.history) > 50:
        st.session_state.history = st.session_state.history[:50]

def render_info_kelas(kelas_terdeteksi):
    kelas_unik = list(set(kelas_terdeteksi))
    for kelas in kelas_unik:
        info = KELAS_INFO.get(kelas)
        if not info:
            continue
        emoji = info["emoji"]
        st.markdown(f"""
        <div class="eco-card" style="border-left: 4px solid {info['warna']};">
            <div class="eco-card-title">{emoji} {kelas}</div>
            <p style="font-size:0.9rem; color:#374151; margin-bottom:14px;">{info['deskripsi']}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            status = "✅ Dapat Didaur Ulang" if info["daur_ulang"] else "🌱 Dapat Dikompos"
            panel_class = "info-daur"
            st.markdown(f"""
            <div class="info-panel {panel_class}">
                <h4>{status}</h4>
                <ul>{''.join(f"<li>{c}</li>" for c in info['cara_daur'])}</ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="info-panel info-tips">
                <h4>💡 Tips & Nilai Ekonomi</h4>
                <p><strong>Nilai jual:</strong> {info['nilai_ekonomi']}</p>
                <p>{info['tips']}</p>
                {f'<p style="color:#e53e3e;"><strong>⚠️ Perhatian:</strong> {info["bahaya"]}</p>' if info.get('bahaya') else ''}
            </div>
            """, unsafe_allow_html=True)

def render_confidence_bars(detections):
    from collections import defaultdict
    kelas_conf = defaultdict(list)
    for d in detections:
        kelas_conf[d["kelas"]].append(d["confidence"])

    for kelas, confs in kelas_conf.items():
        avg_conf = sum(confs) / len(confs)
        warna = get_warna_hex(kelas)
        emoji = KELAS_INFO.get(kelas, {}).get("emoji", "")
        st.markdown(f"""
        <div class="conf-bar-wrap">
            <div class="conf-bar-label">
                <span>{emoji} {kelas} ({len(confs)} objek)</span>
                <span style="font-weight:600; color:{warna};">{avg_conf:.1%}</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{avg_conf*100:.1f}%; background: linear-gradient(90deg, {warna}88, {warna});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_badges(detections):
    kelas_list = [d["kelas"] for d in detections]
    from collections import Counter
    kelas_count = Counter(kelas_list)
    badges_html = '<div class="badge-container">'
    for kelas, count in kelas_count.items():
        emoji = KELAS_INFO.get(kelas, {}).get("emoji", "")
        warna = get_warna_hex(kelas)
        warna_bg = get_warna_bg(kelas)
        badges_html += f'<span class="detect-badge" style="background:{warna_bg}; color:{warna};">{emoji} {kelas} × {count}</span>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 28px;">
        <div style="font-size:2.5rem;">♻️</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; color:white; margin-top:8px;">EcoScan</div>
        <div style="font-size:0.75rem; color:#a8d5bc; margin-top:4px;">Deteksi Sampah Cerdas</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Sumber Input**")
    mode = st.radio(
        "Pilih mode:",
        ["📷 Upload Foto", "🎥 Upload Video", "📹 Webcam Real-time"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**⚙️ Pengaturan**")
    conf_threshold = st.slider("Ambang Kepercayaan (Confidence)", 0.1, 0.95, 0.35, 0.05,
                                help="Semakin tinggi = hanya deteksi yang paling yakin")

    st.markdown("---")
    st.markdown("**🏷️ Kelas yang Didukung**")
    for nama, info in KELAS_INFO.items():
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:8px; padding:5px 0; font-size:0.85rem;">
            <span>{info['emoji']}</span>
            <span style="color:white;">{nama}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Hapus Semua History"):
        st.session_state.history = []
        st.success("History dihapus!")

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"
class_names = list(KELAS_INFO.keys())  # ['Kardus', 'Kertas', 'Metal', 'Organik', 'Plastik']

try:
    model = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Gagal memuat model: {e}")

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="eco-header">
    <h1>EcoScan ♻️</h1>
    <p>Sistem Deteksi & Klasifikasi Sampah Berbasis Kecerdasan Buatan · YOLOv8s</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN TABS ──────────────────────────────────────────────────────────────────
tab_deteksi, tab_history, tab_panduan = st.tabs(["🔍 Deteksi", "📋 History", "📖 Panduan Daur Ulang"])

# ══════════════════════════════════════════════════════════════
# TAB 1: DETEKSI
# ══════════════════════════════════════════════════════════════
with tab_deteksi:

    # ── MODE: UPLOAD FOTO ──────────────────────────────────────
    if mode == "📷 Upload Foto":
        uploaded = st.file_uploader(
            "Unggah foto sampah (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded and model_loaded:
            img = Image.open(uploaded).convert("RGB")
            img_array = np.array(img)

            col_ori, col_hasil = st.columns(2)
            with col_ori:
                st.markdown('<div class="eco-card"><div class="eco-card-title">📸 Foto Asli</div>', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.spinner("Menganalisis sampah..."):
                result = detect_image(model, img_array, conf_threshold)
                detections = parse_detections(result, class_names)
                img_hasil = draw_boxes(img_array, result, class_names)

            with col_hasil:
                st.markdown('<div class="eco-card"><div class="eco-card-title">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)
                st.image(img_hasil, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if detections:
                tambah_history("Foto", detections)

                st.markdown('<div class="eco-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="eco-card-title">📊 Ringkasan Deteksi — {len(detections)} objek ditemukan</div>', unsafe_allow_html=True)
                render_badges(detections)
                st.markdown("<br>", unsafe_allow_html=True)
                render_confidence_bars(detections)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download hasil
                img_pil = Image.fromarray(img_hasil)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Unduh Hasil Deteksi",
                    data=buf.getvalue(),
                    file_name=f"ecoscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )

                st.markdown("---")
                st.markdown("### 📚 Informasi & Panduan Penanganan")
                render_info_kelas([d["kelas"] for d in detections])
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="icon">🔍</div>
                    <p>Tidak ada sampah terdeteksi.<br>Coba turunkan ambang kepercayaan atau gunakan foto yang lebih jelas.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state eco-card">
                <div class="icon">📷</div>
                <p>Unggah foto sampah untuk memulai deteksi.<br>
                Sistem mendukung format JPG, PNG, dan WEBP.</p>
            </div>
            """, unsafe_allow_html=True)

    # ── MODE: UPLOAD VIDEO ─────────────────────────────────────
    elif mode == "🎥 Upload Video":
        uploaded_video = st.file_uploader(
            "Unggah video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed"
        )

        if uploaded_video and model_loaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            skip = max(1, int(fps // 5))  # proses 5 frame/detik

            st.info(f"📹 Video: {total_frames} frame · {fps:.0f} FPS · Memproses setiap {skip} frame")

            stframe = st.empty()
            progress = st.progress(0)
            status_text = st.empty()

            all_detections = []
            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detect_image(model, frame_rgb, conf_threshold)
                    dets = parse_detections(result, class_names)
                    all_detections.extend(dets)
                    frame_out = draw_boxes(frame_rgb, result, class_names)
                    stframe.image(frame_out, use_container_width=True)
                    prog = min(frame_num / total_frames, 1.0)
                    progress.progress(prog)
                    status_text.text(f"Frame {frame_num}/{total_frames}")
                frame_num += 1

            cap.release()
            os.unlink(tfile.name)
            progress.empty()
            status_text.empty()

            if all_detections:
                tambah_history("Video", all_detections)
                st.success(f"✅ Selesai! Total {len(all_detections)} deteksi ditemukan.")
                st.markdown('<div class="eco-card"><div class="eco-card-title">📊 Ringkasan Video</div>', unsafe_allow_html=True)
                render_badges(all_detections)
                st.markdown("<br>", unsafe_allow_html=True)
                render_confidence_bars(all_detections)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
                render_info_kelas([d["kelas"] for d in all_detections])

    # ── MODE: WEBCAM ───────────────────────────────────────────
    elif mode == "📹 Webcam Real-time":
        st.markdown("""
        <div class="eco-card">
            <div class="eco-card-title">📹 Deteksi Real-time via Webcam</div>
        </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2, _ = st.columns([1, 1, 3])
        with col_btn1:
            mulai = st.button("▶️ Mulai Kamera")
        with col_btn2:
            berhenti = st.button("⏹️ Hentikan")

        stframe_cam = st.empty()
        info_cam = st.empty()

        if mulai and model_loaded:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Kamera tidak dapat diakses. Pastikan perangkat memiliki webcam.")
            else:
                all_cam_dets = []
                st.session_state["cam_running"] = True
                while st.session_state.get("cam_running", False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = detect_image(model, frame_rgb, conf_threshold)
                    dets = parse_detections(result, class_names)
                    all_cam_dets.extend(dets)
                    frame_out = draw_boxes(frame_rgb, result, class_names)
                    stframe_cam.image(frame_out, use_container_width=True)

                    if dets:
                        badges_html = '<div class="badge-container">'
                        from collections import Counter
                        for kelas, count in Counter([d["kelas"] for d in dets]).items():
                            emoji = KELAS_INFO.get(kelas, {}).get("emoji", "")
                            warna = get_warna_hex(kelas)
                            warna_bg = get_warna_bg(kelas)
                            badges_html += f'<span class="detect-badge" style="background:{warna_bg}; color:{warna};">{emoji} {kelas} × {count}</span>'
                        badges_html += '</div>'
                        info_cam.markdown(badges_html, unsafe_allow_html=True)

                cap.release()
                if all_cam_dets:
                    tambah_history("Webcam", all_cam_dets)

        if berhenti:
            st.session_state["cam_running"] = False

# ══════════════════════════════════════════════════════════════
# TAB 2: HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="eco-card-title" style="font-size:1.1rem; font-family:Space Grotesk,sans-serif; font-weight:600; color:#1a4731; margin-bottom:16px;">📋 Riwayat Deteksi</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="empty-state eco-card">
            <div class="icon">📋</div>
            <p>Belum ada riwayat deteksi.<br>Lakukan deteksi terlebih dahulu untuk melihat history di sini.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Export history
        history_json = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Unduh History (JSON)",
            data=history_json,
            file_name=f"ecoscan_history_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        for item in st.session_state.history:
            kelas_badges = " ".join([
                f'<span style="background:{get_warna_bg(k)}; color:{get_warna_hex(k)}; padding:2px 10px; border-radius:100px; font-size:0.75rem; font-weight:600;">{KELAS_INFO.get(k,{}).get("emoji","")} {k}</span>'
                for k in item["kelas"]
            ])
            st.markdown(f"""
            <div class="history-item">
                <div>
                    <div class="history-label">🎯 {item['jumlah']} objek · {item['sumber']}</div>
                    <div style="margin-top:6px;">{kelas_badges}</div>
                </div>
                <div class="history-time">{item['tanggal']}<br>{item['waktu']}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3: PANDUAN DAUR ULANG
# ══════════════════════════════════════════════════════════════
with tab_panduan:
    st.markdown('<div class="eco-card-title" style="font-size:1.1rem; font-family:Space Grotesk,sans-serif; font-weight:600; color:#1a4731; margin-bottom:20px;">📖 Panduan Lengkap Penanganan Sampah</div>', unsafe_allow_html=True)

    for nama, info in KELAS_INFO.items():
        with st.expander(f"{info['emoji']} {nama} — {'Dapat Didaur Ulang' if info['daur_ulang'] else 'Dikompos'}"):
            st.markdown(f"**{info['deskripsi']}**")
            st.markdown("**Cara Penanganan:**")
            for step in info["cara_daur"]:
                st.markdown(f"- {step}")
            st.markdown(f"**💰 Nilai Ekonomi:** {info['nilai_ekonomi']}")
            st.markdown(f"**💡 Tips:** {info['tips']}")
            if info.get("bahaya"):
                st.warning(f"⚠️ {info['bahaya']}")

    st.markdown("---")
    st.markdown("""
    <div class="eco-card">
        <div class="eco-card-title">🏦 Di Mana Membuang Sampah Daur Ulang?</div>
        <div class="info-panel info-daur">
            <h4>Bank Sampah</h4>
            <p>Temukan bank sampah terdekat melalui aplikasi <strong>SIPSN</strong> (Sistem Informasi Pengelolaan Sampah Nasional) atau hubungi Dinas Lingkungan Hidup setempat.</p>
        </div>
        <div class="info-panel info-tips">
            <h4>Pengepul & Lapak</h4>
            <p>Pengepul keliling biasanya menerima kardus, kertas, logam, dan botol plastik. Nilai jual bisa lebih tinggi jika sampah sudah dipilah dan dibersihkan.</p>
        </div>
        <div class="info-panel info-tidak">
            <h4>Sampah Organik</h4>
            <p>Buat komposter mandiri di rumah atau titipkan ke program kompos komunitas di RT/RW setempat.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
