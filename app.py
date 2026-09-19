import streamlit as st
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image
import random
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Road Damage Detection System", layout="wide", page_icon="🛣️")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg-0: #05070d;
    --bg-1: #0a0f1c;
    --accent-1: #38bdf8;
    --accent-2: #a78bfa;
    --text-hi: #f2f6ff;
    --text-lo: #93a2c2;
    --border: rgba(148, 163, 184, 0.14);
}

html, body, [class*="st-emotion-cache"] { font-family: 'Inter', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(56,189,248,0.10) 0%, transparent 45%),
        radial-gradient(circle at 88% 0%, rgba(167,139,250,0.10) 0%, transparent 40%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-0) 100%);
}

#MainMenu, footer, header { visibility: hidden; }

.hero-wrap {
    padding: 2.6rem 2.4rem;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(56,189,248,0.10), rgba(167,139,250,0.08));
    border: 1px solid var(--border);
    margin-bottom: 1.8rem;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-1);
    font-weight: 600;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.25);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text-hi);
    margin: 0 0 0.5rem 0;
    line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero-sub { color: var(--text-lo); font-size: 1.02rem; max-width: 640px; margin: 0; }
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero-badge {
    font-size: 0.78rem; color: var(--text-hi);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    padding: 0.35rem 0.8rem; border-radius: 999px;
}

.feature-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.4rem 1.3rem;
    height: 100%;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.feature-card:hover { border-color: rgba(56,189,248,0.35); transform: translateY(-2px); }
.feature-icon {
    width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 11px;
    background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(167,139,250,0.18));
    font-size: 1.25rem;
    margin-bottom: 0.85rem;
}
.feature-card-title { margin: 0 0 0.35rem 0; color: var(--text-hi); font-size: 1.02rem; font-weight: 700; font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }
.feature-card p { margin: 0; color: var(--text-lo); font-size: 0.88rem; line-height: 1.5; }

.section-label {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent-1); font-weight: 700; margin: 0 0 0.4rem 0;
}
.section-title { font-size: 1.35rem; font-weight: 700; color: var(--text-hi); margin: 0 0 1rem 0; }

[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed rgba(56,189,248,0.4);
    border-radius: 16px;
    background: rgba(56,189,248,0.04);
}

.stButton>button {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    color: #05070d;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.8rem;
    box-shadow: 0 8px 24px -8px rgba(56,189,248,0.5);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 28px -6px rgba(56,189,248,0.65); }

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--text-lo) !important; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }

[data-testid="stAlert"] { border-radius: 12px; }
section[data-testid="stSidebar"] { background: var(--bg-0); border-right: 1px solid var(--border); }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; }

p, .stMarkdown, label { color: var(--text-lo); }
h1, h2, h3 { color: var(--text-hi); }

.legend-row {
    display: flex; gap: 1.4rem; flex-wrap: wrap;
    padding: 0.7rem 1rem; border-radius: 12px;
    background: rgba(255,255,255,0.03); border: 1px solid var(--border);
    font-size: 0.85rem; color: var(--text-lo); margin-bottom: 0.8rem;
}
.legend-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return YOLO("road_damage_model.pt")
    except Exception as e:
        st.error(f"Could not load the detection model: {e}")
        st.stop()


model = load_model()

BASE_LAT, BASE_LON = 24.8607, 67.0011


def estimate_severity(box, img_width, img_height):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    box_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    area_pct = (box_area / img_area) * 100
    if area_pct >= 8:
        return "Severe", area_pct
    elif area_pct >= 3:
        return "Moderate", area_pct
    else:
        return "Minor", area_pct


def simulate_gps():
    lat = BASE_LAT + random.uniform(-0.03, 0.03)
    lon = BASE_LON + random.uniform(-0.03, 0.03)
    return lat, lon


# ---------- HERO ----------
st.markdown("""
<div class="hero-wrap">
    <span class="hero-eyebrow">Computer Vision · YOLOv8</span>
    <div class="hero-title">🛣️ Road Damage <span>Detection System</span></div>
    <p class="hero-sub">Upload road photos and get instant pothole &amp; drain detection, severity scoring,
    and a downloadable maintenance report — plotted live on an interactive map.</p>
    <div class="hero-badges">
        <span class="hero-badge">🎯 mAP50: 0.861</span>
        <span class="hero-badge">⚡ Real-time inference</span>
        <span class="hero-badge">📍 GPS-tagged reporting</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.info("📍 Note: GPS coordinates in this demo are simulated for the map view. A production deployment would use the camera device's real GPS.")

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-icon">🕳️</div>
    <div class="feature-card-title">Pothole Detection</div>
    <p>A custom-trained YOLOv8 model spots potholes and drains in any road photo, with a mAP50 of 0.861.</p>
    </div>
    """, unsafe_allow_html=True)
with f2:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-icon">⚠️</div>
    <div class="feature-card-title">Severity Scoring</div>
    <p>Each detection is scored Minor, Moderate, or Severe based on how much of the frame it covers.</p>
    </div>
    """, unsafe_allow_html=True)
with f3:
    st.markdown("""
    <div class="feature-card">
    <div class="feature-icon">🗺️</div>
    <div class="feature-card-title">Live Map + Report</div>
    <p>Results plot on an interactive map and export as a downloadable maintenance report.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown('<p class="section-label">Step 1</p><p class="section-title">Upload road images</p>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag and drop road images here, or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "jfif", "avif", "gif"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if "report_data" not in st.session_state:
    st.session_state.report_data = []
if "annotated_images" not in st.session_state:
    st.session_state.annotated_images = []

if uploaded_files and st.button("🚀 Run Detection", type="primary"):
    st.session_state.report_data = []
    st.session_state.annotated_images = []

    progress = st.progress(0, text="Running detection...")
    total = len(uploaded_files)

    for i, uf in enumerate(uploaded_files):
        try:
            img = Image.open(uf).convert("RGB")
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            results = model.predict(img_array, conf=0.25, verbose=False)
            r = results[0]
            annotated = r.plot()
        except Exception as e:
            st.warning(f"Skipped {uf.name}: could not process this image ({e})")
            continue

        st.session_state.annotated_images.append((uf.name, annotated))

        lat, lon = simulate_gps()
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            if cls_name == "pothole":
                severity, area_pct = estimate_severity(box, w, h)
            else:
                severity, area_pct = "N/A", 0

            st.session_state.report_data.append({
                "image": uf.name,
                "damage_type": cls_name,
                "confidence": f"{conf:.1%}",
                "area_pct": round(area_pct, 1),
                "severity": severity,
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
            })
        progress.progress((i + 1) / total, text=f"Processed {i + 1}/{total} images")
    progress.empty()

if st.session_state.annotated_images:
    st.write("")
    st.markdown('<p class="section-label">Results</p><p class="section-title">🖼️ Detected Images</p>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (name, annotated_img) in enumerate(st.session_state.annotated_images):
        with cols[idx % 2]:
            st.image(annotated_img, caption=name, use_container_width=True)

if st.session_state.report_data:
    st.divider()
    df = pd.DataFrame(st.session_state.report_data)
    severity_order = {"Severe": 0, "Moderate": 1, "Minor": 2, "N/A": 3}
    df["_sort"] = df["severity"].map(severity_order)
    df = df.sort_values(["_sort", "area_pct"], ascending=[True, False]).drop(columns="_sort")

    st.markdown('<p class="section-label">Summary</p><p class="section-title">📋 Maintenance Overview</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Maintenance Report", "🗺️ Damage Map"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Detections", len(df))
        c2.metric("Severe", int((df["severity"] == "Severe").sum()))
        c3.metric("Moderate", int((df["severity"] == "Moderate").sum()))
        c4.metric("Minor", int((df["severity"] == "Minor").sum()))

        st.write("")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Maintenance Report (CSV)", csv, "maintenance_report.csv", "text/csv")

    with tab2:
        severity_colors = {"Severe": "red", "Moderate": "orange", "Minor": "green", "N/A": "blue"}
        m = folium.Map(location=[BASE_LAT, BASE_LON], zoom_start=13, tiles="CartoDB dark_matter")

        for _, row in df.iterrows():
            color = severity_colors.get(row["severity"], "gray")
            popup_text = f"""
            <b>{row['damage_type'].title()}</b><br>
            Severity: {row['severity']}<br>
            Confidence: {row['confidence']}<br>
            Image: {row['image']}
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                popup=folium.Popup(popup_text, max_width=200),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(m)

        st.markdown("""
        <div class="legend-row">
            <span><span class="legend-dot" style="background:#ef4444;"></span>Severe</span>
            <span><span class="legend-dot" style="background:#f97316;"></span>Moderate</span>
            <span><span class="legend-dot" style="background:#22c55e;"></span>Minor</span>
            <span><span class="legend-dot" style="background:#3b82f6;"></span>Drain</span>
        </div>
        """, unsafe_allow_html=True)
        st_folium(m, use_container_width=True, height=500)
else:
    st.write("")
    st.markdown(
        '<div class="legend-row" style="justify-content:center;">Upload road images above and click '
        '<b>&nbsp;Run Detection&nbsp;</b> to see results here.</div>',
        unsafe_allow_html=True,
    )
