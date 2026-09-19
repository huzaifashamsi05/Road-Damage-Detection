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
<style>
.stApp { background: linear-gradient(180deg, #0a0e17 0%, #0d1424 100%); }
h1 { font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(90deg, #38bdf8, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
p, .stMarkdown { color: #b8c4d9; }
[data-testid="stFileUploaderDropzone"] { border: 1.5px dashed rgba(56,189,248,0.35); border-radius: 14px; background: rgba(56,189,248,0.05); }
.stButton>button { background: linear-gradient(90deg, #38bdf8, #8b5cf6); color: #05070d; font-weight: 700; border: none; border-radius: 8px; padding: 0.6rem 1.6rem; }
[data-testid="stMetric"] { background: #121a2c; border: 1px solid rgba(56,189,248,0.18); border-radius: 12px; padding: 1rem; }
[data-testid="stAlert"] { border-radius: 10px; }
section[data-testid="stSidebar"] { background: #0a0e17; border-right: 1px solid rgba(148,163,184,0.08); }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
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

st.title("🛣️ Road Damage Detection System")
st.caption("YOLOv8 Object Detection | Pothole + Drain Classification | mAP50: 0.861")
st.info("📍 Note: GPS coordinates in this demo are simulated for the map view. A production deployment would use the camera device's real GPS.")

uploaded_files = st.file_uploader(
    "Upload road images",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "jfif", "avif", "gif"],
    accept_multiple_files=True
)

if "report_data" not in st.session_state:
    st.session_state.report_data = []
if "annotated_images" not in st.session_state:
    st.session_state.annotated_images = []

if uploaded_files and st.button("Run Detection", type="primary"):
    st.session_state.report_data = []
    st.session_state.annotated_images = []

    for uf in uploaded_files:
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

if st.session_state.annotated_images:
    st.subheader("🖼️ Detected Images")
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

    tab1, tab2 = st.tabs(["📋 Maintenance Report", "🗺️ Damage Map"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Detections", len(df))
        c2.metric("Severe", (df["severity"] == "Severe").sum())
        c3.metric("Moderate", (df["severity"] == "Moderate").sum())
        c4.metric("Minor", (df["severity"] == "Minor").sum())

        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Maintenance Report (CSV)", csv, "maintenance_report.csv", "text/csv")

    with tab2:
        severity_colors = {"Severe": "red", "Moderate": "orange", "Minor": "green", "N/A": "blue"}
        m = folium.Map(location=[BASE_LAT, BASE_LON], zoom_start=13)

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

        st.markdown("🔴 Severe &nbsp;&nbsp; 🟠 Moderate &nbsp;&nbsp; 🟢 Minor &nbsp;&nbsp; 🔵 Drain")
        st_folium(m, use_container_width=True, height=500)
