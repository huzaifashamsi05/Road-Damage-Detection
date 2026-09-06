# Road Damage Detection System

A YOLOv8-based computer vision app that detects potholes and drains from road images, estimates damage severity, and generates a maintenance report with a live map.

Live app: https://road-damage-detection-ljrk3ska9i3uuh8aetpcck.streamlit.app/

## Features

- Detects potholes and drains in uploaded road images using a custom-trained YOLOv8 model (mAP50: 0.861)
- Estimates severity (Minor, Moderate, Severe) based on how much of the frame the damage covers
- Generates a sortable maintenance report with confidence scores and severity, downloadable as a CSV
- Plots detections on an interactive map with severity-coded markers
- Supports batch upload of multiple images at once

## How it works

1. Upload one or more road images (JPG, PNG, WEBP, and other common formats)
The YOLOv8 model runs detection on each image and identifies potholes and drains
Detected potholes are scored for severity based on how much of the frame they cover
Results are shown as annotated images, a sortable maintenance report, and a map vi

## Tech stack

Python, Streamlit, Ultralytics YOLOv8, Folium, Pandas

## Local setup
```
pip install -r requirements.txt
streamlit run app.py
```
 
## Note on GPS data

The map view in this demo uses simulated GPS coordinates. A production deployment would read GPS from the camera device at the time of capture.

## Limitations

- Trained on a limited dataset, so accuracy may vary on road types or lighting conditions not well represented in training
- Severity is estimated from bounding box area, a simple heuristic rather than a calibrated depth or width measurement
