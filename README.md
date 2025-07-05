# Multi-Modal Smart Surveillance System

A real-time intelligent surveillance system that integrates:
- **Object Detection** (YOLOv8)
- **Suspicious Object Alerts**
- **Real-time OCR (Text Extraction)**
- **Voice Alerts using Text-to-Speech (TTS)**

---

## Project Description
This project is a **multi-modal smart surveillance system** designed for real-time security applications. It uses **computer vision and natural language processing (NLP)** techniques to:
- Detect suspicious objects (like backpacks, knives, bottles, suitcases)
- Extract text from live video (OCR)
- Provide voice alerts when a suspicious object is detected

The system is built using **OpenCV, YOLOv8, Pytesseract, and pyttsx3.**

---

## 📂 Project Structure
```text
multi-modal-surveillance/
│
├── app.py # Main application file
├── requirements.txt # List of Python dependencies
├── README.md # Project documentation
├── yolov8n.pt # Pre-trained YOLOv8 nano model 
└── Other support files 
```

---
## Tech Stack
- Python 3.10.11
- OpenCV
- YOLOv8 (Ultralytics)
- Pytesseract (OCR)
- pyttsx3 (Text-to-Speech)

---

## Features
- Real-time object detection using YOLOv8.
- Real-time text extraction (OCR) from camera feed.
- Real-time voice alerts for suspicious objects.
- Multi-modal interaction: Camera + Text + Audio.
- Simple interface (press `Q` to quit)

---

## Project Highlights
- Combines OpenCV + YOLOv8 + OCR + TTS into a seamless real-time system.
- Can be further extended with custom object detection datasets.
- Potential for public safety, smart surveillance, and assistive technologies.

---

by
MOUNESH M
---
