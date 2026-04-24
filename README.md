# Pipeline Defect Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySide6](https://img.shields.io/badge/PySide6-6.5+-green.svg)
![YOLO](https://img.shields.io/badge/YOLOv8-8.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Real-time industrial defect detection system with YOLO object detection and interactive ROI selection**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Configuration](#configuration) • [Project Structure](#project-structure)

</div>

---

## Overview

This is a **PySide6-based desktop application** for real-time pipeline defect detection using YOLOv8. The system processes video streams (files or RTMP) and identifies defects within user-defined regions of interest (ROI).

### Key Features

- **Real-time Detection** — YOLOv8-powered defect detection with adjustable confidence threshold
- **Interactive ROI** — Click 4 points on video to define detection region
- **RTMP Streaming** — Connect to live RTMP streams for real-time monitoring
- **Audio Alarms** — Configurable alarm triggers when defects are detected
- **Recording & Screenshots** — Capture evidence with one click
- **Detection Log** — Track detection events with timestamps
- **Modern UI** — Dark industrial theme with professional look and feel

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# Clone or navigate to the project directory
cd pipeline_detector

# Install dependencies
pip install -r requirements.txt

# Place your trained model (best.pt) in the project root
# If no model is found, the app will use yolov8n.pt as fallback

# Run the application
python main.py
```

### Model Requirements

Place your YOLOv8 model file as `best.pt` in the project root. The application supports:
- Custom trained YOLOv8 models (recommended: `best.pt`)
- Ultralytics standard models (fallback: `yolov8n.pt`)

---

## Usage

### Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  CONTROL PANEL                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    VIDEO DISPLAY AREA                      │
│              (Click 4 points to define ROI)                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  [Open File] [Connect] [Play] [Stop]                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Detection Region (ROI)                               │  │
│  │ Points: 0/4  [Start Detection] [Clear]               │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Detection Log                                        │  │
│  │ [12:30:45] Detected 3 defects (max conf: 0.92)       │  │
│  │ [12:30:40] Detected 1 defects (max conf: 0.87)       │  │
│  └──────────────────────────────────────────────────────┘  │
│  Status: Connected | FPS: 30 | Detections: 0 | Alarm: OK   │
└─────────────────────────────────────────────────────────────┘
```

### Quick Start

1. **Open Video Source**
   - Click **Open File** to load a local video file (MP4, AVI, MOV)
   - Or enter an RTMP URL and click **Connect**

2. **Define Detection Region**
   - Click **4 points** on the video to draw a quadrilateral ROI
   - The region will be highlighted in cyan

3. **Start Detection**
   - Click **Start Detection** to begin monitoring
   - Detected defects will be highlighted with red boxes
   - Audio alarm triggers when defects are found (with configurable cooldown)

4. **Capture Evidence**
   - Click **Screenshot** to save current frame
   - Click **Record** to start/stop video recording

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open video file |
| `Ctrl+S` | Take screenshot |
| `Ctrl+,` | Open settings |

---

## Configuration

### Settings Dialog

Access via **Settings** button or `Ctrl+,`:

| Setting | Description | Default |
|---------|-------------|---------|
| Confidence Threshold | Detection confidence (0.1-1.0) | 0.85 |
| Detection Interval | Frames between detections | 1 |
| Alarm Cooldown | Seconds between alarms | 5s |
| Default RTMP URL | Stream URL preset | - |

### Runtime Configuration

Edit `utils/config.py` to modify:

```python
@dataclass
class Config:
    confidence: float = 0.85
    alarm_cooldown: int = 5000  # milliseconds
    audio_file: str = "jing.mp3"
    model_path: str = "best.pt"
    video_update_interval: int = 30  # ms
```

---

## Project Structure

```
pipeline_detector/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
│
├── ui/                     # User interface module
│   ├── __init__.py
│   ├── main_window.py      # Main application window
│   ├── widgets.py          # Custom widgets (DrawArea, ImagePreviewDialog)
│   └── styles.py           # Dark industrial theme stylesheet
│
├── core/                   # Core detection engine
│   ├── __init__.py
│   ├── detector.py         # YOLO detector wrapper + ROI filtering
│   ├── video_capture.py    # Threaded video capture worker
│   └── alarm.py            # Audio alarm controller
│
├── utils/                  # Utilities
│   ├── __init__.py
│   └── config.py           # Application configuration
│
├── recordings/            # Output: recorded videos
├── screenshots/           # Output: screenshots
└── best.pt               # YOLO model weights (user-provided)
```

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `core/detector.py` | YOLO model loading, inference, ROI filtering |
| `core/video_capture.py` | Threaded frame capture from files/streams |
| `core/alarm.py` | Audio playback with cooldown logic |
| `ui/main_window.py` | Application UI and event handling |
| `ui/widgets.py` | Custom Qt widgets (video display, ROI drawing) |
| `ui/styles.py` | Dark industrial theme CSS stylesheet |

---

## API Reference

### Detector

```python
from core.detector import Detector

detector = Detector("best.pt", confidence=0.85)
detections = detector.detect(frame, roi_points=None)
# Returns: [{'bbox': [x1,y1,x2,y2], 'conf': 0.92, 'cls': 0, 'center': [cx,cy]}, ...]
```

### VideoCapture

```python
from core.video_capture import VideoCaptureWorker

worker = VideoCaptureWorker(queue, max_size=2)
worker.open("video.mp4")  # or "rtmp://example.com/live"
ret, frame = worker.read()
```

---

## Troubleshooting

### Common Issues

**Model fails to load**
- Ensure `best.pt` exists in project root
- Check ultralytics installation: `pip install ultralytics`

**No video display**
- Install OpenCV: `pip install opencv-python`
- Check video file integrity

**RTMP connection fails**
- Verify stream URL is correct
- Check network connectivity
- Ensure SRS/server is running

**Alarm not playing**
- Verify `jing.mp3` exists in project root
- Check system audio volume

---

## License

MIT License - See LICENSE file for details.

---

<div align="center">

**Built with PySide6 + YOLOv8**

</div>
