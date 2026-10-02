# 442OOLS // META-VISION — Football Tactical Intelligence Studio
> **OpenCV AI Competition 2026 (powered by AWS)**  
> *Transforming single-camera broadcast video into real-time spatial digital twins, pose biomechanics, and Agentic AI coaching loops.*

<p align="center">
  <img width="1000" alt="442ools Architecture Banner" src="https://github.com/user-attachments/assets/9d9b33b2-1c85-461d-8253-6e868eca2834" />
</p>

[![OpenCV 5](https://img.shields.io/badge/OpenCV-5.0%20%2F%204.10-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)
[![AWS Bedrock](https://img.shields.io/badge/AWS%20Bedrock-Claude%203.5%20Sonnet-FF9900.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![AWS Graviton](https://img.shields.io/badge/AWS%20Graviton3-ARM64%20COOL%20Optimized-232F3E.svg?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ec2/graviton/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)

---

## ⚽ What is 442ools?

**442ools** (pronounced *"four-four-tools"*) is a professional-grade sports computer vision and agentic artificial intelligence platform designed to democratize high-end tactical performance analysis.

Operating on standard broadcast match footage without requiring expensive multi-camera optical setups or wearable sensors, 442ools leverages **OpenCV 5** geometric algorithms, **YOLOv8** multi-object tracking, and **Amazon Bedrock (Claude 3.5 Sonnet)** to deliver:
1. **Dynamic Pitch Homography**: Real-time projective transformation from broadcast camera pixels to standard $105\text{m} \times 68\text{m}$ pitch coordinates.
2. **Passing Space & Convex Hulls**: Automated space zone generation (`cv2.convexHull`) and passing corridor obstruction calculations.
3. **Dribble Biomechanics**: 17-keypoint skeleton tracking measuring trunk lean, stance balance equilibrium, and defender wrong-footing dynamics.
4. **Expected Threat ($\Delta xT$)**: Quantitative valuation of ball progression and line-breaking passes.
5. **Agentic Vision with AWS Bedrock**: Continuous Perception $\rightarrow$ Reasoning $\rightarrow$ Action loop providing live tactical coaching interventions and dynamic computer vision pipeline adjustments.

---

## 🏛️ System Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      AWS Cloud Environment                             │
 │                                                                        │
 │   ┌─────────────────┐       ┌──────────────────────────────────────┐   │
 │   │  Amazon S3      │       │  Amazon Bedrock                      │   │
 │   │  Match Video &  │       │  (Claude 3.5 Sonnet / Amazon Titan)  │   │
 │   │  Telemetry Sync │       │  Perception -> Decision -> Action    │   │
 │   └────────┬────────┘       └──────────────────▲───────────────────┘   │
 │            │ Video Feeds                       │ Spatial Telemetry     │
 │            ▼                                   ▼ Tactical Prompts      │
 │   ┌────────────────────────────────────────────────────────────────┐   │
 │   │   AWS Graviton3 (c7g) / COOL-Optimized Processing Engine       │   │
 │   │   - OpenCV 5 / COOL Accelerated Computer Vision                │   │
 │   │   - Homography Projection & Voronoi Hulls                      │   │
 │   │   - YOLOv8 Pose Biomechanics & xT Engine                       │   │
 │   │   - FastAPI Real-Time WebSocket / SSE Streaming                │   │
 │   └────────────────────────┬───────────────────────────────────────┘   │
 └────────────────────────────┼───────────────────────────────────────────┘
                              │ HTTP / WebSocket (Port 8000)
                              ▼
                  ┌───────────────────────┐
                  │ Meta-Vision Dashboard │
                  │ Tactical Studio UI    │
                  └───────────────────────┘
```

---

## 🚀 Quick Start & Local Testing

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Insight14/442ools.git
cd 442ools

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Launch Meta-Vision Studio
```bash
python3 server.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

### 3. CLI Direct Video Analysis
```bash
# Full Match Tactics & Passing Radar Analysis
python3 src/detect_track.py \
  --source input_videos/mbappe_clip.mp4 \
  --output output_videos/mbappe_analyzed.mp4 \
  --homography pitch_calibration/homography_mbappe_clip.json

# 1v1 Dribble & Biomechanical Posture Analysis
python3 src/dribble_pose_engine.py \
  --video input_videos/doku_dribble2.mp4 \
  --output output_videos/doku_dribble_analyzed.mp4
```

---

## ⚡ Focus Paths (OpenCV AI Competition 2026)

### 1. Cloud-Optimized Vision with COOL & AWS Graviton3
442ools is optimized for **AWS Graviton3 (ARM64)** using the Cloud-Optimized OpenCV Library (COOL):
- **44.5% faster** Homography Warp & Metric Rectification
- **38.9% faster** Pose Biomechanics Angle Calculations
- **34.0% lower cloud compute cost** compared to legacy x86 instances.
- *See full benchmarks and setup instructions in [`AWS_DEPLOYMENT.md`](AWS_DEPLOYMENT.md).*

### 2. Agentic Vision with AWS Bedrock
An autonomous perception-decision-action loop connects OpenCV 5 spatial telemetry directly to **Amazon Bedrock (Claude 3.5 Sonnet)** via `src/aws_agentic_coach.py` and the `/api/agentic-vision` endpoint.

---

## 📁 Repository Structure

```
├── src/
│   ├── detect_track.py         # OpenCV 5 player detection & tracking pipeline
│   ├── dribble_pose_engine.py  # 1v1 Dribble biomechanics & posture engine
│   ├── play_predictor.py       # Pass probability & Expected Threat (xT) model
│   ├── aws_agentic_coach.py    # AWS Bedrock Agentic Vision controller
│   └── generate_hype_video.py  # Video production & telemetry muxing
├── frontend/                   # Meta-Vision Web Studio HUD & Tactical 2D Radar
├── pitch_calibration/          # Camera calibration & homography matrices
├── aws/
│   └── cloudformation.yaml     # One-click AWS CloudFormation deployment template
├── Dockerfile                  # Multi-arch container for AWS Graviton / x86
├── requirements.txt            # Pinned reproducible dependencies
├── AWS_DEPLOYMENT.md           # AWS Graviton & Bedrock deployment guide
├── OPENCV_AI_COMPETITION_2026_SUBMISSION.md  # Complete Hackathon Submission Doc
└── server.py                   # FastAPI application & real-time streaming backend
```

---

## 📄 License & Team

- **License**: MIT License
- **Author**: Insight14
- **Hackathon Entry**: OpenCV AI Competition 2026, powered by AWS
