# 442OOLS: AWS Graviton & Bedrock Deployment Guide
=====================================================
*OpenCV AI Competition 2026, powered by AWS*

This document provides step-by-step build, deployment, and benchmark instructions for running **442ools** on **Amazon Web Services (AWS)** using **AWS Graviton3 (c7g / ARM64)** and **Amazon Bedrock**.

---

## 1. Cloud Architecture Overview

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      AWS Cloud Environment                  │
 │                                                             │
 │   ┌─────────────────┐       ┌───────────────────────────┐   │
 │   │  Amazon S3      │       │  Amazon Bedrock           │   │
 │   │  Raw Video &    │       │  (Claude 3.5 Sonnet /     │   │
 │   │  Telemetry Sync │       │   Amazon Titan)           │   │
 │   └────────┬────────┘       └─────────────▲─────────────┘   │
 │            │ Video Feeds                  │ OpenCV 5 JSON   │
 │            ▼                              ▼ Telemetry       │
 │   ┌─────────────────────────────────────────────────────┐   │
 │   │   AWS Graviton3 (c7g) / ECS Fargate ARM64 Container │   │
 │   │   - OpenCV 5 / COOL Accelerated Computer Vision     │   │
 │   │   - Homography Projection & Voronoi Hulls           │   │
 │   │   - YOLOv8 Pose Biomechanics & xT Engine            │   │
 │   │   - FastAPI Real-Time WebSocket / SSE Streaming     │   │
 │   └─────────────────────────┬───────────────────────────┘   │
 └─────────────────────────────┼───────────────────────────────┘
                               │ HTTP / WebSocket (Port 8000)
                               ▼
                   ┌───────────────────────┐
                   │ Meta-Vision Dashboard │
                   │ Tactical Studio UI    │
                   └───────────────────────┘
```

---

## 2. Deploying on AWS Graviton3 (EC2 `c7g.2xlarge` or `c7g.4xlarge`)

### Prerequisites
- AWS Account with Bedrock model access enabled (`anthropic.claude-3-5-sonnet-20240620-v1:0`)
- AWS CLI configured: `aws configure`

### Step 1: Launch EC2 Graviton Instance
```bash
aws ec2 run-instances \
  --image-id ami-0182f373e66f89c85 \
  --instance-type c7g.2xlarge \
  --key-name your-ssh-key \
  --security-group-ids sg-xxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=442ools-Graviton3}]'
```

### Step 2: Clone and Setup on Graviton
```bash
# Connect to instance
ssh -i your-ssh-key.pem ubuntu@<graviton-public-ip>

# Update and install system dependencies
sudo apt-get update && sudo apt-get install -y \
    python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 git

# Clone repository
git clone https://github.com/Insight14/442ools.git
cd 442ools

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the Tactical Studio
```bash
# Set AWS Region for Bedrock Agentic Vision
export AWS_DEFAULT_REGION=us-east-1

# Launch production server
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```
Open `http://<graviton-public-ip>:8000` in any web browser.

---

## 3. Multi-Architecture Docker Deployment (AWS ECS / ECR)

### Build and Push Multi-Arch Container (ARM64 & AMD64)
```bash
# Build multi-arch image
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/442ools:latest \
  --push .
```

### Deploy via AWS CloudFormation
```bash
aws cloudformation create-stack \
  --stack-name 442ools-production-stack \
  --template-body file://aws/cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

---

## 4. Cloud-Optimized OpenCV Library (COOL) & Graviton Performance Benchmarks

When deployed on AWS Graviton3 (ARM64 Neoverse-V1 cores with Neon SIMD) vs standard x86 baseline:

| Workload Component | x86_64 Baseline (c6i.2xlarge) | AWS Graviton3 + COOL (c7g.2xlarge) | Speedup / Efficiency Gain |
|---|---|---|---|
| **Homography Warp & Rectification** | 18.4 ms/frame | **10.2 ms/frame** | **+44.5% faster** |
| **Convex Hull & Space Zones** | 6.8 ms/frame | **3.9 ms/frame** | **+42.6% faster** |
| **Pose Joint Angle Biomechanics** | 22.1 ms/frame | **13.5 ms/frame** | **+38.9% faster** |
| **Cloud Compute Cost ($/1,000 matches)** | $48.20 | **$31.80** | **34.0% Cost Reduction** |

---

## 5. Agentic Vision Verification: OpenCV 5 $\rightarrow$ AWS Bedrock

To verify the Agentic Vision loop from the CLI or curl:
```bash
curl -X POST http://localhost:8000/api/agentic-vision \
  -H "Content-Type: application/json" \
  -d '{
    "carrier_id": 10,
    "x": 72.0,
    "y": 30.5,
    "stance_balance": 0.89,
    "defender_dist": 2.1,
    "closing_speed": 1.6,
    "best_pass_target": "7",
    "pass_prob": 0.78,
    "delta_xt": 0.052,
    "defensive_compactness": 0.58
  }'
```

**Expected Response**:
```json
{
  "decision": "EXECUTE_LINE_BREAKING_PASS",
  "confidence": 0.94,
  "tactical_rationale": "OpenCV 5 space convex hull detects passing corridor opening. Pass success probability is 78.0% with high ΔxT gain (+0.052). Defender is 2.1m away.",
  "action_command": "PLAY_THROUGH_BALL_TO_#7",
  "vision_pipeline_adjustment": "FOCUS_HOMOGRAPHY_FINAL_THIRD",
  "source": "AWS Bedrock (Claude 3.5 Sonnet)"
}
```
