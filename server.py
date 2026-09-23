import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_videos"
CALIB_DIR = BASE_DIR / "pitch_calibration"
WEIGHTS_PATH = BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
COCO_WEIGHTS_PATH = BASE_DIR / "yolov8n.pt"
FRONTEND_DIR = BASE_DIR / "frontend"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Meta-Vision 442ools Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analysis_jobs = {}

@app.get("/api/status")
def get_system_status():
    custom_weights_exist = WEIGHTS_PATH.exists()
    coco_weights_exist = COCO_WEIGHTS_PATH.exists()
    
    sample_videos = []
    if INPUT_DIR.exists():
        for f in sorted(INPUT_DIR.iterdir()):
            if f.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
                sample_videos.append({
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "path": f"input_videos/{f.name}"
                })

    homographies = []
    if CALIB_DIR.exists():
        for f in sorted(CALIB_DIR.glob("homography_*.json")):
            homographies.append({
                "filename": f.name,
                "name": f.stem.replace("homography_", "").replace("_", " ").upper(),
                "path": f"pitch_calibration/{f.name}"
            })

    return {
        "status": "ready",
        "custom_model_available": custom_weights_exist,
        "custom_model_path": str(WEIGHTS_PATH) if custom_weights_exist else None,
        "coco_model_available": coco_weights_exist,
        "sample_videos": sample_videos,
        "homographies": homographies
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    target_filename = file.filename.replace(" ", "_")
    target_path = INPUT_DIR / target_filename
    
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "success": True,
        "filename": target_filename,
        "path": f"input_videos/{target_filename}",
        "size_mb": round(target_path.stat().st_size / (1024 * 1024), 2)
    }

def run_analysis_task(job_id: str, video_rel_path: str, homography_rel_path: Optional[str], model_type: str, conf: float, attacking_dir: int):
    job = analysis_jobs[job_id]
    job["status"] = "processing"
    job["logs"] = []
    
    input_file = BASE_DIR / video_rel_path
    base_name = input_file.stem
    output_filename = f"{base_name}_meta_analyzed_{int(time.time())}.mp4"
    output_file = OUTPUT_DIR / output_filename
    events_filename = f"{base_name}_events_{int(time.time())}.jsonl"
    events_file = OUTPUT_DIR / events_filename
    
    model_arg = str(WEIGHTS_PATH) if (model_type == "custom" and WEIGHTS_PATH.exists()) else str(COCO_WEIGHTS_PATH)
    python_bin = sys.executable
    
    cmd = [
        python_bin,
        str(BASE_DIR / "src" / "detect_track.py"),
        "--source", str(input_file),
        "--output", str(output_file),
        "--model", model_arg,
        "--conf", str(conf),
        "--attacking-direction", str(attacking_dir)
    ]
    
    if homography_rel_path:
        homo_full = BASE_DIR / homography_rel_path
        if homo_full.exists():
            cmd.extend(["--homography", str(homo_full)])
            cmd.extend(["--events-output", str(events_file)])
    
    job["logs"].append(f"[INIT] Executing: {' '.join(cmd)}")
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR)
        )
        
        for line in iter(proc.stdout.readline, ''):
            line_clean = line.strip()
            if line_clean:
                job["logs"].append(line_clean)
                if len(job["logs"]) > 200:
                    job["logs"].pop(0)
                    
        proc.wait()
        
        if proc.returncode == 0:
            job["status"] = "completed"
            job["output_video"] = f"/media/output_videos/{output_filename}"
            job["events_file"] = f"/media/output_videos/{events_filename}" if events_file.exists() else None
            
            parsed_events = []
            if events_file.exists():
                with open(events_file, "r") as ef:
                    for l in ef:
                        if l.strip():
                            try:
                                parsed_events.append(json.loads(l))
                            except Exception:
                                pass
            job["events_data"] = parsed_events
        else:
            job["status"] = "failed"
            job["error"] = f"Process exited with code {proc.returncode}"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)

@app.post("/api/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    video_path: str = Form(...),
    homography_path: Optional[str] = Form(None),
    model_type: str = Form("custom"),
    conf: float = Form(0.35),
    attacking_dir: int = Form(1)
):
    job_id = f"job_{int(time.time()*1000)}"
    analysis_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "video_path": video_path,
        "logs": ["Job queued for Meta-Vision Analysis..."]
    }
    
    background_tasks.add_task(
        run_analysis_task,
        job_id=job_id,
        video_rel_path=video_path,
        homography_rel_path=homography_path if (homography_path and homography_path != "none") else None,
        model_type=model_type,
        conf=conf,
        attacking_dir=attacking_dir
    )
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return analysis_jobs[job_id]

app.mount("/media/input_videos", StaticFiles(directory=str(INPUT_DIR)), name="input_videos")
app.mount("/media/output_videos", StaticFiles(directory=str(OUTPUT_DIR)), name="output_videos")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
