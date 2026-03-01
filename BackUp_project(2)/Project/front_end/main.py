"""
FastAPI application for Neural Style Transfer.
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from style_transfer import run_style_transfer

# ──────────────────────────────────────────────
#  SETUP DIRECTORIES
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
#  FASTAPI APP
# ──────────────────────────────────────────────
app = FastAPI(
    title="Neural Style Transfer API",
    description="Upload a content image and a style image to generate artistic style transfer results.",
    version="1.0.0",
)

# Enable CORS (so any frontend can call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (generated images)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# ──────────────────────────────────────────────
#  IN-MEMORY JOB TRACKER (for async processing)
# ──────────────────────────────────────────────
jobs: dict = {}


# ──────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_image(file: UploadFile) -> None:
    """Check if the uploaded file is a valid image."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )


async def save_upload(file: UploadFile, prefix: str) -> str:
    """Save an uploaded file to disk and return the path."""
    ext = Path(file.filename).suffix.lower()
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return str(filepath)


def cleanup_files(*paths: str) -> None:
    """Delete temporary files."""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# ──────────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────────

# ---- HOME PAGE (Web UI) ----
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---- HEALTH CHECK ----
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "style-transfer-api"}


# ---- SYNCHRONOUS STYLE TRANSFER ----
@app.post("/api/style-transfer")
async def style_transfer(
    content_image: UploadFile = File(..., description="The content image (photo to stylize)"),
    style_image: UploadFile = File(..., description="The style image (artistic style to apply)"),
    image_size: int = Form(default=384, description="Max image dimension (lower = faster)", ge=64, le=768),
    num_steps: int = Form(default=150, description="Number of optimization steps", ge=10, le=500),
    style_weight: float = Form(default=1_000_000, description="Style weight (higher = more stylized)"),
    content_weight: float = Form(default=1, description="Content weight"),
):
    """
    Perform style transfer synchronously.
    
    - Upload a **content image** (the photo you want to transform)
    - Upload a **style image** (the artistic style to apply)
    - Adjust parameters as needed
    
    Returns the stylized image directly.
    
    ⚠️ This runs on CPU, so it may take 1-5 minutes depending on image size and steps.
    """
    # Validate files
    validate_image(content_image)
    validate_image(style_image)

    # Save uploads
    content_path = await save_upload(content_image, "content")
    style_path = await save_upload(style_image, "style")

    # Generate output path
    output_filename = f"stylized_{uuid.uuid4().hex[:12]}.jpg"
    output_path = str(OUTPUT_DIR / output_filename)

    try:
        # Run style transfer
        run_style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            image_size=image_size,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
        )

        # Return the generated image
        return FileResponse(
            path=output_path,
            media_type="image/jpeg",
            filename=output_filename,
            headers={"X-Output-Filename": output_filename},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")

    finally:
        # Cleanup uploaded files
        cleanup_files(content_path, style_path)


# ---- ASYNC STYLE TRANSFER (Background Task) ----
def process_style_transfer_job(
    job_id: str,
    content_path: str,
    style_path: str,
    output_path: str,
    image_size: int,
    num_steps: int,
    style_weight: float,
    content_weight: float,
):
    """Background task for processing style transfer."""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["started_at"] = time.time()

        run_style_transfer(
            content_path=content_path,
            style_path=style_path,
            output_path=output_path,
            image_size=image_size,
            num_steps=num_steps,
            style_weight=style_weight,
            content_weight=content_weight,
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = time.time()
        jobs[job_id]["output_url"] = f"/static/outputs/{Path(output_path).name}"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

    finally:
        cleanup_files(content_path, style_path)


@app.post("/api/style-transfer/async")
async def style_transfer_async(
    background_tasks: BackgroundTasks,
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    image_size: int = Form(default=384, ge=64, le=768),
    num_steps: int = Form(default=150, ge=10, le=500),
    style_weight: float = Form(default=1_000_000),
    content_weight: float = Form(default=1),
):
    """
    Submit a style transfer job for background processing.
    Returns a job ID that you can poll with /api/jobs/{job_id}.
    """
    validate_image(content_image)
    validate_image(style_image)

    content_path = await save_upload(content_image, "content")
    style_path = await save_upload(style_image, "style")

    job_id = uuid.uuid4().hex[:16]
    output_filename = f"stylized_{job_id}.jpg"
    output_path = str(OUTPUT_DIR / output_filename)

    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "params": {
            "image_size": image_size,
            "num_steps": num_steps,
            "style_weight": style_weight,
            "content_weight": content_weight,
        },
    }

    # Queue background task
    background_tasks.add_task(
        process_style_transfer_job,
        job_id, content_path, style_path, output_path,
        image_size, num_steps, style_weight, content_weight,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "message": "Job submitted. Poll /api/jobs/{job_id} for status.",
            "poll_url": f"/api/jobs/{job_id}",
        },
    )


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of an async style transfer job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "params": job.get("params"),
    }

    if job["status"] == "completed":
        response["output_url"] = job.get("output_url")
        duration = job.get("completed_at", 0) - job.get("started_at", 0)
        response["processing_time_seconds"] = round(duration, 1)

    elif job["status"] == "failed":
        response["error"] = job.get("error")

    return response


# ---- LIST ALL OUTPUTS ----
@app.get("/api/outputs")
async def list_outputs():
    """List all generated style transfer outputs."""
    files = []
    for f in sorted(OUTPUT_DIR.glob("stylized_*"), key=os.path.getmtime, reverse=True):
        files.append({
            "filename": f.name,
            "url": f"/static/outputs/{f.name}",
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created": time.ctime(f.stat().st_mtime),
        })
    return {"outputs": files, "count": len(files)}


# ---- DOWNLOAD SPECIFIC OUTPUT ----
@app.get("/api/outputs/{filename}")
async def download_output(filename: str):
    """Download a specific output image."""
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(filepath, media_type="image/jpeg", filename=filename)


# ---- CLEANUP OLD OUTPUTS ----
@app.delete("/api/outputs")
async def cleanup_outputs(max_age_hours: int = 24):
    """Delete output images older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    deleted = 0
    for f in OUTPUT_DIR.glob("stylized_*"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            deleted += 1
    return {"deleted": deleted, "message": f"Removed files older than {max_age_hours}h"}


# ──────────────────────────────────────────────
#  STARTUP
# ──────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-download VGG19 weights on startup."""
    print("=" * 60)
    print("  Neural Style Transfer API")
    print("  Loading VGG19 model weights...")
    print("=" * 60)

    import torchvision.models as models
    models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    print("  ✓ VGG19 model ready")
    print(f"  ✓ Uploads dir: {UPLOAD_DIR}")
    print(f"  ✓ Outputs dir: {OUTPUT_DIR}")
    print("=" * 60)


# ──────────────────────────────────────────────
#  RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )