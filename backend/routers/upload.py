"""
Router FastAPI — Upload de fichiers audio.
POST /upload → reçoit le fichier, crée un job, lance le pipeline en arrière-plan.
Le nombre de locuteurs est détecté automatiquement par la diarisation.
"""
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.models.job import Job, get_db
from backend.services.pipeline import process_meeting

router = APIRouter(prefix="/api", tags=["upload"])

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac"}


@router.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Validation de l'extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {ext}. Formats acceptés : {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Sauvegarde du fichier
    job_id = str(uuid.uuid4())
    audio_path = settings.upload_dir / f"{job_id}{ext}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Création du job en base
    job = Job(id=job_id, filename=file.filename, status="pending")
    db.add(job)
    db.commit()

    # Lancement du pipeline en arrière-plan (non bloquant)
    background_tasks.add_task(process_meeting, job_id, audio_path)

    return {"job_id": job_id, "status": "pending", "filename": file.filename}