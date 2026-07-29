"""
Router FastAPI — Statut et résultats des jobs.
GET /status/{job_id} → statut du traitement (avec l'étape en cours)
GET /result/{job_id} → compte rendu complet
GET /meetings        → historique des réunions
"""
import json
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from backend.models.job import Job, get_db

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/status/{job_id}")
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")

    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status,
        "step": job.step,  # transcription / diarisation / analyse / done — pour la barre de progression
        "created_at": job.created_at.isoformat(),
        "error": job.error_message
    }


@router.get("/result/{job_id}")
def get_result(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Traitement non terminé (statut : {job.status})")

    return {
        "job_id": job.id,
        "filename": job.filename,
        "created_at": job.created_at.isoformat(),
        "transcript": json.loads(job.transcript) if job.transcript else [],
        "result": json.loads(job.result) if job.result else {}
    }


@router.get("/meetings")
def list_meetings(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(20).all()
    return [
        {
            "job_id": j.id,
            "filename": j.filename,
            "status": j.status,
            "created_at": j.created_at.isoformat()
        }
        for j in jobs
    ]