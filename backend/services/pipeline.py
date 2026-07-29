"""
Pipeline complet de traitement d'une réunion.
Appelé en arrière-plan par FastAPI BackgroundTasks.

Étapes :
1. Transcription audio → texte (faster-whisper)
2. Diarisation → qui parle quand (Resemblyzer, nombre de locuteurs détecté automatiquement)
3. Fusion transcript + locuteurs
4. Analyse LLM → résumé + décisions + tâches (Groq)
5. Sauvegarde en base de données

Le champ job.step est mis à jour à chaque étape ; il est exposé par
GET /api/status/{job_id} et permet au frontend d'afficher une barre de
progression reflétant l'avancée réelle du traitement.
"""
import json
import logging
from pathlib import Path

from backend.models.job import SessionLocal, Job
from backend.services.transcriber import transcriber_service
from backend.services.diarizer import diarizer_service
from backend.services.analyzer import analyzer_service

logger = logging.getLogger(__name__)


def _set_step(db, job, step: str):
    job.step = step
    db.commit()


def process_meeting(job_id: str, audio_path: Path):
    """
    Pipeline complet de traitement. Appelé en BackgroundTask par FastAPI.
    Met à jour le statut et l'étape du job dans la base à chaque phase.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error("Job %s introuvable en base.", job_id)
            return

        # ── Étape 1 : Transcription ────────────────────────────────
        job.status = "processing"
        _set_step(db, job, "transcription")
        logger.info("[%s] Étape 1 : Transcription...", job_id)

        transcript_segments = transcriber_service.transcribe(audio_path)

        # ── Étape 2 : Diarisation (détection auto du nb de locuteurs) ─
        _set_step(db, job, "diarisation")
        logger.info("[%s] Étape 2 : Diarisation...", job_id)
        speakers = diarizer_service.diarize_segments(audio_path, transcript_segments)

        # ── Étape 3 : Fusion transcript + locuteurs ────────────────
        logger.info("[%s] Étape 3 : Fusion transcript + locuteurs...", job_id)
        merged_segments = []
        lines = []
        for seg, speaker in zip(transcript_segments, speakers):
            merged_segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "speaker": speaker,
                "language": seg.language,
            })
            m, s = int(seg.start // 60), int(seg.start % 60)
            lines.append(f"[{m:02d}:{s:02d}] {speaker}: {seg.text.strip()}")

        full_transcript = "\n".join(lines)

        job.transcript = json.dumps(merged_segments, ensure_ascii=False)
        db.commit()

        # ── Étape 4 : Analyse LLM ──────────────────────────────────
        _set_step(db, job, "analyse")
        logger.info("[%s] Étape 4 : Analyse LLM...", job_id)
        analysis = analyzer_service.analyze(full_transcript)

        # ── Étape 5 : Sauvegarde résultat ─────────────────────────
        job.result = json.dumps(analysis, ensure_ascii=False)
        job.status = "done"
        job.step = "done"
        db.commit()
        logger.info("[%s] Pipeline terminé avec succès.", job_id)

    except Exception as e:
        logger.error("[%s] Erreur pipeline : %s", job_id, e)
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "error"
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()