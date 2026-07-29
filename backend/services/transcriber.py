"""
Service de transcription avec faster-whisper (CTranslate2).

Pourquoi faster-whisper et pas openai-whisper :
- Même architecture, mêmes poids d'origine, qualité de transcription équivalente
- Moteur d'inférence CTranslate2 : 4-8x plus rapide, VRAM réduite (quantization int8_float16)
- VAD Silero intégré : saute les silences au lieu de les transcrire
- Objectif métier : rapport disponible en quelques minutes pour 1h de réunion (vs ~2h avant)

IMPORTANT — contrainte EY (zéro dépendance externe à l'exécution) :
faster_whisper.WhisperModel("medium") télécharge par défaut le modèle
CTranslate2 depuis le Hugging Face Hub (Systran/faster-whisper-medium) au
premier lancement. Pour respecter la même contrainte que sur la version
openai-whisper (CDN OpenAI direct, pas de dépendance HF au runtime), le
modèle doit être converti et stocké localement AVANT le déploiement, puis
chargé avec local_files_only=True pour garantir qu'aucun appel réseau
n'est fait pendant l'exécution.

Conversion à faire une seule fois (build, pas runtime) :
    pip install ctranslate2 "transformers[torch]>=4.23"
    ct2-transformers-converter --model openai/whisper-medium \
        --output_dir ./models/faster-whisper-medium \
        --quantization int8_float16

settings.whisper_ct2_dir doit pointer vers ce dossier local
(ex: "models/faster-whisper-medium").
"""
import gc
import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from faster_whisper import WhisperModel

from backend.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str = ""

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text.strip(),
            "language": self.language,
        }


class TranscriberService:
    """Singleton : le modèle n'est chargé qu'une fois en VRAM."""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            model_dir = Path(settings.whisper_ct2_dir)

            if not model_dir.exists():
                raise FileNotFoundError(
                    f"Modèle CTranslate2 introuvable dans {model_dir}. "
                    "Convertis-le d'abord avec ct2-transformers-converter "
                    "(voir docstring en tête de fichier). Le chargement ne "
                    "doit jamais dépendre d'un téléchargement HuggingFace "
                    "au runtime."
                )

            logger.info(
                "Chargement de faster-whisper depuis %s (int8_float16) sur %s...",
                model_dir, settings.whisper_device,
            )
            self._model = WhisperModel(
                str(model_dir),                    # chemin LOCAL, pas un nom de taille
                device=settings.whisper_device,           # "cuda"
                compute_type=settings.whisper_compute_type,  # "int8_float16"
                local_files_only=True,              # garde-fou : jamais d'appel réseau ici
            )
            logger.info("Modèle chargé avec succès.")
        return self._model

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        model = self._load_model()

        try:
            segments_gen, info = model.transcribe(
            str(audio_path),
            language=settings.whisper_language,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 250,   # coupe dès 0.25s de silence (au lieu de 0.5s)
                "max_speech_duration_s": 12,      # force une coupure : aucun segment > 12s
                "speech_pad_ms": 150,             # garde 150ms de marge autour de la parole
            },
            )   

            logger.info(
                "Langue détectée : %s (prob. %.2f)",
                info.language, info.language_probability,
            )

            # faster-whisper retourne un générateur : la transcription se fait
            # au fur et à mesure de l'itération. Les timestamps sont déjà
            # remappés sur la timeline originale par le VAD (SpeechTimestampsMap),
            # donc compatibles avec ton alignement librosa.load(sr=16000) côté
            # diarisation. À REVALIDER empiriquement sur tes 3 fichiers de test
            # après migration : le VAD peut introduire un léger décalage résiduel
            # sur de l'audio peu silencieux (cas documenté, pas garanti à 100%).
            segments = [
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    language=info.language,
                )
                for seg in segments_gen
            ]

            if not segments:
                logger.warning(
                    "Transcription vide pour %s — fichier silencieux ou VAD trop agressif ?",
                    audio_path.name,
                )

            return segments

        except Exception as e:
            raise RuntimeError(f"Échec de la transcription de {audio_path.name} : {e}") from e
        finally:
            gc.collect()
            if settings.whisper_device == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def segments_to_plain_text(segments: list[TranscriptSegment]) -> str:
        lines = [
            f"[{int(s.start // 60):02d}:{int(s.start % 60):02d}] {s.text.strip()}"
            for s in segments
        ]
        return "\n".join(lines)


transcriber_service = TranscriberService()