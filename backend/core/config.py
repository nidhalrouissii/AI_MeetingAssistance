"""
Configuration centralisée.
Toutes les valeurs "magiques" (modèles, device, langue...) vivent ICI et nulle part ailleurs.
Ça évite d'avoir des "cuda" ou "fr" écrits en dur dans 5 fichiers différents.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # --- Transcription (Whisper converti localement en CTranslate2, zéro HuggingFace) ---
    whisper_model_name: str = "medium"                       # nom du modèle officiel OpenAI
    whisper_original_dir: str = "models/whisper-original"    # cache du .pt téléchargé depuis le CDN OpenAI (si utilisé ailleurs)
    whisper_ct2_dir: str = "models/faster-whisper-medium"    # modèle converti localement, utilisé en prod
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8_float16"
    whisper_batch_size: int = 4             # RTX 3050 4GB VRAM -> batch_size bas obligatoire pour éviter l'OOM
    whisper_language: str | None = None     # None = détection auto par segment (gère le mélange FR/EN)

    # --- Map-Reduce (découpage des longs transcripts) ---
    chunk_duration_minutes: int = 15        # taille des segments envoyés au LLM
    max_tokens_per_chunk: int = 3000        # garde-fou pour rester sous les limites du free tier Groq

    # --- LLM ---
    groq_api_key: str = ""                  # à mettre dans un fichier .env, jamais en dur dans le code
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_model: str = "mistral:7b"        # fallback local si Groq indisponible/quota dépassé
    llm_temperature: float = 0.2            # basse température = extraction fiable, pas de créativité

    # --- Stockage ---
    upload_dir: Path = Path("storage/uploads")
    output_dir: Path = Path("storage/outputs")
    database_url: str = "sqlite:///./meeting_assistant.db"

    class Config:
        env_file = ".env"


settings = Settings()

# Création des dossiers de stockage au démarrage si absents
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)