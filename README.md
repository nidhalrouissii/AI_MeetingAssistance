# AI Meeting Assistant

Génération automatique de comptes rendus de réunion à partir d'un enregistrement audio : transcription, identification des locuteurs, résumé, décisions, tâches et questions ouvertes.

Projet réalisé dans le cadre du stage de fin d'études (ESPRIT, spécialité IA & Data Science) chez **EY Tunisia**.

## Fonctionnalités

- Upload d'un fichier audio de réunion (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.ogg`, `.flac`)
- Transcription automatique avec timestamps (français / anglais, y compris mélangés)
- Identification des locuteurs, avec **détection automatique du nombre de locuteurs** — aucun paramétrage requis
- Résumé exécutif, décisions prises, tâches (avec responsable et deadline si mentionnés), questions ouvertes
- Suivi de progression en temps réel pendant le traitement
- Export du compte rendu en PDF

## Stack technique

| Composant | Technologie |
|---|---|
| Transcription | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2), modèle `medium`, converti et hébergé localement |
| Diarisation | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (embeddings GE2E) + scikit-learn (KMeans, sélection automatique du nombre de locuteurs par score de silhouette) |
| Analyse | Groq API — LLaMA 3.3 70B |
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | React (Vite) |

Aucune donnée n'est envoyée à Hugging Face à l'exécution : les modèles sont convertis et stockés localement lors de l'installation.

## Architecture

```
Audio uploadé → Transcription (faster-whisper) → Diarisation (Resemblyzer,
nombre de locuteurs détecté automatiquement) → Fusion transcript + locuteurs
→ Analyse LLM (Groq) → Compte rendu généré
```

Traitement asynchrone via FastAPI `BackgroundTasks` ; le frontend interroge le statut du traitement toutes les 3 secondes jusqu'à disponibilité du résultat.

Le document `Architecture_AI_Meeting_Assistant.pdf` (à la racine du dépôt) détaille la conception, les choix techniques, les résultats de validation et les limites connues du système.

## Installation

### Prérequis

- Python 3.12, Node.js 20+
- Une clé API [Groq](https://console.groq.com) (gratuite)
- `ffmpeg` installé et accessible dans le PATH

### 1. Backend

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt -c backend/constraints.txt
```

Convertir le modèle de transcription en local (une seule fois) :

```bash
pip install ctranslate2 "transformers[torch]>=4.23"
ct2-transformers-converter --model openai/whisper-medium \
    --output_dir ./models/faster-whisper-medium \
    --quantization int8_float16 \
    --copy_files tokenizer.json preprocessor_config.json
```

Créer un fichier `.env` à la racine du projet :

```
GROQ_API_KEY=votre_clé_ici
```

Lancer le serveur (depuis la racine du projet) :

```bash
uvicorn backend.main:app --reload
```

L'API est disponible sur `http://localhost:8000` (documentation interactive sur `/docs`).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

L'application est disponible sur `http://localhost:5173`.

## Structure du projet

```
AI_MeetingAssistance/
├── backend/
│   ├── core/config.py         # paramètres centralisés
│   ├── models/job.py          # modèle SQLAlchemy (Job)
│   ├── prompts/               # prompt système LLM
│   ├── routers/                # routes FastAPI (upload, status, result, meetings)
│   ├── services/
│   │   ├── transcriber.py      # transcription (faster-whisper)
│   │   ├── diarizer.py         # diarisation + détection auto du nombre de locuteurs
│   │   ├── analyzer.py         # analyse LLM (Groq)
│   │   └── pipeline.py         # orchestration du pipeline complet
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── index.css
├── models/
│   └── faster-whisper-medium/  # modèle converti localement (non versionné)
└── .env                         # GROQ_API_KEY (non versionné)
```

## Limites connues

- Sur des répliques très courtes (< 2 s) en dialogue rapide, l'attribution du locuteur peut être incorrecte (limite du GE2E, documentée dans la littérature).
- La parole superposée (deux locuteurs simultanés) est attribuée à la voix dominante du segment ; il n'y a pas de détection de chevauchement dédiée.
- Précision réduite sur des voix très similaires en audio dégradé (compression, réenregistrement).

Détails complets, mesures et méthodologie de validation dans le document d'architecture.

