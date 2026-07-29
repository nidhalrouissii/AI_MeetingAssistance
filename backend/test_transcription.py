"""
Script de test STANDALONE — pipeline complet : transcription + diarisation.

Usage :
    python test_transcription.py chemin/vers/reunion.mp3 [nombre_de_locuteurs]

Le nombre de locuteurs est optionnel — si tu le connais, ça améliore la précision
de la diarisation. La toute première transcription téléchargera le modèle Whisper
(~3GB pour large-v3) depuis le CDN OpenAI -> ça prend du temps une seule fois.
"""
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.transcriber import transcriber_service  # noqa: E402
from backend.services.diarizer import diarizer_service  # noqa: E402
from backend.services.merger import merge_transcript_and_speakers  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Usage : python test_transcription.py chemin/vers/audio.mp3 [nombre_de_locuteurs]")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    expected_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"\n🎙️  Traitement de : {audio_path.name}")

    # --- Étape 1 : Transcription ---
    print("⏳ Transcription en cours (1er lancement = téléchargement du modèle, patience)...")
    t0 = time.time()
    transcript_segments = transcriber_service.transcribe(audio_path)
    t1 = time.time()
    print(f"✅ Transcription terminée en {t1 - t0:.1f}s — {len(transcript_segments)} segments")

    # --- Étape 2 : Diarisation ---
    print("⏳ Identification des locuteurs en cours...")
    speaker_segments = diarizer_service.diarize(audio_path, expected_speakers=expected_speakers)
    t2 = time.time()
    print(f"✅ Diarisation terminée en {t2 - t1:.1f}s")

    # --- Étape 3 : Fusion ---
    final_transcript = merge_transcript_and_speakers(transcript_segments, speaker_segments)

    print("\n" + "─" * 60)
    print(final_transcript)
    print("─" * 60)
    print(f"\n⏱️  Temps total : {t2 - t0:.1f}s")

    output_path = audio_path.with_suffix(".transcript.txt")
    output_path.write_text(final_transcript, encoding="utf-8")
    print(f"💾 Transcript sauvegardé dans : {output_path}")


if __name__ == "__main__":
    main()