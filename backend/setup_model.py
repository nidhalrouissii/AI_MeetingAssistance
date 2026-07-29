"""
SCRIPT DE SETUP — à lancer UNE SEULE FOIS avant la première utilisation.

Ce script :
1. Télécharge le modèle Whisper "large-v3" depuis le CDN officiel d'OpenAI
   (openaipublic.azureedge.net) — AUCUNE dépendance à HuggingFace.
2. Convertit ce modèle en local vers le format CTranslate2, requis par WhisperX
   pour le batching rapide sur GPU.

Tout se passe sur ta machine, sans connexion à HuggingFace à aucune étape.

Usage :
    python setup_model.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core.config import settings  # noqa: E402


def download_original_model():
    """Étape 1 : télécharge le .pt officiel depuis le CDN OpenAI (pas HuggingFace)."""
    print(f"📥 Téléchargement de Whisper '{settings.whisper_model_name}' depuis le CDN OpenAI...")
    import whisper  # package "openai-whisper"

    # whisper.load_model télécharge dans ~/.cache/whisper/ si pas déjà présent.
    # La source est openaipublic.azureedge.net — vérifiable dans le code source de la lib.
    model = whisper.load_model(settings.whisper_model_name, download_root=settings.whisper_original_dir)
    print(f"✅ Modèle téléchargé dans : {settings.whisper_original_dir}")
    del model  # libère la RAM, on n'a plus besoin de l'objet chargé ici


def convert_to_ctranslate2():
    """Étape 2 : convertit le modèle .pt en format CTranslate2, en local."""
    output_dir = Path(settings.whisper_ct2_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✅ Modèle déjà converti dans : {output_dir} (rien à refaire)")
        return

    print("🔄 Conversion en CTranslate2 (peut prendre quelques minutes)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ct2-whisper-converter est l'outil dédié au format natif OpenAI (.pt),
    # contrairement à ct2-transformers-converter qui s'attend à un modèle
    # au format HuggingFace transformers. C'est l'outil correct ici.
    cmd = [
        "ct2-whisper-converter",
        "--model", settings.whisper_model_name,
        "--output_dir", str(output_dir),
        "--quantization", "float16",
        "--force",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Échec de la conversion CTranslate2.\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )

    print(f"✅ Modèle converti et sauvegardé dans : {output_dir}")


def main():
    print("=" * 60)
    print("SETUP — Préparation du modèle Whisper (sans HuggingFace)")
    print("=" * 60)

    download_original_model()
    convert_to_ctranslate2()

    print("\n🎉 Setup terminé. Le modèle est prêt à être utilisé localement.")
    print(f"   Chemin du modèle final : {settings.whisper_ct2_dir}")


if __name__ == "__main__":
    main()
