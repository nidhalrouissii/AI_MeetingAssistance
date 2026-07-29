"""
Lissage temporel post-diarisation.

Contexte : le d-vector GE2E de Resemblyzer est peu fiable sur des segments
courts (< 2-3s), surtout en fin de tour de parole où l'audio peut contenir
un chevauchement ou une respiration de l'autre locuteur. Ça produit parfois
un segment isolé mal attribué, encadré par deux segments du même autre
locuteur (ex: A A [B] A A alors que la réalité est A A A A A).

Cette fonction ne corrige QUE ce cas précis et isolé : un segment court
dont les deux voisins immédiats partagent le même speaker, différent du
sien. Elle ne touche jamais aux vrais changements de locuteur (alternance
normale de la conversation), et ne fait qu'une seule passe sur l'état
original pour éviter tout effet de cascade.
"""
import logging

logger = logging.getLogger(__name__)


def smooth_isolated_speakers(
    segments: list[dict],
    max_isolated_duration: float = 3.0,
) -> list[dict]:
    """
    Corrige les segments de speaker isolés et courts.

    Args:
        segments: liste de dicts avec au moins "start", "end", "speaker"
                  (le format déjà produit après l'étape de fusion
                  transcript + locuteurs, avant l'appel au LLM).
        max_isolated_duration: durée max (en secondes) d'un segment pour
                  qu'il soit considéré comme "isolé" et corrigible.

    Returns:
        Une NOUVELLE liste (les dicts d'origine ne sont pas mutés),
        avec le champ "speaker" corrigé sur les segments concernés.
    """
    if len(segments) < 3:
        return segments

    corrected = [dict(seg) for seg in segments]  # copie défensive
    nb_corrections = 0

    # On lit toujours les speakers ORIGINAUX des voisins (segments, pas
    # corrected) pour que la correction d'un segment ne pollue pas la
    # décision sur le suivant. Une seule passe, pas de cascade.
    for i in range(1, len(segments) - 1):
        current = segments[i]
        prev_speaker = segments[i - 1]["speaker"]
        next_speaker = segments[i + 1]["speaker"]
        duration = current["end"] - current["start"]

        is_short = duration <= max_isolated_duration
        is_sandwiched = (
            prev_speaker == next_speaker
            and current["speaker"] != prev_speaker
        )

        if is_short and is_sandwiched:
            logger.info(
                "Lissage : segment %.2fs→%.2fs (%.1fs, '%s') réattribué "
                "de %s vers %s (voisins concordants)",
                current["start"], current["end"], duration,
                current["text"][:40] if "text" in current else "",
                current["speaker"], prev_speaker,
            )
            corrected[i]["speaker"] = prev_speaker
            nb_corrections += 1

    if nb_corrections:
        logger.info("Lissage terminé : %d segment(s) corrigé(s).", nb_corrections)

    return corrected