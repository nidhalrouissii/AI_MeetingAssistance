"""
Service de diarisation (identification des locuteurs) avec Resemblyzer.

Pourquoi Resemblyzer et pas pyannote :
- pyannote télécharge ses poids depuis HuggingFace (token + conditions à accepter)
- Resemblyzer embarque ses poids DIRECTEMENT dans le package pip, zéro téléchargement
  externe à l'exécution -> conforme à la contrainte EY.

Approche : un embedding GE2E (d-vector 256 dim) par segment de transcription
Whisper, puis clustering pour regrouper les segments par locuteur.

Détection AUTOMATIQUE du nombre de locuteurs (pas de paramètre utilisateur) :
- On teste k = 2..MAX_SPEAKERS avec KMeans (déterministe via random_state)
- Pour chaque k, on mesure la qualité de séparation des clusters avec le
  score de silhouette (métrique cosine) : proche de 1 = clusters bien
  séparés, proche de 0 = clusters qui se chevauchent.
- On retient le k au meilleur score.
- Si même le meilleur score est très faible, les embeddings ne se séparent
  pas en groupes distincts -> un seul locuteur.

Pourquoi silhouette et pas un distance_threshold fixe sur Agglomerative :
un seuil unique (ex: 0.45) est calibré sur UN enregistrement et ne
généralise pas — sur d'autres voix/qualités audio il sous- ou sur-segmente.
La silhouette compare des partitions candidates entre elles sur CHAQUE
fichier, elle s'adapte donc aux données au lieu de dépendre d'une constante.

Limite à connaître : moins précis que pyannote sur des cas difficiles (voix
très similaires, beaucoup de locuteurs, chevauchements de parole, répliques
très courtes < 2s). Suffisant pour des réunions d'entreprise à 2-6 locuteurs
avec peu de chevauchement.
"""
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import librosa
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

MIN_SEGMENT_SAMPLES = 8000       # 0.5s à 16kHz : en dessous, l'embedding n'est pas fiable
SILENCE_RMS_THRESHOLD = 0.005    # en dessous de cette énergie, le segment est considéré comme du silence
MAX_SPEAKERS = 8                 # borne haute raisonnable pour une réunion d'entreprise
MIN_SILHOUETTE_FOR_MULTI = 0.10  # si le meilleur score est en dessous, on considère 1 seul locuteur
PARSIMONY_MARGIN = 0.05          # un k plus grand n'est retenu que s'il dépasse le score des k plus petits d'au moins cette marge


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker_id: str  # "Person1", "Person2"...


class DiarizerService:
    """Singleton : le VoiceEncoder n'est chargé qu'une fois en mémoire."""
    _instance = None
    _encoder = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_encoder(self):
        if self._encoder is None:
            # device="cpu" volontaire : le VoiceEncoder sur cuda peut produire
            # des embeddings légèrement différents d'un run à l'autre
            # (nondéterminisme des kernels cuDNN/cuBLAS), ce qui fait basculer
            # l'assignation de cluster sur les segments proches de la frontière
            # entre locuteurs. La diarisation n'est pas le goulot d'étranglement
            # du pipeline, le coût CPU est négligeable, et ça libère de la VRAM
            # pour Whisper sur une 3050 4GB.
            logger.info("Chargement du VoiceEncoder Resemblyzer sur CPU (déterminisme, poids inclus dans le package)...")
            self._encoder = VoiceEncoder(device="cpu")
            logger.info("VoiceEncoder chargé.")
        return self._encoder

    @staticmethod
    def _find_best_k(embeddings: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Détecte automatiquement le nombre de locuteurs par score de silhouette.

        Teste k = 2..MAX_SPEAKERS avec KMeans et retient le k dont les clusters
        sont les mieux séparés. Si aucun k ne donne une séparation nette
        (meilleur score < MIN_SILHOUETTE_FOR_MULTI), retourne k=1.

        Returns:
            (k retenu, labels correspondants)
        """
        n = len(embeddings)
        # La silhouette exige au moins k+1 points ; on borne aussi par MAX_SPEAKERS.
        k_max = min(MAX_SPEAKERS, n - 1)

        if k_max < 2:
            return 1, np.zeros(n, dtype=int)

        # Première passe : on calcule le score de chaque k candidat.
        candidates: list[tuple[int, float, np.ndarray]] = []

        for k in range(2, k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(embeddings)

            # Un cluster réduit à 1 seul point rend la silhouette instable
            # et signale presque toujours un outlier isolé, pas un locuteur.
            counts = np.bincount(labels)
            if counts.min() < 2:
                logger.debug("k=%d écarté : un cluster ne contient qu'un seul segment.", k)
                continue

            score = silhouette_score(embeddings, labels, metric="cosine")
            logger.info("Sélection auto : k=%d -> silhouette=%.3f", k, score)
            candidates.append((k, score, labels))

        if not candidates:
            return 1, np.zeros(n, dtype=int)

        best_score = max(score for _, score, _ in candidates)

        if best_score < MIN_SILHOUETTE_FOR_MULTI:
            logger.info(
                "Meilleure silhouette %.3f < %.2f : les voix ne se séparent pas "
                "en groupes distincts -> 1 seul locuteur retenu.",
                best_score, MIN_SILHOUETTE_FOR_MULTI,
            )
            return 1, np.zeros(n, dtype=int)

        # Règle de parcimonie (rasoir d'Occam) : le GE2E a tendance à
        # sur-découper une même voix (segments longs vs courts de la même
        # personne forment des sous-clusters). Un k supplémentaire n'est
        # accepté que s'il apporte une amélioration FRANCHE de la
        # séparation. Concrètement : on retient le PLUS PETIT k dont le
        # score est à moins de PARSIMONY_MARGIN du meilleur score.
        best_k, best_k_score, best_labels = next(
            (k, score, labels)
            for k, score, labels in candidates  # candidates est trié par k croissant
            if score >= best_score - PARSIMONY_MARGIN
        )

        logger.info(
            "Nombre de locuteurs détecté automatiquement : %d "
            "(silhouette=%.3f, meilleur score absolu=%.3f, marge de parcimonie=%.2f)",
            best_k, best_k_score, best_score, PARSIMONY_MARGIN,
        )
        return best_k, best_labels

    @staticmethod
    def _smooth_speakers(transcript_segments, speakers, min_duration=2.0):
        """
        Corrige les segments courts isolés entre deux segments du même locuteur.
        Ex : Person2 → Person1 (1s) → Person2  devient  Person2 → Person2 → Person2.

        Le GE2E est peu fiable sur les répliques très courtes (< 2s) : un
        embedding sur 1s de parole est bruité et bascule facilement du mauvais
        côté. Quand un segment court est encadré par deux segments du même
        locuteur, il lui appartient presque certainement.
        """
        smoothed = speakers.copy()
        for i in range(1, len(speakers) - 1):
            duration = transcript_segments[i].end - transcript_segments[i].start
            if (
                duration < min_duration
                and speakers[i - 1] == speakers[i + 1]
                and speakers[i] != speakers[i - 1]
            ):
                smoothed[i] = speakers[i - 1]
        return smoothed

    def diarize_segments(
        self,
        audio_path: Path,
        transcript_segments: list,  # list[TranscriptSegment] de Whisper
    ) -> list[str]:
        """
        Assigne un locuteur à chaque segment de transcription Whisper.
        Le nombre de locuteurs est détecté automatiquement (score de silhouette).

        Args:
            audio_path: chemin vers le fichier audio.
            transcript_segments: segments produits par le transcriber (avec .start, .end).

        Returns:
            Liste de speaker_ids ("Person1", "Person2"...) alignée index par index
            sur transcript_segments.
        """
        encoder = self._load_encoder()

        # librosa.load préserve la durée originale du fichier (contrairement à
        # preprocess_wav dont le VAD supprime les silences et décale la timeline).
        wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)

        embeddings = []
        valid_indices = []  # indices des segments assez longs/audibles pour être embeddés

        for i, seg in enumerate(transcript_segments):
            start_sample = int(seg.start * 16000)
            end_sample = min(int(seg.end * 16000), len(wav))
            chunk = wav[start_sample:end_sample]

            # Segment trop court ou silencieux : il héritera du speaker voisin
            if len(chunk) < MIN_SEGMENT_SAMPLES:
                continue
            if np.sqrt(np.mean(chunk ** 2)) < SILENCE_RMS_THRESHOLD:
                continue

            embeddings.append(encoder.embed_utterance(chunk))
            valid_indices.append(i)

        n = len(transcript_segments)
        if len(embeddings) < 2:
            return ["Person1"] * n

        embeddings = np.array(embeddings)
        # Normalisation L2 : rend la distance cosine plus stable, et rend la
        # distance euclidienne de KMeans équivalente à la distance cosine.
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        _, labels = self._find_best_k(embeddings)

        # Assigne les labels aux segments valides
        speakers: list[str | None] = [None] * n
        for idx, label in zip(valid_indices, labels):
            speakers[idx] = f"Person{label + 1}"

        # Les segments trop courts/silencieux héritent du speaker précédent
        for i in range(n):
            if speakers[i] is None:
                speakers[i] = speakers[i - 1] if i > 0 and speakers[i - 1] else "Person1"

        # Lissage : corrige les répliques courtes isolées entre deux segments
        # du même locuteur (cas fréquent sur les dialogues rapides).
        speakers = self._smooth_speakers(transcript_segments, speakers)

        n_speakers = len(set(speakers))
        logger.info("Diarisation terminée : %d locuteur(s) détecté(s).", n_speakers)
        for seg, sp in zip(transcript_segments, speakers):
            logger.info("  %.1fs → %.1fs = %s", seg.start, seg.end, sp)

        return speakers


diarizer_service = DiarizerService()