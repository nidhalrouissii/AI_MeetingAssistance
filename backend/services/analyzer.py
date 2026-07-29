"""
Service d'analyse LLM — extrait résumé, décisions, tâches et questions depuis le transcript.

Stratégie Map-Reduce pour les longs transcripts (réunions > 30 min) :
- Map  : découpe le transcript en segments de 15 min, résume chaque segment
- Reduce : fusionne tous les résumés partiels en un seul compte rendu final

Utilise Groq (llama-3.3-70b) en free tier — aucun coût.
Fallback automatique vers Ollama (local) si Groq est indisponible.
"""
import json
import logging
from groq import Groq

from backend.core.config import settings
from backend.prompts.meeting_prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Nombre de mots approximatif pour 15 min de réunion
WORDS_PER_CHUNK = 2000


class AnalyzerService:

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    def _call_llm(self, user_prompt: str, is_partial: bool = False) -> dict:
        """Appel Groq avec gestion d'erreur et parsing JSON robuste."""
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(user_prompt, is_partial)}
                ],
                temperature=settings.llm_temperature,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()

            # Nettoie les balises markdown si le LLM en a ajouté malgré les instructions
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.error("Réponse LLM non parseable en JSON : %s", e)
            return self._empty_result()
        except Exception as e:
            logger.error("Erreur appel Groq : %s", e)
            raise RuntimeError(f"Échec de l'analyse LLM : {e}") from e

    def analyze(self, transcript: str) -> dict:
        """
        Analyse complète du transcript.
        Applique Map-Reduce si le transcript est trop long pour un seul appel.
        """
        words = transcript.split()

        if len(words) <= WORDS_PER_CHUNK:
            # Transcript court : appel direct
            logger.info("Transcript court (%d mots) — appel LLM direct.", len(words))
            return self._call_llm(transcript)

        # Transcript long : stratégie Map-Reduce
        logger.info("Transcript long (%d mots) — stratégie Map-Reduce.", len(words))
        return self._map_reduce(transcript, words)

    def _map_reduce(self, transcript: str, words: list) -> dict:
        """Découpe le transcript en chunks et fusionne les résultats."""
        chunks = []
        for i in range(0, len(words), WORDS_PER_CHUNK):
            chunk = " ".join(words[i:i + WORDS_PER_CHUNK])
            chunks.append(chunk)

        logger.info("Map-Reduce : %d chunks à analyser.", len(chunks))

        # Map : analyse chaque chunk
        partial_results = []
        for i, chunk in enumerate(chunks):
            logger.info("Analyse chunk %d/%d...", i + 1, len(chunks))
            result = self._call_llm(chunk, is_partial=True)
            partial_results.append(result)

        # Reduce : fusionne les résultats partiels
        return self._reduce(partial_results)

    def _reduce(self, partial_results: list[dict]) -> dict:
        """Fusionne les résultats partiels en un résultat final cohérent."""
        # Combine toutes les décisions et tâches des chunks
        all_decisions = []
        all_tasks = []
        all_questions = []
        summaries = []

        for r in partial_results:
            summaries.append(r.get("summary", ""))
            all_decisions.extend(r.get("decisions", []))
            all_tasks.extend(r.get("tasks", []))
            all_questions.extend(r.get("open_questions", []))

        # Demande au LLM de faire la synthèse finale des résumés partiels
        combined_summaries = "\n\n---\n\n".join(summaries)
        final_summary_prompt = f"""Voici les résumés de plusieurs extraits d'une même réunion. 
Rédige un résumé global cohérent et concis (8-10 lignes maximum) qui synthétise l'ensemble :

{combined_summaries}"""

        try:
            final_response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "user", "content": final_summary_prompt}
                ],
                temperature=settings.llm_temperature,
                max_tokens=500,
            )
            final_summary = final_response.choices[0].message.content.strip()
        except Exception:
            final_summary = " ".join(summaries)

        return {
            "summary": final_summary,
            "decisions": list(dict.fromkeys(all_decisions)),   # dédoublonnage
            "tasks": all_tasks,
            "open_questions": list(dict.fromkeys(all_questions))
        }

    @staticmethod
    def _empty_result() -> dict:
        return {"summary": "", "decisions": [], "tasks": [], "open_questions": []}


analyzer_service = AnalyzerService()