"""
Prompt système pour l'extraction structurée depuis un transcript de réunion.
Le LLM doit retourner UNIQUEMENT du JSON valide, sans texte autour.
"""

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de réunions professionnelles.
Tu reçois un transcript de réunion et tu dois extraire les informations clés.
Tu réponds UNIQUEMENT en JSON valide, sans texte avant ou après, sans balises markdown.

Format de réponse attendu :
{
  "summary": "Résumé exécutif de la réunion en 5-10 lignes",
  "decisions": [
    "Décision 1 prise durant la réunion",
    "Décision 2 prise durant la réunion"
  ],
  "tasks": [
    {
      "description": "Description de la tâche",
      "responsible": "Nom ou Person1 si non précisé",
      "deadline": "Date ou délai si mentionné, sinon null"
    }
  ],
  "open_questions": [
    "Question ou point non résolu 1",
    "Question ou point non résolu 2"
  ]
}"""


def build_user_prompt(transcript_chunk: str, is_partial: bool = False) -> str:
    """
    Construit le prompt utilisateur pour un chunk de transcript.
    is_partial=True quand c'est un segment intermédiaire (Map), False pour la synthèse finale (Reduce).
    """
    if is_partial:
        return f"""Voici un extrait de transcript de réunion. Extrais les informations clés de CET EXTRAIT uniquement :

{transcript_chunk}

Retourne le JSON avec les informations de cet extrait."""
    else:
        return f"""Voici le transcript complet d'une réunion professionnelle :

{transcript_chunk}

Analyse ce transcript et extrais toutes les informations clés. Retourne uniquement le JSON demandé."""