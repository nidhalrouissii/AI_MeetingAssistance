"""
Fusionne les segments de transcription (WhisperX) avec les segments de locuteurs
(Resemblyzer) pour produire un transcript final du type :

[00:00] Person1: Bonjour tout le monde
[00:08] Person2: On commence par le point budget
[00:15] Person1: We need to discuss the timeline first
"""
from backend.services.transcriber import TranscriberService, TranscriptSegment
from backend.services.diarizer import DiarizerService, SpeakerSegment


def merge_transcript_and_speakers(transcript_segments, speaker_segments):
    lines = []
    for seg in transcript_segments:
        # Vote majoritaire : quel speaker couvre le plus de temps dans [start, end] ?
        best_speaker = "Person1"
        best_overlap = 0.0

        for sp_seg in speaker_segments:
            overlap = min(seg.end, sp_seg.end) - max(seg.start, sp_seg.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp_seg.speaker_id

        # Fallback si aucun overlap → plus proche
        if best_overlap <= 0:
            best_speaker = DiarizerService.assign_speaker(speaker_segments, (seg.start + seg.end) / 2)

        minutes, seconds = int(seg.start // 60), int(seg.start % 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {best_speaker}: {seg.text.strip()}")

    return "\n".join(lines)
