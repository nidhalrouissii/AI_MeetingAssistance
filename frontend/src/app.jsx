import { useEffect, useMemo, useRef, useState } from "react";

/* ============================================================
   AI Meeting Assistant — Frontend React (v2)
   - Barre de progression déterminée, pilotée par le champ `step`
     retourné par GET /api/status (transcription → diarisation →
     analyse → done). La barre se remplit de 0 à 100 %, une fois.
   - Écran résultat : bandeau de synthèse (durée, locuteurs,
     décisions, tâches), transcript groupé par prise de parole.
   - Export PDF via l'impression navigateur (feuille de style
     print dédiée dans index.css).
   ============================================================ */

const ALLOWED = [".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac"];
const POLL_MS = 3000;
const SPEAKER_COLORS = ["--spk-1", "--spk-2", "--spk-3", "--spk-4", "--spk-5", "--spk-6"];

/* Progression : chaque étape backend a un plancher (atteint dès que
   l'étape démarre) et un plafond (jamais dépassé tant que le backend
   n'a pas confirmé l'étape suivante). Entre les deux, la barre avance
   doucement pour montrer que le traitement est vivant. */
const STEP_RANGES = {
  pending:       { floor: 2,  ceil: 8 },
  transcription: { floor: 10, ceil: 55 },
  diarisation:   { floor: 58, ceil: 78 },
  analyse:       { floor: 80, ceil: 96 },
  done:          { floor: 100, ceil: 100 },
};

const STEP_LABELS = [
  { key: "transcription", label: "Transcription" },
  { key: "diarisation",   label: "Identification des locuteurs" },
  { key: "analyse",       label: "Rédaction du compte rendu" },
];

const STEP_ORDER = ["pending", "transcription", "diarisation", "analyse", "done"];

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

function speakerColor(speaker) {
  const n = parseInt(String(speaker).replace(/\D/g, ""), 10);
  const idx = Number.isFinite(n) && n >= 1 ? (n - 1) % SPEAKER_COLORS.length : 0;
  return `var(${SPEAKER_COLORS[idx]})`;
}

/* Regroupe les segments consécutifs du même locuteur en "prises de
   parole" — beaucoup plus lisible qu'une ligne par segment Whisper. */
function groupBySpeaker(transcript) {
  const turns = [];
  for (const seg of transcript) {
    const last = turns[turns.length - 1];
    if (last && last.speaker === seg.speaker) {
      last.end = seg.end;
      last.texts.push(seg.text);
    } else {
      turns.push({ speaker: seg.speaker, start: seg.start, end: seg.end, texts: [seg.text] });
    }
  }
  return turns;
}

/* ---------- Header ---------- */

function Header() {
  return (
    <header className="header no-print">
      <div className="beam" aria-hidden="true">EY</div>
      <h1>AI Meeting Assistant</h1>
      <span className="sub">Comptes rendus de réunion automatiques</span>
    </header>
  );
}

function ErrorBox({ message }) {
  if (!message) return null;
  return (
    <div className="error-box" role="alert">
      <div className="title">Le traitement a échoué</div>
      <div className="detail">{message}</div>
    </div>
  );
}

/* ---------- Écran upload ---------- */

function UploadScreen({ onStarted, initialError }) {
  const [file, setFile] = useState(null);
  const [drag, setDrag] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(initialError || "");
  const inputRef = useRef(null);

  const pickFile = (f) => {
    if (!f) return;
    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) {
      setError(`Format non supporté : ${ext}. Formats acceptés : ${ALLOWED.join(", ")}`);
      setFile(null);
      return;
    }
    setError("");
    setFile(f);
  };

  const upload = async () => {
    if (!file || sending) return;
    setSending(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/upload", { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Erreur serveur (${res.status})`);
      }
      const data = await res.json();
      onStarted(data.job_id, file.name);
    } catch (e) {
      setError(e.message || "Envoi impossible. Vérifiez que le serveur est démarré.");
      setSending(false);
    }
  };

  return (
    <>
      <div className="intro">
        <h2>Analyser une réunion</h2>
        <p>
          Déposez l'enregistrement audio d'une réunion. La transcription, l'identification
          des locuteurs et le compte rendu sont générés automatiquement.
        </p>
      </div>

      <ErrorBox message={error} />

      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
        role="button"
        tabIndex={0}
        aria-label="Choisir un fichier audio"
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pickFile(e.dataTransfer.files?.[0]);
        }}
      >
        <div className="dz-icon" aria-hidden="true" />
        <div className="dz-main">Glissez un fichier audio ici, ou cliquez pour parcourir</div>
        <div className="dz-hint">{ALLOWED.join("  ·  ")} — jusqu'à 2 h de réunion</div>
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED.join(",")}
          hidden
          onChange={(e) => pickFile(e.target.files?.[0])}
        />
      </div>

      {file && (
        <div className="file-row">
          <span className="name">{file.name}</span>
          <span className="size">{formatSize(file.size)}</span>
        </div>
      )}

      <div className="actions">
        <button className="btn" disabled={!file || sending} onClick={upload}>
          {sending ? "Envoi en cours…" : "Générer le compte rendu"}
        </button>
        {file && !sending && (
          <button className="btn btn-ghost" onClick={() => { setFile(null); setError(""); }}>
            Retirer le fichier
          </button>
        )}
      </div>
    </>
  );
}

/* ---------- Écran traitement (progression déterminée) ---------- */

function ProcessingScreen({ filename, startedAt, step }) {
  const [elapsed, setElapsed] = useState(0);
  const [progress, setProgress] = useState(STEP_RANGES.pending.floor);

  // Horloge du temps écoulé
  useEffect(() => {
    const t = setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => clearInterval(t);
  }, [startedAt]);

  // Progression : saute au plancher de l'étape courante dès qu'elle change,
  // puis avance lentement vers le plafond sans jamais le dépasser.
  useEffect(() => {
    const range = STEP_RANGES[step] || STEP_RANGES.pending;
    setProgress((p) => Math.max(p, range.floor));

    const t = setInterval(() => {
      setProgress((p) => {
        if (p >= range.ceil) return p;
        // Approche asymptotique du plafond : rapide au début, lente à la fin
        return Math.min(range.ceil, p + Math.max(0.2, (range.ceil - p) * 0.03));
      });
    }, 400);
    return () => clearInterval(t);
  }, [step]);

  const currentIdx = STEP_ORDER.indexOf(step || "pending");

  return (
    <div className="card processing">
      <div className="progress-track" role="progressbar" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}>
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="progress-pct">{Math.round(progress)} %</div>

      <ol className="steps">
        {STEP_LABELS.map((s) => {
          const idx = STEP_ORDER.indexOf(s.key);
          const state = idx < currentIdx ? "done" : idx === currentIdx ? "active" : "todo";
          return (
            <li key={s.key} className={`step ${state}`}>
              <span className="step-mark" aria-hidden="true" />
              {s.label}
            </li>
          );
        })}
      </ol>

      <div className="status-label">Traitement de « {filename} »</div>
      <div className="elapsed">Temps écoulé : {formatTime(elapsed)}</div>
    </div>
  );
}

/* ---------- Écran résultat ---------- */

function StatCard({ value, label }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function TasksTable({ tasks }) {
  if (!tasks?.length) return <p className="empty">Aucune tâche identifiée dans cette réunion.</p>;
  return (
    <table className="tasks-table">
      <thead>
        <tr>
          <th style={{ width: "55%" }}>Tâche</th>
          <th>Responsable</th>
          <th>Deadline</th>
        </tr>
      </thead>
      <tbody>
        {tasks.map((t, i) => (
          <tr key={i}>
            <td>{t.description}</td>
            <td>
              {t.responsible
                ? <span className="badge">{t.responsible}</span>
                : <span className="badge empty">Non assigné</span>}
            </td>
            <td>
              {t.deadline
                ? <span className="badge">{t.deadline}</span>
                : <span className="badge empty">Non précisée</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ResultScreen({ data, onReset }) {
  const analysis = data.result || {};
  const transcript = data.transcript || [];

  const turns = useMemo(() => groupBySpeaker(transcript), [transcript]);
  const speakers = useMemo(
    () => [...new Set(transcript.map((s) => s.speaker))],
    [transcript]
  );
  const duration = transcript.length ? transcript[transcript.length - 1].end : 0;

  return (
    <>
      <div className="result-actions no-print">
        <button className="btn btn-ghost" onClick={onReset}>← Nouvelle réunion</button>
        <span className="spacer" />
        <button className="btn" onClick={() => window.print()}>Exporter en PDF</button>
      </div>

      {/* Bandeau de synthèse */}
      <div className="hero">
        <div className="hero-beam" aria-hidden="true" />
        <div className="hero-body">
          <div className="hero-title">
            <span className="hero-kicker">Compte rendu de réunion</span>
            <h2>{data.filename || "Réunion"}</h2>
            <span className="hero-date">
              {data.created_at ? new Date(data.created_at).toLocaleString("fr-FR", { dateStyle: "long", timeStyle: "short" }) : ""}
            </span>
          </div>
          <div className="hero-stats">
            <StatCard value={formatTime(duration)} label="Durée" />
            <StatCard value={speakers.length} label={speakers.length > 1 ? "Locuteurs" : "Locuteur"} />
            <StatCard value={analysis.decisions?.length ?? 0} label="Décisions" />
            <StatCard value={analysis.tasks?.length ?? 0} label="Tâches" />
          </div>
        </div>
      </div>

      <section className="card">
        <h2>Résumé exécutif</h2>
        {analysis.summary
          ? <p className="summary-text">{analysis.summary}</p>
          : <p className="empty">Aucun résumé généré.</p>}
      </section>

      <div className="two-col">
        <section className="card">
          <h2>Décisions prises</h2>
          {analysis.decisions?.length ? (
            <ul className="decisions-list">
              {analysis.decisions.map((d, i) => <li key={i}>{d}</li>)}
            </ul>
          ) : <p className="empty">Aucune décision identifiée.</p>}
        </section>

        <section className="card">
          <h2>Questions ouvertes</h2>
          {analysis.open_questions?.length ? (
            <ul className="questions-list">
              {analysis.open_questions.map((q, i) => <li key={i}>{q}</li>)}
            </ul>
          ) : <p className="empty">Aucun point resté en suspens.</p>}
        </section>
      </div>

      <section className="card">
        <h2>Tâches à réaliser</h2>
        <TasksTable tasks={analysis.tasks} />
      </section>

      <section className="card">
        <div className="transcript-head">
          <h2>Transcription ({turns.length} prises de parole)</h2>
          <div className="legend no-print">
            {speakers.map((sp) => (
              <span key={sp} className="legend-item">
                <span className="dot" style={{ background: speakerColor(sp) }} />
                {sp}
              </span>
            ))}
          </div>
        </div>
        {turns.length ? (
          <div className="transcript">
            {turns.map((t, i) => (
              <div className="turn" key={i}>
                <div className="turn-meta">
                  <span className="chip" style={{ background: speakerColor(t.speaker) }}>{t.speaker}</span>
                  <span className="turn-time">{formatTime(t.start)} – {formatTime(t.end)}</span>
                </div>
                <p className="turn-text">{t.texts.join(" ")}</p>
              </div>
            ))}
          </div>
        ) : <p className="empty">Transcription indisponible.</p>}
      </section>
    </>
  );
}

/* ---------- App ---------- */

export default function App() {
  const [view, setView] = useState("upload");   // upload | processing | result
  const [jobId, setJobId] = useState(null);
  const [filename, setFilename] = useState("");
  const [startedAt, setStartedAt] = useState(null);
  const [step, setStep] = useState("pending");
  const [result, setResult] = useState(null);
  const [fatalError, setFatalError] = useState("");

  const onStarted = (id, name) => {
    setJobId(id);
    setFilename(name);
    setStartedAt(Date.now());
    setStep("pending");
    setFatalError("");
    setView("processing");
  };

  const onReset = () => {
    setJobId(null);
    setFilename("");
    setResult(null);
    setStep("pending");
    setFatalError("");
    setView("upload");
  };

  useEffect(() => {
    if (view !== "processing" || !jobId) return;

    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) throw new Error(`Statut indisponible (${res.status})`);
        const data = await res.json();
        if (cancelled) return;

        if (data.step) setStep(data.step);

        if (data.status === "done") {
          setStep("done");
          const r = await fetch(`/api/result/${jobId}`);
          if (!r.ok) throw new Error(`Résultat indisponible (${r.status})`);
          const payload = await r.json();
          if (cancelled) return;
          setResult(payload);
          // Laisse la barre atteindre visuellement 100 % avant de basculer
          setTimeout(() => { if (!cancelled) setView("result"); }, 500);
        } else if (data.status === "error") {
          setFatalError(data.error || "Une erreur est survenue pendant le traitement. Réessayez avec le même fichier ; si l'erreur persiste, vérifiez les logs du serveur.");
          setView("upload");
        }
      } catch (e) {
        if (cancelled) return;
        setFatalError(e.message || "Connexion au serveur perdue.");
        setView("upload");
      }
    };

    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => { cancelled = true; clearInterval(timer); };
  }, [view, jobId]);

  return (
    <>
      <Header />
      <main className="container">
        {view === "upload" && <UploadScreen onStarted={onStarted} initialError={fatalError} />}
        {view === "processing" && <ProcessingScreen filename={filename} startedAt={startedAt} step={step} />}
        {view === "result" && result && <ResultScreen data={result} onReset={onReset} />}
      </main>
    </>
  );
}