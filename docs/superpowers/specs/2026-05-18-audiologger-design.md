# AudioLogger — Design

**Date:** 2026-05-18
**Status:** Approved for planning
**Platform:** Windows 10/11 only
**Primary user:** Chris (RTX 4090 Laptop GPU), distributable to others (Python + GPU/CPU extras)

## Goal

Background utility for meetings (Slack/Discord/Teams) and voice notes. Global hotkey toggles recording of microphone + system audio. After stop, produces a Markdown transcript with at least "Ich" vs. "Others" separation, with full pyannote-based speaker diarization on the system-audio side.

## Non-Goals (v1)

- Cloud transcription — fully local
- GUI settings window — tray menu + YAML config only
- Live transcription during recording — only post-stop
- Speaker name mapping (Sprecher 1 → "Max") — generic labels only
- Pause/resume during recording — only start/stop toggle
- Voice-activity-based auto-stop
- macOS/Linux support — WASAPI is Windows-only
- PyInstaller standalone EXE — deferred to a later phase

## Architecture Overview

Two-process model: a long-lived tray daemon, and one transcription worker subprocess spawned per recording (re-used for consecutive jobs to amortize model load).

```
┌──────────────────────────────────────────────────┐
│  Tray-Daemon-Prozess (Python, immer im RAM)      │
│                                                  │
│  pystray ── keyboard ── Settings (YAML)          │
│        │         │                               │
│        ▼         ▼                               │
│  RecordingController  (IDLE / RECORDING)         │
│        │                                         │
│        ▼                                         │
│  AudioCaptureThread                              │
│   - 2 streams: mic + WASAPI-loopback             │
│   - writes 2 WAV files in parallel               │
│        │                                         │
│        ▼ (on stop)                               │
│  TranscriptionJobQueue (in-process FIFO)         │
│   - spawnt Worker-Subprozess                     │
└────────────────────┬─────────────────────────────┘
                     ▼ subprocess
       ┌──────────────────────────────┐
       │ transcribe_worker.py         │
       │  - WhisperX large-v3         │
       │  - pyannote diarization      │
       │  - schreibt .md + Job-Log    │
       │  - bleibt 30s warm für       │
       │    nächsten Queue-Job        │
       └──────────────────────────────┘
```

### Output-Layout pro Aufnahme

```
<output-dir>/
  2026-05-18_14-32-15_meeting/
    mic.wav                  ← deine Spur
    system.wav               ← andere
    mixed.wav                ← gemischt (für späteres Anhören)
    transcript.md            ← finales Ergebnis
    transcript.json          ← Roh-Output WhisperX (Reprocessing)
    job.log                  ← Worker-Logs
    RECORDING_IN_PROGRESS    ← Marker-File während Aufnahme; bei Crash da → Recovery
```

Session-Verzeichnisname: `YYYY-MM-DD_HH-MM-SS` (Suffix `_meeting` ist Platzhalter; v1 lässt das Suffix weg, der Pfad ist `YYYY-MM-DD_HH-MM-SS/`).

## Components

### `config.py`

YAML-backed dataclass. Persistiert in `%APPDATA%/AudioLogger/config.yaml`. Fehlende Felder werden beim Laden mit Defaults aufgefüllt und zurückgeschrieben.

```python
@dataclass
class Config:
    hotkey: str = "ctrl+alt+r"           # keyboard-lib Syntax
    output_dir: Path = Path("./recordings")
    whisper_model: str = "large-v3"      # tiny|base|small|medium|large-v3
    device: str = "cuda"                 # cuda|cpu
    compute_type: str = "float16"        # float16|int8|float32
    diarization_enabled: bool = True
    huggingface_token: str | None = None # pyannote benötigt es
    audio_source: str = "all"            # "all" | "apps"
    filtered_app_names: list[str] = []   # nur bei audio_source="apps"
    notification_enabled: bool = True
```

### `RecordingController`

State-Machine, orchestriert Aufnahme-Lifecycle.

- States: `IDLE → RECORDING → STOPPING → IDLE`
- `toggle()` — vom Hotkey-Handler aufgerufen
- Bei Start: erzeugt Session-Verzeichnis, legt Marker-File an, startet `AudioCaptureThread`, ändert Tray-Icon, sendet Toast
- Bei Stop: signalisiert Thread, wartet auf WAVs, generiert `mixed.wav`, löscht Marker-File, enqueued Job

### `AudioCaptureThread`

Ein Thread-Container mit zwei parallelen Capture-Threads (mic + loopback) und einem `threading.Event` als Stop-Signal.

- Format: 48 kHz mono 16-bit PCM für beide Streams (~5,5 MB/min/Stream). Mono reicht für Speech, Whisper resampled intern eh auf 16 kHz.
- Library: `soundcard` (`default_microphone()` und `default_speaker()` mit Loopback-Recorder)
- Schreibt direkt in `wave.Wave_write`-Objekte in 1-Sekunden-Chunks → keine großen In-Memory-Buffer (wichtig für Stunden-Meetings)
- Bei `audio_source="apps"`: eigener Code-Pfad in `process_loopback.py` — wrappt `ActivateAudioInterfaceAsync` mit `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS` via `ctypes` (Windows 10 21H2+).
- Default-Device-Snapshot zu Aufnahmebeginn; bei Wechsel während Aufnahme → Toast-Warnung (kein Abbruch)

### `TranscriptionJobQueue`

In-Process FIFO. Genau ein Worker-Subprozess gleichzeitig.

- `enqueue(session_dir: Path)` — fügt zu `pending.txt` hinzu (eine Session pro Zeile, absolute Pfade)
- Spawnt `python -m audiologger.transcribe_worker` wenn kein Worker läuft
- Worker liest `pending.txt`, arbeitet Einträge der Reihe nach ab und entfernt sie nach Erfolg
- Worker bleibt 30 s nach letztem Job idle, prüft `pending.txt` periodisch (1 s), beendet sich wenn Liste leer
- Tray fragt Status via Heartbeat-File `worker_status.json` (`{ "running": "...", "queued": [...] }`)

### `transcribe_worker.py`

Eigenständiger Prozess, gestartet vom Tray.

Pro Session:
1. Transkribiere `mic.wav` mit WhisperX → alle Segmente bekommen Label "Ich"
2. Transkribiere `system.wav` mit WhisperX, dann pyannote-Diarization auf demselben Audio → Segmente bekommen Labels "Sprecher 1", "Sprecher 2", …
3. Merge beider Segment-Listen chronologisch nach Start-Zeitstempel
4. Schreibe `transcript.md` und `transcript.json`
5. Aktualisiere `pending.txt`, prüfe auf weitere Jobs

Modell wird einmal pro Worker-Start geladen — der 30-s-Warmhalter spart Reload-Kosten zwischen Jobs.

### `TrayApp`

`pystray` + PIL für dynamische Icons.

- Menü: Start/Stop (mirror Hotkey) · Output-Ordner ändern · Hotkey ändern · Audio-Quelle (Alles / Nur diese Apps) · App-Filter konfigurieren · Job-Status · Beenden
- Icon-Farben: grau (idle) · rot (recording) · gelb (transcribing in worker)
- Toasts via `winotify` bei Start, Stop, Worker-Done, Worker-Fail

## Audio-Capture-Details

### Standard-Modus (`audio_source: "all"`)

`soundcard.default_speaker().recorder(samplerate=48000, channels=[0])` öffnet einen Loopback-Stream auf dem Default-Output. Funktioniert auch wenn nichts läuft (silent samples).

Bei Default-Device-Wechsel während Aufnahme (z.B. Bluetooth-Headset connect): Toast-Warnung, Aufnahme läuft auf ursprünglichem Device weiter.

### App-Filter-Modus (`audio_source: "apps"`)

Windows 10 21H2+ Process Loopback API. Keine bestehende Python-Lib deckt das sauber ab — eigenes `process_loopback.py`-Modul wrappt die COM-Calls via `ctypes`.

Settings-UI listet laufende Audio-Sessions (via `pycaw`) zur Auswahl. Auswahl wird per Exe-Name persistiert (PIDs ändern sich).

Fallback: scheitert der Modus (alte Windows-Version, fehlende Rechte), Toast + automatisches Downgrade auf `"all"` für die laufende Aufnahme. Config bleibt unverändert.

### `mixed.wav`

Wird **nicht** live erzeugt. Nach Stop, vor Enqueue, wird ein Mix per `numpy` aus mic+system gebaut (sample-aligned dank gleicher Startzeit & Sample-Rate). Macht den Worker unabhängig vom Mix-Schritt.

### Max-Aufnahmelänge

Keine harte Grenze. Praktisches Limit = Disk-Space. Bei >3 h Toast-Warnung ("Aufnahme läuft 3+ Stunden — alles ok?"), kein automatischer Abbruch.

## Transcript-Output (Markdown)

```markdown
# Aufnahme 2026-05-18 14:32:15

**Dauer:** 47:21
**Quelle:** mic + system (loopback, all)
**Modell:** WhisperX large-v3 + pyannote/speaker-diarization-3.1

---

**[00:00:03] Ich:** Hi zusammen, könnt ihr mich hören?
**[00:00:06] Sprecher 1:** Ja, alles klar bei dir.
**[00:00:09] Sprecher 2:** Bei mir auch. Sollen wir anfangen?
**[00:00:12] Ich:** Ja. Also der erste Punkt ist...
```

- "Ich" kommt immer aus `mic.wav`
- "Sprecher N" aus diarisiertem `system.wav`. IDs sind nur innerhalb einer Aufnahme stabil
- Bei Crosstalk: beide Sprecher in Reihenfolge der Start-Zeitstempel
- Zeitstempel = Segment-Start relativ zur Aufnahme
- Falls Diarization fehlschlägt oder deaktiviert: alle System-Segmente bekommen Label "Andere"; Warnung im Markdown-Header

## Error-Handling

| Szenario | Verhalten |
|---|---|
| Hotkey-Bind fehlgeschlagen (Konflikt) | Toast: "Hotkey X belegt — bitte in Tray ändern". App läuft weiter, Hotkey inaktiv bis Änderung. |
| Mic nicht verfügbar | Aufnahme startet mit nur System-Stream + Toast-Warnung. `mic.wav` fehlt im Output. |
| Loopback nicht verfügbar | Symmetrisch: nur Mic, Toast-Warnung. |
| Disk voll während Aufnahme | Aufnahme wird sauber gestoppt, bis dahin geschriebenes Audio bleibt erhalten, Toast: "Aufnahme gestoppt — kein Plattenplatz". |
| Worker-Crash (OOM, CUDA-Fehler) | `job.log` enthält Traceback. Tray-Status zeigt "Fehlgeschlagen". WAVs bleiben → manueller Retry über Tray-Menü ("Letzte Aufnahme erneut transkribieren"). |
| HuggingFace-Token fehlt, Diarization an | Worker fällt zurück auf "ohne Diarization", schreibt Warnung in Markdown-Header. |
| App-Crash während Recording | WAVs werden in 1-s-Chunks geschrieben → bis kurz vorm Crash erhalten. Beim nächsten Start prüft Tray auf Session-Verzeichnisse mit Marker-File `RECORDING_IN_PROGRESS` und bietet "Diese Aufnahme transkribieren" an. |
| Modell-Download (erstes Mal) | Worker zeigt Status "Lade Modell (~3 GB)..." in Tray-Heartbeat. Blockiert ersten Job, danach gecached in `%USERPROFILE%/.cache/whisperx`. |

## Testing-Strategie

### Unit-Tests (pytest, schnell, kein Audio/GPU)

- `config`: Laden/Speichern, Default-Fallback bei fehlenden Feldern
- `RecordingController`: State-Machine (IDLE↔RECORDING), Toggle-Verhalten, Session-Verzeichnis-Naming
- `TranscriptionJobQueue`: Enqueue/Spawn-Logik, FIFO-Reihenfolge, Worker-Reuse (Mock-Worker)
- `transcript_merger`: Merge zweier Segment-Listen nach Zeitstempel, Markdown-Rendering (Snapshot-Tests mit fixierten Beispiel-Segmenten)

### Integration-Tests (langsamer, echte WAVs, gemockter Whisper)

- Full-Cycle mit 30-s-Sample-Audio (mic + system fixtures im Repo): Trigger → WAV-Schreiben → Worker-Spawn → Markdown-Output. Whisper wird durch Stub ersetzt, der deterministische Segmente liefert.
- App-Crash-Simulation: Marker-File-Recovery beim Neustart.

### Manueller Test-Plan (eigene Checkliste)

Wegen GPU + Hardware-Abhängigkeiten nicht automatisierbar:

- Echte WhisperX-Transkription mit 1-min DE+EN Mix-Audio, Vergleich gegen Referenz-Transkript
- Hotkey-Erkennung bei diversen Vordergrund-Apps (Discord-Vollbild, Slack-Call, Teams)
- Default-Device-Wechsel mid-recording (Bluetooth-Headset rein/raus)
- 3-h-Stress-Test (Speicher, Disk, Stabilität)
- App-Filter-Modus mit Discord + Slack parallel

## Packaging & Distribution

### Entwicklung

- Python 3.11+, `uv` für Dependencies (lockfile)
- `pyproject.toml` mit zwei Extras: `[gpu]` (torch + cuda), `[cpu]` (torch-cpu)
- Entry-Point: `audiologger` startet `tray_app.main`

### Distribution

**Phase 1 (v1):** README mit `uv sync --extra gpu` bzw. `--extra cpu` + `audiologger` starten. Setzt Python voraus — ok für technisch versierte Empfänger.

**Phase 2 (später):** PyInstaller-Bundle (Onefile-EXE, ~1,5 GB mit CUDA). Nicht in Scope für v1.
