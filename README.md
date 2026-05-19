# AudioLogger

Windows tray utility for recording meetings (Slack, Discord, Teams, Zoom, ...) and producing local Markdown transcripts using WhisperX large-v3 with pyannote speaker diarization.

## What it does

- **Three global hotkeys** (configurable):
  - `Ctrl+Alt+R` — Meeting: mic + system audio, Markdown transcript with speaker diarization.
  - `Ctrl+Alt+D` — Dictation: mic-only, plain text, copied to clipboard for instant paste.
  - `Ctrl+Alt+E` — Extend: appends a new chunk (audio + text) to the most recent dictation session.
- For meetings, both streams are transcribed independently:
  - Mic audio → labeled "Me".
  - System audio → diarized into "Speaker 1", "Speaker 2", ...
- Merged chronological Markdown transcript saved next to the audio.
- Multilingual model (DE / EN / mixed handled out of the box).

## Requirements

- Windows 10 (Build 19044 / 21H2 or newer recommended) or Windows 11.
- Python 3.11+.
- For GPU acceleration: NVIDIA GPU with CUDA 11.8 or 12.x.
- A free [HuggingFace token](https://huggingface.co/settings/tokens) for pyannote diarization. Accept the model terms at:
  - https://huggingface.co/pyannote/speaker-diarization-community-1 (used by whisperx ≥ 3.8)
  - https://huggingface.co/pyannote/speaker-diarization-3.1 (older whisperx; harmless to accept both)

## Install

Install `uv`: <https://docs.astral.sh/uv/>

Clone and install (GPU build):
```bash
git clone https://github.com/cwolfertstetter/AudioLogger.git
cd AudioLogger
uv venv
uv pip install -e ".[gpu,dev]"
# Install torch with CUDA support (adjust cu121 to match your CUDA version):
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only (slow but works):
```bash
uv pip install -e ".[cpu,dev]"
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Configure

First launch writes `%APPDATA%/AudioLogger/config.yaml`. Edit it directly, or use the tray menu — most settings have a "Settings" submenu for in-place changes, and "Open Config File..." opens the file in your default editor.

| Key                     | Default                | Notes                                                   |
|-------------------------|------------------------|---------------------------------------------------------|
| `hotkey`                | `ctrl+alt+r`           | Meeting hotkey. `keyboard`-lib syntax, e.g. `f8`, `ctrl+shift+space` |
| `dictation_hotkey`      | `ctrl+alt+d`           | Dictation hotkey (mic-only, plain text, clipboard)      |
| `extend_hotkey`         | `ctrl+alt+e`           | Append a new chunk to the most recent dictation session |
| `output_dir`            | `./recordings`         | Where session folders are written                       |
| `whisper_model`         | `large-v3`             | Meeting model. `tiny` / `base` / `small` / `medium` / `large-v3` |
| `dictation_model`       | `medium`               | Dictation model. Same options; smaller = faster, slightly less accurate |
| `device`                | `cuda`                 | `cuda` or `cpu`                                         |
| `compute_type`          | `float16`              | GPU: `float16`. CPU: use `int8`                         |
| `diarization_enabled`   | `true`                 | Requires `huggingface_token`                            |
| `huggingface_token`     | `null`                 | Paste your HF token here for diarization                |
| `audio_source`          | `all`                  | `all` (system loopback) or `apps` (per-app filter)      |
| `filtered_app_names`    | `[]`                   | e.g. `["Discord.exe", "Slack.exe"]` when `audio_source: apps` |
| `notification_enabled`  | `true`                 | Windows toast notifications                             |
| `worker_prewarm`        | `true`                 | Load transcription models when tray starts (~3-5 GB VRAM, ~15 s startup). Set `false` to lazy-load. |
| `worker_warm_seconds`   | `600`                  | Seconds the worker stays alive idle between jobs. Lower = less VRAM held, higher = faster repeat transcriptions. Set very large for "always warm". |

## Run

From a terminal:
```bash
uv run audiologger
```

Or, **without a terminal** (Explorer / pinned shortcut / autostart): double-click
`scripts/start-audiologger.vbs`. The launcher resolves paths relative to itself
and starts the tray silently — no CMD window appears.

A tray icon appears (grey = idle, red = recording, yellow = transcribing).
Right-click for menu: start/stop each mode, settings submenu (model/device/diarization/notifications/pre-warm), output folder, audio source, open config file, re-transcribe last recording, quit.

**Three recording modes:**
- `Ctrl+Alt+R` — Meeting: mic + system audio, Markdown transcript with speaker diarization.
- `Ctrl+Alt+D` — Dictation: mic-only, plain text, copied to clipboard for instant paste.
- `Ctrl+Alt+E` — Extend: appends a new chunk to the most recent dictation session (audio + text). If no previous dictation exists, behaves like Ctrl+Alt+D.

### Autostart on login

1. Press `Win + R`, type `shell:startup`, hit Enter — Windows opens the per-user
   startup folder.
2. Drag `scripts/start-audiologger.vbs` in there (hold Alt while dragging to
   create a shortcut instead of moving the file).
3. Sign out + back in to verify the tray icon appears automatically.

### Pin to taskbar

Right-click `scripts/start-audiologger.vbs` → **Create shortcut** → drag the
shortcut onto your taskbar. (Windows refuses to pin `.vbs` directly, but it
accepts a `.lnk` that points at one.)

## Output layout

```
recordings/
  2026-05-18_14-32-15/
    mic.wav           ← your microphone
    system.wav        ← everything from system audio
    mixed.wav         ← sum for easy playback
    transcript.md     ← final result
    transcript.json   ← raw WhisperX output (for re-processing)
    job.log           ← present only if transcription errored
```

## Troubleshooting

- **"Hotkey conflict" toast:** Edit the relevant `*_hotkey` field in config.yaml, restart.
- **Diarization disabled warning in transcript:** Set `huggingface_token` in config and accept model terms at <https://huggingface.co/pyannote/speaker-diarization-community-1> (and <https://huggingface.co/pyannote/speaker-diarization-3.1> for older whisperx versions). Worker logs a `GatedRepoError` when the right model hasn't been accepted yet.
- **First-run transcription hangs for several minutes:** WhisperX is downloading the ~3 GB model. Subsequent runs use the cache in `%USERPROFILE%/.cache`.
- **"App-Filter nicht verfügbar" toast:** Per-app loopback needs Windows 10 21H2+ and a `pyaudiowpatch` build that exposes `PaWasapiStreamInfo`. The app silently falls back to full-system loopback for that recording.
- **App crashed mid-recording:** Restart `audiologger` — orphan sessions are auto-detected and queued for transcription.

## Development

```bash
uv run pytest                       # unit tests
uv run pytest tests/test_config.py  # one file
uv run audiologger                  # run the app
```

Manual test checklist: `docs/MANUAL_TEST_PLAN.md`.

## License

MIT — see `LICENSE`.
