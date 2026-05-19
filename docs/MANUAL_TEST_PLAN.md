# Manual Test Plan

Automated tests cover state machines and pure functions. The following scenarios require a real machine, GPU, microphone, and speakers — run before any release.

## Setup

- Windows 11 with NVIDIA GPU
- AudioLogger installed per README (GPU build)
- HuggingFace token configured
- WhisperX model already downloaded (run one transcription first)

## Test Cases

### TC-1: Full cycle with DE+EN mixed audio
1. Start AudioLogger.
2. Press hotkey to record.
3. Speak ~30 s mixing German and English ("Hallo zusammen, today we discuss the roadmap, also nochmal: was war der nächste Punkt?").
4. Play a short YouTube clip in English in the background.
5. Press hotkey to stop.
6. Wait for transcription.
7. **Expected:** `transcript.md` contains both DE and EN text correctly; mic audio labeled "Me"; video audio labeled "Speaker 1" (and possibly more if multiple speakers).

### TC-2: Hotkey works across foreground apps
1. Start AudioLogger.
2. Open Discord in full-screen voice call.
3. Press hotkey — expect "Recording started" toast.
4. Switch to Slack call.
5. Press hotkey — expect "Recording stopped" toast.
6. **Expected:** Hotkey triggers regardless of focused app.

### TC-3: Default device change mid-recording
1. Start recording with built-in mic + speakers as default.
2. After ~10 s, connect Bluetooth headset (set as default automatically).
3. Continue recording another 10 s.
4. Stop.
5. **Expected:** Toast warning about device change; recording continues on original device; transcript is coherent for the original-device portion.

### TC-4: 3-hour stress test
1. Start a 3-hour recording.
2. Periodically check task manager: RAM should be stable (<500 MB tray + capture).
3. Check disk usage grows roughly linearly (~660 MB/hr/stream).
4. **Expected:** No crash, no out-of-disk, transcription completes within reasonable time (<30 min on RTX 4090 with large-v3).

### TC-5: App-filter mode (Discord + Slack)
1. Set `audio_source: apps` and `filtered_app_names: ["Discord.exe", "Slack.exe"]` in config.
2. Restart AudioLogger.
3. Play audio in Discord and Chrome simultaneously.
4. Record 15 s, stop.
5. **Expected:** `system.wav` contains only Discord audio (Chrome filtered out). If unsupported on the OS, toast warns and full-system loopback is used.

### TC-6: Crash recovery
1. Start recording.
2. After 10 s, force-kill the AudioLogger process (Task Manager).
3. Restart `audiologger`.
4. **Expected:** Tray re-appears; the partial session in `recordings/` is silently mixed and enqueued for transcription. After processing, transcript.md exists.

### TC-7: Worker reuse warm window
1. Start recording, stop after 5 s.
2. Wait for transcription to begin (icon yellow).
3. After it finishes (icon grey), within 30 s, start a new recording and stop.
4. **Expected:** Second transcription starts without re-loading the model (much faster). Check `%APPDATA%/AudioLogger/worker_state/worker.log` for single "Loading WhisperX model" line.

### TC-8: Config hand-edit and reload
1. Quit AudioLogger.
2. Edit `config.yaml`: change `hotkey` to `f8`.
3. Start AudioLogger.
4. Press F8 from a foreground app.
5. **Expected:** Recording starts.
