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

### TC-9: Capture warning is visible after stopping
1. Unplug/disable the default microphone (Sound settings → disable the input device).
2. Start a meeting recording, let it run ~10 s, stop.
3. **Expected:** A "Recording warning" toast appears immediately on stop, naming the
   failed channel (e.g. "Microphone not available." or "Microphone recording aborted.").
4. Click **Open details** → `capture_warnings.txt` opens with the same text.
5. Click **Open folder** (or the toast body) → the session folder opens.
6. **Expected:** `%APPDATA%/AudioLogger/tray.log` also contains a `Capture warning for <session>: ...` line.
7. Re-enable the mic, record again, stop.
8. **Expected:** No warning toast for the clean recording.

### TC-10: Screenshots during a meeting
Only AudioLogger's own toasts count here ("Screenshot N saved"), not the Snipping Tool's
notification from Windows.

1. Get ready: a window with visible content (for `Alt+Print`) and Excel with a few
   filled cells (Word with some text will do if Excel is not available).
2. Take a region snip with `Win+Shift+S` and copy nothing else, so the clipboard already
   holds an image, not text.
3. Start a meeting recording (`Ctrl+Alt+R`) and keep talking, or play a clip, so the
   transcript has speech around the screenshots.
4. **Expected:** No toast for the image that was already on the clipboard (step 18 checks
   that no PNG was saved for it either).
5. After ~10 s, take another region snip with `Win+Shift+S`.
6. **Expected:** A "Screenshot 1 saved" toast appears within a few seconds (it needs 1–2 s),
   naming the moment, e.g. "at 00:10".
7. After a few more seconds, press `Alt+Print` with the prepared window focused.
8. **Expected:** A "Screenshot 2 saved" toast.
9. In Excel, select a few cells and copy them (`Ctrl+C`); in Word, select some text and
   copy it instead if Excel is not available.
10. **Expected:** No toast and nothing saved: a copy that also carries text is content,
    not a screenshot.
11. Copy plain text, e.g. a line from Notepad.
12. **Expected:** Nothing.
13. **Optional (Notifications switch):** In the tray menu, choose Settings → Notifications →
    Disabled. Take a `Win+Shift+S` snip, then choose Settings → Notifications → Enabled.
14. **Expected (optional):** Switching off shows no confirmation (it is muted at once).
    The snip gives no toast but is still saved, as a third PNG. Switching on shows
    "Notifications set to Enabled." without "Restart may be required." If you did this
    step, N is 3 in the checks below, otherwise 2.
15. Stop the recording (`Ctrl+Alt+R`). Once the tray icon is no longer red, take another
    `Win+Shift+S` snip.
16. **Expected:** No toast and no extra PNG: nothing is saved after the recording has stopped.
17. Wait for the transcript.
18. **Expected:** `screenshots/` in the session folder holds exactly N PNGs, named after
    their offsets (`screenshot_HH-MM-SS.png`), and no `.part` files. Nothing was saved for
    the image from before the recording, the Excel or Word copy, the text copy or the snip
    after stopping.
19. **Expected:** `transcript.md` has `**Screenshots:** N` in the header and N `Screenshot`
    lines at the right places (matching the moments in the toasts), rendering as images in
    a Markdown preview. `transcript.json` lists the same N under `"screenshots"`.
20. Open the PNG from the `Alt+Print` capture (the transcript's Screenshot 2).
21. **Expected:** It is opaque and shows the window, not transparent or blank (regression
    check for the DIBV5 alpha bug). Quick check:
    `Image.open(path).convert("RGBA").getpixel((10, 10))[3]` is `255`.
22. Start a dictation (`Ctrl+Alt+D`), take a `Win+Shift+S` snip, stop.
23. **Expected:** No screenshot toast and no `screenshots/` folder for the dictation.
