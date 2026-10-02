# Screenshots in Meeting Transcripts — Design

**Date:** 2026-10-02
**Status:** Implemented; hardware acceptance pending
**Builds on:** [2026-05-18-audiologger-design.md](2026-05-18-audiologger-design.md)

## Goal

Screenshots taken during a meeting recording are saved next to the recording and
appear in the transcript at the moment they were taken. On 2026-09-18 the user
did this by hand — three Win+Shift+S snips pasted into Paint and saved into the
session folder — and found it extremely helpful. This makes it automatic.

## Approach

Watch the clipboard instead of taking screenshots ourselves. The user already
captures with Win+Shift+S (region, window or full screen via the Snipping Tool)
and Alt+Print (active window); both put an image on the clipboard. AudioLogger
saves every screenshot that appears there during a meeting recording: an image
that does not come with text (see Capture).

Rejected alternatives:
- *Own capture hotkeys* — fixed framing where the user deliberately framed
  regions (their manual snips were 1204–1460 px wide, never a full monitor),
  new shortcuts to learn, DPI pitfalls on a scaled second monitor, and the
  "active window" is not always the one being looked at.
- *Both* — more surface for a case Alt+Print already covers.

## Non-Goals

- Own screenshot hotkeys
- Captions or annotations on screenshots
- OCR or automatic image description
- Screenshots during dictation or "Append Note" recordings
- De-duplicating identical images
- Downscaling or recompressing images beyond PNG

## Behaviour

### Capture

- Active only during **meeting** recordings. Dictation and "Append Note" are
  voice notes; "Append Note" also extends an older session, which would
  complicate offsets.
- A watcher thread starts right after audio capture starts and stops when the
  recording stops. When it is told to stop, the thread makes one final poll
  before it ends, so a snip taken just before stopping is kept.
- Every 0.25 s it reads `GetClipboardSequenceNumber()`, a counter Windows
  increments on every clipboard change. Reading it changes nothing and needs no
  clipboard lock. Only when the counter moved does the watcher read the
  clipboard.
- The counter value at start is taken as already handled: whatever was on the
  clipboard before the recording is ignored.
- The clipboard is only ever **read**, never written. Dictation writes its
  result to the clipboard, and the user's clipboard must stay theirs.
- If the new content is an image, it is saved as PNG. The clipboard is read as
  PNG if offered, else as DIB, else as DIBV5. Text, file lists and anything else
  are ignored and count as handled.
- A copy that also offers text — Excel cells, a Word selection — is skipped as
  well. Office adds a picture of such copies, but they are content, not
  screenshots; screenshots come without text.
- A change is marked handled only after a successful read. If reading fails —
  typically another program holding the clipboard open, or an image that cannot
  be decoded — the same change is retried on the next poll. After 20 consecutive
  failures (5 s) for one change it is given up and logged. It is also re-read if
  the counter moved during the read: writers add formats one by one, and reading
  mid-way would save the same copy twice. The change's moment then restarts at
  the next poll.
- After a save, the tray shows a toast: *"Screenshot N saved"* / *"at 12:31"*.
  It follows the Notifications setting, which applies immediately. N counts
  screenshots saved in this recording. The toast goes through powershell.exe and
  appears 1–2 s after the save.

### Storage

- `<session>/screenshots/`, created on the first save attempt.
- Each PNG is written as `<name>.png.part` and then renamed with
  `os.replace`, so a half-written file never carries a final name. A crash
  mid-save leaves only a `.part`, which `find_screenshots` ignores.
- A watcher is single-use: a second `start()` raises `RuntimeError`. Each
  recording gets its own watcher, with its own start time and screenshot count.
- File name carries the offset from the start of the recording, in whole
  seconds: `screenshot_HH-MM-SS.png`. The offset runs from the watcher's
  `start()`, which happens right after audio capture starts, to the moment the
  change was first noticed — not when a busy clipboard finally let it be read.
  A second screenshot in the same second gets `screenshot_HH-MM-SS_2.png`, then
  `_3`, and so on.
- The name is the only record of the timestamp. That survives a crash and lets
  re-transcription find screenshots without a separate list. Deleting a file by
  hand removes it from the next transcript.

### Transcript

- One line per screenshot, sorted by time among the speech lines:

  ```
  **[00:12:28] Speaker 1:** Hier seht ihr die neue Rezeptstruktur.
  **[00:12:31] Screenshot 3:** ![Screenshot 3](screenshots/screenshot_00-12-31.png)
  **[00:12:35] Me:** Okay, und wo kommt der Teig rein?
  ```

- Sorting is by start time in whole seconds, as displayed (`int(seg.start)`). On
  a tie the speech line comes first: speech starting anywhere within the
  screenshot's second sits above it. A speech line that starts before a
  screenshot and runs past it stays above it.
- Screenshots are numbered 1..N in time order when the transcript is rendered.
- The path is relative to the session folder (forward slashes), so moving the
  folder keeps the links intact. Markdown viewers (VS Code preview, Obsidian,
  GitHub) show the image inline.
- Header gains `**Screenshots:** N` after the `**Model:**` line, only when N > 0.
- `transcript.json` gains
  `"screenshots": [{"at_s": 751, "path": "screenshots/screenshot_00-12-31.png"}]`
  (an empty list without screenshots).
- **Without screenshots, `transcript.md` is byte-for-byte what it is today.**

## Components

### `screenshot_watch.py` (new)

- `screenshot_file_name(offset_s: float, taken: set[str]) -> str` and
  `find_screenshots(session_dir: Path) -> list[Screenshot]` — the naming scheme
  lives here and nowhere else. `find_screenshots` parses
  `^screenshot_(\d{2})-(\d{2})-(\d{2})(?:_(\d+))?\.png$`, ignores other files,
  returns entries sorted by (offset, suffix).
- `Screenshot` — frozen dataclass: `at_s: int`, `path: str` (relative, forward
  slashes).
- `WindowsClipboard` — reads the clipboard through the Win32 API with ctypes:
  `GetClipboardSequenceNumber`, a format check with `IsClipboardFormatAvailable`
  (so a copy without an image never opens the clipboard), then `OpenClipboard` /
  `GetClipboardData`, decoding with Pillow's PNG and DIB plugins; the image is
  decoded right there, so a bad one is a failed read. Not
  `PIL.ImageGrab.grabclipboard()`: on a busy clipboard it sleeps 500 ms holding
  the GIL, which stalls the soundcard loopback threads and drops audio.
  `choose_image_format` picks PNG, then DIB, then DIBV5 (Pillow decodes DIBV5's
  usually all-zero alpha as fully transparent), and nothing when text is
  present.
- `ClipboardScreenshotWatcher(session_dir, *, on_saved=None,
  clock=time.monotonic, poll_s=0.25, clipboard=None)` with `start()`, `stop()`
  and `poll_once()`. The thread is `poll_once()` in a loop, plus one last
  `poll_once()` after the loop ends. `stop()` only sets the stop flag and joins
  for at most 2 s, so a clipboard owner that hangs cannot hold up the hotkey
  thread. A watcher that was never started does nothing in `stop()`. `on_saved`,
  which receives `(index, offset_s, path)`, runs on the watcher thread. Every
  exception inside a poll is caught and logged; the watcher never ends because
  of one, and never touches audio.

### Changed

- **`controller.py`** — optional constructor argument
  `screenshot_watcher_factory: Callable[[Path], watcher] | None`. In meeting mode
  the controller creates and starts the watcher right after `capture.start()`
  and stops it at the beginning of `_stop()`, before `capture.stop()`. A watcher
  that cannot be created or started is logged and the recording continues
  without it; a `stop()` that raises is logged too.
- **`tray_app.py`** — passes a factory whose `on_saved` shows the toast. The
  Notifications switch now applies immediately (it updates `notifier.enabled`
  instead of waiting for a restart), so the screenshot toasts follow it; its
  confirmation no longer says "Restart may be required."
- **`transcript_merger.py`** — `render_markdown(..., screenshots=())`; compares
  whole seconds, as displayed.
- **`transcribe_worker.py`** — `_process_meeting_session` calls
  `find_screenshots` inside a try/except and passes the result to
  `render_markdown` and into `transcript.json`; a failing lookup is logged and
  the transcript is written without screenshots. The dictation path is
  unchanged.

## Error Handling

| Situation | Behaviour |
|---|---|
| Clipboard locked by another program | Change stays pending, retried next poll; given up after 20 failures with a log entry |
| Image cannot be decoded | A failed read like the one above: retried, given up after 20 |
| Clipboard holds text or files | Ignored, marked handled |
| Copy that also offers text (Excel cells, Word selection) | Skipped, marked handled |
| Saving the PNG fails (disk full, permissions) | Logged, `.part` file removed, screenshot skipped, watcher continues |
| Crash mid-save | Only a `.part` file is left; `find_screenshots` ignores it |
| Any other error in a poll | Logged, watcher continues |
| Watcher cannot be created or started | Logged, recording continues without screenshots |
| Screenshot lookup fails in the worker | Logged, transcript written without screenshots |
| Crash mid-recording | PNGs are already on disk; recovery transcribes them because the worker finds them by name |
| Watcher thread does not stop in time | `stop()` joins with a 2 s timeout and returns; the daemon thread finishes its poll and the final one on its own |

Audio capture never depends on the watcher.

## Testing

`poll_once()` makes the watcher testable without threads or waiting, using a
fake clipboard and a controllable clock. Tests are written first and must fail
first.

- **Watcher:** pre-start content ignored · new image saved under the right
  offset name · text and file lists ignored · locked clipboard retried, not
  lost · gives up after 20 failures, counted per change · same-second suffix
  `_2` · save error does not stop it and leaves no file behind · a copy still
  being written is read again, not saved twice · the moment is when the copy
  appeared · a failing callback does not stop it · `stop()` ends the thread and
  catches a snip taken just before it · single-use
- **Clipboard format choice:** PNG preferred · Alt+Print bitmap read as plain DIB
  · a copy that also offers text is not a screenshot · no image format, nothing
  to save
- **`find_screenshots`:** parses offsets and suffixes, ignores other files,
  sorts (suffixes as numbers)
- **Transcript:** screenshot lines interleaved by time, tie order in whole
  seconds, numbering, header line · **identical output without screenshots**
- **Worker:** screenshots found on disk end up in `transcript.md` and
  `transcript.json` · a failing lookup never costs the transcript
- **Controller:** meeting mode starts and stops the watcher; dictation and
  "Append Note" do not · a watcher that cannot be created or started does not
  stop the recording
- **Tray:** toast text on save, with hours in long recordings · the controller
  is wired to the toast · the Notifications switch silences toasts at once

## Acceptance on Real Hardware

1. A short real recording during which a test image is put on the real
   clipboard programmatically; the PNG and the transcript line must appear.
   This overwrites the user's clipboard — announced before running.
2. The user takes one real Win+Shift+S snip during a recording. The Snipping
   Tool path cannot be triggered programmatically, so this is the proof.
3. TC-10 in the manual test plan walks the branches only real clipboard traffic
   reaches: an image already on the clipboard at the start, Alt+Print, an Excel
   copy, a snip after stopping.
