# Transcript Railway Comparator

Copyright (C) 2026 Svetlin Tassev. Licensed under the GNU General Public License v3.0; see [`LICENSE`](LICENSE).

`transcript_railway_standalone.html` is a self-contained, browser-based tool for comparing and editing transcripts from multiple speech-recognition runs. It is separate from the command-line transcription toolkit: it does not execute models and does not need Python, a server, model weights, or external JavaScript/CSS files.

## Open it

Open `transcript_railway_standalone.html` in a modern desktop browser. For easiest folder selection, use a browser that supports selecting a directory from an HTML file input. No installation or build step is required.

## Load transcripts

Use **Add model folder** to select one or more model-run folders, or load transcript files directly. The importer handles common TXT, SRT, VTT, and supported timed-JSON formats. For model output folders it normally prefers a `primary_transcript` declared in `run.json`, otherwise `transcript.txt`. Matching subtitle and timing sidecars can add caption and word-seek timing; raw or chunk sidecars are not treated as independent canonical transcripts.

A saved session JSON can restore the railway, branch choices, and edits. Older sessions are upgraded using the current equivalence/auto-collapse rules when loaded. Load the associated audio separately if you want playback.

## Compare and edit

Common wording is shown on the main track; alternative wording appears in branches. You can choose branch wording, collapse recognized equivalent forms, and edit the main track. The statistics view compares normalized words in order and reports model overlap, word counts that do not align, and credit for explicitly selected, unedited human branch choices. The score is a comparison of the loaded sources and edited final text, not an objective measure of transcription accuracy.

## Local-file handling

Files selected in the browser are parsed locally. The page contains its own CSS and JavaScript and does not upload transcript or audio content to a server. A browser (notably Firefox) may label folder selection as an “upload” confirmation; this is the native file-picker permission allowing the page to read the files you selected. It is not itself a network transfer. The app does not update an imported JSON file in place: use the export controls to create a new file when you choose to save changes.

Use **Export** to download a new session or transcript. Treat exported files as sensitive if the original transcripts contain private information.

## License and development note

This standalone comparator is distributed under GPL-3.0; see [`LICENSE`](LICENSE).

**Development note:** AI tools were used in creating and editing parts of this codebase. The published source is licensed under GPL-3.0.
