# collective-chant
A website to facilitate a collective chant or prayer.
Under development.

## Installation
During development I am running the Collective Chant software on a local Linux box, inside a Python virtual environment. The software uses the Python FastAPI module to serve web pages over my local network to a browser running on my Windows system. I edited my Windows `hosts` file so that I can use the URL https://fastapi.local to open the main page. To use HTTPS, certificate files were created and placed in the project directory, but a port other than 443 could have been used for testing so that a certificate would not be needed.

### On the Linux system:
1. Install system dependencies: `sudo apt update && sudo apt install ffmpeg && sudo apt install rubberband-cli`
2. Install Python.
3. Download the project software: https://github.com/metamagical-musings/collective-chant/archive/refs/heads/main.zip
4. Unzip it then optionally move and/or rename the collective-chant-main directory as desired.
5. Inside this directory, create a virtual environment: `python3 -m venv .venv`
6. Activate the venv: `source .venv/bin/activate`
7. Install Python dependencies: `pip install -r requirements.txt`
8. Place a reference audio (named chant.mp3 by default) into the `reference` directory. (Use a short, single-stanza chant for testing.)
9. Place its associated transcription (named chant_stanzas.txt by default) into the same directory. (See below about stanzas.)
10. Start mix_worker.py: `python3 mix_worker.py test` (This should create `mixes/test.webm` using only the reference audio.)
11. If using HTTPS, run: `sudo setcap 'cap_net_bind_service=+ep' $(readlink -f .venv/bin/python3)` (to allow binding to port 443)
12. If using HTTPS, create a certificate using mkcert, for example, and place fastapi.crt, fastapi.key, fastapi.pfx in the project directory.
13. In a separate terminal, start the main program (optionally using the `screen` program to allow for disconnection): `python3 main.py`

### On the remote system:
1. Add a line like `xxx.xxx.xxx.xxx fastapi.local` to the `hosts` file, in admin/superuser mode.
2. Open the main Collective Chant page in a browser: https://fastapi.local

## Software

| Component | Purpose |
|-----------|---------|
| `index.html` | Client: uploads user submitted audios to `main.py` via websockets. |
| `main.py` | Server: serves `index.html` to clients, saves uploaded audios, signals `mix_worker.py` to begin a mix. |
| `mix_worker.py` | Audio processing worker: gathers uploaded audios and mixes them upon signal. |

### `mix_worker.py`
A standalone program that mixes `.webm` audio files found in the `uploads` directory.
The resulting mix is saved to the `mixes` directory.

The only command line argument is `mix_id` and it is optional. After the optional initial mix is done, the program
starts polling the `mixes` directory for signal files of the form `<mix_id>.mix`. If any are found, a mix is started.
The signal file is removed and the resulting mix is placed in the `mixes` directory with the name `<mix_id>.webm`,
along with a `<mix_id>.json` file that contains a dictionary of contributors to the mix. Uploaded audio files are then
removed. Polling resumes.

Automatic Speech Recognition (ASR) is used to ensure that uploaded audios are faithful recitations of the reference audio
at `ref_orig_path`. To avoid ASR degredation, audios are split into stanzas that should be less than 30 seconds in duration,
as determined by gaps of (near) silence between them. After splitting, stanzas are compressed/stretched
to match the reference audio, then transcribed using ASR and checked for accuracy relative to the reference text
in `ref_stanzas_path` (which uses blank lines to separate stanzas — the number of which must match the number found
by splitting the audio), then reassembled with a fixed gap of silence between them in preparation for the mix.

When `mix_worker.py` starts, the original reference audio `ref_orig_path` (which can be any type recognized by ffmpeg,
such as *.mp3 or *.webm) is split then reassembled with both fixed silence gaps between stanzas and with 3-tick count-ins
between them (to help chanters synchronize with it when recording audios for upload). The former is stored in
`ref_audio_path` and the latter is stored in `ref_ticks_path`, both for use by `main.py`.

ASR is done by the module `faster_whisper`, which automatically downloads from HuggingFace and caches a language model
when `mix_worker.py` is first run. (There will be an extra delay during the first run and a warning if you do not have
a HuggingFace account — just ignore it.) `mix_worker.py` can run as a service started manually or automatically at
bootup by registering `mix_worker.service` with the system. Example commands are included as comments inside the service
file itself. Making `mix_worker.py` a service ensures that the very large language model is only loaded into memory once
per session (and not per mix). Also, it can be run in the background at a lower priority than `main.py`.

For best ASR performance — and best mix results — correctly pronounce and clearly enunciate recorded words in a
quiet environment while avoiding touching your computer or other recording device while chanting.

There are many tunable constants in the `Const` object of `mix_worker.py`. Some pertaining to stanzas are:

| Constant | Suggested | Comment |
|----------|:---------:|---------|
|min_gap_ms | 1000 | Minimum (near) silence gap in uploaded audio to qualify as a stanza boundary.|
|min_speech_ms | 250 | Overlook nonsilent segments shorter than this when detecting silence gap.|
|silence_threshold_offset | 15 | dBFS offset below overall level for silence detection.|
|stanza_gap_ms | 1500 | Gap of silence to be inserted between stanzas when reassembled.|
|max_stanza_word_errors | 1 | If any single stanza exceeds this word error limit, the entire audio is rejected.|
|ratio_tolerance | 0.1 | Stanzas that must be stretched/compressed more than a factor of 1 +/- this value are silenced.|
|max_stanza_duration_errors | 1 | If more than this number of stanzas are silenced, the entire audio is rejected.|

### `main.py`
A Python server based on the FastAPI module. It serves an HTML/JS webpage to browser clients which allows users
to listen to a recorded chant and to upload their own recordings to the server to be mixed into a chorus.

`main.py` synchronizes with universal time. Chants are repeated every ´time_slot_duration_sec´ so their duration
must be shorter (by at least 5 seconds) to allow for short delays in browser audio timing and to give users chanting
along with the recording a chance to pause between rounds.

The first chant played is the reference chant. When `uploads_per_mix` user recordings have been successfully uploaded,
the reference chant is replaced by a mix. New mixes are created as additional recordings are uploaded.

The number of uploads per `time_slot_duration_sec` is limited to `max_uploads_per_time_slot`. This upload quota
is randomly distributed among connected users at the beginning of each time_slot by means of a random token consisting
of `words_per_token` words per token. The use of such tokens provides the following benefits:

- No user account (and no login) needed, ensuring anonymity.
- Ensures that `mix_worker.py` has enough time to mix uploaded recordings.
- Limits intentional upload flooding.

Tokens must be used within `token_expiration_sec` before they expire.

Uploaded recordings are screened before sending them to `mix_worker.py` as follows:

- Must be WebM audio.
- Must not exceed `max_upload_size_bytes` bytes.
- Duplicate recordings not allowed during a mix cycle.

Although `main.py` associates uploaded recordings with their corresponding IP addresses, both recordings and
IP addresses are erased after each mix completes to protect users' anonymity. `uploads_per_mix`
should be set large enough to provide addtional anonymity protection (10 - 20 recommended).

Recordings are also associated with corresponding upload tokens. When a mix is downloaded to connected client
browsers, a JSON file containing the tokens associated with the recordings contributed to the mix is also downloaded,
so that users will know that their contribution is included in the mix even if they cannot recognize their own voice
in the chorus. Although IP addresses are hidden from users, they are used to identify the countries associated with
contributions so that all users can know which countries contributed to the mix.

When a mix fails to complete within `mix_timeout_sec`, the mix is discarded and an error is logged.

Some tunable constants in the `Const` object of `main.py`.

| Constant | Suggested | Comment |
|----------|:---------:|---------|
|time_slot_duration_sec | 60 | Playback of the mix is repeated at this interval.|
|max_uploads_per_time_slot | 1.5 | Global upload quota of tokens per time_slot.|
|token_expiration_sec | 360 | Can use 6 * time_slot_duration_sec to keep proportion.|
|max_upload_size_bytes | 10 * 1024 * 1024 | Maximum audio recording size in bytes.|
|uploads_per_mix | 2 | Set large enough to provide anonymity in the mix chorus.|
|words_per_token | 3 | Number of BIP39 words per token.|
|mix_timeout_sec | 300 | Maximum time allowed for a mix to complete.|