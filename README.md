# collective-chant
A website to facilitate a collective chant or prayer.
Under development: main.py is temporarily out of sync with mix_worker.py

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
8. Place a reference audio (named ref_chant.webm by default) into the `reference` directory. (Use a short, single-stanza chant for testing.)
9. Place its associated transcription (named ref_stanzas.txt by default) into the same directory. (See below about stanzas.)
10. Start mix_worker.py: `python3 mix_worker.py test` (This should create `mixes/test.webm` using only the reference audio.)
11. If using HTTPS, run: `sudo setcap 'cap_net_bind_service=+ep' $(readlink -f .venv/bin/python3)` (to allow binding to port 443)
12. If using HTTPS, create a certificate using mkcert, for example, and place fastapi.crt, fastapi.key, fastapi.pfx in the project directory.
13. In a separate terminal, start the main program (optionally using the `screen` program to allow for disconnection): `python3 main.py`

### On the remote system:
1. Add a line like **xxx.xxx.xxx.xxx fastapi.local**  to the `hosts` file, in admin/superuser mode.
2. Open the main Collective Chant page in a browser on your remote system: https://fastapi.local

## `mix_worker.py`
A standalone program that mixes `.webm` audio files found in the `uploads` directory; result saved to the `mixes` directory.

The only command line argument is `mix_id` and it is optional. After the optional initial mix is done, the program
starts polling the `mixes` directory for signal files of the form `<mix_id>.mix`. If any are found, a mix is started.
The signal file is removed and the resulting mix is placed in the same directory with the name `<mix_id>.webm`,
along with a `<mix_id>.json` file that contains a dictionary of contributors to the mix. Uploaded audio files are then
removed. Polling resumes.

Automatic Speech Recognition (ASR) is used to ensure that uploaded audios are faithful recitations of a reference audio
at `ref_path`. To avoid ASR degredation, audios are split into stanzas that should be less than 30 seconds in duration,
as determined by gaps of (near) silence between them. After splitting, stanzas are compressed/stretched
to match the reference audio, then transcribed using ASR and checked for accuracy relative to the reference text
in `ref_stanzas_path` (which uses blank lines to separate stanzas -- the number of which must match the number found
by splitting the audio), then reassembled with a fixed gap of silence between them in preparation for the mix.
The reference audio itself is split then reassembled with 3-tick count-ins between them (to help chanters synchronize
with it when recording audio for upload), then stored in `ref_path_ticks` for use by `main.py`.

ASR is done by the module `faster_whisper`, which automatically downloads and caches a language model when `mix_worker.py`
is first run. (There will be an extra delay and a warning if you do not have a HuggingFace account -- just ignore it.)
`mix_worker.py` can run as a service started manually or automatically at bootup by registering `mix_worker.service`
with the system. Example commands are included as comments inside the service file itself. Making `mix_worker.py`
a service ensures that the very large language model is only loaded into memory once per session (and not per mix).

For best ASR performance, and best mix results, correctly pronounce and clearly enunciate recorded words in a quiet
environment while avoiding touching your computer or other recording device while chanting.

There are many tunable constants in `mix_worker.py`. Some that are more likely to be tuned are:

| Constant | Suggested | Comment |
|----------|-----------|---------|
|min_gap_ms | 1000 | Minimum (near) silence gap in uploaded audio to qualify as a stanza boundary.|
|min_speech_ms | 250 | Overlook nonsilent segments shorter than this when detecting silence gap.|
|silence_threshold_offset | 15 | dBFS offset below overall level for silence detect.|
|stanza_gap_ms | 1500 | Gap of silence to be inserted between stanzas when reassembled.|
|max_stanza_word_errors | 1 | If any single stanza exceeds this word error limit, the entire audio is rejected.|
|ratio_tolerance | 0.1 | Stanzas that must be stretched/compressed more than a factor of 1 +/- this value are silenced.|
|max_stanza_duration_errors | 1 | If more than this number of stanzas are silenced, the entire audio is rejected.|
|num_ref_audio_in_mix | 1 | Number of times to include the reference audio in the mix for stability.|
