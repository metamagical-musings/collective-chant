#!/usr/bin/env python3

"""
mix_worker.py - standalone program to mix audio files found in uploads/*.webm and save the result to the mixes directory.
                The only command line argument is the mix number.

To improve effectiveness, audio blobs are split into stanzas (that should be less than 30 seconds in duration, to avoid ASR degredation)
according to >1 second gaps of silence between them. After splitting, stanzas are shifted and compressed/stretched to match a
reference audio in static/ref_path.

Automatic Speech Recognition is done by faster_whisper, which downloads and caches a model when first run. Recognized text
must match the reference text in stanzas_file within max_stanza_errors per stanza. Stanzas in stanzas_file are separated by blank lines.

Currently works only on Linux. Must install: ffmpeg and pyrubberband-cli
"""
import json, os, re, shutil, sys, time, subprocess, math
import numpy as np, pyrubberband as pyrb, soundfile as sf
from datetime import datetime
from pydub import AudioSegment, silence
from difflib import SequenceMatcher
from pathlib import Path
from faster_whisper import WhisperModel
from io import BytesIO

class Const:
    error_level = 3 # 0==FATAL, 1==COURSE, 2==FINE, 3==TIMING
    uploads_dir = "uploads"
    mixes_dir = "mixes"
    tmp_dir = "tmp"
    ref_path = "static/ref_chant.webm"
    stanzas_file = "static/stanzas.txt"
    stanza_tmp_dir = "tmp/chant_stanzas"
    stanza_tmp_cleanup = False
    # prepare_audio()
    ffmpeg_timeout = 60.0
    target_sample_rate = 16000
    target_channels = 1
    loudnorm_filter = "loudnorm=I=-16:TP=-3:LRA=11"
    # compute_ratio()
    ratio_tolerance = 0.10
    # normalize_to_target()
    headroom_db = 6.0
    target_db = -14.0
    bitrate = "24k" # bps
    # split_into_stanzas()
    gap_threshold_ms = 1000      # minimum gap to qualify as a stanza boundary
    min_speech_ms = 250          # discard nonsilent segments shorter than this
    silence_thresh_offset = 15   # dBFS offset below overall level for silence detect
    seek_step_ms = 10            # resolution of silence detection
    # Model.load_model()
    device = "cpu"
    compute_type = "int8"
    model_name = "base.en"
    # score_stanzas()
    beam_size = 5
    temperature = 0.0
    word_timestamps = False
    condition_on_previous_text = False
    compression_ratio_threshold = 2.0
    log_prob_threshold = -1.0
    no_speech_threshold = 0.6
    vad_filter = True
    max_stanza_errors = 1

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.stages = {}

    def tick(self, label):
        now = time.perf_counter()
        elapsed = now - self.last
        self.stages[label] = self.stages.get(label, 0.0) + elapsed
        self.last = now
        total = now - self.t0
        log(f"  [t={total:6.2f}s | step {elapsed:5.2f}s] {label}", 3)

    def report(self):
        log("Timing summary:", 3)
        for label, secs in self.stages.items(): log(f"    {secs:7.2f}s  {label}", 3)
        log(f"    {time.perf_counter() - self.t0:7.2f}s  TOTAL", 3)

class Model:
    model = None
    @classmethod
    def load_model(cls):
        if cls.model is None: cls.model = WhisperModel(Const.model_name, device=Const.device, compute_type=Const.compute_type)

class Stanzas:
    words = []
    num_stanzas = 0
    @classmethod
    def load_stanzas(cls):
        with open(Path(Const.stanzas_file), 'r', encoding='utf-8') as f:
            current_block = []
            for line in f:
                line = line.rstrip()
                if line: current_block.append(line)
                elif current_block:
                    cls.words.append(' '.join(current_block))
                    current_block = []
            if current_block: cls.words.append(' '.join(current_block))
        cls.num_stanzas = len(cls.words)

class StanzaChunk:
    def __init__(self, index, start_ms, end_ms, audio):
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.audio = audio
        
    def duration_ms(self):
        return self.end_ms - self.start_ms

    def score_chunk(self, tmp_dir):
        try:
            wav_file = str(tmp_dir / f"stanza_{self.index}.wav")
            self.audio.export(wav_file, format="wav")
            segments, info = Model.model.transcribe(
                wav_file,
                language="en",
                beam_size=Const.beam_size,
                temperature=Const.temperature,
                initial_prompt=Stanzas.words[self.index],
                word_timestamps=Const.word_timestamps,
                condition_on_previous_text=Const.condition_on_previous_text,
                compression_ratio_threshold=Const.compression_ratio_threshold,
                log_prob_threshold=Const.log_prob_threshold,
                no_speech_threshold=Const.no_speech_threshold,
                vad_filter=Const.vad_filter
            )
            parts = []
            for seg in segments: parts.append(seg.text.strip())
            language = (info.language or "").lower()
            transcript = " ".join(parts).strip()
            wed = self.word_edit_distance(Stanzas.words[self.index], transcript)
            stanza_results = {
                "index": self.index,
                "transcript": transcript,
                "language": language,
                "word_edit_distance": wed
            }
            results_file = str(tmp_dir / f"stanza_{self.index}.json")
            with open(results_file, "w") as f: json.dump(stanza_results, f, indent=4)
        except Exception as exc:
            log(f"  REJECT: Scoring aborted for chunk {self.index} ({exc})", 1)
            return False
        if wed > Const.max_stanza_errors:
            log(f"  REJECT: {wed} stanza word errors > {Const.max_stanza_errors}", 1)
            return False
        return True

    def word_edit_distance(self, reference, hypothesis):
        ref = self.normalize_words(reference)
        hyp = self.normalize_words(hypothesis)
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)
        for i in range(len(ref) + 1): d[i, 0] = i  # deletions
        for j in range(len(hyp) + 1): d[0, j] = j  # insertions
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i-1] == hyp[j-1]: d[i, j] = d[i-1, j-1]
                else:
                    sub = d[i-1, j-1] + 1
                    ins = d[i,   j-1] + 1
                    dele = d[i-1, j]   + 1
                    d[i, j] = min(sub, ins, dele)
        return int(d[-1, -1])

    def normalize_words(self, text):
        """Lowercase, strip punctuation, split into bare word tokens."""
        cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower())
        return cleaned.split()

def parse_filename(fname):
    """Return (md5, token, ip) or None if the name doesn't match."""
    m = re.compile(r"^([^.]+)\.([^.]+)\.(.+)\.webm$").match(fname)
    if not m:
        log(f"  SKIP (bad filename pattern): {fname}", 1)
        return None
    return m.groups()

def prepare_audio(path):
    try:
        dst_path = Path(Const.tmp_dir) / Path(path.name + "_asr.wav")
        cmd = ["ffmpeg", "-y", "-i", str(path), "-af", Const.loudnorm_filter, "-ar", str(Const.target_sample_rate), "-ac", str(Const.target_channels), str(dst_path)]
        try: proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=Const.ffmpeg_timeout)
        except FileNotFoundError as exc: raise RuntimeError("ffmpeg executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc: raise RuntimeError(f"ffmpeg timed out after {Const.ffmpeg_timeout}s") from exc
        if proc.returncode != 0 or not dst_path.exists():
            tail = proc.stderr.decode(errors="replace")[-400:] if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}) on {path}: {tail}")
        audio = AudioSegment.from_wav(str(dst_path))
        try: dst_path.unlink()
        except OSError as exc: raise RuntimeError("Unable to delete temporary wav file {dst_path}") from exc
    except RuntimeError as exc:
        log(f"  REJECT: {path.name}: {exc}", 1)
        return None
    return audio.set_frame_rate(Const.target_sample_rate).set_channels(1).set_sample_width(2)

def create_silence(duration_ms):
    return AudioSegment.silent(duration=duration_ms, frame_rate=Const.target_sample_rate)

def compute_ratio(duration_ms, ref_duration_ms):
    if duration_ms == 0 or ref_duration_ms == 0: return False
    ratio = duration_ms / ref_duration_ms
    if abs(1 - ratio) > Const.ratio_tolerance:
        log(f"  SKIP: {ratio:.2f} duration ratio exceeds allowed tolerance {Const.ratio_tolerance}", 1)
        return False
    return ratio

def normalize_to_target(audio):
    if audio.max_dBFS <= -180: return audio # digital silence
    return audio.apply_gain(min(Const.target_db - audio.dBFS, -Const.headroom_db - audio.max_dBFS))

def split_into_stanzas(audio):
    silence_thresh = audio.dBFS - Const.silence_thresh_offset
    nonsilent = silence.detect_nonsilent(audio, min_silence_len=Const.seek_step_ms, silence_thresh=silence_thresh, seek_step=Const.seek_step_ms)
    # Filter out short noise artifacts within silence regions.
    real_speech = [(s, e) for s, e in nonsilent if (e - s) >= Const.min_speech_ms]
    if not real_speech:
        log(f"  REJECT: stanzas split failed (No speech detected)", 1)
        return None
    # Group speech regions into stanzas: a new stanza starts when the gap before it exceeds Const.gap_threshold_ms.
    stanzas = [[real_speech[0]]]
    for i in range(1, len(real_speech)):
        prev_end = real_speech[i - 1][1]
        curr_start = real_speech[i][0]
        gap = curr_start - prev_end
        if gap >= Const.gap_threshold_ms: stanzas.append([real_speech[i]])
        else: stanzas[-1].append(real_speech[i])
    chunks = []
    for idx, regions in enumerate(stanzas):
        start_ms = regions[0][0]
        end_ms = regions[-1][1]
        blob = audio[start_ms:end_ms]
        chunks.append(StanzaChunk(index=idx, start_ms=start_ms, end_ms=end_ms, audio=blob))
    return chunks

def score_stanzas(mix_id, chunks):
    if len(chunks) != Stanzas.num_stanzas:
        log(f"  REJECT: detected {len(chunks)} stanzas, expected {Stanzas.num_stanzas}", 1)
        return False
    tmp_dir = Path(Const.stanza_tmp_dir + f"_{mix_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for chunk in chunks:
        if not chunk.score_chunk(tmp_dir):
            passed = False
            break
    if Const.stanza_tmp_cleanup: shutil.rmtree(tmp_dir, ignore_errors=True)
    return passed

def match_duration_preserve_pitch(audio, reference):
    ref_ms = len(reference)
    seg_ms = len(audio)
    ratio = compute_ratio(seg_ms, ref_ms)
    if ratio is False: return create_silence(ref_ms)
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2: samples = samples.reshape((-1, 2))
    samples = samples.astype(np.float32) / (2 ** (8 * audio.sample_width - 1))
    stretched = pyrb.time_stretch(samples, audio.frame_rate, ratio)
    buf = BytesIO()
    sf.write(buf, stretched, audio.frame_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return AudioSegment.from_file(buf, format="wav")

def align_stanzas(up_chunks, ref_chunks):
    if len(up_chunks) != len(ref_chunks):
        log(f"  REJECT: Stanzas mismatch (upload {len(up_chunks)} vs reference {len(ref_chunks)})", 1)
        return None
    total_ms = ref_chunks[-1].end_ms
    canvas = create_silence(total_ms)
    for upload, ref in zip(up_chunks, ref_chunks):
        seg = match_duration_preserve_pitch(upload.audio, ref.audio)
        if ref.start_ms + len(seg) > total_ms: seg = seg[: max(total_ms - ref.start_ms, 0)] # Clamp to avoid overflow.
        canvas = canvas.overlay(seg, position=max(ref.start_ms, 0))
    return canvas

def equal_gain_mix(segments):
    segments = [t.set_frame_rate(Const.target_sample_rate).set_channels(2).set_sample_width(2) for t in segments]
    attenuation_db = 20.0 * math.log10(len(segments)) # safe starting value for coherent case; 16–18 for naturally variable voices
    attenuated = [t - attenuation_db for t in segments]
    canvas = create_silence(max(len(t) for t in attenuated))
    for t in attenuated: canvas = canvas.overlay(t)
    return canvas

def encode_webm(audio, out_path):
    part = out_path.with_suffix(out_path.suffix + ".part")
    with open(part, "wb") as fh:
        audio.export(fh, format="webm", codec="libopus", bitrate=Const.bitrate)
    os.replace(part, out_path)  # atomic on same filesystem

def write_json_atomic(obj, out_path):
    part = out_path.with_suffix(out_path.suffix + ".part")
    with open(part, "w") as fh: json.dump(obj, fh)
    os.replace(part, out_path)

def delete_uploads(uploads):
    for path in uploads:
        try:
            os.remove(path)
            log(f"Deleted: {path.name}", 2)
        except FileNotFoundError: log(f"File not found (may have been deleted): {path.name}", 2)
        except PermissionError: log(f"Permission denied: {path.name}", 2)

def run(mix_id, uploads_dir, mixes_dir, ref_path, timer):
    def print_stats(path, audio):
        log(f"  OK {path.name}  {round(len(audio) / 1000.0, 3)}s  {round(audio.dBFS, 2)}dBFS", 1)
    log(f"Start reference audio processing...", 1)
    ref_seg = prepare_audio(ref_path)
    timer.tick("prepare_audio")
    if ref_seg is None:
        log("ERROR: Reference audio preparation failed", 0)
        return
    ref_duration_ms = len(ref_seg)
    ref_seg = normalize_to_target(ref_seg)
    timer.tick("normalize")
    ref_chunks = split_into_stanzas(ref_seg)
    timer.tick("split_stanzas")
    if ref_chunks is None:
        log("ERROR: Reference audio is inaudible", 0)
        return
    passed = score_stanzas(mix_id, ref_chunks)
    timer.tick("score_stanzas")
    if not passed:
        log("ERROR: Reference audio stanza score insufficient", 0)
        return
    print_stats(ref_path, ref_seg)
    mixed_parts = []
    mixed_parts.append(ref_seg) # append reference multiple times?
    log(f"Starting processing of uploads for mix {mix_id}...", 1)
    uploads = sorted(p for p in uploads_dir.iterdir() if p.is_file() and p.name.endswith(".webm"))
    log(f"Found {len(uploads)} submitted file(s).", 1)
    contributors = []
    for path in uploads:
        log(f"Processing {path.name}...", 1)
        # parsed = parse_filename(path.name)
        # timer.tick("parse")
        # if not parsed: continue
        # md5, token, ip = parsed
        token, ip = "token", "ip"
        seg = prepare_audio(path)
        timer.tick("prepare_audio")
        if seg is None: continue
        accepted = compute_ratio(len(seg), ref_duration_ms)
        timer.tick("compute_ratio")
        if not accepted: continue
        seg = normalize_to_target(seg)
        timer.tick("normalize")
        chunks = split_into_stanzas(seg)
        timer.tick("split_stanzas")
        if chunks is None: continue
        passed = score_stanzas(mix_id, chunks)
        timer.tick("score_stanzas")
        if not passed: continue
        seg = align_stanzas(chunks, ref_chunks)
        timer.tick("align_stanzas")
        if seg is None: continue
        mixed_parts.append(seg)
        contributors.append({token: ip})
        print_stats(path, seg)
    timer.tick("all_uploads")
    log(f"Accepted {len(contributors)} of {len(uploads)}.", 1)
    log(f"Mixing {len(mixed_parts)} voices (equal-gain sum)...", 1)
    mixed = equal_gain_mix(mixed_parts)
    timer.tick("mix")
    mixed = normalize_to_target(mixed)
    timer.tick("normalize")
    log(f"Mix length {len(mixed)/1000:.2f}s, peak {mixed.max_dBFS:.1f} dBFS, loudness {mixed.dBFS:.1f} dBFS", 1)
    mix_path = mixes_dir / f"{mix_id}.webm"
    encode_webm(mixed, mix_path)
    timer.tick("encode")
    contributors_path = mixes_dir / f"{mix_id}.json"
    write_json_atomic(contributors, contributors_path)
    timer.tick("publish")
    #delete_uploads(uploads)
    timer.report()
    log(f"Process complete. Mix: {mix_path}", 1)

def main():
    if len(sys.argv) < 2:
        log("Usage: mix_worker.py <mix_id>", 0)
        return
    mix_id = sys.argv[1]
    uploads_dir = Path(Const.uploads_dir)
    if not uploads_dir.is_dir():
        log(f"ERROR: uploads dir not found: {uploads_dir}", 0)
        return
    mixes_dir = Path(Const.mixes_dir)
    if not mixes_dir.is_dir():
        log(f"ERROR: mixes dir not found: {mixes_dir}", 0)
        return
    tmp_dir = Path(Const.tmp_dir)
    if not tmp_dir.is_dir():
        log(f"ERROR: tmp dir not found: {tmp_dir}", 0)
        return
    ref_path = Path(Const.ref_path)
    if not ref_path.is_file():
        log(f"ERROR: reference audio not found: {ref_path}", 0)
        return
    stanzas_file = Path(Const.stanzas_file)
    if not stanzas_file.is_file():
        log(f"ERROR: reference audio not found: {stanzas_file}", 0)
        return
    Stanzas.load_stanzas()
    Model.load_model()
    timer = Timer()
    run(mix_id, uploads_dir, mixes_dir, ref_path, timer)

def log(message, level):
    if level > Const.error_level: return
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

if __name__ == "__main__":
    main()
