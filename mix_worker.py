#!/usr/bin/env python3

"""
mix_worker.py - standalone program to mix audio files found in uploads/*.webm and save the result to the mixes directory.
                The only command line argument is the mix number.

To improve effectiveness, audio blobs are split into stanzas (that should be less than 30 seconds in duration, to avoid ASR degredation)
according to >1 second gaps of silence between them. After splitting, stanzas are shifted and compressed/stretched to match a
reference    in static/ref_path.

Automatic Speech Recognition is done by faster_whisper, which downloads and caches a model when first run. Recognized text
must match the reference text in stanzas_file within max_stanza_errors per stanza. Stanzas in stanzas_file are separated by blank lines.

Currently works only on Linux. Must install: ffmpeg and pyrubberband-cli
"""
import json, os, re, shutil, sys, time, subprocess, math
import numpy as np, pyrubberband as pyrb, soundfile as sf
from datetime import datetime
from pydub import AudioSegment, silence
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
    # Whisper.load_model()
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

class Util:
    @staticmethod
    def create_silence(duration_ms):
        return AudioSegment.silent(duration=duration_ms, frame_rate=Const.target_sample_rate)

    @staticmethod
    def compute_ratio(duration_ms, ref_duration_ms, error_code):
        if duration_ms == 0 or ref_duration_ms == 0: return False
        ratio = duration_ms / ref_duration_ms
        if abs(1 - ratio) > Const.ratio_tolerance:
            log(f"  REJECT: {ratio:.2f} duration ratio exceeds allowed tolerance {Const.ratio_tolerance}", error_code)
            return False
        return ratio

    @staticmethod
    def normalize_to_target(audio):
        if audio.max_dBFS <= -180: return audio # digital silence
        return audio.apply_gain(min(Const.target_db - audio.dBFS, -Const.headroom_db - audio.max_dBFS))

    @staticmethod
    def equal_gain_mix(segments):
        segments = [t.set_frame_rate(Const.target_sample_rate).set_channels(2).set_sample_width(2) for t in segments]
        attenuation_db = 20.0 * math.log10(len(segments)) # safe starting value for coherent case; 16–18 for naturally variable voices
        attenuated = [t - attenuation_db for t in segments]
        canvas = Util.create_silence(max(len(t) for t in attenuated))
        for t in attenuated: canvas = canvas.overlay(t)
        return canvas

    @staticmethod
    def write_audio_atomic(audio, out_path):
        part = out_path.with_suffix(out_path.suffix + ".part")
        with open(part, "wb") as fh:
            audio.export(fh, format="webm", codec="libopus", bitrate=Const.bitrate)
        os.replace(part, out_path)

    @staticmethod
    def write_json_atomic(obj, out_path):
        part = out_path.with_suffix(out_path.suffix + ".part")
        with open(part, "w") as fh: json.dump(obj, fh)
        os.replace(part, out_path)

    @staticmethod
    def delete_uploads(uploads):
        for path in uploads:
            try:
                os.remove(path)
                log(f"Deleted: {path.name}", 2)
            except FileNotFoundError: log(f"File not found (may have been deleted): {path.name}", 2)
            except PermissionError: log(f"Permission denied: {path.name}", 2)

class Whisper:
    model = None
    @classmethod
    def load_model(cls):
        if cls.model is None: cls.model = WhisperModel(Const.model_name, device=Const.device, compute_type=Const.compute_type)

class ReferenceText:
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

class Stanza:
    def __init__(self, index, start_ms, end_ms, audio):
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.audio = audio

    def word_edit_distance(self, reference, hypothesis):
        def normalize_words(text):
            """Lowercase, strip punctuation, split into bare word tokens."""
            cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower())
            return cleaned.split()
        ref = normalize_words(reference)
        hyp = normalize_words(hypothesis)
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

    def score_stanza(self, tmp_dir, error_code):
        try:
            wav_file = str(tmp_dir / f"stanza_{self.index}.wav")
            self.audio.export(wav_file, format="wav")
            segments, info = Whisper.model.transcribe(
                wav_file,
                language="en",
                beam_size=Const.beam_size,
                temperature=Const.temperature,
                initial_prompt=ReferenceText.words[self.index],
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
            wed = self.word_edit_distance(ReferenceText.words[self.index], transcript)
            stanza_results = {
                "index": self.index,
                "transcript": transcript,
                "language": language,
                "word_edit_distance": wed
            }
            results_file = str(tmp_dir / f"stanza_{self.index}.json")
            with open(results_file, "w") as f: json.dump(stanza_results, f, indent=4)
        except Exception as exc:
            log(f"  REJECT: Scoring aborted for stanza {self.index} ({exc})", error_code)
            return False
        if wed > Const.max_stanza_errors:
            log(f"  REJECT: {wed} stanza word errors > {Const.max_stanza_errors}", error_code)
            return False
        return True

    @staticmethod
    def score_stanzas(stanzas, error_code):
        if len(stanzas) != ReferenceText.num_stanzas:
            log(f"  REJECT: detected {len(stanzas)} stanzas, expected {ReferenceText.num_stanzas}", 1)
            return False
        tmp_dir = Path(Const.stanza_tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        passed = True
        for stanza in stanzas:
            if not stanza.score_stanza(tmp_dir, error_code):
                passed = False
                break
        if Const.stanza_tmp_cleanup: shutil.rmtree(tmp_dir, ignore_errors=True)
        return passed

    @staticmethod
    def align_stanzas(up_stanzas, ref_stanzas, error_code):
        def match_duration_preserve_pitch(audio, reference):
            ref_ms = len(reference)
            seg_ms = len(audio)
            ratio = Util.compute_ratio(seg_ms, ref_ms, error_code)
            if ratio is False: return Util.create_silence(ref_ms)
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2: samples = samples.reshape((-1, 2))
            samples = samples.astype(np.float32) / (2 ** (8 * audio.sample_width - 1))
            stretched = pyrb.time_stretch(samples, audio.frame_rate, ratio)
            buf = BytesIO()
            sf.write(buf, stretched, audio.frame_rate, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return AudioSegment.from_file(buf, format="wav")
        if len(up_stanzas) != len(ref_stanzas):
            log(f"  REJECT: Stanzas mismatch (upload {len(up_stanzas)} vs reference {len(ref_stanzas)})", error_code)
            return False
        total_ms = ref_stanzas[-1].end_ms
        canvas = Util.create_silence(total_ms)
        for up, ref in zip(up_stanzas, ref_stanzas):
            seg = match_duration_preserve_pitch(up.audio, ref.audio)
            if ref.start_ms + len(seg) > total_ms: seg = seg[: max(total_ms - ref.start_ms, 0)] # Clamp to avoid overflow.
            canvas = canvas.overlay(seg, position=max(ref.start_ms, 0))
        return canvas

class Audio:
    def __init__(self, path, error_code, ref_duration_ms, ref_stanzas):
        self.path = path
        self.error_code = error_code
        self.ref_duration_ms = ref_duration_ms
        self.ref_stanzas = ref_stanzas
        self.audio = None
        self.stanzas = []
        self.duration_ms = 0

    def prepare_audio(self):
        try:
            dst_path = Path(Const.tmp_dir) / Path(self.path.name + "_asr.wav")
            cmd = ["ffmpeg", "-y", "-i", str(self.path), "-af", Const.loudnorm_filter, "-ar", str(Const.target_sample_rate), "-ac", str(Const.target_channels), str(dst_path)]
            try: proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=Const.ffmpeg_timeout)
            except FileNotFoundError as exc: raise RuntimeError("ffmpeg executable not found on PATH") from exc
            except subprocess.TimeoutExpired as exc: raise RuntimeError(f"ffmpeg timed out after {Const.ffmpeg_timeout}s") from exc
            if proc.returncode != 0 or not dst_path.exists():
                tail = proc.stderr.decode(errors="replace")[-400:] if proc.stderr else ""
                raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}) on {self.path}: {tail}")
            audio = AudioSegment.from_wav(str(dst_path))
            try: dst_path.unlink()
            except OSError as exc: raise RuntimeError("Unable to delete temporary wav file {dst_path}") from exc
        except RuntimeError as exc:
            log(f"  REJECT {self.path.name}: Audio preparation failed ({exc})", self.error_code)
            return False
        self.audio = audio.set_frame_rate(Const.target_sample_rate).set_channels(1).set_sample_width(2)
        return True

    def split_into_stanzas(self):
        silence_thresh = self.audio.dBFS - Const.silence_thresh_offset
        nonsilent = silence.detect_nonsilent(self.audio, min_silence_len=Const.seek_step_ms, silence_thresh=silence_thresh, seek_step=Const.seek_step_ms)
        # Filter out short noise artifacts within silence regions.
        real_speech = [(s, e) for s, e in nonsilent if (e - s) >= Const.min_speech_ms]
        if not real_speech:
            log(f"  REJECT {self.path.name}: stanzas split failed (No speech detected)", self.error_code)
            return False
        # Group speech regions into stanzas: a new stanza starts when the gap before it exceeds Const.gap_threshold_ms.
        stanzas = [[real_speech[0]]]
        for i in range(1, len(real_speech)):
            prev_end = real_speech[i - 1][1]
            curr_start = real_speech[i][0]
            gap = curr_start - prev_end
            if gap >= Const.gap_threshold_ms: stanzas.append([real_speech[i]])
            else: stanzas[-1].append(real_speech[i])
        for idx, regions in enumerate(stanzas):
            start_ms = regions[0][0]
            end_ms = regions[-1][1]
            blob = self.audio[start_ms:end_ms]
            self.stanzas.append(Stanza(index=idx, start_ms=start_ms, end_ms=end_ms, audio=blob))
        return True

    def process(self, timer):
        if not self.prepare_audio(): return False
        timer.tick("prepare_audio")
        self.duration_ms = len(self.audio)
        if self.ref_duration_ms is not False:
            if Util.compute_ratio(self.duration_ms, self.ref_duration_ms, self.error_code) is False: return False
            timer.tick("check_ratio")
        self.audio = Util.normalize_to_target(self.audio)
        timer.tick("normalize")
        if not self.split_into_stanzas(): return False
        timer.tick("split_stanzas")
        if not Stanza.score_stanzas(self.stanzas, self.error_code): return False
        timer.tick("score_stanzas")
        if self.ref_duration_ms is not False and self.ref_stanzas is not False:
            self.audio = Stanza.align_stanzas(self.stanzas, self.ref_stanzas, self.error_code)
            timer.tick("align_stanzas")
            if self.audio is False: return False
        return self.audio, self.duration_ms, self.stanzas

def run(mix_id, uploads_dir, mixes_dir, ref_path, timer):
    def log_stats(path, audio):
        log(f"  OK {path.name}  {round(len(audio) / 1000.0, 3)}s  {round(audio.dBFS, 2)}dBFS", 1)
    log(f"Start reference audio processing...", 1)
    ref_audio, ref_duration_ms, ref_stanzas = Audio(ref_path, 0, False, False).process(timer) # never returns False
    log_stats(ref_path, ref_audio)
    mix_parts = []
    mix_parts.append(ref_audio) # append reference multiple times?
    log(f"Starting processing of uploads for mix {mix_id}...", 1)
    uploads = sorted(p for p in uploads_dir.iterdir() if p.is_file() and p.name.endswith(".webm"))
    log(f"Found {len(uploads)} submitted file(s).", 1)
    contributors = []
    for path in uploads:
        log(f"Processing {path.name}...", 1)
        audio, duration_ms, stanzas = Audio(path, 1, ref_duration_ms, ref_stanzas).process(timer)
        mix_parts.append(audio)
        matches = re.compile(r"^([^.]+)\.([^.]+)\.(.+)\.webm$").match(path.name)
        if matches: md5, token, ip = matches.groups()
        else: token, ip = "token", "ip" # for testing
        contributors.append({token: ip})
        log_stats(path, audio)
    timer.tick("all_uploads")
    log(f"Accepted {len(contributors)} of {len(uploads)}.", 1)
    log(f"Mixing {len(mix_parts)} voices (equal-gain sum)...", 1)
    mixed = Util.equal_gain_mix(mix_parts)
    timer.tick("mix")
    mixed = Util.normalize_to_target(mixed)
    timer.tick("normalize")
    log(f"Mix length {len(mixed)/1000:.2f}s, peak {mixed.max_dBFS:.1f} dBFS, loudness {mixed.dBFS:.1f} dBFS", 1)
    mix_path = mixes_dir / f"{mix_id}.webm"
    Util.write_audio_atomic(mixed, mix_path)
    timer.tick("encode")
    contributors_path = mixes_dir / f"{mix_id}.json"
    Util.write_json_atomic(contributors, contributors_path)
    timer.tick("publish")
    # Util.delete_uploads(uploads)
    timer.report()
    log(f"Process complete. Mix: {mix_path}", 1)

def main():
    if len(sys.argv) < 2: log("Usage: mix_worker.py <mix_id>", 0)
    mix_id = sys.argv[1]
    uploads_dir = Path(Const.uploads_dir)
    if not uploads_dir.is_dir(): log(f"ERROR: uploads dir not found: {uploads_dir}", 0)
    mixes_dir = Path(Const.mixes_dir)
    if not mixes_dir.is_dir(): log(f"ERROR: mixes dir not found: {mixes_dir}", 0)
    tmp_dir = Path(Const.tmp_dir)
    if not tmp_dir.is_dir(): log(f"ERROR: tmp dir not found: {tmp_dir}", 0)
    ref_path = Path(Const.ref_path)
    if not ref_path.is_file(): log(f"ERROR: reference audio not found: {ref_path}", 0)
    stanzas_file = Path(Const.stanzas_file)
    if not stanzas_file.is_file(): log(f"ERROR: reference audio not found: {stanzas_file}", 0)
    ReferenceText.load_stanzas()
    Whisper.load_model()
    timer = Timer()
    run(mix_id, uploads_dir, mixes_dir, ref_path, timer)

def log(message, level):
    if level > Const.error_level: return
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    if level == 0: sys.exit(1)

if __name__ == "__main__":
    main()
    sys.exit(0)
