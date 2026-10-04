#!/usr/bin/env python3

from pydub import AudioSegment, silence
from difflib import SequenceMatcher
import json, os, re, shutil, sys, time, subprocess
from pathlib import Path
from pydub import AudioSegment, silence
from faster_whisper import WhisperModel

class Const:
    uploads_dir = "uploads"
    mixes_dir = "mixes"
    tmp_dir = "tmp"
    ref_path = "static/ref_audio.webm"
    stanza_file = "static/stanzas.txt"
    stanza_tmp_dir = "tmp/chant_stanzas"
    stanza_tmp_cleanup = True
    # prepare_audio()
    target_sample_rate = 16000
    target_channels = 1
    loudnorm_filter = "loudnorm=I=-16:TP=-3:LRA=11"
    # check_duration()
    tolerance_sec = 1.0
    # normalize_to_target()
    headroom_db = 6.0
    target_db = -14.0
    bitrate = "48k" # bps
    # split_into_stanzas()
    gap_threshold_ms = 1000      # minimum gap to qualify as a stanza boundary
    min_speech_ms = 250          # discard nonsilent segments shorter than this
    silence_thresh_offset = 15   # dBFS offset below overall level for silence detect
    seek_step_ms = 10            # resolution of silence detection
    # get_model()
    device = "cpu"
    compute_type = "int8"
    model_name = "base.en"
    model = None
    # score_stanzas()
    beam_size = 5
    temperature = 0.0
    word_timestamps = False
    condition_on_previous_text = False
    compression_ratio_threshold = 2.0
    log_prob_threshold = -1.0
    no_speech_threshold = 0.6
    vad_filter = True
    max_word_errors = 3

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
        print(f"  [t={total:6.2f}s | step {elapsed:5.2f}s] {label}")

    def report(self):
        print("Timing summary:")
        for label, secs in self.stages.items(): print(f"    {secs:7.2f}s  {label}")
        print(f"    {time.perf_counter() - self.t0:7.2f}s  TOTAL")

class WorkerLock:
    def __init__(self, mixes_dir):
        self.path = mixes_dir / ".mix_worker.lock"
        self.fh = None
    def acquire(self):
        try:
            self.fh = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fh, str(os.getpid()).encode())
            return True
        except FileExistsError: return False
    def release(self) -> None:
        if self.fh is not None:
            os.close(self.fh)
            try: os.unlink(self.path)
            except FileNotFoundError: pass

class StanzaChunk:
    def __init__(self, index, start_ms, end_ms, audio):
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.audio = audio
        
    def duration_ms(self):
        return self.end_ms - self.start_ms

def parse_filename(fname):
    """Return (md5, token, ip) or None if the name doesn't match."""
    m = re.compile(r"^([^.]+)\.([^.]+)\.(.+)\.webm$").match(fname)
    if not m:
        print(f"  SKIP (bad filename pattern): {fname}")
        return None
    return m.groups()

def decode_webm(path):
    try:
        audio = AudioSegment.from_file(str(path), format="webm")
    except Exception as exc:  # corrupt / truncated / not really webm
        print(f"  REJECT {path.name}: decode failed ({exc})")
        return None
    # Force a common format so overlay arithmetic is well defined.
    return audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)

def prepare_audio(path, timeout=60.0):
    try:
        dst_path = Path(Const.tmp_dir) / Path(path.name + "_asr.wav")
        cmd = ["ffmpeg", "-y", "-i", str(path), "-af", Const.loudnorm_filter, "-ar", str(Const.target_sample_rate), "-ac", str(Const.target_channels), str(dst_path)]
        try: proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
        except FileNotFoundError as exc: raise RuntimeError("ffmpeg executable not found on PATH") from exc
        except subprocess.TimeoutExpired as exc: raise RuntimeError(f"ffmpeg timed out after {timeout}s on {path}") from exc
        if proc.returncode != 0 or not dst_path.exists():
            tail = proc.stderr.decode(errors="replace")[-400:] if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}) on {path}: {tail}")
        audio = AudioSegment.from_wav(str(dst_path))
        try: dst_path.unlink()
        except OSError as exc: raise RuntimeError("Unable to delete temporary wav file {dst_path}") from exc
    except RuntimeError as exc:
        print(f"  REJECT: {path.name}: unlink failed ({exc})")
        return None
    return audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)

def check_duration(audio, ref_duration_sec):
    dur_sec = len(audio) / 1000.0
    delta_sec = abs(dur_sec - ref_duration_sec);
    if (delta_sec > Const.tolerance_sec):
        print(f"  REJECT: {delta_sec:.1f}s delta duration exceeds allowed tolerance {Const.tolerance_sec}s")
        return False
    return True

def normalize_to_target(audio):
    peak = audio.max_dBFS
    if peak > -180: audio = audio.apply_gain(-peak - Const.headroom_db) # not digital silence
    current = audio.dBFS
    if current > -180: audio = audio.apply_gain(Const.target_db - current)
    return audio

def split_into_stanzas(audio):
    try:
        silence_thresh = audio.dBFS - Const.silence_thresh_offset
        nonsilent = silence.detect_nonsilent(audio, min_silence_len=Const.seek_step_ms, silence_thresh=silence_thresh, seek_step=Const.seek_step_ms)
        # Filter out short noise artifacts within silence regions.
        real_speech = [(s, e) for s, e in nonsilent if (e - s) >= Const.min_speech_ms]
        if not real_speech: raise ValueError(f"No speech detected in {wav_path}")
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
    except ValueError as exc:
        print(f"  REJECT: {path.name}: stanzas split failed ({exc})")
        return None
    return chunks

def get_model():
    if Const.model is None: Const.model = WhisperModel(Const.model_name, device=Const.device, compute_type=Const.compute_type)
    return Const.model

def normalize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into bare word tokens."""
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower())
    return cleaned.split()

def count_word_errors(reference, hypothesis):
    matcher = SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
    replace, delete, insert = 0, 0, 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal": continue
        elif tag == "replace": replace += max(i2 - i1, j2 - j1) # Substitution(s)
        elif tag == "delete": delete += i2 - i1 # Words present in reference but missing from hypothesis
        elif tag == "insert": insert += j2 - j1 # Extra words that appear only in the hypothesis
    total = replace + delete + insert
    denom = max(len(reference), 1)
    return {
        "word_errors": total,
        "substitutions": replace,
        "missing": delete,
        "extra": insert,
        "error_rate": total / denom,
        "ref_count": len(reference),
        "hyp_count": len(hypothesis),
    }

def score_stanzas(mix_id, chunks):
    with open(Path(Const.stanza_file), 'r', encoding='utf-8') as f:
        chant_texts = []
        current_block = []
        for line in f:
            line = line.rstrip()
            if line: current_block.append(line)
            elif current_block:
                chant_texts.append(' '.join(current_block))
                current_block = []
        if current_block: chant_texts.append(' '.join(current_block))
    stanza_texts = [normalize_words(line) for line in chant_texts]
    num_ref_stanzas = len(stanza_texts)
    if len(chunks) != num_ref_stanzas:
        print(f"  REJECT: detected {len(chunks)} stanzas, expected {num_ref_stanzas}")
        return None
    model = get_model()
    tmp_dir = Path(Const.stanza_tmp_dir + f"_{mix_id}.wav")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    total_errors = 0
    stanza_windows = []
    saved_exc = None
    try:
        for chunk in chunks:
            wav_file = str(tmp_dir / f"stanza_{chunk.index}.wav")
            chunk.audio.export(wav_file, format="wav")
            segments, info = model.transcribe(
                wav_file,
                language="en",
                initial_prompt=chant_texts[chunk.index],
                beam_size=Const.beam_size,
                temperature=Const.temperature,
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
            hyp_words = normalize_words(transcript)
            ref_words = stanza_texts[chunk.index]
            err = count_word_errors(ref_words, hyp_words)
            stanza_results = {
                "index": chunk.index,
                "transcript": transcript,
                "language": language,
            }
            stanza_results.update(err)
            results_file = str(tmp_dir / f"stanza_{chunk.index}.json")
            with open(results_file, "w") as f: json.dump(stanza_results, f, indent=4)
            total_errors += err["word_errors"]
            stanza_windows.append((chunk.start_ms / 1000.0, chunk.end_ms / 1000.0))
    except Exception as exc:
        saved_exc = exc
    finally:
        if Const.stanza_tmp_cleanup: shutil.rmtree(tmp_dir, ignore_errors=True)
    if saved_exc is not None:
        print(f"  REJECT: Scoring aborted ({saved_exc})")
        return None
    elif total_errors > Const.max_word_errors:
        print(f"  REJECT: {total_errors} total word errors > {Const.max_word_errors}")
        return None
    return stanza_windows

def compress_or_stretch_to_target(audio, ref_duration_ms):
    factor = len(audio) / ref_duration_ms # >1 means audio is too long
    return audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * factor)}).set_frame_rate(audio.frame_rate)

def equal_gain_mix(segments):
    width = max(len(s) for s in segments)
    out = AudioSegment.silent(duration=width, frame_rate=48000)
    for seg in segments: out = out.overlay(seg)  # pydub overlays with saturating add
#    if loop_pause_ms > 0: out = out + AudioSegment.silent(duration=loop_pause_ms, frame_rate=48000)
    return out

def apply_safety_limiter(audio, ceiling_dbfs=-1.0):
    peak = audio.max_dBFS
    if peak > ceiling_dbfs:
        print(f"  Limiter engaged: peak {peak:.1f} dBFS -> {ceiling_dbfs:.1f} dBFS")
        audio = audio.apply_gain(ceiling_dbfs - peak)
    return audio

def encode_webm(audio, out_path, bitrate="48k"):
    part = out_path.with_suffix(out_path.suffix + ".part")
    with open(part, "wb") as fh:
        audio.export(fh, format="webm", codec="libopus", bitrate=bitrate)
    os.replace(part, out_path)  # atomic on same filesystem

def write_json_atomic(obj, out_path):
    part = out_path.with_suffix(out_path.suffix + ".part")
    with open(part, "w") as fh: json.dump(obj, fh)
    os.replace(part, out_path)

def run(mix_id, uploads_dir, mixes_dir, ref_path, timer):
    def print_stats(name, audio):
        print(f"  OK {name}  {round(len(audio) / 1000.0, 3)}s  {round(audio.dBFS, 2)}dBFS")
    print(f"[Worker {mix_id}] Start reference audio processing...")
    ref_seg = decode_webm(ref_path)
    if ref_seg is None: raise ValueError("Invalid reference audio")
    ref_seg = normalize_to_target(ref_seg)
    print_stats(ref_path.name, ref_seg)
    ref_duration_ms = len(ref_seg)
    ref_duration_sec = ref_duration_ms / 1000.0
    mixed_parts = []
    mixed_parts.append(ref_seg) # append reference multiple times?
    print(f"[Worker {mix_id}] Starting processing of candidate uploads...")
    candidates = sorted(p for p in uploads_dir.iterdir() if p.is_file() and p.name.endswith(".webm"))
    print(f"Found {len(candidates)} candidate file(s).")
    contributors = []
    for path in candidates:
        print(f"Processing {path.name}...")
        parsed = parse_filename(path.name)
        timer.tick("parse")
        if not parsed: continue
        md5, token, ip = parsed
        seg = prepare_audio(path)
        timer.tick("prepare_audio")
        if seg is None: continue
        accepted = check_duration(seg, ref_duration_sec)
        timer.tick("check_duration")
        if not accepted: continue
        seg = normalize_to_target(seg)
        timer.tick("normalize")
        chunks = split_into_stanzas(seg)
        timer.tick("stanzas_split")
        if chunks is None: continue
        stanza_windows = score_stanzas(mix_id, chunks)
        timer.tick("score_stanzas")
        if stanza_windows is None: continue

        continue

        seg = compress_or_stretch_to_target(seg, ref_duration_ms)
        timer.tick("strech_compress")
        mixed_parts.append(seg)
        contributors.append({token: ip})
        print_stats(path.name, seg)
    timer.tick("all_candidates")
    sys.exit()
    print(f"Accepted {len(contributors)} of {len(candidates)}.")
    print(f"Mixing {len(mixed_parts)} voices (equal-gain sum)...")
    mixed = equal_gain_mix(mixed_parts)
    timer.tick("mix")
    mixed = apply_safety_limiter(mixed)
    timer.tick("limit")
    print(f"Mix length {len(mixed)/1000:.2f}s, peak {mixed.max_dBFS:.1f} dBFS, loudness {mixed.dBFS:.1f} dBFS")
    mix_path = mixes_dir / f"{mix_id}.webm"
    encode_webm(mixed, mix_path, bitrate=Const.bitrate)
    timer.tick("encode")
    contributors_path = mixes_dir / f"{mix_id}.json"
    write_json_atomic(contributors, contributors_path)
    timer.tick("publish")
    # for path in candidates:
        # try:
            # os.remove(path)
            # print(f"Deleted: {path.name}")
        # except FileNotFoundError: print(f"File not found (may have been deleted): {path.name}")
        # except PermissionError: print(f"Permission denied: {path.name}")
    timer.report()
    print(f"Process complete. Mix: {mix_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: mix_worker.py <mix_id>")
        return 1
    mix_id = sys.argv[1]
    uploads_dir = Path(Const.uploads_dir)
    if not uploads_dir.is_dir():
        print(f"ERROR: uploads dir not found: {uploads_dir}")
        return 1
    mixes_dir = Path(Const.mixes_dir)
    if not mixes_dir.is_dir():
        print(f"ERROR: mixes dir not found: {mixes_dir}")
        return 1
    tmp_dir = Path(Const.tmp_dir)
    if not tmp_dir.is_dir():
        print(f"ERROR: tmp dir not found: {mixes_dir}")
        return 1
    ref_path = Path(Const.ref_path)
    if not ref_path.is_file():
        print(f"ERROR: reference audio not found: {ref_path}")
        return 1
    # lock = WorkerLock(mixes_dir)
    # if not lock.acquire():
        # print("Another worker holds the lock; exiting.")
        # return 0
    timer = Timer()
    try:
        run(mix_id, uploads_dir, mixes_dir, ref_path, timer)
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        raise
        return 1
    # finally: lock.release()

if __name__ == "__main__":
    sys.exit(main())

