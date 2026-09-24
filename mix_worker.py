#!/usr/bin/env python3

import json, os, re, shutil, sys, time
from pathlib import Path
from pydub import AudioSegment, effects, silence

class Const:
    uploads_dir = "uploads"
    mixes_dir = "mixes"
    ref_path = "static/mantra.webm"
    tolerance_sec = 1.0
    headroom_db = 6.0
    target_db = -14.0
    silence_thresh_db = -50.0
    min_silence_ms = 150
    bitrate = "48k" # bps

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

def trim_leading_trailing_silence(audio):
    ranges = silence.detect_nonsilent(audio, min_silence_len=Const.min_silence_ms, silence_thresh=Const.silence_thresh_db)
    if not ranges:
        print(f"  REJECT: Entirely silent (or below threshold)")
        return None
    start, end = ranges[0][0], ranges[-1][1]
    return audio[start:end]  

def check_duration(audio, ref_duration_sec):
    dur_sec = len(audio) / 1000.0
    delta_sec = abs(dur_sec - ref_duration_sec);
    if (delta_sec > Const.tolerance_sec):
        print(f"  REJECT: {delta_sec:.1f}s delta duration exceeds allowed tolerance {Const.tolerance_sec}s")
        return False
    return True
    
def normalize_to_target(audio, target_db, headroom_db):
    peak = audio.max_dBFS
    if peak > -180: audio = audio.apply_gain(-peak - headroom_db) # not digital silence
    current = audio.dBFS
    if current > -180: audio = audio.apply_gain(target_db - current)
    return audio

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
        print(f"ERROR: uploads dir not found: {mixes_dir}")
        return 1
    ref_path = Path(Const.ref_path)
    if not ref_path.is_file():
        print(f"ERROR: reference audio not found: {ref_path}")
        return 1
    lock = WorkerLock(mixes_dir)
    if not lock.acquire():
        print("Another worker holds the lock; exiting.")
        return 0
    timer = Timer()
    try:
        run(mix_id, uploads_dir, mixes_dir, ref_path, timer)
        return 0
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        #raise
        return 1
    finally: lock.release()

def run(mix_id, uploads_dir, mixes_dir, ref_path, timer):
    def print_stats(name, audio):
        print(f"  OK {name}  {round(len(audio) / 1000.0, 3)}s  {round(audio.dBFS, 2)}dBFS")
    print(f"[Worker {mix_id}] Start reference audio processing...")
    ref_seg = decode_webm(ref_path)
    if ref_seg is None: raise ValueError("Invalid reference audio")
    ref_seg = normalize_to_target(ref_seg, Const.target_db, Const.headroom_db)
    print_stats(ref_path.name, ref_seg)
    ref_duration_sec = len(ref_seg) / 1000.0
    mixed_parts = []
    mixed_parts.append(ref_seg) # append reference multiple times?
    print(f"[Worker {mix_id}] Starting processing of candidate uploads...")
    candidates = sorted(p for p in uploads_dir.iterdir() if p.is_file() and p.name.endswith(".webm"))
    print(f"Found {len(candidates)} candidate file(s).")
    contributors = []
    for path in candidates:
        print(f"Processing {path.name}...")
        parsed = parse_filename(path.name)
        if not parsed: continue
        md5, token, ip = parsed
        seg = decode_webm(path)
        timer.tick("decode")
        if seg is None: continue
        seg = trim_leading_trailing_silence(seg)
        timer.tick("trim_silence")
        if seg is None: continue
        if not check_duration(seg, ref_duration_sec): continue
        seg = normalize_to_target(seg, Const.target_db, Const.headroom_db)
        timer.tick("normalize")
        mixed_parts.append(seg)
        contributors.append({token: ip})
        print_stats(path.name, seg)
    timer.tick("all_candidates")
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
    for path in candidates:
        try:
            os.remove(path)
            print(f"Deleted: {path.name}")
        except FileNotFoundError: print(f"File not found (may have been deleted): {path.name}")
        except PermissionError: print(f"Permission denied: {path.name}")
    timer.report()
    print(f"Process complete. Mix: {mix_path}")

if __name__ == "__main__":
    sys.exit(main())

