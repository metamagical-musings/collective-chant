# whisper_cli.py
"""
Test harness for whisper_tools.prepare_audio_for_whisper + score_stanzas.

Usage:

    python whisper_cli.py clip1 [clip2 ...] { [--model tiny.en] [--details] [--errors 3] [--stanza_file stanzas_clip.txt]

Behavior per clip:
    raw audio -> prepare_audio_for_whisper() -> split_into_stanzas() -> score_stanzas() -> report
The ffmpeg-produced WAV is written next to the source as <stem>_asr.wav and is deleted after scoring.
Each run prints the resolved config (so you can eyeball which model produced which numbers) and,
at the end, a one-line-per-clip summary table plus aggregate timing.

Examples:
    # quick pass with default base.en
    python whisper_cli.py take01.webm take02.webm

    # A/B three models on the same clips
    for m in tiny.en base.en small.en; do
        python whisper_cli.py take*.webm --model "$m" > "run_$m.txt"
    done
"""

import argparse, sys, time, os, pprint
from pathlib import Path
import whisper_tools as wt

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score chant submissions via faster-whisper.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("clips", nargs="+", help="Uploaded .webm (or any ffmpeg-readable) files")
    p.add_argument('--details', action='store_true', help='Print all score_stanzas results, including error details')
    p.add_argument("--model", default=None, help=f"Model name override (e.g. tiny.en/base.en/small.en). Default: {wt.MODEL_NAME}")
    p.add_argument("--errors", type=int, default=None, help=f"Max word errors before rejection. Default: {wt.MAX_WORD_ERRORS}")
    p.add_argument("--stanza_file", type=str, default=None, help="Text file with (possibly multi-line) stanzas separated by blank lines representing a >1 second gap of silence")
    return p.parse_args(argv)

def apply_overrides(args) -> None:
    """Mutate module-level config BEFORE the model singleton loads."""
    if args.model: wt.MODEL_NAME = args.model
    if args.errors is not None: wt.MAX_WORD_ERRORS = args.errors
    if args.stanza_file is not None: wt.STANZA_FILE = args.stanza_file

def print_header(args) -> None:
    print("=" * 72)
    print(f"model={wt.MODEL_NAME}")
    print(f"beam_size={wt.BEAM_SIZE}  temperature={wt.TEMPERATURE}")
    print(f"max_word_errors={wt.MAX_WORD_ERRORS}  stanza_file={wt.STANZA_FILE}")
    print("=" * 72)

def process_one(clip: Path, args) -> dict:
    wav = clip.with_name(clip.stem + "_asr.wav")
    t0 = time.perf_counter()
    try:
        wt.prepare_audio_for_whisper(clip, wav)
    except RuntimeError as exc:
        dt = time.perf_counter() - t0
        print(f"\n[REJECT/prep] {clip.name}: {exc}  ({dt:.1f}s)")
        return {"clip": clip.name, "status": "PREP_FAIL", "secs": dt}

    try:
        len_audio, chunks = wt.split_into_stanzas(wav)
    except ValueError as exc:
        dt = time.perf_counter() - t0
        print(f"\n[REJECT/split] {clip.name}: {exc}  ({dt:.1f}s)")
        return {"clip": clip.name, "status": "SPLIT_FAIL", "secs": dt}

    print(f"\nAudio {clip.name} length: {len_audio}ms")
    print(f"\nStanzas detected: {len(chunks)}")
    len_chunks = len(chunks)
    for i in range(len_chunks):
        c = chunks[i]
        print(f"  stanza {c.index}: {c.start_ms}ms → {c.end_ms}ms ({c.duration_ms()}ms)")
        if i == len_chunks - 1: break
        print(f"  gap: {chunks[i+1].start_ms - c.end_ms}ms")

    t1 = time.perf_counter()
    res = wt.score_stanzas(chunks)
    t2 = time.perf_counter()

    prep_s, tr_s = t1 - t0, t2 - t1

    verdict = "ACCEPT" if res.accepted else "REJECT"
    print(f"\n[{verdict}] {clip.name}")
    print(f"  reason            : {res.reason}")
    print(f"  stanzas           : {res.num_stanzas} / {res.num_ref_stanzas}")
    print(f"  word_errors       : {res.word_errors}  (rate={res.error_rate:.1%})")
    print(f"  timing            : prep+split={prep_s:.1f}s  transcribe={tr_s:.1f}s  total={t2-t0:.1f}s")

    for ss in res.stanza_scores:
        flag = "OK" if ss.accepted else "ERR"
        print(f"  [{flag}] stanza {ss.index}: {ss.word_errors} errors  | {ss.transcript!r}")
    if args.details: pprint.pprint(res.to_dict(), indent=2)

    try: wav.unlink()
    except OSError: pass

    return {"clip": clip.name, "status": verdict, "reason": res.reason,
            "stanzas": res.num_stanzas, "errs": res.word_errors,
            "secs": t2 - t0, "transcribe_s": tr_s}

def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("-" * 72)
    print(f"{'clip':<28}{'verdict':<11}{'stz':>4}{'err':>5}{'sec':>8}")
    for r in rows:
        stz = r.get("stanzas", "-")
        err = r.get("errs", "-")
        print(f"{r['clip']:<28}{r['status']:<11}{str(stz):>4}{str(err):>5}{r['secs']:>8.1f}")
    n_acc = sum(1 for r in rows if r["status"] == "ACCEPT")
    total = sum(r["secs"] for r in rows)
    tr_total = sum(r.get("transcribe_s", 0) for r in rows)
    print("-" * 72)
    print(f"{len(rows)} clip(s): {n_acc} accepted, {len(rows)-n_acc} rejected.  total={total:.1f}s  transcribe-only={tr_total:.1f}s")

def main(argv=None) -> int:
    args = parse_args(argv)
    apply_overrides(args)
    print_header(args)

    rows = []
    for raw in args.clips:
        clip = Path(raw)
        if not clip.exists():
            print(f"\n[SKIP] missing file: {clip}", file=sys.stderr)
            continue
        rows.append(process_one(clip, args))

    if not rows:
        print("No clips processed.", file=sys.stderr)
        return 1
    print_summary(rows)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
