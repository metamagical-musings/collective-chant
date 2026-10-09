# whisper_tools.py
"""
Whisper-based scoring helpers for mix_worker.py.

Pipeline position:
    raw .webm upload
        -> [pydub] trim silence + duration check      (existing)
        -> prepare_audio_for_whisper()                (this file)
        -> split_into_stanzas()                       (this file)
        -> score_stanzas()                            (this file)
        -> [downstream] stanza alignment / shifting   (later)
        -> [post-process] normalize / sum / limiter   (existing)

Design notes:
- faster-whisper (CTranslate2 backend) is used instead of openai-whisper:
  same models, ~2-4x faster on CPU, int8 quantization, deterministic output.
- The model is loaded ONCE at module import. When mix_worker becomes a
  long-running daemon, this means no per-job cold start.
- Stanza boundaries come from ENERGY detection (pydub), NOT Whisper segmentation.
  This is deterministic, sample-accurate, and immune to Whisper's non-determinism.
- Each stanza blob is transcribed independently by Whisper. No chunking issues.
- Reference texts are per-stanza word lists (may be identical or different).
- The caller imposes a recording standard: inter-stanza pauses >1000ms.
"""

import json, subprocess, shutil, sys
from pathlib import Path
from faster_whisper import WhisperModel
from pydub import AudioSegment, silence
from difflib import SequenceMatcher

MAX_WORD_ERRORS = 3
MODEL_NAME = "base.en"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
TEMPERATURE = 0.0
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
LOUDNORM_FILTER = "loudnorm=I=-16:TP=-3:LRA=11"
GAP_THRESHOLD_MS = 1000      # minimum gap to qualify as a stanza boundary
MIN_SPEECH_MS = 250          # discard nonsilent segments shorter than this
SILENCE_THRESH_OFFSET = 15   # dBFS offset below overall level for silence detect
SEEK_STEP_MS = 10            # resolution of silence detection
STANZA_FILE = "stanzas.txt"
STANZA_TMP_DIR = "chant_stanzas"
STANZA_TMP_CLEANUP = True

_MODEL = None

class StanzaChunk:
    def __init__(self, index, start_ms, end_ms, audio):
        self.index = index
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.audio = audio
        
    def duration_ms(self):
        return self.end_ms - self.start_ms

class StanzaScore:
    def __init__(self, index, transcript, words, word_errors, error_detail, accepted):
        self.index = index
        self.transcript = transcript
        self.words = words
        self.word_errors = word_errors
        self.error_detail = error_detail
        self.accepted = accepted

class ScoreResult:
    def __init__(self, accepted, reason, num_stanzas, num_ref_stanzas, word_errors, error_rate, stanza_windows=None, transcript="", detected_language="en", language_probability=1.0, stanza_scores=None):
        self.accepted = accepted
        self.reason = reason
        self.num_stanzas = num_stanzas
        self.num_ref_stanzas = num_ref_stanzas
        self.word_errors = word_errors
        self.error_rate = error_rate
        self.stanza_windows = stanza_windows if stanza_windows is not None else []
        self.transcript = transcript
        self.detected_language = detected_language
        self.language_probability = language_probability
        self.stanza_scores = stanza_scores if stanza_scores is not None else []

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "stanza_scores" and k != "stanza_windows"}
        d["stanza_windows"] = [list(w) for w in self.stanza_windows]
        d["stanza_scores"] = [
            {"index": s.index, "transcript": s.transcript,
             "word_errors": s.word_errors, "accepted": s.accepted, "error_detail": s.error_detail}
            for s in self.stanza_scores
        ]
        return d

def prepare_audio_for_whisper(src_path, dst_path, timeout=60.0):
    src_path, dst_path = Path(src_path), Path(dst_path)
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-af", LOUDNORM_FILTER, "-ar", str(TARGET_SAMPLE_RATE), "-ac", str(TARGET_CHANNELS), str(dst_path)]
    try: proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    except FileNotFoundError as exc: raise RuntimeError("ffmpeg executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc: raise RuntimeError(f"ffmpeg timed out after {timeout}s on {src_path}") from exc
    if proc.returncode != 0 or not dst_path.exists():
        tail = proc.stderr.decode(errors="replace")[-400:] if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}) on {src_path}: {tail}")
    return dst_path

def split_into_stanzas(wav_path, *, gap_threshold_ms = GAP_THRESHOLD_MS, min_speech_ms = MIN_SPEECH_MS, silence_thresh_offset = SILENCE_THRESH_OFFSET, seek_step_ms = SEEK_STEP_MS, max_stanzas = None):
    """Split a pre-processed WAV into per-stanza chunks at large energy gaps.
    Uses pydub.silence.detect_nonsilent to find speech regions, then groups
    consecutive regions into stanzas separated by gaps >= gap_threshold_ms.
    Returns a list of StanzaChunk objects, one per detected stanza.
    Raises ValueError if no speech is detected or if max_stanzas is exceeded.
    """
    audio = AudioSegment.from_wav(str(wav_path))
    audio = audio.set_frame_rate(48000).set_channels(1).set_sample_width(2)
    silence_thresh = audio.dBFS - silence_thresh_offset
    nonsilent = silence.detect_nonsilent(audio, min_silence_len=seek_step_ms, silence_thresh=silence_thresh, seek_step=seek_step_ms)
    # Filter out short noise artifacts within silence regions.
    real_speech = [(s, e) for s, e in nonsilent if (e - s) >= min_speech_ms]
    if not real_speech: raise ValueError(f"No speech detected in {wav_path}")
    # Group speech regions into stanzas: a new stanza starts when the gap before it exceeds gap_threshold_ms.
    stanzas = [[real_speech[0]]]
    for i in range(1, len(real_speech)):
        prev_end = real_speech[i - 1][1]
        curr_start = real_speech[i][0]
        gap = curr_start - prev_end
        if gap >= gap_threshold_ms: stanzas.append([real_speech[i]])
        else: stanzas[-1].append(real_speech[i])
    if max_stanzas is not None and len(stanzas) > max_stanzas: raise ValueError(f"Detected {len(stanzas)} stanzas, exceeds max {max_stanzas}")
    chunks = []
    for idx, regions in enumerate(stanzas):
        start_ms = regions[0][0]
        end_ms = regions[-1][1]
        blob = audio[start_ms:end_ms]
        chunks.append(StanzaChunk(index=idx, start_ms=start_ms, end_ms=end_ms, audio=blob))
    return len(audio), chunks

def get_model():
    global _MODEL
    if _MODEL is None: _MODEL = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _MODEL

def _audio_to_temp_wav(blob, tmp_dir, idx):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"stanza_{idx}.wav"
    blob.export(str(path), format="wav")
    return path

def normalize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into bare word tokens."""
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text.lower())
    return cleaned.split()

def count_word_errors(reference, hypothesis):
    """
    Count the number of word-level errors (substitutions + insertions + deletions)
    between a reference sequence and a hypothesis sequence.

    The alignment is computed with difflib.SequenceMatcher, which produces
    the same style of “keep-in-sync” matching that the classic diff algorithm uses.
    """
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

def score_stanzas(chunks, *, max_word_errors=MAX_WORD_ERRORS, tmp_dir=STANZA_TMP_DIR, cleanup_tmp=STANZA_TMP_CLEANUP):
    """Transcribe each stanza chunk independently and accumulate global WER.
    Args:
        chunks: Output from split_into_stanzas().
        max_word_errors: Global threshold across all stanzas.
        tmp_dir: Where to write temporary per-stanza WAVs for Whisper.
        cleanup_tmp: Delete temp files after scoring.
    Returns:
        ScoreResult with per-stanza details and overall verdict.
    """

    with open(Path(STANZA_FILE), 'r', encoding='utf-8') as f:
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
    tmp_path = Path(tmp_dir)
    if len(chunks) != num_ref_stanzas:
        return ScoreResult(accepted=False, reason=f"detected {len(chunks)} stanzas, expected {num_ref_stanzas}", num_stanzas=len(chunks), num_ref_stanzas=num_ref_stanzas, word_errors=0, error_rate=0.0)
    model = get_model()
    stanza_scores = []
    total_errors = 0
    total_ref_words = 0
    all_transcripts = []
    stanza_windows = []
    try:
        for chunk in chunks:
            ref_words = stanza_texts[chunk.index]
            total_ref_words += len(ref_words)
            wav_file = _audio_to_temp_wav(chunk.audio, tmp_path, chunk.index)
            segments, info = model.transcribe(
                str(wav_file),
                language="en",
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                word_timestamps=False,
                condition_on_previous_text=False,
                initial_prompt=chant_texts[chunk.index],
                compression_ratio_threshold=2.0,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                vad_filter=True
            )
            words_list, parts = [], []
            for seg in segments:
                parts.append(seg.text.strip())
                for w in (seg.words or []): words_list.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
            transcript = " ".join(parts).strip()
            lang = (info.language or "").lower()
            hyp_words = normalize_words(transcript)
            err = count_word_errors(ref_words, hyp_words)
            stanza_err_count = err["word_errors"]
            total_errors += stanza_err_count
            stanza_scores.append(StanzaScore(index=chunk.index, transcript=transcript, words=hyp_words, word_errors=stanza_err_count, error_detail=err, accepted=(stanza_err_count <= max_word_errors)))
            all_transcripts.append(transcript)
            stanza_windows.append((chunk.start_ms / 1000.0, chunk.end_ms / 1000.0))
    finally:
        if cleanup_tmp: shutil.rmtree(tmp_path, ignore_errors=True)
    denom = max(total_ref_words, 1)
    accepted = total_errors <= max_word_errors
    reason = "ok" if accepted else (f"{total_errors} total word errors > {max_word_errors} threshold")
    return ScoreResult(accepted=accepted, reason=reason, num_stanzas=len(chunks), num_ref_stanzas=num_ref_stanzas, word_errors=total_errors, error_rate=total_errors / denom, stanza_windows=stanza_windows, transcript=" | ".join(all_transcripts), stanza_scores=stanza_scores)


