#!/usr/bin/env python3
import json, os, re, shutil, sys, time, math, webrtcvad, soundfile
import numpy as np, pyrubberband as pyrb
from filelock import FileLock, Timeout
from datetime import datetime
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.exceptions import CouldntDecodeError
from pathlib import Path
from faster_whisper import WhisperModel

MAX_INT16 = 32768

class Const:
    error_level = 3 # 0==FATAL, 1==COURSE, 2==FINE, 3==TIMING
    lock_file = "mix_worker.lock"
    uploads_dir = "uploads"
    mixes_dir = "mixes"
    tmp_dir = "tmp"
    ref_orig_path = "reference/chant.mp3"
    ref_audio_path = "reference/chant.webm"
    ref_ticks_path = "reference/chant_ticks.webm"
    ref_stanzas_path = "reference/chant_stanzas.txt"
    stanza_tmp_dir = "tmp/chant_stanzas"
    sample_rate = 16000
    channels = 1
    # create_count_in_audio()
    num_ticks = 3
    tick_duration = 0.04
    tick_interval = 0.8
    tick_frequency = 1200
    tick_amplitude = 0.3
    tick_decay = 40.0
    bitrate = "24k" # bps
    # split_into_stanzas()
    min_gap_ms = 1000              # minimum silence gap to qualify as a stanza boundary
    min_speech_ms = 250            # skip speech segments shorter than this
    min_speech_rms = 0.02          # skip speech segments weaker than this
    vad_aggressiveness = 3         # maximum 3 for precise word boundaries
    frame_ms = 10
    # score_stanza()
    save_stanzas = True            # True to save stanza results in tmp_dir during testing
    beam_size = 10
    temperature = 0.0
    word_timestamps = False
    condition_on_previous_text = False
    compression_ratio_threshold = 2.0
    log_prob_threshold = -1.0
    no_speech_threshold = 0.6
    vad_filter = False
    max_stanza_word_errors = 1     # stanzas that exceed this word error limit are silenced
    # compute_ratio()
    ratio_tolerance = 0.10         # stanzas that must be stretched/compressed more than a factor of 1 +/- this value are silenced
    # align_stanzas()
    max_stanzas_silenced = 1       # if this number silenced stanzas is exceeded, the entire audio is rejected
    stanza_gap_ms = 1500           # gap of silence to be inserted between stanzas when reassembled
    # Whisper.load_model()
    device = "cpu"
    compute_type = "int8"
    model_name = "base.en"         # alternatives: tiny.en, base.en, small.en, or drop .en
    # run()
    num_ref_audio_in_mix = 1
    save_uploads = True            # True to save uploaded recordings in uploads_dir during testing

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
        Util.log(f"  [t={total:6.2f}s | step {elapsed:5.2f}s] {label}", 3)

    def report(self):
        Util.log("Timing summary:", 3)
        for label, secs in self.stages.items(): Util.log(f"    {secs:7.2f}s  {label}", 3)
        Util.log(f"    {time.perf_counter() - self.t0:7.2f}s  TOTAL", 3)

class Util:
    @staticmethod
    def log(message, level):
        if level > Const.error_level: return
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
        if level == 0: sys.exit(1)
    @staticmethod
    def log_stats(path, audio):
        Util.log(f"  OK {path.name}  {round(len(audio) / 1000.0, 3)}s  {round(audio.dBFS, 2)}dBFS", 1)
    @staticmethod
    def compute_ratio(duration_ms, ref_duration_ms):
        if duration_ms == 0 or ref_duration_ms == 0: return False
        ratio = duration_ms / ref_duration_ms
        if abs(1 - ratio) > Const.ratio_tolerance: return False
        return ratio
    @staticmethod
    def check_dirs_files(*dfs):
        df_paths = []
        for df in dfs:
            df_path = Path(df)
            if not df_path.exists(): Util.log(f"ERROR: directory/file not found: {df_path.name}", 0)
            df_paths.append(df_path)
        return tuple(df_paths)
    @staticmethod
    def get_mix_signal(mixes_dir):
        matches = [p for p in mixes_dir.iterdir() if p.is_file() and p.name.endswith(".mix")]
        if not matches: return False
        path = matches[0]
        path.unlink()
        return path.stem
    @staticmethod
    def get_uploads(uploads_dir):
        return sorted(p for p in uploads_dir.iterdir() if p.is_file() and p.name.endswith(".webm"))
    @staticmethod
    def delete_uploads(uploads):
        for path in uploads:
            try:
                os.remove(path)
                Util.log(f"Deleted: {path.name}", 2)
            except FileNotFoundError: Util.log(f"File not found (may have been deleted): {path.name}", 2)
            except PermissionError: Util.log(f"Permission denied: {path.name}", 2)
    @staticmethod
    def write_audio_atomic(audio, out_path):
        part = out_path.with_suffix(out_path.suffix + ".part")
        with open(part, "wb") as fh: audio.export(fh, format="webm", codec="libopus", bitrate=Const.bitrate)
        os.replace(part, out_path)
    @staticmethod
    def write_json_atomic(obj, out_path):
        part = out_path.with_suffix(out_path.suffix + ".part")
        with open(part, "w") as fh: json.dump(obj, fh)
        os.replace(part, out_path)

class VAD:
    vad = None
    @classmethod
    def load(cls):
        if cls.vad is None: cls.vad = webrtcvad.Vad(Const.vad_aggressiveness)

class Whisper:
    model = None
    @classmethod
    def load_model(cls, model_name):
        if cls.model is None: cls.model = WhisperModel(model_name, device=Const.device, compute_type=Const.compute_type)

class Reference:
    words = []
    stanza_durations_ms = []
    num_stanzas = 0
    ref_audio = None
    @classmethod
    def load_stanzas(cls, ref_stanzas_path):
        with open(ref_stanzas_path, 'r', encoding='utf-8') as f:
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
            wav_path = str(tmp_dir / f"stanza_{self.index}.wav")
            self.audio.export(wav_path, format="wav")
            segments, info = Whisper.model.transcribe(
                wav_path,
                language="en",
                beam_size=Const.beam_size,
                temperature=Const.temperature,
                initial_prompt=Reference.words[self.index],
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
            wed = self.word_edit_distance(Reference.words[self.index], transcript)
            stanza_results = {
                "index": self.index,
                "transcript": transcript,
                "language": language,
                "word_edit_distance": wed
            }
            results_file = str(tmp_dir / f"stanza_{self.index}.json")
            with open(results_file, "w") as f: json.dump(stanza_results, f, indent=4)
        except Exception as exc:
            Util.log(f"  REJECT STANZA: Scoring aborted for stanza {self.index} ({exc})", error_code)
            return False
        if wed > Const.max_stanza_word_errors:
            Util.log(f"  REJECT STANZA: {wed} word errors in stanza {self.index} exceeds allowed maximum {Const.max_stanza_word_errors}", error_code)
            return False
        return True
    @staticmethod
    def score_stanzas(stanzas, error_code):
        tmp_dir = Path(Const.stanza_tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        num_silenced_stanzas = 0
        for stanza in stanzas:
            if not stanza.score_stanza(tmp_dir, error_code):
                stanzas[stanza.index].audio = False
                num_silenced_stanzas += 1
        if not Const.save_stanzas: shutil.rmtree(tmp_dir, ignore_errors=True)
        if num_silenced_stanzas > Const.max_stanzas_silenced: return False
        return True
    @staticmethod
    def combine_stanzas(stanzas, ticks):
        silence = Audio.create_silence(Const.stanza_gap_ms)
        if ticks:
            combined = silence
            gap = Audio.count_in_audio
        else:
            combined = AudioSegment.empty()
            gap = silence
        for seg in stanzas: combined += gap + seg.audio
        return combined
    @staticmethod
    def align_stanzas(up_stanzas, error_code):
        def match_duration_preserve_pitch(audio, index, ref_ms):
            stanza_ms = len(audio)
            ratio = Util.compute_ratio(stanza_ms, ref_ms)
            if ratio is False:
                Util.log(f"  REJECT STANZA: duration ratio of stanza {index} exceeds allowed tolerance {Const.ratio_tolerance}", error_code)
                return Audio.create_silence(ref_ms), True
            stretched = pyrb.time_stretch(Audio.audio_to_float_array(audio), audio.frame_rate, ratio)
            if stretched.ndim == 2 and audio.channels == 1: stretched = stretched.flatten()
            return Audio.float_array_to_audio(stretched), False
        combined = AudioSegment.empty()
        gap = Audio.create_silence(Const.stanza_gap_ms)
        num_silenced_stanzas = 0
        for i, (up, ref_ms) in enumerate(zip(up_stanzas, Reference.stanza_durations_ms)):
            if up.audio is False:
                stanza = Audio.create_silence(ref_ms)
                num_silenced_stanzas += 1
            else:
                stanza, silent = match_duration_preserve_pitch(up.audio, i, ref_ms)
                if silent: num_silenced_stanzas += 1
            combined += gap + stanza
        if num_silenced_stanzas > Const.max_stanzas_silenced:
            Util.log(f"  REJECT AUDIO: {num_silenced_stanzas} silenced stazas exceeds allowed maximum {Const.max_stanzas_silenced}", error_code)
            return False
        return combined

class Audio:
    count_in_audio = None
    def __init__(self, path, error_code):
        self.path = path
        self.error_code = error_code
        self.audio = None
        self.stanzas = []
    def prepare_audio(self):
        try:
            audio = AudioSegment.from_file(self.path)
        except CouldntDecodeError as e:
            Util.log(f"  REJECT AUDIO {self.path.name}: Audio decoding failed", self.error_code)
            return False
        except FileNotFoundError:
            Util.log(f"  REJECT AUDIO {self.path.name}: Audio file not found", self.error_code)
            return False
        self.audio = audio.set_frame_rate(Const.sample_rate).set_channels(1).set_sample_width(2)
        return True
    def split_into_stanzas(self):
        # detect speech frames
        frame_size = (Const.sample_rate * Const.frame_ms) // 1000
        frame_bytes = frame_size * 2  # 2 bytes per int16 sample
        samples = self.audio.raw_data
        speech_frames = []
        for i in range(0, len(samples) - frame_bytes + 1, frame_bytes):
            speech_frames.append(VAD.vad.is_speech(samples[i : i + frame_bytes], Const.sample_rate))
        # group frames into speech segments
        speech = []
        in_speech = False
        start = 0
        for i, is_speech in enumerate(speech_frames):
            frame = i * Const.frame_ms
            if is_speech and not in_speech:
                in_speech = True
                start = frame
            elif not is_speech and in_speech:
                in_speech = False
                speech.append((start, frame))
        if in_speech: speech.append((start, frame + Const.frame_ms))
        # remove very short or weak speech segments
        speech = [(s, e) for s, e in speech if (e - s) > Const.min_speech_ms and Audio.compute_rms(self.audio[s:e]) > Const.min_speech_rms]
        # group speech segments into stanzas
        stanzas = []
        for s, e in speech:
            if stanzas and (s - stanzas[-1][1]) < Const.min_gap_ms: stanzas[-1] = (stanzas[-1][0], e)
            else: stanzas.append((s, e))
        if not stanzas:
            Util.log(f"  REJECT AUDIO {self.path.name}: stanzas split failed (No speech detected)", self.error_code)
            return False
        if len(stanzas) != Reference.num_stanzas:
            Util.log(f"  REJECT AUDIO: Detected {len(stanzas)} upload stanzas, expected {Reference.num_stanzas}", self.error_code)
            return False
        # normalize and save stanzas
        self.stanzas = [Stanza(index=i, start_ms=s, end_ms=e, audio=normalize(self.audio[s:e])) for i, (s, e) in enumerate(stanzas)]
        return True
    def process(self, timer, process_ref):
        if not self.prepare_audio(): return False
        timer.tick("prepare_audio")
        if not self.split_into_stanzas(): return False
        timer.tick("split_stanzas")
        if not Stanza.score_stanzas(self.stanzas, self.error_code): return False
        timer.tick("score_stanzas")
        if process_ref:
            Reference.stanza_durations_ms = [len(stanza.audio) for stanza in self.stanzas]
            combined = Stanza.combine_stanzas(self.stanzas, False)
            timer.tick("combine_stanzas_silence")
            Util.write_audio_atomic(combined, Path(Const.ref_audio_path))
            timer.tick("publish_ref_audio")
            combined = Stanza.combine_stanzas(self.stanzas, True)
            timer.tick("combine_stanzas_ticks")
            Util.write_audio_atomic(combined, Path(Const.ref_ticks_path))
            timer.tick("publish_count_in_audio")
        self.audio = Stanza.align_stanzas(self.stanzas, self.error_code)
        if self.audio is False: return False
        timer.tick("align_stanzas")
        return self.audio
    @classmethod
    def compute_rms(cls, audio):
        return float(np.sqrt(np.mean(cls.audio_to_float_array(audio) ** 2)))
    @staticmethod
    def create_silence(duration_ms):
        return AudioSegment.silent(duration=duration_ms, frame_rate=Const.sample_rate)
    @classmethod
    def create_count_in_audio(cls):
        tick_interval = max(Const.tick_interval, Const.tick_duration)
        t = np.linspace(0, Const.tick_duration, int(Const.sample_rate * Const.tick_duration), endpoint=False)
        envelope = np.exp(-Const.tick_decay * t)
        tick = Const.tick_amplitude * envelope * np.sin(2 * np.pi * Const.tick_frequency * t)
        tick = tick.astype(np.float32)
        silence = np.zeros(int(Const.sample_rate * tick_interval) - len(tick), dtype=np.float32)
        segments = []
        segments.append(silence)
        for _ in range(Const.num_ticks):
            segments.append(tick)
            segments.append(silence)
        count_in = np.concatenate(segments)
        count_in = np.clip(count_in, -1.0, 1.0) # safety clip
        cls.count_in_audio = cls.float_array_to_audio(count_in)
    @staticmethod
    def audio_to_float_array(audio):
        """Convert AudioSegment to float32 numpy array in [-1, 1].
        Returns 1D array for mono, 2D (n_frames, 2) for stereo.
        """
        samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / MAX_INT16
        if audio.channels == 2: samples = samples.reshape(-1, 2)
        return samples
    @staticmethod
    def float_array_to_audio(samples):
        """Convert mono float32 array in [-1, 1] to a pydub AudioSegment."""
        samples = np.asarray(samples, dtype=np.float32)
        samples_int16 = np.clip(samples * MAX_INT16 - 1, -MAX_INT16, MAX_INT16 - 1).astype(np.int16)
        return AudioSegment(data=samples_int16.tobytes(), frame_rate=Const.sample_rate, sample_width=2, channels=Const.channels)

class Mix:
    def __init__(self, audios, contributors):
        self.audios = audios
        self.contributors = contributors
        self.mixed_audio = None
    def equal_gain_mix(self):
        segments = [t.set_frame_rate(Const.sample_rate).set_channels(Const.channels).set_sample_width(2) for t in self.audios]
        attenuation_db = 18.0 * math.log10(len(segments)) # 20 safe starting value for coherent case; 16–18 for naturally variable voices
        attenuated = [t - attenuation_db for t in segments]
        canvas = Audio.create_silence(max(len(t) for t in attenuated))
        for t in attenuated: canvas = canvas.overlay(t)
        self.mixed_audio = normalize(canvas)
    def process(self, mix_id, mixes_dir, timer):
        self.equal_gain_mix()
        timer.tick("mix_audio")
        mixed_audio_path = mixes_dir / f"{mix_id}.webm"
        Util.write_audio_atomic(self.mixed_audio, mixed_audio_path)
        timer.tick("publish_audio")
        Util.write_json_atomic(self.contributors, mixes_dir / f"{mix_id}.json")
        timer.tick("publish_json")
        return mixed_audio_path, self.mixed_audio

def run(mix_id, uploads_dir, mixes_dir, timer):
    contributions = []
    for _ in range(Const.num_ref_audio_in_mix): contributions.append(Reference.ref_audio)
    Util.log(f"Starting processing of uploads for mix {mix_id}...", 1)
    uploads = Util.get_uploads(uploads_dir)
    Util.log(f"Found {len(uploads)} submitted file(s).", 1)
    contributors = []
    for path in uploads:
        Util.log(f"Processing {path.name}...", 1)
        audio = Audio(path, 1).process(timer, False)
        if audio is False: continue
        contributions.append(audio)
        matches = re.compile(r"^([^.]+)\.([^.]+)\.(.+)\.webm$").match(path.name)
        if matches: md5, token, ip = matches.groups()
        else: token, ip = "token", "ip" # for testing
        contributors.append({token: ip})
        Util.log_stats(path, audio)
    timer.tick("all_uploads")
    Util.log(f"Mixing audios ({Const.num_ref_audio_in_mix} reference + {len(contributions) - Const.num_ref_audio_in_mix} contributed)...", 1)
    Util.log_stats(*Mix(contributions, contributors).process(mix_id, mixes_dir, timer))
    if not Const.save_uploads: Util.delete_uploads(uploads)
    timer.report()

def main():
    Util.log("Collective Chant program", 1)
    timer = Timer()
    Reference.load_stanzas(*Util.check_dirs_files(Const.ref_stanzas_path))
    Audio.create_count_in_audio()
    VAD.load()
    Whisper.load_model(Const.model_name)
    timer.tick("load_model")
    Util.log(f"Start reference audio processing...", 1)
    Reference.ref_audio = Audio(*Util.check_dirs_files(Const.ref_orig_path), 0).process(timer, True) # never returns False
    Util.log_stats(Path(Const.ref_orig_path), Reference.ref_audio)
    uploads_dir, mixes_dir, tmp_dir = Util.check_dirs_files(Const.uploads_dir, Const.mixes_dir, Const.tmp_dir)
    Util.log(f"Waiting for mix signal...", 1)
    try:
        while True:
            if len(sys.argv) > 1:
                mix_id = sys.argv[1] # optional one-time command line mix request to be followed by polling
                sys.argv = [sys.argv[0]]
            else: mix_id = Util.get_mix_signal(*Util.check_dirs_files(Const.mixes_dir))
            if not mix_id:
                time.sleep(2)
                continue
            Util.log(f"Processing mix {mix_id}...", 1)
            run(mix_id, uploads_dir, mixes_dir, Timer())
            Util.log(f"Waiting for mix signal...", 1)
    except KeyboardInterrupt:
        Util.log("Program terminated by user.", 1)
        return
if __name__ == "__main__":
    lock = FileLock(Const.lock_file)
    try: lock.acquire(timeout=0)
    except Timeout:
        Util.log("Another instance is already running.", 0)
        sys.exit(1)
    main()
    sys.exit(0)
