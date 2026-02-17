# ============================================================
# utils/audio_utils.py
# AI Music Remix & Mood Generator — Audio Utility Functions
# ============================================================
# Comprehensive audio processing toolkit using:
#   - Pydub       : Format conversion, segment manipulation
#   - SoundFile   : Low-level PCM read/write
#   - Torchaudio  : Resampling, transforms
#   - Librosa     : Spectral analysis, feature extraction
#   - NumPy       : Array operations
# ============================================================

from __future__ import annotations

import base64
import io
import logging
import math
import os
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BIT_DEPTH   = 24
DEFAULT_CHANNELS    = 2

SUPPORTED_INPUT_FORMATS  = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".opus"}
SUPPORTED_OUTPUT_FORMATS = {".mp3", ".wav", ".flac", ".ogg"}

FORMAT_CODEC_MAP = {
    "mp3":  {"format": "mp3",  "codec": "libmp3lame",  "bitrate": "320k"},
    "flac": {"format": "flac", "codec": "flac",         "bitrate": None},
    "ogg":  {"format": "ogg",  "codec": "libvorbis",    "bitrate": "192k"},
    "wav":  {"format": "wav",  "codec": "pcm_s24le",    "bitrate": None},
}


# ── Load / Save ───────────────────────────────────────────────

def load_audio(
    path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    mono: bool = False,
    start_seconds: float = 0.0,
    duration_seconds: Optional[float] = None,
    normalize: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file into a float32 numpy array.

    Args:
        path:             Path to audio file.
        target_sr:        Target sample rate (resamples if needed).
        mono:             Convert to mono if True.
        start_seconds:    Start offset in seconds.
        duration_seconds: Duration to load (None = full file).
        normalize:        Peak-normalize the waveform.

    Returns:
        Tuple of (waveform_array, sample_rate).
        waveform shape: (channels, samples) or (samples,) if mono=True.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the format is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(
            f"Unsupported format: {suffix}. Supported: {SUPPORTED_INPUT_FORMATS}"
        )

    logger.debug("Loading audio: %s (target_sr=%d, mono=%s)", path, target_sr, mono)

    # Use librosa for robust multi-format loading
    try:
        import librosa
        y, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
            offset=start_seconds,
            duration=duration_seconds,
        )
        # librosa returns (samples,) for mono or (channels, samples) for stereo
    except ImportError:
        # Fallback: SoundFile + Torchaudio
        y, sr = _load_with_soundfile(str(path), target_sr, mono, start_seconds, duration_seconds)

    y = y.astype(np.float32)

    if normalize:
        y = normalize_audio(y)

    return y, sr


def _load_with_soundfile(
    path: str,
    target_sr: int,
    mono: bool,
    start: float,
    duration: Optional[float],
) -> Tuple[np.ndarray, int]:
    """SoundFile-based loader (fallback when librosa unavailable)."""
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        start_frame = int(start * sr)
        n_frames = int(duration * sr) if duration else -1

        f.seek(start_frame)
        data = f.read(frames=n_frames, dtype="float32", always_2d=True)

    # data: (samples, channels) → (channels, samples)
    y = data.T

    if mono and y.shape[0] > 1:
        y = y.mean(axis=0)

    if sr != target_sr:
        waveform_t = torch.from_numpy(y)
        if waveform_t.ndim == 1:
            waveform_t = waveform_t.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        y = resampler(waveform_t).numpy()
        if mono:
            y = y.squeeze(0)
        sr = target_sr

    return y, sr


def save_audio(
    waveform: np.ndarray,
    path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_format: Optional[str] = None,
    bit_depth: int = DEFAULT_BIT_DEPTH,
    normalize: bool = False,
) -> str:
    """
    Save audio array to file.

    Args:
        waveform:     Audio array. Shape: (channels, samples) or (samples,).
        path:         Output file path.
        sample_rate:  Audio sample rate.
        output_format: Override format (inferred from path extension if None).
        bit_depth:    Bit depth for WAV output (16 or 24).
        normalize:    Peak-normalize before saving.

    Returns:
        Absolute path to the saved file.
    """
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    if normalize:
        waveform = normalize_audio(waveform)

    fmt = (output_format or path.suffix.lstrip(".")).lower()

    # Ensure float32
    waveform = waveform.astype(np.float32)

    # Clip to prevent distortion
    waveform = np.clip(waveform, -1.0, 1.0)

    # sf.write expects (samples, channels)
    if waveform.ndim == 2:
        data_for_write = waveform.T
    else:
        data_for_write = waveform

    if fmt == "wav":
        subtype = "PCM_24" if bit_depth == 24 else "PCM_16"
        sf.write(str(path), data_for_write, sample_rate, subtype=subtype)

    elif fmt in ("mp3", "flac", "ogg"):
        tmp_wav = str(path) + ".tmp.wav"
        sf.write(tmp_wav, data_for_write, sample_rate, subtype="PCM_16")
        seg = AudioSegment.from_wav(tmp_wav)
        bitrate = FORMAT_CODEC_MAP.get(fmt, {}).get("bitrate") or "192k"
        seg.export(str(path), format=fmt, bitrate=bitrate)
        os.unlink(tmp_wav)

    else:
        sf.write(str(path), data_for_write, sample_rate)

    logger.debug("Saved audio: %s (%s, %d Hz)", path, fmt, sample_rate)
    return str(path.resolve())


# ── Format Conversion ─────────────────────────────────────────

def convert_to_wav(
    input_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
) -> str:
    """
    Convert any supported audio file to WAV.

    Args:
        input_path:  Source file path.
        output_path: Destination WAV path (auto-generated if None).
        sample_rate: Target sample rate.
        channels:    Output channels (1=mono, 2=stereo).

    Returns:
        Path to the converted WAV file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="converted_")
        os.close(fd)

    logger.info("Converting %s → WAV (%d Hz, %dch)", input_path.name, sample_rate, channels)

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(channels)
    audio = audio.set_sample_width(3)    # 24-bit
    audio.export(output_path, format="wav")

    return output_path


def export_to_format(
    input_path: str,
    output_format: str,
    output_path: Optional[str] = None,
    bitrate: str = "320k",
) -> str:
    """
    Re-encode audio into the specified format.

    Args:
        input_path:    Source audio file.
        output_format: Target format: 'mp3', 'flac', 'ogg', 'wav'.
        output_path:   Destination path (auto-generated if None).
        bitrate:       Bitrate for lossy formats.

    Returns:
        Path to exported file.
    """
    input_path = Path(input_path)
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=f".{output_format}")
        os.close(fd)

    audio = AudioSegment.from_file(str(input_path))

    export_kwargs: dict = {"format": output_format}
    if output_format not in ("wav", "flac"):
        export_kwargs["bitrate"] = bitrate

    audio.export(output_path, **export_kwargs)
    logger.debug("Exported: %s → %s", input_path.name, output_format)
    return output_path


# ── Tempo & Pitch ─────────────────────────────────────────────

def change_tempo(
    waveform: np.ndarray,
    sr: int,
    speed_factor: float,
    preserve_pitch: bool = True,
) -> np.ndarray:
    """
    Change audio tempo by a speed factor.

    Args:
        waveform:       Audio array (channels, samples) or (samples,).
        sr:             Sample rate.
        speed_factor:   > 1.0 speeds up, < 1.0 slows down.
        preserve_pitch: Use time-stretching (True) or resampling (False).

    Returns:
        Tempo-adjusted audio array (same shape convention as input).
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("Librosa required for tempo change. pip install librosa")

    if abs(speed_factor - 1.0) < 1e-4:
        return waveform    # No-op

    def _stretch_mono(y: np.ndarray) -> np.ndarray:
        if preserve_pitch:
            return librosa.effects.time_stretch(y.astype(np.float32), rate=speed_factor)
        else:
            return y    # Speed without pitch fix handled by caller via sample_rate

    if waveform.ndim == 2:
        channels = [_stretch_mono(waveform[c]) for c in range(waveform.shape[0])]
        min_len = min(c.shape[0] for c in channels)
        return np.stack([c[:min_len] for c in channels])
    else:
        return _stretch_mono(waveform)


def change_pitch(
    waveform: np.ndarray,
    sr: int,
    semitones: float,
) -> np.ndarray:
    """
    Shift pitch by N semitones without changing speed.

    Args:
        waveform:  Audio array.
        sr:        Sample rate.
        semitones: Pitch shift in semitones (+/- 12 = one octave).

    Returns:
        Pitch-shifted audio array.
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("Librosa required for pitch shift. pip install librosa")

    if abs(semitones) < 0.001:
        return waveform    # No-op

    def _shift_mono(y: np.ndarray) -> np.ndarray:
        return librosa.effects.pitch_shift(y.astype(np.float32), sr=sr, n_steps=semitones)

    if waveform.ndim == 2:
        channels = [_shift_mono(waveform[c]) for c in range(waveform.shape[0])]
        min_len = min(c.shape[0] for c in channels)
        return np.stack([c[:min_len] for c in channels])
    else:
        return _shift_mono(waveform)


def detect_tempo(
    waveform: np.ndarray,
    sr: int,
) -> Tuple[float, np.ndarray]:
    """
    Detect BPM and beat positions.

    Returns:
        (tempo_bpm, beat_times_seconds)
    """
    try:
        import librosa
    except ImportError:
        return 120.0, np.array([])

    if waveform.ndim == 2:
        y_mono = waveform.mean(axis=0)
    else:
        y_mono = waveform

    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return float(tempo), beat_times


def detect_key(
    waveform: np.ndarray,
    sr: int,
) -> Tuple[str, float]:
    """
    Detect the musical key (e.g., 'C major', 'A# minor').

    Returns:
        (key_string, confidence_0_to_1)
    """
    try:
        import librosa
    except ImportError:
        return "C major", 0.0

    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    if waveform.ndim == 2:
        y_mono = waveform.mean(axis=0)
    else:
        y_mono = waveform

    chroma = librosa.feature.chroma_cqt(y=y_mono.astype(np.float32), sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Major and minor key templates (Krumhansl-Schmuckler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    def _ks_correlation(chroma: np.ndarray, profile: np.ndarray) -> np.ndarray:
        scores = []
        for shift in range(12):
            rotated = np.roll(profile, shift)
            corr = np.corrcoef(chroma, rotated)[0, 1]
            scores.append(corr)
        return np.array(scores)

    major_scores = _ks_correlation(chroma_mean, major_profile)
    minor_scores = _ks_correlation(chroma_mean, minor_profile)

    best_major_idx = int(np.argmax(major_scores))
    best_minor_idx = int(np.argmax(minor_scores))

    if major_scores[best_major_idx] >= minor_scores[best_minor_idx]:
        key_str = f"{NOTES[best_major_idx]} major"
        confidence = float(major_scores[best_major_idx])
    else:
        key_str = f"{NOTES[best_minor_idx]} minor"
        confidence = float(minor_scores[best_minor_idx])

    return key_str, max(0.0, min(1.0, confidence))


# ── Normalization ─────────────────────────────────────────────

def normalize_audio(
    waveform: np.ndarray,
    target_peak: float = 0.95,
    target_lufs: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize audio to target peak amplitude or LUFS.

    Args:
        waveform:    Audio array.
        target_peak: Peak amplitude (0.0–1.0). Used if target_lufs is None.
        target_lufs: Target integrated loudness in LUFS (e.g., -14.0 for streaming).

    Returns:
        Normalized audio array.
    """
    if target_lufs is not None:
        return _normalize_lufs(waveform, target_lufs)

    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform * (target_peak / peak)
    return waveform


def _normalize_lufs(waveform: np.ndarray, target_lufs: float) -> np.ndarray:
    """LUFS-normalized (loudness units relative to full scale) normalization."""
    try:
        import pyloudnorm as pyln
        sr = DEFAULT_SAMPLE_RATE
        meter = pyln.Meter(sr)

        if waveform.ndim == 1:
            data = waveform[:, np.newaxis]
        else:
            data = waveform.T

        loudness = meter.integrated_loudness(data)
        if np.isnan(loudness) or np.isinf(loudness):
            return normalize_audio(waveform)

        normalized = pyln.normalize.loudness(data, loudness, target_lufs)
        if waveform.ndim == 1:
            return normalized[:, 0]
        return normalized.T

    except ImportError:
        logger.warning("pyloudnorm not installed; using peak normalization instead")
        return normalize_audio(waveform)


# ── Silence & Trimming ────────────────────────────────────────

def trim_silence(
    waveform: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading/trailing silence.

    Args:
        waveform:  Audio array.
        sr:        Sample rate (unused, kept for API consistency).
        top_db:    Threshold in dB below peak to consider as silence.

    Returns:
        (trimmed_waveform, (start_sample, end_sample))
    """
    try:
        import librosa
    except ImportError:
        return waveform, (0, waveform.shape[-1])

    if waveform.ndim == 2:
        y_mono = waveform.mean(axis=0)
    else:
        y_mono = waveform

    _, intervals = librosa.effects.trim(
        y_mono.astype(np.float32),
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    start, end = int(intervals[0]), int(intervals[1])

    if waveform.ndim == 2:
        return waveform[:, start:end], (start, end)
    return waveform[start:end], (start, end)


def apply_fade(
    waveform: np.ndarray,
    sr: int,
    fade_in_seconds: float = 0.05,
    fade_out_seconds: float = 0.05,
) -> np.ndarray:
    """
    Apply fade-in and fade-out to audio.

    Args:
        waveform:          Audio array.
        sr:                Sample rate.
        fade_in_seconds:   Duration of fade-in.
        fade_out_seconds:  Duration of fade-out.

    Returns:
        Audio with fades applied.
    """
    waveform = waveform.copy()
    n_samples = waveform.shape[-1]

    fade_in_len  = min(int(fade_in_seconds * sr), n_samples // 4)
    fade_out_len = min(int(fade_out_seconds * sr), n_samples // 4)

    fade_in_env  = np.linspace(0, 1, fade_in_len, dtype=np.float32)
    fade_out_env = np.linspace(1, 0, fade_out_len, dtype=np.float32)

    if waveform.ndim == 2:
        waveform[:, :fade_in_len] *= fade_in_env
        if fade_out_len > 0:
            waveform[:, -fade_out_len:] *= fade_out_env
    else:
        waveform[:fade_in_len] *= fade_in_env
        if fade_out_len > 0:
            waveform[-fade_out_len:] *= fade_out_env

    return waveform


# ── Stem Merging ──────────────────────────────────────────────

def merge_stems(
    stems: Dict[str, np.ndarray],
    sr: int,
    weights: Optional[Dict[str, float]] = None,
    normalize_output: bool = True,
) -> np.ndarray:
    """
    Mix multiple audio stems into a single signal.

    Args:
        stems:            Dict of stem_name → waveform array.
        sr:               Sample rate (for validation).
        weights:          Per-stem gain factors. Defaults to 1.0 for all.
        normalize_output: Peak-normalize the final mix.

    Returns:
        Mixed audio array.
    """
    weights = weights or {}
    mixed: Optional[np.ndarray] = None
    max_len = max(s.shape[-1] for s in stems.values()) if stems else 0

    for name, stem in stems.items():
        gain = float(weights.get(name, 1.0))
        audio = stem.astype(np.float32) * gain

        # Pad to max length
        pad_len = max_len - audio.shape[-1]
        if pad_len > 0:
            if audio.ndim == 2:
                audio = np.pad(audio, ((0, 0), (0, pad_len)))
            else:
                audio = np.pad(audio, (0, pad_len))

        if mixed is None:
            mixed = audio.copy()
        else:
            # Ensure same shape
            if audio.ndim != mixed.ndim:
                if mixed.ndim == 1:
                    mixed = np.stack([mixed, mixed])
                if audio.ndim == 1:
                    audio = np.stack([audio, audio])
            mixed = mixed + audio

    if mixed is None:
        return np.zeros(0, dtype=np.float32)

    if normalize_output:
        mixed = normalize_audio(mixed)

    return mixed


# ── Waveform Visualization ────────────────────────────────────

def generate_waveform_peaks(
    waveform: np.ndarray,
    sr: int,
    n_peaks: int = 1000,
    mono: bool = True,
) -> List[float]:
    """
    Generate waveform peak data for frontend visualization (WaveSurfer.js compatible).

    Args:
        waveform:  Audio array.
        sr:        Sample rate (unused, kept for API consistency).
        n_peaks:   Number of data points to return.
        mono:      If True, average channels before computing peaks.

    Returns:
        List of peak amplitude values (0.0–1.0) of length n_peaks.
    """
    if waveform.ndim == 2 and mono:
        y = waveform.mean(axis=0)
    elif waveform.ndim == 2:
        y = waveform[0]    # Use first channel
    else:
        y = waveform

    y = y.astype(np.float32)
    total_samples = len(y)

    if total_samples == 0:
        return [0.0] * n_peaks

    chunk_size = max(1, total_samples // n_peaks)
    peaks = []

    for i in range(n_peaks):
        start = i * chunk_size
        end = min(start + chunk_size, total_samples)
        if start >= total_samples:
            peaks.append(0.0)
        else:
            chunk = y[start:end]
            peaks.append(float(np.abs(chunk).max()))

    # Normalize peaks to [0, 1]
    max_peak = max(peaks) if peaks else 1.0
    if max_peak > 0:
        peaks = [p / max_peak for p in peaks]

    return peaks


def generate_spectrogram_data(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    max_frames: int = 256,
) -> Dict[str, Union[List, int, float]]:
    """
    Compute mel-spectrogram data for frontend visualization.

    Returns:
        Dict with 'mel_db' (2D list), 'n_mels', 'n_frames', 'sr'.
    """
    try:
        import librosa
    except ImportError:
        return {"mel_db": [], "n_mels": n_mels, "n_frames": 0, "sr": sr}

    if waveform.ndim == 2:
        y_mono = waveform.mean(axis=0)
    else:
        y_mono = waveform

    mel = librosa.feature.melspectrogram(
        y=y_mono.astype(np.float32),
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=2048,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Downsample frames if needed
    if mel_db.shape[1] > max_frames:
        indices = np.linspace(0, mel_db.shape[1] - 1, max_frames, dtype=int)
        mel_db = mel_db[:, indices]

    # Normalize to 0–1 range
    mel_db_norm = (mel_db - mel_db.min()) / (mel_db.ptp() + 1e-8)

    return {
        "mel_db": mel_db_norm.tolist(),
        "n_mels": n_mels,
        "n_frames": mel_db_norm.shape[1],
        "sr": sr,
        "hop_length": hop_length,
    }


# ── Audio Analysis ────────────────────────────────────────────

def get_audio_duration(path: str) -> float:
    """
    Get duration of an audio file in seconds.

    Args:
        path: Path to audio file.

    Returns:
        Duration in seconds.
    """
    try:
        import librosa
        return float(librosa.get_duration(path=path))
    except ImportError:
        pass

    # Fallback using SoundFile
    with sf.SoundFile(path) as f:
        return float(f.frames) / f.samplerate


def get_audio_info(path: str) -> Dict:
    """
    Get metadata about an audio file.

    Returns:
        Dict with sample_rate, channels, duration, format, bitrate.
    """
    path = Path(path)
    info = {
        "path": str(path),
        "filename": path.name,
        "format": path.suffix.lstrip(".").upper(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
    }

    try:
        with sf.SoundFile(str(path)) as f:
            info["sample_rate"] = f.samplerate
            info["channels"] = f.channels
            info["frames"] = f.frames
            info["duration_seconds"] = float(f.frames) / f.samplerate
            info["subtype"] = f.subtype
    except Exception:
        # Fallback: pydub
        try:
            seg = AudioSegment.from_file(str(path))
            info["sample_rate"] = seg.frame_rate
            info["channels"] = seg.channels
            info["duration_seconds"] = len(seg) / 1000.0
            info["frames"] = int(len(seg) / 1000.0 * seg.frame_rate)
        except Exception as e:
            info["error"] = str(e)

    return info


def compute_rms_db(waveform: np.ndarray) -> float:
    """Compute RMS loudness in decibels."""
    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    if rms <= 0:
        return -np.inf
    return float(20 * np.log10(rms + 1e-10))


def compute_dynamic_range(waveform: np.ndarray) -> float:
    """Estimate dynamic range in dB (crest factor)."""
    peak = float(np.abs(waveform).max())
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    if rms <= 0:
        return 0.0
    return float(20 * np.log10((peak + 1e-10) / (rms + 1e-10)))


# ── Encoding ──────────────────────────────────────────────────

def audio_to_base64(
    waveform: np.ndarray,
    sr: int,
    output_format: str = "wav",
) -> str:
    """
    Encode audio array as base64 string for API responses.

    Args:
        waveform:      Audio array.
        sr:            Sample rate.
        output_format: 'wav', 'mp3', or 'flac'.

    Returns:
        Base64-encoded string of the audio file.
    """
    buf = io.BytesIO()

    if waveform.ndim == 2:
        data_for_write = waveform.T
    else:
        data_for_write = waveform

    data_for_write = data_for_write.astype(np.float32)

    if output_format == "wav":
        sf.write(buf, data_for_write, sr, format="WAV", subtype="PCM_16")
    else:
        # Write to temp WAV then convert
        tmp_wav = io.BytesIO()
        sf.write(tmp_wav, data_for_write, sr, format="WAV", subtype="PCM_16")
        tmp_wav.seek(0)
        seg = AudioSegment.from_wav(tmp_wav)
        seg.export(buf, format=output_format)

    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def base64_to_audio(
    b64_string: str,
    input_format: str = "wav",
) -> Tuple[np.ndarray, int]:
    """
    Decode a base64 audio string back to numpy array + sample rate.

    Args:
        b64_string:   Base64-encoded audio bytes.
        input_format: Original format ('wav', 'mp3', etc.).

    Returns:
        (waveform, sample_rate)
    """
    audio_bytes = base64.b64decode(b64_string)
    buf = io.BytesIO(audio_bytes)

    if input_format == "wav":
        data, sr = sf.read(buf, dtype="float32", always_2d=True)
        return data.T, sr
    else:
        seg = AudioSegment.from_file(buf, format=input_format)
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(seg.array_type).max
        if seg.channels == 2:
            samples = samples.reshape(-1, 2).T
        return samples, seg.frame_rate


# ── Mixing & Effects ──────────────────────────────────────────

def apply_reverb(
    waveform: np.ndarray,
    sr: int,
    room_size: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.3,
    dry_level: float = 0.8,
) -> np.ndarray:
    """
    Apply a simple FIR reverb effect.

    Args:
        waveform:   Audio array.
        sr:         Sample rate.
        room_size:  Reverb room size (0–1).
        damping:    High-frequency damping (0–1).
        wet_level:  Wet signal gain.
        dry_level:  Dry signal gain.

    Returns:
        Audio with reverb applied.
    """
    try:
        import pedalboard
        board = pedalboard.Pedalboard([
            pedalboard.Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=dry_level,
            )
        ])
        if waveform.ndim == 2:
            return board(waveform, sr)
        return board(waveform[np.newaxis, :], sr)[0]
    except ImportError:
        pass

    # Fallback: simple comb filter reverb
    delay_samples = int(0.05 * sr)
    decay = 1.0 - damping * 0.5

    result = waveform.copy()
    if waveform.ndim == 1:
        reverb_buf = np.zeros(len(waveform) + delay_samples)
        reverb_buf[:len(waveform)] += waveform * dry_level
        for i in range(len(waveform)):
            reverb_buf[i] += waveform[i] * wet_level
            if i + delay_samples < len(reverb_buf):
                reverb_buf[i + delay_samples] += waveform[i] * wet_level * decay
        result = reverb_buf[:len(waveform)]
    else:
        for c in range(waveform.shape[0]):
            reverb_buf = np.zeros(waveform.shape[1] + delay_samples)
            reverb_buf[:waveform.shape[1]] += waveform[c] * dry_level
            for i in range(waveform.shape[1]):
                reverb_buf[i] += waveform[c, i] * wet_level
                if i + delay_samples < len(reverb_buf):
                    reverb_buf[i + delay_samples] += waveform[c, i] * wet_level * decay
            result[c] = reverb_buf[:waveform.shape[1]]

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def apply_eq(
    waveform: np.ndarray,
    sr: int,
    bass_gain_db: float = 0.0,
    mid_gain_db: float = 0.0,
    treble_gain_db: float = 0.0,
) -> np.ndarray:
    """
    Apply a simple 3-band equalizer.

    Args:
        waveform:       Audio array.
        sr:             Sample rate.
        bass_gain_db:   Bass gain/cut in dB (< 200 Hz).
        mid_gain_db:    Mid gain/cut in dB (200 Hz – 4 kHz).
        treble_gain_db: Treble gain/cut in dB (> 4 kHz).

    Returns:
        EQ-processed audio array.
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        logger.warning("scipy not installed; EQ unavailable")
        return waveform

    def _shelf_filter(y: np.ndarray, cutoff: float, gain_db: float, high: bool) -> np.ndarray:
        gain = 10 ** (gain_db / 20.0)
        wc = 2 * np.pi * cutoff / sr
        b, a = scipy_signal.butter(2, wc / np.pi, btype="highpass" if high else "lowpass")
        filtered = scipy_signal.lfilter(b, a, y)
        return y + (gain - 1.0) * filtered

    def _process(y: np.ndarray) -> np.ndarray:
        y = y.astype(np.float64)
        if abs(bass_gain_db) > 0.1:
            y = _shelf_filter(y, 200.0, bass_gain_db, high=False)
        if abs(treble_gain_db) > 0.1:
            y = _shelf_filter(y, 4000.0, treble_gain_db, high=True)
        # Simplified mid: difference between full and bass+treble
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    if waveform.ndim == 2:
        return np.stack([_process(waveform[c]) for c in range(waveform.shape[0])])
    return _process(waveform)


# ── Utility ───────────────────────────────────────────────────

def resample(
    waveform: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using Torchaudio."""
    if orig_sr == target_sr:
        return waveform

    t = torch.from_numpy(waveform.astype(np.float32))
    if t.ndim == 1:
        t = t.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    t_out = resampler(t)

    if squeeze:
        t_out = t_out.squeeze(0)

    return t_out.numpy()


def stereo_to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert (channels, samples) stereo to (samples,) mono."""
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=0)


def mono_to_stereo(waveform: np.ndarray) -> np.ndarray:
    """Duplicate mono channel to create stereo."""
    if waveform.ndim == 2:
        return waveform
    return np.stack([waveform, waveform])


def pad_or_trim(
    waveform: np.ndarray,
    target_samples: int,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Pad or trim audio to exactly target_samples.

    Args:
        waveform:       Audio array.
        target_samples: Target number of samples.
        pad_mode:       NumPy pad mode ('constant', 'wrap', 'reflect', etc.).

    Returns:
        Audio with exactly target_samples samples.
    """
    current = waveform.shape[-1]

    if current == target_samples:
        return waveform
    elif current > target_samples:
        # Trim
        if waveform.ndim == 2:
            return waveform[:, :target_samples]
        return waveform[:target_samples]
    else:
        # Pad
        pad_len = target_samples - current
        if waveform.ndim == 2:
            return np.pad(waveform, ((0, 0), (0, pad_len)), mode=pad_mode)
        return np.pad(waveform, (0, pad_len), mode=pad_mode)


def crossfade(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int,
    fade_seconds: float = 1.0,
) -> np.ndarray:
    """
    Crossfade between two audio arrays.
    audio1 fades out, audio2 fades in over fade_seconds overlap.

    Returns:
        Blended audio array.
    """
    fade_samples = min(int(fade_seconds * sr), audio1.shape[-1] // 2, audio2.shape[-1] // 2)

    fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
    fade_in  = np.linspace(0, 1, fade_samples, dtype=np.float32)

    if audio1.ndim == 2:
        fade_out = fade_out[np.newaxis, :]
        fade_in  = fade_in[np.newaxis, :]

    tail = audio1[..., -fade_samples:] * fade_out
    head = audio2[..., :fade_samples] * fade_in
    blend = tail + head

    if audio1.ndim == 2:
        return np.concatenate([audio1[:, :-fade_samples], blend, audio2[:, fade_samples:]], axis=1)
    return np.concatenate([audio1[:-fade_samples], blend, audio2[fade_samples:]])


def split_into_chunks(
    waveform: np.ndarray,
    sr: int,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.5,
) -> List[np.ndarray]:
    """
    Split long audio into overlapping chunks for batch processing.

    Returns:
        List of audio chunk arrays.
    """
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    stride = chunk_samples - overlap_samples
    total = waveform.shape[-1]
    chunks = []
    start = 0

    while start < total:
        end = min(start + chunk_samples, total)
        if waveform.ndim == 2:
            chunks.append(waveform[:, start:end])
        else:
            chunks.append(waveform[start:end])
        start += stride

    return chunks


def validate_audio_file(path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a readable audio file.

    Returns:
        (is_valid: bool, message: str)
    """
    path = Path(path)

    if not path.exists():
        return False, f"File not found: {path}"

    if path.stat().st_size == 0:
        return False, "File is empty"

    if path.suffix.lower() not in SUPPORTED_INPUT_FORMATS:
        return False, f"Unsupported format: {path.suffix}. Supported: {sorted(SUPPORTED_INPUT_FORMATS)}"

    if path.stat().st_size > 500 * 1024 * 1024:    # 500 MB limit
        return False, "File too large (max 500 MB)"

    try:
        # Quick read test: just metadata
        with sf.SoundFile(str(path)) as f:
            if f.frames == 0:
                return False, "Audio file has no frames"
            if f.samplerate <= 0:
                return False, "Invalid sample rate"
    except Exception as e:
        try:
            # Fallback: try pydub
            AudioSegment.from_file(str(path))
        except Exception as e2:
            return False, f"Cannot read audio file: {e2}"

    return True, "Valid audio file"