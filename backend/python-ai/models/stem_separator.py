# ============================================================
# models/stem_separator.py
# AI Music Remix & Mood Generator — Stem Separator Model
# ============================================================
# Separates audio into stems: vocals, drums, bass, other
# Primary: Meta Demucs (htdemucs)  Fallback: Spleeter 4stems
# ============================================================

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment

logger = logging.getLogger(__name__)


# ── Enumerations ────────────────────────────────────────────

class SeparationBackend(str, Enum):
    DEMUCS = "demucs"
    SPLEETER = "spleeter"


class StemType(str, Enum):
    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    PIANO = "piano"       # Demucs 6-stem model
    GUITAR = "guitar"     # Demucs 6-stem model


# ── Data Classes ────────────────────────────────────────────

@dataclass
class StemResult:
    """Holds separated stem data and metadata."""
    stem_type: StemType
    waveform: np.ndarray          # shape: (channels, samples)
    sample_rate: int
    duration_seconds: float
    output_path: Optional[str] = None
    peak_amplitude: float = 0.0
    rms_energy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stem_type": self.stem_type.value,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "output_path": self.output_path,
            "peak_amplitude": round(float(self.peak_amplitude), 6),
            "rms_energy": round(float(self.rms_energy), 6),
            "channels": self.waveform.shape[0] if self.waveform.ndim > 1 else 1,
        }


@dataclass
class SeparationResult:
    """Container for all separated stems."""
    stems: Dict[str, StemResult] = field(default_factory=dict)
    backend_used: SeparationBackend = SeparationBackend.DEMUCS
    model_name: str = "htdemucs"
    processing_time_seconds: float = 0.0
    input_duration_seconds: float = 0.0
    sample_rate: int = 44100
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and len(self.stems) > 0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "backend_used": self.backend_used.value,
            "model_name": self.model_name,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "input_duration_seconds": round(self.input_duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "stems": {k: v.to_dict() for k, v in self.stems.items()},
            "error": self.error,
        }


# ── Stem Separator ──────────────────────────────────────────

class StemSeparator:
    """
    Multi-backend stem separator.

    Supports:
    - Meta Demucs (htdemucs, htdemucs_ft, htdemucs_6s)
    - Deezer Spleeter (4stems, 5stems)

    Usage:
        sep = StemSeparator(backend="demucs", model="htdemucs")
        result = sep.separate("/path/to/song.mp3", output_dir="/tmp/stems")
        vocals_np = result.stems["vocals"].waveform
    """

    # Map model names → expected stem names
    MODEL_STEM_MAP: Dict[str, List[str]] = {
        "htdemucs":    ["vocals", "drums", "bass", "other"],
        "htdemucs_ft": ["vocals", "drums", "bass", "other"],
        "htdemucs_6s": ["vocals", "drums", "bass", "other", "piano", "guitar"],
        "mdx":         ["vocals", "drums", "bass", "other"],
        "mdx_extra":   ["vocals", "drums", "bass", "other"],
        "spleeter:4stems": ["vocals", "drums", "bass", "other"],
        "spleeter:5stems": ["vocals", "drums", "bass", "other", "piano"],
    }

    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_SEGMENT_LENGTH = 7.8          # seconds – Demucs default chunk
    SUPPORTED_INPUT_FORMATS = {
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff"
    }

    def __init__(
        self,
        backend: str = "demucs",
        model: str = "htdemucs",
        device: Optional[str] = None,
        segment: float = DEFAULT_SEGMENT_LENGTH,
        overlap: float = 0.25,
        shifts: int = 1,
        clip_mode: str = "rescale",
        jobs: int = 0,
    ):
        self.backend = SeparationBackend(backend)
        self.model_name = model
        self.segment = segment
        self.overlap = overlap
        self.shifts = shifts
        self.clip_mode = clip_mode
        self.jobs = jobs or os.cpu_count() or 1

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._demucs_model = None      # lazy-loaded
        self._spleeter_separator = None

        logger.info(
            "StemSeparator initialized | backend=%s | model=%s | device=%s",
            self.backend, self.model_name, self.device,
        )

    # ── Public API ─────────────────────────────────────────

    def separate(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
        stems_to_extract: Optional[List[str]] = None,
        output_format: str = "wav",
        normalize_stems: bool = True,
    ) -> SeparationResult:
        """
        Separate audio file into individual stems.

        Args:
            input_path:       Path to input audio file.
            output_dir:       Where to save stem files. Uses tempdir if None.
            stems_to_extract: Subset of stems to return. None = all.
            output_format:    'wav', 'mp3', or 'flac'.
            normalize_stems:  Normalize peak amplitude of each stem.

        Returns:
            SeparationResult with per-stem numpy arrays and file paths.
        """
        start_time = time.monotonic()
        result = SeparationResult(
            backend_used=self.backend,
            model_name=self.model_name,
        )

        # ── Validate input ──────────────────────────────────
        input_path = Path(input_path)
        if not input_path.exists():
            result.error = f"Input file not found: {input_path}"
            logger.error(result.error)
            return result

        if input_path.suffix.lower() not in self.SUPPORTED_INPUT_FORMATS:
            result.error = (
                f"Unsupported format: {input_path.suffix}. "
                f"Supported: {self.SUPPORTED_INPUT_FORMATS}"
            )
            logger.error(result.error)
            return result

        # ── Prepare output directory ────────────────────────
        own_tempdir = False
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="stems_")
            own_tempdir = True
        else:
            os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)

        try:
            # ── Convert to WAV if necessary ─────────────────
            wav_path = self._ensure_wav(input_path, output_dir)
            waveform, sr = torchaudio.load(str(wav_path))
            result.input_duration_seconds = waveform.shape[-1] / sr
            result.sample_rate = sr

            # ── Run separation ──────────────────────────────
            if self.backend == SeparationBackend.DEMUCS:
                stem_paths = self._separate_demucs(wav_path, output_dir)
            else:
                stem_paths = self._separate_spleeter(wav_path, output_dir)

            # ── Load & post-process stems ───────────────────
            expected_stems = self.MODEL_STEM_MAP.get(self.model_name, ["vocals", "drums", "bass", "other"])

            for stem_name, stem_path in stem_paths.items():
                if stems_to_extract and stem_name not in stems_to_extract:
                    continue

                audio_np, stem_sr = self._load_stem_file(stem_path, sr)

                if normalize_stems:
                    audio_np = self._normalize(audio_np)

                # Save in requested format
                out_filename = f"{stem_name}.{output_format}"
                final_path = output_dir / out_filename
                self._save_stem(audio_np, stem_sr, final_path, output_format)

                duration = audio_np.shape[-1] / stem_sr
                peak = float(np.abs(audio_np).max())
                rms = float(np.sqrt(np.mean(audio_np ** 2)))

                result.stems[stem_name] = StemResult(
                    stem_type=StemType(stem_name) if stem_name in StemType._value2member_map_ else StemType.OTHER,
                    waveform=audio_np,
                    sample_rate=stem_sr,
                    duration_seconds=duration,
                    output_path=str(final_path),
                    peak_amplitude=peak,
                    rms_energy=rms,
                )

            result.processing_time_seconds = time.monotonic() - start_time
            logger.info(
                "Separation complete | stems=%d | time=%.2fs",
                len(result.stems),
                result.processing_time_seconds,
            )

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Stem separation failed: %s", exc)
        finally:
            # Clean temp WAV conversion (not the output dir)
            if wav_path != input_path and wav_path.exists():
                wav_path.unlink(missing_ok=True)

        return result

    def separate_from_bytes(
        self,
        audio_bytes: bytes,
        original_filename: str = "upload.mp3",
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> SeparationResult:
        """
        Convenience method for API endpoints that receive raw bytes.
        Writes bytes to a temp file, then calls separate().
        """
        suffix = Path(original_filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.separate(tmp_path, output_dir=output_dir, **kwargs)
        finally:
            os.unlink(tmp_path)

    def get_stem_names(self) -> List[str]:
        """Return stem names for the current model."""
        return self.MODEL_STEM_MAP.get(self.model_name, ["vocals", "drums", "bass", "other"])

    def is_gpu_available(self) -> bool:
        return torch.cuda.is_available()

    # ── Demucs Backend ─────────────────────────────────────

    def _load_demucs_model(self):
        """Lazy-load Demucs model into memory."""
        if self._demucs_model is not None:
            return self._demucs_model

        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            logger.info("Loading Demucs model: %s", self.model_name)
            model = get_model(self.model_name)
            model.eval()
            if self.device == "cuda":
                model.cuda()
            self._demucs_model = model
            self._demucs_apply = apply_model
            logger.info("Demucs model loaded successfully on %s", self.device)
            return self._demucs_model
        except ImportError:
            raise RuntimeError(
                "Demucs not installed. Run: pip install demucs"
            )

    def _separate_demucs(
        self,
        wav_path: Path,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Run Demucs separation using Python API.
        Falls back to CLI subprocess if Python API fails.
        """
        try:
            return self._separate_demucs_python(wav_path, output_dir)
        except Exception as py_err:
            logger.warning("Demucs Python API failed (%s), trying CLI...", py_err)
            return self._separate_demucs_cli(wav_path, output_dir)

    def _separate_demucs_python(
        self,
        wav_path: Path,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Demucs via Python API (preferred – no subprocess overhead)."""
        from demucs.audio import AudioFile, save_audio as demucs_save
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        model = self._load_demucs_model()

        # Load audio with Demucs utilities
        audio_file = AudioFile(wav_path)
        wav = audio_file.read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels,
        )
        wav = wav.to(self.device)

        # Apply model
        with torch.no_grad():
            sources = apply_model(
                model,
                wav[None],                  # add batch dimension
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=False,
                num_workers=self.jobs,
            )[0]                            # remove batch dimension

        # Save each stem
        stem_paths: Dict[str, Path] = {}
        for i, stem_name in enumerate(model.sources):
            stem_audio = sources[i].cpu()   # (channels, samples)
            out_path = output_dir / f"{stem_name}_raw.wav"
            demucs_save(stem_audio, str(out_path), samplerate=model.samplerate)
            stem_paths[stem_name] = out_path

        return stem_paths

    def _separate_demucs_cli(
        self,
        wav_path: Path,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Demucs via CLI subprocess (fallback)."""
        cmd = [
            "python", "-m", "demucs",
            "--name", self.model_name,
            "--out", str(output_dir),
            "--device", self.device,
            "--jobs", str(self.jobs),
            "--segment", str(self.segment),
            "--overlap", str(self.overlap),
            "--shifts", str(self.shifts),
            "--clip-mode", self.clip_mode,
            str(wav_path),
        ]

        logger.debug("Demucs CLI: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            raise RuntimeError(f"Demucs CLI failed:\n{proc.stderr}")

        # Demucs writes: output_dir/{model_name}/{track_name}/{stem}.wav
        track_name = wav_path.stem
        stems_dir = output_dir / self.model_name / track_name

        stem_paths: Dict[str, Path] = {}
        for stem_name in self.get_stem_names():
            p = stems_dir / f"{stem_name}.wav"
            if p.exists():
                stem_paths[stem_name] = p

        if not stem_paths:
            raise RuntimeError(
                f"Demucs produced no stems. Expected in: {stems_dir}"
            )

        return stem_paths

    # ── Spleeter Backend ────────────────────────────────────

    def _separate_spleeter(
        self,
        wav_path: Path,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Spleeter separation via Python API."""
        try:
            from spleeter.separator import Separator as SpleeterSep
        except ImportError:
            raise RuntimeError(
                "Spleeter not installed. Run: pip install spleeter"
            )

        config_map = {
            "spleeter:4stems": "spleeter:4stems",
            "spleeter:5stems": "spleeter:5stems",
        }
        config = config_map.get(self.model_name, "spleeter:4stems")

        if self._spleeter_separator is None:
            logger.info("Loading Spleeter: %s", config)
            self._spleeter_separator = SpleeterSep(config)

        self._spleeter_separator.separate_to_file(
            str(wav_path),
            str(output_dir),
        )

        # Spleeter writes: output_dir/{track_name}/{stem}.wav
        track_name = wav_path.stem
        stems_dir = output_dir / track_name

        stem_names = ["vocals", "drums", "bass", "other"]
        if "5stems" in config:
            stem_names.append("piano")

        stem_paths: Dict[str, Path] = {}
        for stem_name in stem_names:
            p = stems_dir / f"{stem_name}.wav"
            if p.exists():
                stem_paths[stem_name] = p

        return stem_paths

    # ── Internal Helpers ────────────────────────────────────

    def _ensure_wav(self, input_path: Path, output_dir: Path) -> Path:
        """Convert non-WAV files to WAV for processing."""
        if input_path.suffix.lower() == ".wav":
            return input_path

        wav_path = output_dir / f"{input_path.stem}_converted.wav"
        logger.info("Converting %s → %s", input_path, wav_path)

        audio = AudioSegment.from_file(str(input_path))
        audio = audio.set_frame_rate(self.DEFAULT_SAMPLE_RATE)
        audio = audio.set_channels(2)
        audio.export(str(wav_path), format="wav")
        return wav_path

    def _load_stem_file(
        self,
        stem_path: Path,
        target_sr: int,
    ) -> Tuple[np.ndarray, int]:
        """Load a stem WAV file as numpy array."""
        data, sr = sf.read(str(stem_path), dtype="float32", always_2d=True)
        # sf.read → (samples, channels); transpose → (channels, samples)
        data = data.T

        if sr != target_sr:
            waveform_t = torch.from_numpy(data)
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            data = resampler(waveform_t).numpy()
            sr = target_sr

        return data, sr

    def _normalize(self, audio: np.ndarray, target_peak: float = 0.99) -> np.ndarray:
        """Peak-normalize audio array."""
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio * (target_peak / peak)
        return audio

    def _save_stem(
        self,
        audio: np.ndarray,
        sr: int,
        out_path: Path,
        fmt: str,
    ) -> None:
        """Save stem numpy array to file in the requested format."""
        # sf.write expects (samples, channels)
        data_for_write = audio.T if audio.ndim == 2 else audio

        if fmt == "wav":
            sf.write(str(out_path), data_for_write, sr, subtype="PCM_24")
        elif fmt in ("mp3", "flac"):
            tmp_wav = out_path.with_suffix(".tmp.wav")
            sf.write(str(tmp_wav), data_for_write, sr, subtype="PCM_16")
            seg = AudioSegment.from_wav(str(tmp_wav))
            seg.export(str(out_path), format=fmt)
            tmp_wav.unlink(missing_ok=True)
        else:
            sf.write(str(out_path), data_for_write, sr)

    # ── Waveform Analysis ───────────────────────────────────

    @staticmethod
    def compute_stem_features(stem_result: StemResult) -> dict:
        """
        Compute basic spectral features for a separated stem.
        Used for quality metrics and downstream mood classification.
        """
        try:
            import librosa
        except ImportError:
            return {}

        y = stem_result.waveform
        sr = stem_result.sample_rate

        # Convert stereo to mono for librosa
        if y.ndim == 2:
            y_mono = y.mean(axis=0)
        else:
            y_mono = y

        y_mono = y_mono.astype(np.float32)

        features = {}

        # RMS energy
        features["rms"] = float(np.sqrt(np.mean(y_mono ** 2)))

        # Spectral centroid (brightness)
        sc = librosa.feature.spectral_centroid(y=y_mono, sr=sr)
        features["spectral_centroid_mean"] = float(sc.mean())
        features["spectral_centroid_std"] = float(sc.std())

        # Spectral rolloff
        sr_feat = librosa.feature.spectral_rolloff(y=y_mono, sr=sr, roll_percent=0.85)
        features["spectral_rolloff_mean"] = float(sr_feat.mean())

        # Zero crossing rate (percussion indicator)
        zcr = librosa.feature.zero_crossing_rate(y_mono)
        features["zcr_mean"] = float(zcr.mean())

        # Tempo estimate
        try:
            tempo, _ = librosa.beat.beat_track(y=y_mono, sr=sr)
            features["estimated_tempo"] = float(tempo)
        except Exception:
            features["estimated_tempo"] = 0.0

        return features

    # ── Stem Manipulation ───────────────────────────────────

    @staticmethod
    def mute_stem(result: SeparationResult, stem_name: str) -> SeparationResult:
        """Zero out a specific stem (mute it in the mix)."""
        if stem_name in result.stems:
            stem = result.stems[stem_name]
            result.stems[stem_name] = StemResult(
                stem_type=stem.stem_type,
                waveform=np.zeros_like(stem.waveform),
                sample_rate=stem.sample_rate,
                duration_seconds=stem.duration_seconds,
                output_path=stem.output_path,
            )
        return result

    @staticmethod
    def mix_stems(
        result: SeparationResult,
        volumes: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Mix separated stems back together with optional per-stem volume.

        Args:
            result:  SeparationResult from separate().
            volumes: Dict of stem_name → gain factor (0.0–2.0).
                     Missing stems default to 1.0.

        Returns:
            Mixed audio as numpy array (channels, samples).
        """
        volumes = volumes or {}
        mixed = None

        for stem_name, stem_data in result.stems.items():
            gain = volumes.get(stem_name, 1.0)
            audio = stem_data.waveform * gain

            if mixed is None:
                mixed = audio.copy()
            else:
                # Pad shorter array to match
                if audio.shape[-1] < mixed.shape[-1]:
                    pad = mixed.shape[-1] - audio.shape[-1]
                    audio = np.pad(audio, ((0, 0), (0, pad)) if audio.ndim == 2 else (0, pad))
                elif audio.shape[-1] > mixed.shape[-1]:
                    pad = audio.shape[-1] - mixed.shape[-1]
                    mixed = np.pad(mixed, ((0, 0), (0, pad)) if mixed.ndim == 2 else (0, pad))
                mixed += audio

        if mixed is None:
            return np.zeros((2, 0), dtype=np.float32)

        # Prevent clipping
        peak = np.abs(mixed).max()
        if peak > 1.0:
            mixed /= peak

        return mixed

    def __repr__(self) -> str:
        return (
            f"StemSeparator(backend={self.backend.value!r}, "
            f"model={self.model_name!r}, device={self.device!r})"
        )