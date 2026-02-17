# ============================================================
# models/music_generator.py
# AI Music Remix & Mood Generator — Music Generator Model
# ============================================================
# Generates new music from text prompts, remixes audio into
# new genres, and applies tempo/pitch transformations.
#
# Primary generation backend:   Meta MusicGen (audiocraft)
# Secondary:                    Riffusion (diffusers)
# Fallback:                     Torchaudio synthesis
# ============================================================

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


# ── Enumerations ─────────────────────────────────────────────

class GenerationBackend(str, Enum):
    MUSICGEN  = "musicgen"
    RIFFUSION = "riffusion"
    TORCHAUDIO = "torchaudio"   # fallback synthesis


class MusicGenModel(str, Enum):
    SMALL   = "facebook/musicgen-small"      # 300M params — fast
    MEDIUM  = "facebook/musicgen-medium"     # 1.5B params — quality
    LARGE   = "facebook/musicgen-large"      # 3.3B params — best
    STEREO  = "facebook/musicgen-stereo-small"
    MELODY  = "facebook/musicgen-melody"     # melody conditioning

class RemixGenre(str, Enum):
    EDM           = "edm"
    HOUSE         = "house"
    TECHNO        = "techno"
    DRUM_AND_BASS = "drum_and_bass"
    POP           = "pop"
    HIP_HOP       = "hip_hop"
    JAZZ          = "jazz"
    CLASSICAL     = "classical"
    ROCK          = "rock"
    REGGAE        = "reggae"
    FUNK          = "funk"
    AMBIENT       = "ambient"
    LO_FI         = "lo_fi"
    METAL         = "metal"
    RNB           = "rnb"
    BOSSA_NOVA    = "bossa_nova"


# Genre → text prompt augmentation
GENRE_PROMPT_MAP: Dict[str, str] = {
    RemixGenre.EDM:           "electronic dance music, synthesizer, 128 BPM, bass drop, euphoric",
    RemixGenre.HOUSE:         "deep house music, four-on-the-floor kick, warm bassline, soulful",
    RemixGenre.TECHNO:        "dark techno, industrial, driving 130 BPM, hypnotic repetition",
    RemixGenre.DRUM_AND_BASS: "drum and bass, fast breakbeats 174 BPM, heavy bass, jungle",
    RemixGenre.POP:           "catchy pop music, upbeat, radio-friendly, polished production",
    RemixGenre.HIP_HOP:       "hip hop beat, trap drums, 808 bass, urban, rhythmic flow",
    RemixGenre.JAZZ:          "smooth jazz, saxophone, piano trio, walking bass, swing feel",
    RemixGenre.CLASSICAL:     "orchestral classical music, strings, piano, symphonic, cinematic",
    RemixGenre.ROCK:          "electric guitar rock, distorted riffs, powerful drums, anthem",
    RemixGenre.REGGAE:        "reggae, offbeat guitar skank, bass-heavy, relaxed groove, roots",
    RemixGenre.FUNK:          "funky groove, slap bass, brass section, wah guitar, danceable",
    RemixGenre.AMBIENT:       "ambient soundscape, ethereal pads, slow evolving textures, peaceful",
    RemixGenre.LO_FI:         "lo-fi hip hop, dusty vinyl crackle, relaxed beats, mellow chords",
    RemixGenre.METAL:         "heavy metal, distorted guitar, double kick drums, aggressive, loud",
    RemixGenre.RNB:           "R&B soul, smooth vocals, lush chords, sensual groove, neo-soul",
    RemixGenre.BOSSA_NOVA:    "bossa nova, nylon guitar, soft samba rhythm, intimate, Brazilian",
}

# Mood → generation prompt template
MOOD_PROMPT_MAP: Dict[str, str] = {
    "energetic":   "high energy, driving rhythm, powerful, exciting, uplifting",
    "happy":       "joyful, bright, cheerful, major key, sunny, positive",
    "calm":        "peaceful, relaxing, gentle, serene, meditative, soft",
    "melancholic": "sad, emotional, minor key, longing, introspective, tender",
    "tense":       "suspenseful, dark, building tension, dramatic, unsettling",
    "angry":       "aggressive, intense, raw, powerful, fierce, heavy",
    "romantic":    "romantic, intimate, warm, loving, lush, tender",
    "euphoric":    "euphoric, transcendent, soaring, triumphant, uplifting",
    "dark":        "dark, brooding, mysterious, heavy, ominous, deep",
    "dreamy":      "dreamy, ethereal, hazy, soft focus, floating, hypnotic",
}


# ── Data Classes ─────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Configuration for a music generation request."""
    prompt: str
    duration_seconds: float = 8.0
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    cfg_coef: float = 3.0               # classifier-free guidance
    sample_rate: int = 32000            # MusicGen default
    two_step_cfg: bool = False
    extend_stride: float = 18.0

    # Optional conditioning
    melody_audio: Optional[np.ndarray] = None
    melody_sr: Optional[int] = None
    continuation_audio: Optional[np.ndarray] = None
    continuation_sr: Optional[int] = None

    # Genre / Mood overrides
    genre: Optional[str] = None
    mood: Optional[str] = None

    def build_prompt(self) -> str:
        """Build enriched prompt from genre + mood + base prompt."""
        parts = [self.prompt]
        if self.genre and self.genre in GENRE_PROMPT_MAP:
            parts.insert(0, GENRE_PROMPT_MAP[self.genre])
        if self.mood and self.mood in MOOD_PROMPT_MAP:
            parts.append(MOOD_PROMPT_MAP[self.mood])
        return ", ".join(filter(None, parts))


@dataclass
class GenerationResult:
    """Result of a music generation operation."""
    audio: Optional[np.ndarray] = None     # (samples,) or (channels, samples)
    sample_rate: int = 32000
    duration_seconds: float = 0.0
    output_path: Optional[str] = None
    backend_used: GenerationBackend = GenerationBackend.MUSICGEN
    model_name: str = ""
    prompt_used: str = ""
    processing_time_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.audio is not None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 3),
            "output_path": self.output_path,
            "backend_used": self.backend_used.value,
            "model_name": self.model_name,
            "prompt_used": self.prompt_used,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "error": self.error,
        }


@dataclass
class RemixResult:
    """Result of a genre remix operation."""
    original_path: str = ""
    remixed_audio: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duration_seconds: float = 0.0
    target_genre: str = ""
    output_path: Optional[str] = None
    tempo_adjusted: bool = False
    pitch_adjusted: bool = False
    processing_time_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.remixed_audio is not None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "original_path": self.original_path,
            "target_genre": self.target_genre,
            "duration_seconds": round(self.duration_seconds, 3),
            "output_path": self.output_path,
            "tempo_adjusted": self.tempo_adjusted,
            "pitch_adjusted": self.pitch_adjusted,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "error": self.error,
        }


# ── Music Generator ───────────────────────────────────────────

class MusicGenerator:
    """
    Multi-backend AI music generator.

    Capabilities:
    1. Text-to-music generation (MusicGen / Riffusion)
    2. Genre-conditioned remix of existing audio
    3. Melody-conditioned generation
    4. Tempo and pitch manipulation
    5. Audio continuation / extension

    Usage:
        gen = MusicGenerator(backend="musicgen", model="facebook/musicgen-small")
        result = gen.generate_from_text("relaxing piano jazz, 80 BPM", duration=15.0)
        remix = gen.remix_to_genre("/path/to/song.mp3", genre="lo_fi")
    """

    SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3", "flac", "ogg"}
    MAX_MUSICGEN_DURATION = 30.0           # seconds (MusicGen limitation per call)
    DEFAULT_SR_MUSICGEN = 32000
    DEFAULT_SR_RIFFUSION = 22050

    def __init__(
        self,
        backend: str = "musicgen",
        model: str = MusicGenModel.SMALL,
        device: Optional[str] = None,
        use_float16: bool = False,
    ):
        self.backend = GenerationBackend(backend)
        self.model_name = model
        self.use_float16 = use_float16

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Lazy-loaded model references
        self._musicgen_model = None
        self._musicgen_processor = None
        self._riffusion_pipeline = None

        logger.info(
            "MusicGenerator initialized | backend=%s | model=%s | device=%s",
            self.backend, self.model_name, self.device,
        )

    # ── Public API ─────────────────────────────────────────

    def generate_from_text(
        self,
        prompt: str,
        duration: float = 8.0,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 250,
        cfg_coef: float = 3.0,
        output_path: Optional[str] = None,
        output_format: str = "wav",
    ) -> GenerationResult:
        """
        Generate music from a text description.

        Args:
            prompt:        Natural language description of desired music.
            duration:      Target duration in seconds (max 30 for MusicGen).
            genre:         Optional genre key from RemixGenre enum.
            mood:          Optional mood key for prompt augmentation.
            temperature:   Sampling temperature (higher = more creative).
            top_k:         Top-k sampling parameter.
            cfg_coef:      Classifier-free guidance coefficient.
            output_path:   Where to save generated audio. Auto-temp if None.
            output_format: 'wav', 'mp3', or 'flac'.

        Returns:
            GenerationResult with audio array and file path.
        """
        start_time = time.monotonic()
        config = GenerationConfig(
            prompt=prompt,
            duration_seconds=min(duration, self.MAX_MUSICGEN_DURATION),
            temperature=temperature,
            top_k=top_k,
            cfg_coef=cfg_coef,
            genre=genre,
            mood=mood,
        )
        enriched_prompt = config.build_prompt()
        logger.info("Generating music | prompt=%r | duration=%.1fs", enriched_prompt, config.duration_seconds)

        result = GenerationResult(
            backend_used=self.backend,
            model_name=self.model_name,
            prompt_used=enriched_prompt,
        )

        try:
            if self.backend == GenerationBackend.MUSICGEN:
                audio, sr = self._generate_musicgen(config)
            elif self.backend == GenerationBackend.RIFFUSION:
                audio, sr = self._generate_riffusion(config)
            else:
                audio, sr = self._generate_fallback(config)

            result.audio = audio
            result.sample_rate = sr
            result.duration_seconds = len(audio.flatten()) / sr

            # Save to file
            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(audio, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Generation failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    def generate_from_melody(
        self,
        prompt: str,
        melody_path: str,
        duration: float = 15.0,
        output_path: Optional[str] = None,
        output_format: str = "wav",
    ) -> GenerationResult:
        """
        Generate music conditioned on a melody (MusicGen-Melody).

        Args:
            prompt:       Text description of desired style.
            melody_path:  Path to melody audio file (WAV/MP3).
            duration:     Duration of generated audio.
        """
        start_time = time.monotonic()
        result = GenerationResult(
            backend_used=GenerationBackend.MUSICGEN,
            model_name="facebook/musicgen-melody",
            prompt_used=prompt,
        )

        try:
            # Load melody
            melody_wav, melody_sr = torchaudio.load(melody_path)
            melody_np = melody_wav.numpy()

            config = GenerationConfig(
                prompt=prompt,
                duration_seconds=min(duration, self.MAX_MUSICGEN_DURATION),
                melody_audio=melody_np,
                melody_sr=melody_sr,
            )

            # Ensure melody model is loaded
            original_model = self.model_name
            self.model_name = MusicGenModel.MELODY
            audio, sr = self._generate_musicgen(config)
            self.model_name = original_model

            result.audio = audio
            result.sample_rate = sr
            result.duration_seconds = audio.flatten().shape[0] / sr

            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(audio, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Melody generation failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    def generate_long(
        self,
        prompt: str,
        total_duration: float = 60.0,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        output_path: Optional[str] = None,
        output_format: str = "wav",
    ) -> GenerationResult:
        """
        Generate audio longer than MusicGen's 30-second limit.
        Uses sliding window continuation approach.
        """
        chunk_duration = 20.0
        overlap = 3.0
        chunks: List[np.ndarray] = []
        sr = self.DEFAULT_SR_MUSICGEN

        start_time = time.monotonic()
        result = GenerationResult(
            backend_used=self.backend,
            model_name=self.model_name,
            prompt_used=prompt,
        )

        try:
            n_chunks = int(np.ceil(total_duration / (chunk_duration - overlap)))

            for i in range(n_chunks):
                is_last = (i == n_chunks - 1)
                dur = min(chunk_duration, total_duration - i * (chunk_duration - overlap))

                config = GenerationConfig(
                    prompt=prompt,
                    duration_seconds=dur,
                    genre=genre,
                    mood=mood,
                )

                if i > 0 and chunks:
                    # Use tail of previous chunk as continuation context
                    tail_samples = int(overlap * sr)
                    config.continuation_audio = chunks[-1][:, -tail_samples:]
                    config.continuation_sr = sr

                chunk_audio, sr = self._generate_musicgen(config)
                chunks.append(chunk_audio)

                if is_last:
                    break

            # Crossfade and concatenate chunks
            full_audio = self._crossfade_chunks(chunks, sr, overlap)

            result.audio = full_audio
            result.sample_rate = sr
            result.duration_seconds = full_audio.flatten().shape[0] / sr

            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(full_audio, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Long generation failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    def remix_to_genre(
        self,
        input_path: str,
        genre: str,
        preserve_tempo: bool = True,
        duration_seconds: Optional[float] = None,
        output_path: Optional[str] = None,
        output_format: str = "wav",
        stems: Optional[Dict[str, np.ndarray]] = None,
        stems_sr: Optional[int] = None,
    ) -> RemixResult:
        """
        Remix an existing track into a target genre.

        Strategy:
        1. Extract tempo from original
        2. Build genre-conditioned prompt
        3. Generate new music using MusicGen
        4. Optionally mix in original stems (e.g., vocals)

        Args:
            input_path:       Original audio file path.
            genre:            Target genre (RemixGenre enum value).
            preserve_tempo:   Use original track's tempo in generation.
            duration_seconds: Output duration (defaults to input duration).
            stems:            Pre-separated stems to blend in (e.g., vocals).
            stems_sr:         Sample rate of provided stems.
        """
        start_time = time.monotonic()
        result = RemixResult(
            original_path=input_path,
            target_genre=genre,
        )

        try:
            import librosa

            # ── Analyze original ────────────────────────────
            y_orig, sr_orig = librosa.load(input_path, sr=None, mono=True)
            orig_duration = len(y_orig) / sr_orig

            if duration_seconds is None:
                duration_seconds = min(orig_duration, self.MAX_MUSICGEN_DURATION)

            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=y_orig, sr=sr_orig)
            tempo_str = f"{int(round(float(tempo)))} BPM"

            # ── Build prompt ────────────────────────────────
            genre_desc = GENRE_PROMPT_MAP.get(genre, genre)
            prompt_parts = [genre_desc]
            if preserve_tempo:
                prompt_parts.append(tempo_str)
            prompt = ", ".join(prompt_parts)

            logger.info("Remixing | genre=%s | tempo=%s | duration=%.1fs", genre, tempo_str, duration_seconds)

            # ── Generate new music ──────────────────────────
            config = GenerationConfig(
                prompt=prompt,
                duration_seconds=duration_seconds,
                genre=genre,
            )

            if self.backend == GenerationBackend.MUSICGEN:
                audio, sr = self._generate_musicgen(config)
            else:
                audio, sr = self._generate_riffusion(config)

            # ── Blend original vocals if stems provided ─────
            if stems and "vocals" in stems and stems_sr:
                vocals = stems["vocals"]
                if vocals.ndim == 2:
                    vocals_mono = vocals.mean(axis=0)
                else:
                    vocals_mono = vocals

                # Resample vocals to match generated audio SR
                if stems_sr != sr:
                    vocals_t = torch.from_numpy(vocals_mono).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(stems_sr, sr)
                    vocals_mono = resampler(vocals_t).squeeze(0).numpy()

                # Trim or pad to match length
                gen_len = audio.shape[-1]
                voc_len = len(vocals_mono)
                if voc_len < gen_len:
                    vocals_mono = np.pad(vocals_mono, (0, gen_len - voc_len))
                else:
                    vocals_mono = vocals_mono[:gen_len]

                # Mix: 70% instrumental + 30% original vocals
                if audio.ndim == 1:
                    audio = 0.7 * audio + 0.3 * vocals_mono
                else:
                    audio = 0.7 * audio + 0.3 * vocals_mono[np.newaxis, :]

            result.remixed_audio = audio
            result.sample_rate = sr
            result.duration_seconds = audio.flatten().shape[0] / sr

            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(audio, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Remix failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    def change_tempo(
        self,
        input_path: str,
        target_bpm: Optional[float] = None,
        speed_factor: Optional[float] = None,
        output_path: Optional[str] = None,
        output_format: str = "wav",
        preserve_pitch: bool = True,
    ) -> GenerationResult:
        """
        Change the tempo of an audio file.

        Args:
            input_path:     Source audio file.
            target_bpm:     Target BPM (detected automatically from source).
            speed_factor:   Direct speed multiplier (1.0 = no change, 1.5 = 50% faster).
            preserve_pitch: Use time-stretching (pitch preserved) vs speed-up (pitch changes).
        """
        start_time = time.monotonic()
        result = GenerationResult(backend_used=GenerationBackend.TORCHAUDIO)

        try:
            import librosa

            y, sr = librosa.load(input_path, sr=None, mono=False)

            if speed_factor is None:
                if target_bpm is None:
                    raise ValueError("Provide either target_bpm or speed_factor")
                detected_tempo, _ = librosa.beat.beat_track(y=y if y.ndim == 1 else y[0], sr=sr)
                speed_factor = float(target_bpm) / float(detected_tempo)

            logger.info("Changing tempo | factor=%.3f | preserve_pitch=%s", speed_factor, preserve_pitch)

            if preserve_pitch:
                # Time-stretch using librosa (phase vocoder)
                if y.ndim == 2:
                    channels = [
                        librosa.effects.time_stretch(y[c], rate=speed_factor)
                        for c in range(y.shape[0])
                    ]
                    y_out = np.stack(channels)
                else:
                    y_out = librosa.effects.time_stretch(y, rate=speed_factor)
            else:
                # Simple resampling (changes pitch too)
                new_sr = int(sr * speed_factor)
                y_out = y
                sr = new_sr

            result.audio = y_out
            result.sample_rate = sr
            result.duration_seconds = y_out.flatten().shape[0] / sr

            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(y_out, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Tempo change failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    def change_pitch(
        self,
        input_path: str,
        semitones: float = 0.0,
        output_path: Optional[str] = None,
        output_format: str = "wav",
    ) -> GenerationResult:
        """
        Shift the pitch of an audio file by N semitones.

        Args:
            input_path:  Source audio file.
            semitones:   Pitch shift in semitones (+12 = up one octave).
        """
        start_time = time.monotonic()
        result = GenerationResult(backend_used=GenerationBackend.TORCHAUDIO)

        try:
            import librosa

            y, sr = librosa.load(input_path, sr=None, mono=False)

            logger.info("Shifting pitch | semitones=%.2f", semitones)

            if y.ndim == 2:
                channels = [
                    librosa.effects.pitch_shift(y[c], sr=sr, n_steps=semitones)
                    for c in range(y.shape[0])
                ]
                y_out = np.stack(channels)
            else:
                y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

            result.audio = y_out
            result.sample_rate = sr
            result.duration_seconds = y_out.flatten().shape[0] / sr

            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(y_out, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Pitch shift failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    def apply_tempo_and_pitch(
        self,
        input_path: str,
        tempo_factor: float = 1.0,
        pitch_semitones: float = 0.0,
        output_path: Optional[str] = None,
        output_format: str = "wav",
    ) -> GenerationResult:
        """Apply both tempo change and pitch shift in a single pass."""
        start_time = time.monotonic()
        result = GenerationResult(backend_used=GenerationBackend.TORCHAUDIO)

        try:
            import librosa

            y, sr = librosa.load(input_path, sr=None, mono=False)

            logger.info(
                "Applying tempo=%.3f pitch=%.2f semitones", tempo_factor, pitch_semitones
            )

            def process_channel(ch):
                out = ch
                if abs(tempo_factor - 1.0) > 0.001:
                    out = librosa.effects.time_stretch(out, rate=tempo_factor)
                if abs(pitch_semitones) > 0.001:
                    out = librosa.effects.pitch_shift(out, sr=sr, n_steps=pitch_semitones)
                return out

            if y.ndim == 2:
                channels = [process_channel(y[c]) for c in range(y.shape[0])]
                # Align lengths
                min_len = min(c.shape[0] for c in channels)
                y_out = np.stack([c[:min_len] for c in channels])
            else:
                y_out = process_channel(y)

            result.audio = y_out
            result.sample_rate = sr
            result.duration_seconds = y_out.flatten().shape[0] / sr

            if output_path is None:
                output_path = self._make_temp_path(output_format)
            self._save_audio(y_out, sr, output_path, output_format)
            result.output_path = output_path

        except Exception as exc:
            result.error = str(exc)
            logger.exception("Tempo+Pitch failed: %s", exc)

        result.processing_time_seconds = time.monotonic() - start_time
        return result

    # ── MusicGen Backend ────────────────────────────────────

    def _load_musicgen(self) -> None:
        """Lazy-load MusicGen model."""
        if self._musicgen_model is not None:
            return

        try:
            from audiocraft.models import MusicGen
            from audiocraft.data.audio import audio_write

            logger.info("Loading MusicGen: %s", self.model_name)
            self._musicgen_model = MusicGen.get_pretrained(
                self.model_name,
                device=self.device,
            )
            self._audio_write_fn = audio_write
            logger.info("MusicGen loaded on %s", self.device)

        except ImportError:
            raise RuntimeError(
                "audiocraft not installed. Run: pip install audiocraft"
            )

    def _generate_musicgen(
        self,
        config: GenerationConfig,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio using Meta MusicGen."""
        self._load_musicgen()
        model = self._musicgen_model

        model.set_generation_params(
            duration=config.duration_seconds,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            cfg_coef=config.cfg_coef,
            two_step_cfg=config.two_step_cfg,
            extend_stride=config.extend_stride,
        )

        prompt = config.build_prompt()

        with torch.no_grad():
            if config.melody_audio is not None:
                # Melody conditioning
                melody_tensor = torch.from_numpy(config.melody_audio).float()
                if melody_tensor.ndim == 1:
                    melody_tensor = melody_tensor.unsqueeze(0).unsqueeze(0)
                elif melody_tensor.ndim == 2:
                    melody_tensor = melody_tensor.unsqueeze(0)
                wav = model.generate_with_chroma(
                    descriptions=[prompt],
                    melody_wavs=melody_tensor.to(self.device),
                    melody_sample_rate=config.melody_sr or 32000,
                    progress=False,
                )
            else:
                wav = model.generate([prompt], progress=False)

        # wav: Tensor (batch, channels, samples)
        audio_np = wav[0].cpu().numpy()   # (channels, samples) or (samples,)
        sr = model.sample_rate

        return audio_np, sr

    # ── Riffusion Backend ───────────────────────────────────

    def _load_riffusion(self) -> None:
        """Lazy-load Riffusion diffusion pipeline."""
        if self._riffusion_pipeline is not None:
            return

        try:
            from diffusers import StableDiffusionPipeline

            logger.info("Loading Riffusion pipeline...")
            pipe = StableDiffusionPipeline.from_pretrained(
                "riffusion/riffusion-model-v1",
                torch_dtype=torch.float16 if self.use_float16 else torch.float32,
                safety_checker=None,
            )
            pipe = pipe.to(self.device)
            self._riffusion_pipeline = pipe
            logger.info("Riffusion loaded on %s", self.device)

        except ImportError:
            raise RuntimeError(
                "diffusers not installed. Run: pip install diffusers"
            )

    def _generate_riffusion(
        self,
        config: GenerationConfig,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio using Riffusion (spectrogram diffusion)."""
        self._load_riffusion()

        try:
            from riffusion.spectrogram_image_converter import SpectrogramImageConverter
            from riffusion.spectrogram_params import SpectrogramParams
        except ImportError:
            logger.warning("riffusion package not found; using diffusers directly")
            return self._generate_riffusion_diffusers(config)

        params = SpectrogramParams(
            stereo=False,
            sample_rate=self.DEFAULT_SR_RIFFUSION,
            min_frequency=20,
            max_frequency=10000,
        )
        converter = SpectrogramImageConverter(params=params)

        prompt = config.build_prompt()

        # Generate spectrogram image
        image = self._riffusion_pipeline(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]

        # Convert spectrogram image back to audio
        audio_segment = converter.audio_from_spectrogram_image(image)

        # Convert to numpy
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(audio_segment.array_type).max
        sr = audio_segment.frame_rate

        return samples, sr

    def _generate_riffusion_diffusers(
        self,
        config: GenerationConfig,
    ) -> Tuple[np.ndarray, int]:
        """Riffusion fallback: generate spectrogram image → convert via librosa."""
        import librosa
        from PIL import Image

        prompt = config.build_prompt()
        image = self._riffusion_pipeline(
            prompt=prompt,
            num_inference_steps=40,
            guidance_scale=7.0,
        ).images[0]

        # Convert grayscale spectrogram image to audio
        img_gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        # Flip vertically (low freq at bottom in spectrograms)
        spec = np.flipud(img_gray).T

        # Approximate ISTFT
        sr = self.DEFAULT_SR_RIFFUSION
        n_fft = 2048
        hop_length = 512
        audio = librosa.griffinlim(
            spec * 100.0,
            n_iter=64,
            hop_length=hop_length,
            n_fft=n_fft,
        )

        return audio, sr

    # ── Fallback Synthesis ──────────────────────────────────

    def _generate_fallback(
        self,
        config: GenerationConfig,
    ) -> Tuple[np.ndarray, int]:
        """
        Pure Torchaudio/NumPy fallback synthesis.
        Generates a simple melodic texture when no AI backend is available.
        """
        logger.warning("Using fallback synthesis (no AI backend available)")

        sr = 22050
        duration = config.duration_seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Simple harmonic synthesis based on genre
        genre = config.genre or "ambient"
        if genre in ("edm", "techno", "house"):
            freq = 110.0    # bass kick-ish
            env = np.sin(2 * np.pi * freq * t)
            env += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        elif genre in ("classical", "ambient", "lo_fi"):
            freqs = [261.63, 329.63, 392.0, 523.25]  # C major chord
            env = sum(0.25 * np.sin(2 * np.pi * f * t) for f in freqs)
        else:
            # Generic drone
            env = np.sin(2 * np.pi * 220 * t) * 0.5

        # Apply fade
        fade_len = int(0.1 * sr)
        env[:fade_len] *= np.linspace(0, 1, fade_len)
        env[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Normalize
        env = env / (np.abs(env).max() + 1e-8) * 0.8

        return env.astype(np.float32), sr

    # ── Audio Utilities ─────────────────────────────────────

    def _crossfade_chunks(
        self,
        chunks: List[np.ndarray],
        sr: int,
        overlap_seconds: float,
    ) -> np.ndarray:
        """Crossfade and concatenate audio chunks."""
        if len(chunks) == 1:
            return chunks[0]

        overlap_samples = int(overlap_seconds * sr)
        result = chunks[0]

        for i in range(1, len(chunks)):
            nxt = chunks[i]

            # Ensure matching channels
            if result.ndim != nxt.ndim:
                if result.ndim == 1:
                    result = result[np.newaxis, :]
                if nxt.ndim == 1:
                    nxt = nxt[np.newaxis, :]

            # Crossfade window
            fade_out = np.linspace(1, 0, overlap_samples)
            fade_in = np.linspace(0, 1, overlap_samples)

            if result.ndim == 2:
                fade_out = fade_out[np.newaxis, :]
                fade_in = fade_in[np.newaxis, :]
                tail = result[:, -overlap_samples:] * fade_out
                head = nxt[:, :overlap_samples] * fade_in
                crossfade = tail + head
                result = np.concatenate([
                    result[:, :-overlap_samples],
                    crossfade,
                    nxt[:, overlap_samples:],
                ], axis=1)
            else:
                tail = result[-overlap_samples:] * fade_out.squeeze()
                head = nxt[:overlap_samples] * fade_in.squeeze()
                crossfade = tail + head
                result = np.concatenate([
                    result[:-overlap_samples],
                    crossfade,
                    nxt[overlap_samples:],
                ])

        return result

    def _save_audio(
        self,
        audio: np.ndarray,
        sr: int,
        output_path: str,
        fmt: str,
    ) -> None:
        """Save numpy audio array to file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # sf.write expects (samples, channels)
        if audio.ndim == 2:
            data = audio.T                # (samples, channels)
        else:
            data = audio                  # (samples,)

        fmt_lower = fmt.lower()
        if fmt_lower == "wav":
            sf.write(output_path, data, sr, subtype="PCM_24")
        elif fmt_lower in ("mp3", "flac", "ogg"):
            from pydub import AudioSegment
            tmp_wav = output_path + ".tmp.wav"
            sf.write(tmp_wav, data, sr, subtype="PCM_16")
            seg = AudioSegment.from_wav(tmp_wav)
            seg.export(output_path, format=fmt_lower)
            os.unlink(tmp_wav)
        else:
            sf.write(output_path, data, sr)

    @staticmethod
    def _make_temp_path(fmt: str = "wav") -> str:
        fd, path = tempfile.mkstemp(suffix=f".{fmt}", prefix="generated_")
        os.close(fd)
        return path

    # ── Utility / Info ──────────────────────────────────────

    def list_genres(self) -> Dict[str, str]:
        """Return all supported genres and their prompt descriptions."""
        return {k: v for k, v in GENRE_PROMPT_MAP.items()}

    def is_gpu_available(self) -> bool:
        return torch.cuda.is_available()

    def get_model_info(self) -> dict:
        return {
            "backend": self.backend.value,
            "model_name": self.model_name,
            "device": self.device,
            "gpu_available": self.is_gpu_available(),
            "musicgen_loaded": self._musicgen_model is not None,
            "riffusion_loaded": self._riffusion_pipeline is not None,
            "max_duration_single_call": self.MAX_MUSICGEN_DURATION,
            "supported_genres": list(GENRE_PROMPT_MAP.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"MusicGenerator(backend={self.backend.value!r}, "
            f"model={self.model_name!r}, device={self.device!r})"
        )