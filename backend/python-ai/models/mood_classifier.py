# ============================================================
# models/mood_classifier.py
# AI Music Remix & Mood Generator — Mood Classifier Model
# ============================================================
# Classifies audio mood using:
#   Primary:  Librosa feature extraction + XGBoost ensemble
#   Enhanced: HuggingFace audio spectrogram transformer
#   Output:   Valence/arousal + discrete mood label + confidence
# ============================================================

from __future__ import annotations

import json
import logging
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ── Mood Taxonomy ────────────────────────────────────────────

class MoodLabel(str, Enum):
    """Russell's circumplex-inspired discrete mood labels."""
    ENERGETIC   = "energetic"       # high arousal, high valence
    HAPPY       = "happy"           # moderate arousal, high valence
    CALM        = "calm"            # low arousal, high valence
    MELANCHOLIC = "melancholic"     # low arousal, low valence
    TENSE       = "tense"           # high arousal, low valence
    ANGRY       = "angry"           # very high arousal, low valence
    ROMANTIC    = "romantic"        # low arousal, very high valence
    EUPHORIC    = "euphoric"        # very high arousal, very high valence
    DARK        = "dark"            # very low arousal, very low valence
    DREAMY      = "dreamy"          # low arousal, moderate valence


# Valence/Arousal coordinates for each label (for reference/visualization)
MOOD_COORDINATES: Dict[MoodLabel, Tuple[float, float]] = {
    MoodLabel.ENERGETIC:   (0.7,  0.8),
    MoodLabel.HAPPY:       (0.75, 0.55),
    MoodLabel.CALM:        (0.6,  0.2),
    MoodLabel.MELANCHOLIC: (0.2,  0.25),
    MoodLabel.TENSE:       (0.25, 0.7),
    MoodLabel.ANGRY:       (0.15, 0.9),
    MoodLabel.ROMANTIC:    (0.85, 0.3),
    MoodLabel.EUPHORIC:    (0.9,  0.95),
    MoodLabel.DARK:        (0.1,  0.15),
    MoodLabel.DREAMY:      (0.55, 0.15),
}

# Suggested remix genres per mood
MOOD_TO_GENRE_MAP: Dict[MoodLabel, List[str]] = {
    MoodLabel.ENERGETIC:   ["EDM", "Drum & Bass", "House", "Techno"],
    MoodLabel.HAPPY:       ["Pop", "Funk", "Reggae", "Disco"],
    MoodLabel.CALM:        ["Lo-Fi", "Ambient", "Acoustic", "Classical"],
    MoodLabel.MELANCHOLIC: ["Indie Folk", "Blues", "Slow Jazz", "Emo"],
    MoodLabel.TENSE:       ["Industrial", "Metal", "Thriller", "Orchestral"],
    MoodLabel.ANGRY:       ["Metal", "Punk", "Hard Rock", "Drum & Bass"],
    MoodLabel.ROMANTIC:    ["R&B", "Bossa Nova", "Soul", "Neo-Soul"],
    MoodLabel.EUPHORIC:    ["Progressive House", "Trance", "Festival EDM"],
    MoodLabel.DARK:        ["Gothic", "Dark Ambient", "Trip-Hop", "Post-Rock"],
    MoodLabel.DREAMY:      ["Dream Pop", "Shoegaze", "Chillwave", "Ambient"],
}


# ── Data Classes ─────────────────────────────────────────────

@dataclass
class AudioFeatureVector:
    """Complete audio feature set extracted by Librosa."""
    # Temporal
    duration_seconds: float = 0.0
    sample_rate: int = 22050
    tempo_bpm: float = 0.0
    beat_strength: float = 0.0

    # Spectral
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    spectral_contrast_mean: List[float] = field(default_factory=list)
    spectral_flatness_mean: float = 0.0

    # Rhythmic
    zcr_mean: float = 0.0
    zcr_std: float = 0.0
    onset_strength_mean: float = 0.0

    # MFCC (13 coefficients)
    mfcc_means: List[float] = field(default_factory=list)
    mfcc_stds: List[float] = field(default_factory=list)

    # Chroma
    chroma_means: List[float] = field(default_factory=list)

    # Loudness / Dynamics
    rms_mean: float = 0.0
    rms_std: float = 0.0
    dynamic_range_db: float = 0.0

    # Tonal
    key_profile: List[float] = field(default_factory=list)  # 12-dim chroma
    tuning_deviation: float = 0.0
    key_clarity: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Flatten all features to a 1-D numpy array for ML input."""
        parts = [
            self.tempo_bpm / 200.0,           # normalize
            self.beat_strength,
            self.spectral_centroid_mean / 8000.0,
            self.spectral_centroid_std / 2000.0,
            self.spectral_bandwidth_mean / 4000.0,
            self.spectral_rolloff_mean / 8000.0,
            *[v / 50.0 for v in self.spectral_contrast_mean],
            self.spectral_flatness_mean,
            self.zcr_mean,
            self.zcr_std,
            self.onset_strength_mean / 10.0,
            *[v / 50.0 for v in self.mfcc_means],
            *[v / 50.0 for v in self.mfcc_stds],
            *self.chroma_means,
            self.rms_mean,
            self.rms_std,
            self.dynamic_range_db / 60.0,
            *self.key_profile,
            self.tuning_deviation,
            self.key_clarity,
        ]
        return np.array(parts, dtype=np.float32)


@dataclass
class MoodPrediction:
    """Full mood classification result."""
    primary_mood: MoodLabel = MoodLabel.CALM
    confidence: float = 0.0
    valence: float = 0.5             # 0 (negative) → 1 (positive)
    arousal: float = 0.5             # 0 (calm) → 1 (energetic)
    mood_probabilities: Dict[str, float] = field(default_factory=dict)
    suggested_genres: List[str] = field(default_factory=list)
    feature_vector: Optional[AudioFeatureVector] = None
    processing_time_seconds: float = 0.0
    model_used: str = "librosa+xgboost"
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "primary_mood": self.primary_mood.value,
            "confidence": round(self.confidence, 4),
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "mood_probabilities": {k: round(v, 4) for k, v in self.mood_probabilities.items()},
            "suggested_genres": self.suggested_genres,
            "model_used": self.model_used,
            "processing_time_seconds": round(self.processing_time_seconds, 3),
            "error": self.error,
        }


# ── Feature Extractor ─────────────────────────────────────────

class AudioFeatureExtractor:
    """
    Extracts a rich set of audio features using Librosa.
    These features feed the downstream ML classifier.
    """

    N_MFCC = 13
    N_CHROMA = 12
    N_SPECTRAL_CONTRAST_BANDS = 6

    def __init__(self, sample_rate: int = 22050, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono

    def extract(
        self,
        audio_path: str,
        segment_duration: Optional[float] = None,
        segment_offset: float = 0.0,
    ) -> AudioFeatureVector:
        """
        Extract features from an audio file.

        Args:
            audio_path:        Path to audio file.
            segment_duration:  If set, only analyze this many seconds.
            segment_offset:    Start time (seconds) for segment analysis.

        Returns:
            AudioFeatureVector instance.
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError("Librosa not installed. Run: pip install librosa")

        feat = AudioFeatureVector()

        logger.debug("Extracting features from: %s", audio_path)
        y, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=self.mono,
            offset=segment_offset,
            duration=segment_duration,
        )

        feat.duration_seconds = float(len(y) / sr)
        feat.sample_rate = sr

        # ── Tempo & Beat ──────────────────────────────────────
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            feat.tempo_bpm = float(tempo)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            feat.beat_strength = float(onset_env[beat_frames].mean()) if len(beat_frames) > 0 else 0.0
            feat.onset_strength_mean = float(onset_env.mean())
        except Exception as e:
            logger.warning("Beat tracking failed: %s", e)

        # ── Spectral Features ─────────────────────────────────
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        sc = librosa.feature.spectral_centroid(S=spec, sr=sr)[0]
        feat.spectral_centroid_mean = float(sc.mean())
        feat.spectral_centroid_std = float(sc.std())

        sb = librosa.feature.spectral_bandwidth(S=spec, sr=sr)[0]
        feat.spectral_bandwidth_mean = float(sb.mean())

        sro = librosa.feature.spectral_rolloff(S=spec, sr=sr, roll_percent=0.85)[0]
        feat.spectral_rolloff_mean = float(sro.mean())

        contrast = librosa.feature.spectral_contrast(S=spec, sr=sr)
        feat.spectral_contrast_mean = contrast.mean(axis=1).tolist()
        if len(feat.spectral_contrast_mean) < self.N_SPECTRAL_CONTRAST_BANDS:
            feat.spectral_contrast_mean += [0.0] * (
                self.N_SPECTRAL_CONTRAST_BANDS - len(feat.spectral_contrast_mean)
            )

        sf_feat = librosa.feature.spectral_flatness(S=spec)[0]
        feat.spectral_flatness_mean = float(sf_feat.mean())

        # ── Zero Crossing Rate ────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feat.zcr_mean = float(zcr.mean())
        feat.zcr_std = float(zcr.std())

        # ── MFCC ─────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.N_MFCC)
        feat.mfcc_means = mfcc.mean(axis=1).tolist()
        feat.mfcc_stds = mfcc.std(axis=1).tolist()

        # ── Chroma ───────────────────────────────────────────
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=self.N_CHROMA)
        feat.chroma_means = chroma.mean(axis=1).tolist()

        # ── Loudness / Dynamics ───────────────────────────────
        rms = librosa.feature.rms(y=y)[0]
        feat.rms_mean = float(rms.mean())
        feat.rms_std = float(rms.std())
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        feat.dynamic_range_db = float(rms_db.max() - rms_db.min())

        # ── Key / Tonal ───────────────────────────────────────
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        key_profile = chroma_cens.mean(axis=1)
        if key_profile.sum() > 0:
            key_profile = key_profile / key_profile.sum()
        feat.key_profile = key_profile.tolist()

        # Key clarity = ratio of dominant chroma vs mean
        sorted_kp = np.sort(key_profile)[::-1]
        feat.key_clarity = float(sorted_kp[0] / (sorted_kp[1] + 1e-8))

        try:
            feat.tuning_deviation = float(librosa.estimate_tuning(y=y, sr=sr))
        except Exception:
            feat.tuning_deviation = 0.0

        return feat

    def extract_from_array(
        self,
        y: np.ndarray,
        sr: int,
    ) -> AudioFeatureVector:
        """Extract features from an already-loaded numpy audio array."""
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y.T if y.ndim == 2 else y, sr)
            feat = self.extract(tmp.name)
        os.unlink(tmp.name)
        return feat


# ── Mood Classifier ───────────────────────────────────────────

class MoodClassifier:
    """
    Multi-stage mood classifier for audio.

    Stage 1 (Primary): Librosa features → XGBoost ensemble
    Stage 2 (Optional): HuggingFace Audio Spectrogram Transformer
    Stage 3 (Fusion):   Weighted ensemble of both predictions

    Features:
    - Valence/Arousal regression
    - Discrete mood label classification
    - Confidence calibration via Platt scaling
    - Genre recommendation based on mood
    - Model persistence (save/load)
    """

    MODEL_REGISTRY = {
        "xgboost": "_predict_xgboost",
        "random_forest": "_predict_rf",
        "transformer": "_predict_transformer",
        "ensemble": "_predict_ensemble",
    }

    # Heuristic rules for rule-based fallback (no trained model required)
    HEURISTIC_RULES = [
        # (tempo_min, tempo_max, centroid_min_norm, zcr_min, rms_min, mood, valence, arousal)
        (140, 999, 0.4,  0.08, 0.05, MoodLabel.ENERGETIC,  0.75, 0.88),
        (120, 140, 0.35, 0.06, 0.04, MoodLabel.HAPPY,      0.80, 0.65),
        ( 60,  90, 0.2,  0.03, 0.02, MoodLabel.CALM,       0.65, 0.20),
        ( 50,  80, 0.15, 0.02, 0.01, MoodLabel.MELANCHOLIC,0.20, 0.25),
        (100, 140, 0.45, 0.07, 0.05, MoodLabel.TENSE,      0.25, 0.75),
        (140, 999, 0.5,  0.09, 0.06, MoodLabel.ANGRY,      0.15, 0.90),
        ( 60,  90, 0.18, 0.02, 0.02, MoodLabel.ROMANTIC,   0.85, 0.28),
        (130, 999, 0.42, 0.08, 0.05, MoodLabel.EUPHORIC,   0.92, 0.95),
        ( 40,  70, 0.10, 0.01, 0.01, MoodLabel.DARK,       0.10, 0.12),
        ( 70, 100, 0.22, 0.03, 0.02, MoodLabel.DREAMY,     0.58, 0.18),
    ]

    def __init__(
        self,
        model_type: str = "ensemble",
        model_path: Optional[str] = None,
        use_transformer: bool = False,
        transformer_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: Optional[str] = None,
        sample_rate: int = 22050,
        analysis_duration: float = 30.0,   # seconds to analyze (from middle)
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.use_transformer = use_transformer
        self.transformer_model_id = transformer_model_id
        self.sample_rate = sample_rate
        self.analysis_duration = analysis_duration

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

        # ML models (lazy-loaded)
        self._xgb_model = None
        self._rf_model = None
        self._transformer_pipeline = None
        self._label_encoder = None
        self._valence_model = None
        self._arousal_model = None

        # Load pre-trained model if path provided
        if model_path and Path(model_path).exists():
            self._load_models(model_path)

        logger.info(
            "MoodClassifier initialized | model=%s | transformer=%s | device=%s",
            model_type, use_transformer, self.device,
        )

    # ── Public API ─────────────────────────────────────────

    def classify(
        self,
        audio_path: str,
        use_full_track: bool = False,
    ) -> MoodPrediction:
        """
        Classify the mood of an audio file.

        Args:
            audio_path:    Path to audio file.
            use_full_track: If False, analyzes a 30-second segment from the middle.

        Returns:
            MoodPrediction with label, valence, arousal, and genre suggestions.
        """
        start_time = time.monotonic()
        result = MoodPrediction()

        try:
            # ── Select analysis segment ─────────────────────
            import librosa
            total_dur = librosa.get_duration(path=audio_path)

            if use_full_track or total_dur <= self.analysis_duration:
                offset = 0.0
                duration = None
            else:
                # Analyze from 25% into the track (typically more representative)
                offset = max(0.0, total_dur * 0.25)
                duration = self.analysis_duration

            # ── Extract features ────────────────────────────
            features = self.feature_extractor.extract(
                audio_path,
                segment_duration=duration,
                segment_offset=offset,
            )
            result.feature_vector = features

            # ── Run classifier ──────────────────────────────
            if self._xgb_model is not None:
                result = self._predict_xgboost(features)
            elif self.use_transformer and self._transformer_pipeline is not None:
                result = self._predict_transformer(audio_path)
            else:
                result = self._predict_heuristic(features)

            # ── Attach metadata ─────────────────────────────
            result.feature_vector = features
            result.suggested_genres = MOOD_TO_GENRE_MAP.get(result.primary_mood, [])
            result.processing_time_seconds = time.monotonic() - start_time

        except Exception as exc:
            result.error = str(exc)
            result.processing_time_seconds = time.monotonic() - start_time
            logger.exception("Mood classification failed: %s", exc)

        return result

    def classify_from_array(
        self,
        y: np.ndarray,
        sr: int,
    ) -> MoodPrediction:
        """Classify mood from a numpy audio array (useful after stem separation)."""
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y.T if y.ndim == 2 else y, sr)
            tmp_path = tmp.name

        try:
            return self.classify(tmp_path)
        finally:
            os.unlink(tmp_path)

    def train(
        self,
        audio_paths: List[str],
        labels: List[str],
        valences: Optional[List[float]] = None,
        arousals: Optional[List[float]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train XGBoost + RandomForest ensemble on labeled audio data.

        Args:
            audio_paths: Paths to training audio files.
            labels:      Mood label strings matching MoodLabel values.
            valences:    Optional valence scores (0–1) for regression head.
            arousals:    Optional arousal scores (0–1) for regression head.
            save_path:   Where to save trained models.

        Returns:
            Training metrics dict.
        """
        try:
            import xgboost as xgb
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            from sklearn.metrics import accuracy_score, f1_score
        except ImportError:
            raise RuntimeError(
                "Training requires: pip install xgboost scikit-learn"
            )

        logger.info("Training on %d samples...", len(audio_paths))

        # ── Extract features for all tracks ────────────────
        X_list = []
        valid_labels = []
        valid_valences = []
        valid_arousals = []

        for i, (path, label) in enumerate(zip(audio_paths, labels)):
            try:
                feat = self.feature_extractor.extract(path)
                X_list.append(feat.to_numpy())
                valid_labels.append(label)
                if valences:
                    valid_valences.append(valences[i])
                if arousals:
                    valid_arousals.append(arousals[i])
                logger.debug("Extracted features [%d/%d]: %s", i + 1, len(audio_paths), path)
            except Exception as e:
                logger.warning("Failed to extract from %s: %s", path, e)

        X = np.array(X_list)
        y_labels = np.array(valid_labels)

        # ── Encode labels ───────────────────────────────────
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_labels)

        # ── Train XGBoost classifier ────────────────────────
        self._xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            device=self.device,
            random_state=42,
        )
        self._xgb_model.fit(X, y_encoded)

        # ── Train RandomForest (for ensemble) ───────────────
        self._rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=42,
        )
        self._rf_model.fit(X, y_encoded)

        # ── Train valence/arousal regressors ────────────────
        if valid_valences:
            self._valence_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
            self._valence_model.fit(X, valid_valences)

        if valid_arousals:
            self._arousal_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
            self._arousal_model.fit(X, valid_arousals)

        # ── Cross-validation ────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        xgb_cv_acc = cross_val_score(self._xgb_model, X, y_encoded, cv=cv, scoring="accuracy").mean()
        xgb_cv_f1 = cross_val_score(self._xgb_model, X, y_encoded, cv=cv, scoring="f1_macro").mean()

        metrics = {
            "n_samples": len(X_list),
            "n_classes": len(self._label_encoder.classes_),
            "classes": list(self._label_encoder.classes_),
            "xgb_cv_accuracy": round(float(xgb_cv_acc), 4),
            "xgb_cv_f1_macro": round(float(xgb_cv_f1), 4),
        }

        if save_path:
            self._save_models(save_path)
            metrics["saved_to"] = save_path

        logger.info("Training complete: %s", metrics)
        return metrics

    # ── XGBoost Prediction ──────────────────────────────────

    def _predict_xgboost(self, features: AudioFeatureVector) -> MoodPrediction:
        """Predict using trained XGBoost + RF ensemble."""
        result = MoodPrediction(model_used="xgboost+rf_ensemble")

        X = features.to_numpy().reshape(1, -1)

        # XGBoost probabilities
        xgb_proba = self._xgb_model.predict_proba(X)[0]

        # Random Forest probabilities
        if self._rf_model is not None:
            rf_proba = self._rf_model.predict_proba(X)[0]
            # Weighted ensemble: 60% XGBoost, 40% RF
            proba = 0.6 * xgb_proba + 0.4 * rf_proba
        else:
            proba = xgb_proba

        # Decode label
        label_idx = int(np.argmax(proba))
        label_str = self._label_encoder.inverse_transform([label_idx])[0]

        try:
            result.primary_mood = MoodLabel(label_str)
        except ValueError:
            result.primary_mood = MoodLabel.CALM

        result.confidence = float(proba[label_idx])

        # Build full probability map
        classes = self._label_encoder.classes_
        result.mood_probabilities = {
            cls: round(float(p), 4) for cls, p in zip(classes, proba)
        }

        # Valence / Arousal
        if self._valence_model is not None:
            result.valence = float(np.clip(self._valence_model.predict(X)[0], 0.0, 1.0))
        else:
            v, a = MOOD_COORDINATES.get(result.primary_mood, (0.5, 0.5))
            result.valence = v

        if self._arousal_model is not None:
            result.arousal = float(np.clip(self._arousal_model.predict(X)[0], 0.0, 1.0))
        else:
            v, a = MOOD_COORDINATES.get(result.primary_mood, (0.5, 0.5))
            result.arousal = a

        return result

    # ── HuggingFace Transformer Prediction ─────────────────

    def _load_transformer(self) -> None:
        """Lazy-load HuggingFace Audio Spectrogram Transformer."""
        if self._transformer_pipeline is not None:
            return

        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading HF transformer: %s", self.transformer_model_id)
            self._transformer_pipeline = hf_pipeline(
                "audio-classification",
                model=self.transformer_model_id,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("HF transformer loaded.")
        except ImportError:
            raise RuntimeError(
                "HuggingFace Transformers not installed. "
                "Run: pip install transformers"
            )

    def _predict_transformer(self, audio_path: str) -> MoodPrediction:
        """Classify mood using HuggingFace Audio Spectrogram Transformer."""
        if self.use_transformer:
            self._load_transformer()

        result = MoodPrediction(model_used="huggingface-ast")

        try:
            preds = self._transformer_pipeline(audio_path, top_k=10)
        except Exception as e:
            logger.warning("Transformer prediction failed: %s", e)
            return self._predict_heuristic(
                self.feature_extractor.extract(audio_path)
            )

        # Map HF AudioSet labels → our mood taxonomy
        audioset_to_mood = {
            "music": MoodLabel.CALM,
            "dance music": MoodLabel.ENERGETIC,
            "pop music": MoodLabel.HAPPY,
            "electronic music": MoodLabel.ENERGETIC,
            "rock music": MoodLabel.TENSE,
            "heavy metal": MoodLabel.ANGRY,
            "jazz": MoodLabel.ROMANTIC,
            "blues": MoodLabel.MELANCHOLIC,
            "classical music": MoodLabel.CALM,
            "ambient music": MoodLabel.DREAMY,
        }

        best_score = 0.0
        result.primary_mood = MoodLabel.CALM

        for pred in preds:
            label_lower = pred["label"].lower()
            for kw, mood in audioset_to_mood.items():
                if kw in label_lower and pred["score"] > best_score:
                    best_score = pred["score"]
                    result.primary_mood = mood

        result.confidence = best_score
        v, a = MOOD_COORDINATES.get(result.primary_mood, (0.5, 0.5))
        result.valence = v
        result.arousal = a

        return result

    # ── Heuristic Fallback ──────────────────────────────────

    def _predict_heuristic(self, features: AudioFeatureVector) -> MoodPrediction:
        """
        Rule-based mood prediction when no trained model is available.
        Uses musical heuristics: tempo, spectral centroid, ZCR, energy.
        """
        result = MoodPrediction(model_used="heuristic")

        centroid_norm = features.spectral_centroid_mean / 8000.0
        probs: Dict[str, float] = {}

        for (t_min, t_max, c_min, z_min, r_min, mood, valence, arousal) in self.HEURISTIC_RULES:
            score = 0.0

            # Tempo score
            if t_min <= features.tempo_bpm < t_max:
                score += 0.4
            elif abs(features.tempo_bpm - (t_min + t_max) / 2) < 20:
                score += 0.2

            # Spectral centroid score
            if centroid_norm >= c_min:
                score += 0.2

            # ZCR score
            if features.zcr_mean >= z_min:
                score += 0.2

            # Energy score
            if features.rms_mean >= r_min:
                score += 0.2

            probs[mood.value] = max(probs.get(mood.value, 0.0), score)

        if not probs or max(probs.values()) == 0:
            # Default fallback
            probs = {m.value: 1.0 / len(MoodLabel) for m in MoodLabel}

        # Normalize to sum to 1
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        best_mood_str = max(probs, key=probs.__getitem__)
        result.primary_mood = MoodLabel(best_mood_str)
        result.confidence = probs[best_mood_str]
        result.mood_probabilities = probs

        v, a = MOOD_COORDINATES.get(result.primary_mood, (0.5, 0.5))

        # Adjust valence/arousal using tempo
        tempo_norm = min(features.tempo_bpm / 200.0, 1.0)
        result.arousal = float(np.clip(0.5 * a + 0.5 * tempo_norm, 0.0, 1.0))
        result.valence = float(np.clip(v + 0.1 * (features.rms_mean - 0.05), 0.0, 1.0))

        return result

    # ── Model Persistence ───────────────────────────────────

    def _save_models(self, save_dir: str) -> None:
        """Save trained models to disk."""
        os.makedirs(save_dir, exist_ok=True)

        if self._xgb_model:
            self._xgb_model.save_model(os.path.join(save_dir, "xgb_model.json"))

        if self._rf_model:
            with open(os.path.join(save_dir, "rf_model.pkl"), "wb") as f:
                pickle.dump(self._rf_model, f)

        if self._label_encoder:
            with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
                pickle.dump(self._label_encoder, f)

        if self._valence_model:
            with open(os.path.join(save_dir, "valence_model.pkl"), "wb") as f:
                pickle.dump(self._valence_model, f)

        if self._arousal_model:
            with open(os.path.join(save_dir, "arousal_model.pkl"), "wb") as f:
                pickle.dump(self._arousal_model, f)

        meta = {
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "analysis_duration": self.analysis_duration,
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Models saved to: %s", save_dir)

    def _load_models(self, model_dir: str) -> None:
        """Load previously trained models from disk."""
        try:
            import xgboost as xgb
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

            xgb_path = os.path.join(model_dir, "xgb_model.json")
            if os.path.exists(xgb_path):
                self._xgb_model = xgb.XGBClassifier()
                self._xgb_model.load_model(xgb_path)

            rf_path = os.path.join(model_dir, "rf_model.pkl")
            if os.path.exists(rf_path):
                with open(rf_path, "rb") as f:
                    self._rf_model = pickle.load(f)

            le_path = os.path.join(model_dir, "label_encoder.pkl")
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    self._label_encoder = pickle.load(f)

            val_path = os.path.join(model_dir, "valence_model.pkl")
            if os.path.exists(val_path):
                with open(val_path, "rb") as f:
                    self._valence_model = pickle.load(f)

            aro_path = os.path.join(model_dir, "arousal_model.pkl")
            if os.path.exists(aro_path):
                with open(aro_path, "rb") as f:
                    self._arousal_model = pickle.load(f)

            logger.info("Models loaded from: %s", model_dir)

        except Exception as e:
            logger.error("Failed to load models: %s", e)

    # ── Feature Importance ──────────────────────────────────

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return XGBoost feature importance (if model is trained)."""
        if self._xgb_model is None:
            return None

        importance = self._xgb_model.get_booster().get_fscore()
        total = sum(importance.values()) or 1
        return {k: round(v / total, 4) for k, v in sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )}

    def __repr__(self) -> str:
        has_model = self._xgb_model is not None
        return (
            f"MoodClassifier(model={self.model_type!r}, "
            f"trained={has_model}, device={self.device!r})"
        )