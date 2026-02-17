# ============================================================
# models/__init__.py
# AI Music Remix & Mood Generator — Model Package Initializer
# ============================================================

"""
models/
  __init__.py         ← This file. Exposes all model classes.
  stem_separator.py   ← Demucs/Spleeter-based stem separation
  mood_classifier.py  ← Librosa + XGBoost mood classification
  music_generator.py  ← MusicGen / Riffusion text-to-music
"""

from models.stem_separator import StemSeparator
from models.mood_classifier import MoodClassifier
from models.music_generator import MusicGenerator

__all__ = [
    "StemSeparator",
    "MoodClassifier",
    "MusicGenerator",
]