# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import os
# import logging
# from pathlib import Path
# import tempfile

# # Import AI models
# from models.stem_separator import StemSeparator
# from models.mood_classifier import MoodClassifier
# from models.music_generator import MusicGenerator
# from utils.audio_utils import AudioProcessor

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Initialize AI models
# try:
#     stem_separator = StemSeparator()
#     mood_classifier = MoodClassifier()
#     music_generator = MusicGenerator()
#     audio_processor = AudioProcessor()
#     logger.info("✅ All AI models loaded successfully")
# except Exception as e:
#     logger.error(f"❌ Error loading AI models: {e}")
#     stem_separator = None
#     mood_classifier = None
#     music_generator = None
#     audio_processor = None

# # Configuration
# UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/tmp/audio-uploads')
# OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', '/tmp/audio-outputs')
# Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
# Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     models_status = {
#         'stem_separator': stem_separator is not None,
#         'mood_classifier': mood_classifier is not None,
#         'music_generator': music_generator is not None,
#         'audio_processor': audio_processor is not None
#     }
    
#     return jsonify({
#         'status': 'healthy',
#         'models': models_status,
#         'all_models_loaded': all(models_status.values())
#     })


# @app.route('/separate-stems', methods=['POST'])
# def separate_stems():
#     """Separate audio into stems (vocals, drums, bass, other)"""
#     try:
#         if stem_separator is None:
#             return jsonify({'error': 'Stem separator model not loaded'}), 503

#         # Get uploaded file
#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio_file = request.files['audio']
#         model_name = request.form.get('model', 'htdemucs')
        
#         # Save uploaded file
#         input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#         audio_file.save(input_path)
        
#         logger.info(f"Processing stem separation for: {audio_file.filename}")
        
#         # Separate stems
#         output_paths = stem_separator.separate(
#             audio_path=input_path,
#             output_dir=OUTPUT_FOLDER,
#             model=model_name
#         )
        
#         return jsonify({
#             'success': True,
#             'stems': output_paths,
#             'message': 'Stems separated successfully'
#         })
        
#     except Exception as e:
#         logger.error(f"Stem separation error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/classify-mood', methods=['POST'])
# def classify_mood():
#     """Classify the mood of an audio file"""
#     try:
#         if mood_classifier is None:
#             return jsonify({'error': 'Mood classifier model not loaded'}), 503

#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio_file = request.files['audio']
        
#         # Save uploaded file
#         input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#         audio_file.save(input_path)
        
#         logger.info(f"Classifying mood for: {audio_file.filename}")
        
#         # Classify mood
#         result = mood_classifier.classify(input_path)
        
#         return jsonify({
#             'success': True,
#             'mood': result['mood'],
#             'confidence': result['confidence'],
#             'mood_scores': result['mood_scores'],
#             'features': result['features']
#         })
        
#     except Exception as e:
#         logger.error(f"Mood classification error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/generate-music', methods=['POST'])
# def generate_music():
#     """Generate music from text prompt"""
#     try:
#         if music_generator is None:
#             return jsonify({'error': 'Music generator model not loaded'}), 503

#         data = request.get_json()
        
#         if 'prompt' not in data:
#             return jsonify({'error': 'No prompt provided'}), 400

#         prompt = data['prompt']
#         duration = data.get('duration', 10)
#         temperature = data.get('temperature', 1.0)
        
#         logger.info(f"Generating music from prompt: {prompt}")
        
#         # Generate music
#         output_path = music_generator.generate(
#             prompt=prompt,
#             duration=duration,
#             temperature=temperature,
#             output_dir=OUTPUT_FOLDER
#         )
        
#         return jsonify({
#             'success': True,
#             'audio_path': output_path,
#             'message': 'Music generated successfully'
#         })
        
#     except Exception as e:
#         logger.error(f"Music generation error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/extract-features', methods=['POST'])
# def extract_features():
#     """Extract audio features for analysis"""
#     try:
#         if audio_processor is None:
#             return jsonify({'error': 'Audio processor not loaded'}), 503

#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio_file = request.files['audio']
        
#         # Save uploaded file
#         input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#         audio_file.save(input_path)
        
#         # Extract features
#         features = audio_processor.extract_features(input_path)
        
#         return jsonify({
#             'success': True,
#             'features': features
#         })
        
#     except Exception as e:
#         logger.error(f"Feature extraction error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/waveform', methods=['POST'])
# def get_waveform():
#     """Generate waveform data"""
#     try:
#         if audio_processor is None:
#             return jsonify({'error': 'Audio processor not loaded'}), 503

#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio_file = request.files['audio']
#         samples = int(request.form.get('samples', 1000))
        
#         # Save uploaded file
#         input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#         audio_file.save(input_path)
        
#         # Generate waveform
#         waveform_data = audio_processor.get_waveform(input_path, samples)
        
#         return jsonify({
#             'success': True,
#             'waveform': waveform_data
#         })
        
#     except Exception as e:
#         logger.error(f"Waveform generation error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/spectrogram', methods=['POST'])
# def get_spectrogram():
#     """Generate spectrogram image"""
#     try:
#         if audio_processor is None:
#             return jsonify({'error': 'Audio processor not loaded'}), 503

#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio_file = request.files['audio']
        
#         # Save uploaded file
#         input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#         audio_file.save(input_path)
        
#         # Generate spectrogram
#         spec_path = audio_processor.generate_spectrogram(
#             input_path,
#             OUTPUT_FOLDER
#         )
        
#         return send_file(spec_path, mimetype='image/png')
        
#     except Exception as e:
#         logger.error(f"Spectrogram generation error: {e}")
#         return jsonify({'error': str(e)}), 500


# @app.route('/apply-effects', methods=['POST'])
# def apply_effects():
#     """Apply audio effects"""
#     try:
#         if audio_processor is None:
#             return jsonify({'error': 'Audio processor not loaded'}), 503

#         if 'audio' not in request.files:
#             return jsonify({'error': 'No audio file provided'}), 400

#         audio_file = request.files['audio']
#         effects_json = request.form.get('effects', '{}')
        
#         import json
#         effects = json.loads(effects_json)
        
#         # Save uploaded file
#         input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#         audio_file.save(input_path)
        
#         # Apply effects
#         output_path = audio_processor.apply_effects(
#             input_path,
#             effects,
#             OUTPUT_FOLDER
#         )
        
#         return jsonify({
#             'success': True,
#             'audio_path': output_path,
#             'message': 'Effects applied successfully'
#         })
        
#     except Exception as e:
#         logger.error(f"Effects application error: {e}")
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     port = int(os.getenv('PORT', 5001))
#     app.run(
#         host='0.0.0.0',
#         port=port,
#         debug=os.getenv('FLASK_ENV') == 'development'
#     )


# ============================================================
# app.py
# AI Music Remix & Mood Generator — FastAPI Application
# ============================================================
# Full-stack REST API with:
#   - JWT Authentication
#   - Celery background task queue
#   - PostgreSQL persistence via SQLAlchemy
#   - AWS S3 audio storage
#   - Swagger/OpenAPI documentation
# ============================================================

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import boto3
import uvicorn
from botocore.exceptions import ClientError
from celery import Celery
from celery.result import AsyncResult
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

# ── Local imports ────────────────────────────────────────────
from models import MoodClassifier, MusicGenerator, StemSeparator
from utils.audio_utils import (
    audio_to_base64,
    generate_waveform_peaks,
    get_audio_info,
    validate_audio_file,
)

load_dotenv()

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

class Settings:
    # Application
    APP_NAME: str = "AI Music Remix & Mood Generator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-use-secrets-module")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/music_remix_db"
    )

    # Redis / Celery
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

    # AWS S3
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "music-remix-audio")
    S3_PRESIGNED_URL_EXPIRY: int = int(os.getenv("S3_PRESIGNED_URL_EXPIRY", "3600"))

    # AI Models
    AI_DEVICE: str = os.getenv("AI_DEVICE", "cpu")           # 'cuda' or 'cpu'
    MUSICGEN_MODEL: str = os.getenv("MUSICGEN_MODEL", "facebook/musicgen-small")
    DEMUCS_MODEL: str = os.getenv("DEMUCS_MODEL", "htdemucs")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "/tmp/models")

    # Upload limits
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    ALLOWED_AUDIO_EXTENSIONS: set = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,https://your-frontend.vercel.app"
    ).split(",")


settings = Settings()


# ═══════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═══════════════════════════════════════════════════════════════

Base = declarative_base()


class UserDB(Base):
    """User account table."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    audio_files = relationship("AudioFileDB", back_populates="user", cascade="all, delete-orphan")
    tasks = relationship("TaskDB", back_populates="user", cascade="all, delete-orphan")


class AudioFileDB(Base):
    """Uploaded and generated audio files."""
    __tablename__ = "audio_files"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=True)
    s3_key = Column(String(1000), nullable=True)
    s3_bucket = Column(String(255), nullable=True)
    file_type = Column(String(50), nullable=False)   # 'upload', 'stem', 'remix', 'generated'
    format = Column(String(20), nullable=False)       # 'mp3', 'wav', 'flac'
    duration_seconds = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    parent_file_id = Column(String(36), ForeignKey("audio_files.id"), nullable=True)
    stem_type = Column(String(50), nullable=True)    # 'vocals', 'drums', 'bass', 'other'
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("UserDB", back_populates="audio_files")
    parent = relationship("AudioFileDB", remote_side=[id])
    tasks = relationship("TaskDB", back_populates="audio_file")
    mood_results = relationship("MoodResultDB", back_populates="audio_file")


class TaskDB(Base):
    """Celery background task tracking."""
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    celery_task_id = Column(String(255), unique=True, nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    audio_file_id = Column(String(36), ForeignKey("audio_files.id"), nullable=True)
    task_type = Column(String(100), nullable=False)  # 'stem_separation', 'mood_classify', etc.
    status = Column(String(50), default="pending")   # pending, running, success, failure
    progress = Column(Float, default=0.0)
    result_data = Column(Text, nullable=True)        # JSON blob
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("UserDB", back_populates="tasks")
    audio_file = relationship("AudioFileDB", back_populates="tasks")


class MoodResultDB(Base):
    """Mood classification results."""
    __tablename__ = "mood_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    audio_file_id = Column(String(36), ForeignKey("audio_files.id"), nullable=False)
    primary_mood = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    valence = Column(Float, nullable=False)
    arousal = Column(Float, nullable=False)
    suggested_genres = Column(Text, nullable=True)   # JSON array string
    mood_probabilities = Column(Text, nullable=True) # JSON object string
    tempo_bpm = Column(Float, nullable=True)
    key_signature = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    audio_file = relationship("AudioFileDB", back_populates="mood_results")


# ═══════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized.")


# ═══════════════════════════════════════════════════════════════
# CELERY SETUP
# ═══════════════════════════════════════════════════════════════

celery_app = Celery(
    "music_remix",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=86400,          # 24h
)


# ── Celery Tasks ─────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.separate_stems", max_retries=2)
def task_separate_stems(
    self,
    audio_path: str,
    task_id: str,
    user_id: str,
    model: str = "htdemucs",
    output_dir: Optional[str] = None,
) -> dict:
    """Background task: separate audio into stems using Demucs."""
    db = SessionLocal()
    try:
        task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
        if task_db:
            task_db.status = "running"
            task_db.started_at = datetime.now(timezone.utc)
            db.commit()

        separator = StemSeparator(backend="demucs", model=model, device=settings.AI_DEVICE)
        result = separator.separate(audio_path, output_dir=output_dir)

        if not result.success:
            raise RuntimeError(result.error or "Separation failed")

        # Upload stems to S3
        stem_s3_paths = {}
        for stem_name, stem_data in result.stems.items():
            if stem_data.output_path:
                s3_key = f"stems/{user_id}/{task_id}/{stem_name}.wav"
                s3_url = upload_to_s3(stem_data.output_path, s3_key)
                stem_s3_paths[stem_name] = {
                    "s3_key": s3_key,
                    "s3_url": s3_url,
                    "duration": stem_data.duration_seconds,
                    "peak_amplitude": stem_data.peak_amplitude,
                }

        result_data = {
            "stems": stem_s3_paths,
            "model_used": result.model_name,
            "processing_time": result.processing_time_seconds,
            "input_duration": result.input_duration_seconds,
        }

        if task_db:
            import json
            task_db.status = "success"
            task_db.progress = 1.0
            task_db.result_data = json.dumps(result_data)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()

        return result_data

    except Exception as exc:
        if task_db:
            task_db.status = "failure"
            task_db.error_message = str(exc)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()
        logger.exception("Stem separation task failed: %s", exc)
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


@celery_app.task(bind=True, name="tasks.classify_mood", max_retries=2)
def task_classify_mood(self, audio_path: str, task_id: str, user_id: str) -> dict:
    """Background task: classify mood of audio."""
    db = SessionLocal()
    try:
        task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
        if task_db:
            task_db.status = "running"
            task_db.started_at = datetime.now(timezone.utc)
            db.commit()

        classifier = MoodClassifier(model_path=settings.MODEL_DIR)
        result = classifier.classify(audio_path)

        if not result.success:
            raise RuntimeError(result.error or "Classification failed")

        result_data = result.to_dict()

        if task_db:
            import json
            task_db.status = "success"
            task_db.progress = 1.0
            task_db.result_data = json.dumps(result_data)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()

        return result_data

    except Exception as exc:
        if task_db:
            task_db.status = "failure"
            task_db.error_message = str(exc)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


@celery_app.task(bind=True, name="tasks.generate_music", max_retries=2)
def task_generate_music(
    self,
    prompt: str,
    task_id: str,
    user_id: str,
    duration: float = 10.0,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    output_format: str = "mp3",
) -> dict:
    """Background task: generate music from text prompt."""
    db = SessionLocal()
    try:
        task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
        if task_db:
            task_db.status = "running"
            task_db.started_at = datetime.now(timezone.utc)
            db.commit()

        generator = MusicGenerator(
            backend="musicgen",
            model=settings.MUSICGEN_MODEL,
            device=settings.AI_DEVICE,
        )

        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
            tmp_path = tmp.name

        result = generator.generate_from_text(
            prompt=prompt,
            duration=duration,
            genre=genre,
            mood=mood,
            output_path=tmp_path,
            output_format=output_format,
        )

        if not result.success:
            raise RuntimeError(result.error or "Generation failed")

        # Upload to S3
        s3_key = f"generated/{user_id}/{task_id}/generated.{output_format}"
        s3_url = upload_to_s3(result.output_path, s3_key)

        result_data = {
            **result.to_dict(),
            "s3_key": s3_key,
            "s3_url": s3_url,
        }

        if task_db:
            import json
            task_db.status = "success"
            task_db.progress = 1.0
            task_db.result_data = json.dumps(result_data)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()

        # Cleanup temp file
        os.unlink(tmp_path)
        return result_data

    except Exception as exc:
        if task_db:
            task_db.status = "failure"
            task_db.error_message = str(exc)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


@celery_app.task(bind=True, name="tasks.remix_genre", max_retries=2)
def task_remix_genre(
    self,
    audio_path: str,
    genre: str,
    task_id: str,
    user_id: str,
    output_format: str = "mp3",
) -> dict:
    """Background task: remix audio into a target genre."""
    db = SessionLocal()
    try:
        task_db = db.query(TaskDB).filter(TaskDB.id == task_id).first()
        if task_db:
            task_db.status = "running"
            task_db.started_at = datetime.now(timezone.utc)
            db.commit()

        generator = MusicGenerator(
            backend="musicgen",
            model=settings.MUSICGEN_MODEL,
            device=settings.AI_DEVICE,
        )

        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp:
            tmp_path = tmp.name

        result = generator.remix_to_genre(
            input_path=audio_path,
            genre=genre,
            output_path=tmp_path,
            output_format=output_format,
        )

        if not result.success:
            raise RuntimeError(result.error or "Remix failed")

        s3_key = f"remixes/{user_id}/{task_id}/remix_{genre}.{output_format}"
        s3_url = upload_to_s3(result.output_path, s3_key)

        result_data = {
            **result.to_dict(),
            "s3_key": s3_key,
            "s3_url": s3_url,
        }

        if task_db:
            import json
            task_db.status = "success"
            task_db.progress = 1.0
            task_db.result_data = json.dumps(result_data)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()

        os.unlink(tmp_path)
        return result_data

    except Exception as exc:
        if task_db:
            task_db.status = "failure"
            task_db.error_message = str(exc)
            task_db.completed_at = datetime.now(timezone.utc)
            db.commit()
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload.update({"exp": expire, "type": "access"})
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload.update({"exp": expire, "type": "refresh"})
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> UserDB:
    payload = decode_token(credentials.credentials)
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


# ═══════════════════════════════════════════════════════════════
# S3 UTILITIES
# ═══════════════════════════════════════════════════════════════

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION,
    )


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a file to S3 and return a presigned URL."""
    if not settings.AWS_ACCESS_KEY_ID:
        logger.warning("AWS credentials not configured; skipping S3 upload")
        return local_path

    try:
        s3 = get_s3_client()
        s3.upload_file(
            local_path,
            settings.S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": "audio/wav"},
        )
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": s3_key},
            ExpiresIn=settings.S3_PRESIGNED_URL_EXPIRY,
        )
        logger.info("Uploaded to S3: %s", s3_key)
        return url
    except ClientError as e:
        logger.error("S3 upload failed: %s", e)
        return local_path


def download_from_s3(s3_key: str, local_path: str) -> str:
    """Download a file from S3."""
    s3 = get_s3_client()
    s3.download_file(settings.S3_BUCKET_NAME, s3_key, local_path)
    return local_path


# ═══════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores/hyphens allowed)")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    is_premium: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60


class TaskResponse(BaseModel):
    task_id: str
    celery_task_id: Optional[str]
    status: str
    progress: float
    task_type: str
    created_at: datetime
    result_data: Optional[dict] = None
    error_message: Optional[str] = None


class GenerateMusicRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    duration: float = Field(default=10.0, ge=1.0, le=30.0)
    genre: Optional[str] = None
    mood: Optional[str] = None
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    output_format: str = Field(default="mp3", pattern="^(mp3|wav|flac)$")


class RemixRequest(BaseModel):
    audio_file_id: str
    genre: str
    output_format: str = Field(default="mp3", pattern="^(mp3|wav|flac)$")
    preserve_tempo: bool = True


class TempoRequest(BaseModel):
    audio_file_id: str
    target_bpm: Optional[float] = Field(default=None, ge=40.0, le=300.0)
    speed_factor: Optional[float] = Field(default=None, ge=0.25, le=4.0)
    preserve_pitch: bool = True
    output_format: str = Field(default="wav", pattern="^(mp3|wav|flac)$")


class PitchRequest(BaseModel):
    audio_file_id: str
    semitones: float = Field(..., ge=-24.0, le=24.0)
    output_format: str = Field(default="wav", pattern="^(mp3|wav|flac)$")


class AudioFileResponse(BaseModel):
    id: str
    filename: str
    original_filename: Optional[str]
    file_type: str
    format: str
    duration_seconds: Optional[float]
    sample_rate: Optional[int]
    file_size_bytes: Optional[int]
    s3_key: Optional[str]
    created_at: datetime


class WaveformResponse(BaseModel):
    peaks: List[float]
    duration_seconds: float
    sample_rate: int
    channels: int


class MoodResponse(BaseModel):
    primary_mood: str
    confidence: float
    valence: float
    arousal: float
    mood_probabilities: Dict[str, float]
    suggested_genres: List[str]
    tempo_bpm: Optional[float]
    key_signature: Optional[str]


# ═══════════════════════════════════════════════════════════════
# APPLICATION LIFESPAN
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("🎵  %s v%s starting...", settings.APP_NAME, settings.APP_VERSION)
    init_db()
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs("/tmp/uploads", exist_ok=True)
    os.makedirs("/tmp/stems", exist_ok=True)
    os.makedirs("/tmp/generated", exist_ok=True)
    logger.info("✅  Application ready.")
    yield
    logger.info("👋  Application shutting down.")


# ═══════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title=settings.APP_NAME,
    description="""
## AI Music Remix & Mood Generator API

Powerful AI-driven music processing:

- 🎵 **Stem Separation** — Separate audio into vocals, drums, bass, other
- 🎭 **Mood Classification** — Detect energy, valence, and discrete mood
- 🎸 **Genre Remixing** — Transform tracks into EDM, jazz, lo-fi, and more
- 🤖 **Text-to-Music** — Generate music from natural language prompts
- 🎼 **Tempo & Pitch** — Time-stretch and pitch-shift with precision
- 📊 **Waveform Preview** — Visual waveform data for frontend players

### Authentication
Use JWT Bearer tokens. Get a token via `/auth/login` then include:
```
Authorization: Bearer <your-token>
```
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ═══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register a new user account",
)
async def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Create a new user account."""
    if db.query(UserDB).filter(UserDB.email == user_in.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(UserDB).filter(UserDB.username == user_in.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = UserDB(
        email=user_in.email,
        username=user_in.username,
        hashed_password=hash_password(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("New user registered: %s", user.email)
    return user


@app.post(
    "/auth/login",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Authenticate and receive JWT tokens",
)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access + refresh tokens."""
    user = db.query(UserDB).filter(UserDB.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    access_token = create_access_token({"sub": user.id, "email": user.email})
    refresh_token = create_refresh_token({"sub": user.id})
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@app.post(
    "/auth/refresh",
    response_model=TokenResponse,
    tags=["Authentication"],
    summary="Refresh access token",
)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """Use a refresh token to get a new access token."""
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    user = db.query(UserDB).filter(UserDB.id == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    access_token = create_access_token({"sub": user.id, "email": user.email})
    new_refresh = create_refresh_token({"sub": user.id})
    return TokenResponse(access_token=access_token, refresh_token=new_refresh)


@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_me(current_user: UserDB = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return current_user


# ═══════════════════════════════════════════════════════════════
# AUDIO UPLOAD ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/audio/upload",
    response_model=AudioFileResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Audio Management"],
    summary="Upload an audio file",
)
async def upload_audio(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload an audio file (MP3, WAV, FLAC, OGG, M4A, AAC).
    Max size: 100 MB. File is stored in S3 and metadata persisted in DB.
    """
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {suffix}"
        )

    # Read file
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.MAX_FILE_SIZE_MB} MB)"
        )

    # Save to temp location
    file_id = str(uuid.uuid4())
    tmp_path = f"/tmp/uploads/{file_id}{suffix}"

    with open(tmp_path, "wb") as f:
        f.write(content)

    # Validate audio
    is_valid, msg = validate_audio_file(tmp_path)
    if not is_valid:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"Invalid audio file: {msg}")

    # Get audio info
    info = get_audio_info(tmp_path)

    # Upload to S3
    s3_key = f"uploads/{current_user.id}/{file_id}{suffix}"
    s3_url = upload_to_s3(tmp_path, s3_key)

    # Persist metadata
    audio_file = AudioFileDB(
        id=file_id,
        user_id=current_user.id,
        filename=f"{file_id}{suffix}",
        original_filename=file.filename,
        s3_key=s3_key,
        s3_bucket=settings.S3_BUCKET_NAME,
        file_type="upload",
        format=suffix.lstrip("."),
        duration_seconds=info.get("duration_seconds"),
        sample_rate=info.get("sample_rate"),
        channels=info.get("channels"),
        file_size_bytes=len(content),
    )
    db.add(audio_file)
    db.commit()
    db.refresh(audio_file)

    logger.info("Audio uploaded: %s (%s)", file.filename, file_id)
    return audio_file


@app.get(
    "/audio",
    response_model=List[AudioFileResponse],
    tags=["Audio Management"],
    summary="List user's audio files",
)
async def list_audio_files(
    file_type: Optional[str] = Query(None, description="Filter by type: upload, stem, remix, generated"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List audio files belonging to the current user."""
    query = db.query(AudioFileDB).filter(AudioFileDB.user_id == current_user.id)
    if file_type:
        query = query.filter(AudioFileDB.file_type == file_type)
    return query.order_by(AudioFileDB.created_at.desc()).offset(offset).limit(limit).all()


@app.get(
    "/audio/{file_id}",
    response_model=AudioFileResponse,
    tags=["Audio Management"],
)
async def get_audio_file(
    file_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get details of a specific audio file."""
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return audio_file


@app.delete("/audio/{file_id}", status_code=204, tags=["Audio Management"])
async def delete_audio_file(
    file_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an audio file and its S3 object."""
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Delete from S3
    if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
        try:
            s3 = get_s3_client()
            s3.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=audio_file.s3_key)
        except ClientError:
            pass

    db.delete(audio_file)
    db.commit()


# ═══════════════════════════════════════════════════════════════
# WAVEFORM ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/audio/{file_id}/waveform",
    response_model=WaveformResponse,
    tags=["Visualization"],
    summary="Get waveform peak data for visualization",
)
async def get_waveform(
    file_id: str,
    n_peaks: int = Query(1000, ge=100, le=5000),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return waveform peak data suitable for WaveSurfer.js.
    Downloads from S3 if needed, then computes peaks.
    """
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    tmp_path = f"/tmp/waveform_{file_id}.{audio_file.format}"

    try:
        if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
            download_from_s3(audio_file.s3_key, tmp_path)
        elif not os.path.exists(tmp_path):
            raise HTTPException(status_code=503, detail="Audio file not accessible")

        from utils.audio_utils import load_audio
        waveform, sr = load_audio(tmp_path, mono=True)
        peaks = generate_waveform_peaks(waveform, sr, n_peaks=n_peaks)

        return WaveformResponse(
            peaks=peaks,
            duration_seconds=audio_file.duration_seconds or len(waveform) / sr,
            sample_rate=audio_file.sample_rate or sr,
            channels=audio_file.channels or 1,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════
# STEM SEPARATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/audio/{file_id}/separate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Stem Separation"],
    summary="Start stem separation (background task)",
)
async def start_stem_separation(
    file_id: str,
    model: str = Query("htdemucs", description="Demucs model: htdemucs, htdemucs_6s, mdx"),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Queue stem separation job. Returns a task_id to poll for results.
    Stems (vocals, drums, bass, other) will be uploaded to S3 when complete.
    """
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Download from S3 to temp
    tmp_path = f"/tmp/uploads/{file_id}.{audio_file.format}"
    if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
        download_from_s3(audio_file.s3_key, tmp_path)

    # Create task record
    task_id = str(uuid.uuid4())
    task_db = TaskDB(
        id=task_id,
        user_id=current_user.id,
        audio_file_id=file_id,
        task_type="stem_separation",
        status="pending",
    )
    db.add(task_db)
    db.commit()

    # Dispatch Celery task
    celery_task = task_separate_stems.delay(tmp_path, task_id, current_user.id, model)

    task_db.celery_task_id = celery_task.id
    db.commit()
    db.refresh(task_db)

    return TaskResponse(
        task_id=task_db.id,
        celery_task_id=celery_task.id,
        status=task_db.status,
        progress=task_db.progress,
        task_type=task_db.task_type,
        created_at=task_db.created_at,
    )


# ═══════════════════════════════════════════════════════════════
# MOOD CLASSIFICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/audio/{file_id}/classify",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Mood Classification"],
    summary="Classify the mood of an audio file",
)
async def classify_mood(
    file_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Queue mood classification. Returns a task_id for polling."""
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    tmp_path = f"/tmp/uploads/{file_id}.{audio_file.format}"
    if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
        download_from_s3(audio_file.s3_key, tmp_path)

    task_id = str(uuid.uuid4())
    task_db = TaskDB(
        id=task_id,
        user_id=current_user.id,
        audio_file_id=file_id,
        task_type="mood_classification",
        status="pending",
    )
    db.add(task_db)
    db.commit()

    celery_task = task_classify_mood.delay(tmp_path, task_id, current_user.id)

    task_db.celery_task_id = celery_task.id
    db.commit()
    db.refresh(task_db)

    return TaskResponse(
        task_id=task_db.id,
        celery_task_id=celery_task.id,
        status=task_db.status,
        progress=task_db.progress,
        task_type=task_db.task_type,
        created_at=task_db.created_at,
    )


# ═══════════════════════════════════════════════════════════════
# MUSIC GENERATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/generate",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Music Generation"],
    summary="Generate music from a text prompt",
)
async def generate_music(
    request: GenerateMusicRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate original music from a text description using Meta MusicGen.

    Example prompts:
    - "relaxing lo-fi hip hop with soft piano, 80 BPM"
    - "energetic EDM drop with massive bass and synths"
    - "sad indie folk guitar ballad in minor key"
    """
    task_id = str(uuid.uuid4())
    task_db = TaskDB(
        id=task_id,
        user_id=current_user.id,
        task_type="music_generation",
        status="pending",
    )
    db.add(task_db)
    db.commit()

    celery_task = task_generate_music.delay(
        request.prompt,
        task_id,
        current_user.id,
        request.duration,
        request.genre,
        request.mood,
        request.output_format,
    )

    task_db.celery_task_id = celery_task.id
    db.commit()
    db.refresh(task_db)

    return TaskResponse(
        task_id=task_db.id,
        celery_task_id=celery_task.id,
        status=task_db.status,
        progress=task_db.progress,
        task_type=task_db.task_type,
        created_at=task_db.created_at,
    )


@app.post(
    "/remix",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Music Generation"],
    summary="Remix an uploaded track into a different genre",
)
async def remix_to_genre(
    request: RemixRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remix an existing audio file into a new genre using AI generation."""
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == request.audio_file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    tmp_path = f"/tmp/uploads/{audio_file.id}.{audio_file.format}"
    if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
        download_from_s3(audio_file.s3_key, tmp_path)

    task_id = str(uuid.uuid4())
    task_db = TaskDB(
        id=task_id,
        user_id=current_user.id,
        audio_file_id=request.audio_file_id,
        task_type="genre_remix",
        status="pending",
    )
    db.add(task_db)
    db.commit()

    celery_task = task_remix_genre.delay(
        tmp_path, request.genre, task_id, current_user.id, request.output_format
    )

    task_db.celery_task_id = celery_task.id
    db.commit()
    db.refresh(task_db)

    return TaskResponse(
        task_id=task_db.id,
        celery_task_id=celery_task.id,
        status=task_db.status,
        progress=task_db.progress,
        task_type=task_db.task_type,
        created_at=task_db.created_at,
    )


# ═══════════════════════════════════════════════════════════════
# TEMPO / PITCH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/audio/{file_id}/tempo",
    tags=["Audio Processing"],
    summary="Change tempo of an audio file",
)
async def adjust_tempo(
    file_id: str,
    request: TempoRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Time-stretch or speed-change an audio file.
    `target_bpm` auto-detects source tempo and rescales.
    `speed_factor` applies a direct multiplier (1.5 = 50% faster).
    """
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    tmp_path = f"/tmp/uploads/{file_id}.{audio_file.format}"
    if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
        download_from_s3(audio_file.s3_key, tmp_path)

    gen = MusicGenerator(device=settings.AI_DEVICE)
    output_path = f"/tmp/generated/{file_id}_tempo.{request.output_format}"

    result = gen.change_tempo(
        tmp_path,
        target_bpm=request.target_bpm,
        speed_factor=request.speed_factor,
        output_path=output_path,
        output_format=request.output_format,
        preserve_pitch=request.preserve_pitch,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    s3_key = f"processed/{current_user.id}/{file_id}_tempo.{request.output_format}"
    s3_url = upload_to_s3(result.output_path, s3_key)

    return {"success": True, "download_url": s3_url, "duration_seconds": result.duration_seconds}


@app.post(
    "/audio/{file_id}/pitch",
    tags=["Audio Processing"],
    summary="Shift pitch of an audio file",
)
async def adjust_pitch(
    file_id: str,
    request: PitchRequest,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Shift the pitch of an audio file by N semitones (±24)."""
    audio_file = db.query(AudioFileDB).filter(
        AudioFileDB.id == file_id,
        AudioFileDB.user_id == current_user.id,
    ).first()
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found")

    tmp_path = f"/tmp/uploads/{file_id}.{audio_file.format}"
    if audio_file.s3_key and settings.AWS_ACCESS_KEY_ID:
        download_from_s3(audio_file.s3_key, tmp_path)

    gen = MusicGenerator(device=settings.AI_DEVICE)
    output_path = f"/tmp/generated/{file_id}_pitch.{request.output_format}"

    result = gen.change_pitch(
        tmp_path,
        semitones=request.semitones,
        output_path=output_path,
        output_format=request.output_format,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    s3_key = f"processed/{current_user.id}/{file_id}_pitch.{request.output_format}"
    s3_url = upload_to_s3(result.output_path, s3_key)

    return {"success": True, "download_url": s3_url, "duration_seconds": result.duration_seconds}


# ═══════════════════════════════════════════════════════════════
# TASK STATUS ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get(
    "/tasks/{task_id}",
    response_model=TaskResponse,
    tags=["Tasks"],
    summary="Poll background task status",
)
async def get_task_status(
    task_id: str,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Poll the status of a background task (stem separation, generation, etc.).
    Status values: pending | running | success | failure
    """
    task = db.query(TaskDB).filter(
        TaskDB.id == task_id,
        TaskDB.user_id == current_user.id,
    ).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Sync with Celery if still pending/running
    if task.celery_task_id and task.status in ("pending", "running"):
        celery_result = AsyncResult(task.celery_task_id, app=celery_app)
        if celery_result.state == "SUCCESS":
            task.status = "success"
            task.progress = 1.0
            db.commit()
        elif celery_result.state == "FAILURE":
            task.status = "failure"
            task.error_message = str(celery_result.info)
            db.commit()

    import json
    result_data = None
    if task.result_data:
        try:
            result_data = json.loads(task.result_data)
        except Exception:
            pass

    return TaskResponse(
        task_id=task.id,
        celery_task_id=task.celery_task_id,
        status=task.status,
        progress=task.progress,
        task_type=task.task_type,
        created_at=task.created_at,
        result_data=result_data,
        error_message=task.error_message,
    )


@app.get(
    "/tasks",
    response_model=List[TaskResponse],
    tags=["Tasks"],
    summary="List user's tasks",
)
async def list_tasks(
    status_filter: Optional[str] = Query(None, description="Filter: pending, running, success, failure"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all background tasks for the current user."""
    import json
    query = db.query(TaskDB).filter(TaskDB.user_id == current_user.id)
    if status_filter:
        query = query.filter(TaskDB.status == status_filter)

    tasks = query.order_by(TaskDB.created_at.desc()).offset(offset).limit(limit).all()

    results = []
    for task in tasks:
        result_data = None
        if task.result_data:
            try:
                result_data = json.loads(task.result_data)
            except Exception:
                pass
        results.append(TaskResponse(
            task_id=task.id,
            celery_task_id=task.celery_task_id,
            status=task.status,
            progress=task.progress,
            task_type=task.task_type,
            created_at=task.created_at,
            result_data=result_data,
            error_message=task.error_message,
        ))
    return results


# ═══════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/genres", tags=["Utilities"], summary="List available remix genres")
async def list_genres():
    """Return all supported genres for remix operations."""
    from models.music_generator import GENRE_PROMPT_MAP
    return {"genres": list(GENRE_PROMPT_MAP.keys())}


@app.get("/moods", tags=["Utilities"], summary="List mood taxonomy")
async def list_moods():
    """Return available mood labels and their valence/arousal coordinates."""
    from models.mood_classifier import MoodLabel, MOOD_COORDINATES, MOOD_TO_GENRE_MAP
    return {
        "moods": [
            {
                "label": mood.value,
                "valence": MOOD_COORDINATES[mood][0],
                "arousal": MOOD_COORDINATES[mood][1],
                "suggested_genres": MOOD_TO_GENRE_MAP[mood],
            }
            for mood in MoodLabel
        ]
    }


@app.get("/health", tags=["Utilities"], summary="Health check")
async def health_check():
    """API health status."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "ok",
            "celery": "ok" if _check_celery() else "degraded",
            "database": "ok" if _check_db() else "degraded",
        },
    }


def _check_celery() -> bool:
    try:
        celery_app.control.ping(timeout=1)
        return True
    except Exception:
        return False


def _check_db() -> bool:
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
