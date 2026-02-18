# 🎵 SonicAI Music Remix & Mood Generator

A production-grade, full-stack AI-powered music platform that enables users to upload audio files, separate stems (vocals, drums, bass), classify moods using ML, remix songs into different genres, generate new music from text prompts, manipulate tempo and pitch, preview interactive waveforms, and download results — all backed by scalable cloud infrastructure.

---

---

## 🛠️ Technology Stack

### Backend — Python / FastAPI

| Layer | Technology |
|---|---|
| Framework | FastAPI 0.111+ |
| ASGI Server | Uvicorn + Gunicorn |
| ORM | SQLAlchemy 2.x (async) |
| Migrations | Alembic |
| Task Queue | Celery 5.x |
| Message Broker | Redis 7+ |
| Authentication | JWT (python-jose) + passlib (bcrypt) |
| Object Storage | boto3 → AWS S3 (or MinIO/Backblaze) |
| Validation | Pydantic v2 |
| API Docs | Swagger UI + ReDoc (auto-generated) |

### AI / ML Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch 2.x · Torchaudio |
| Audio Analysis | Librosa · SoundFile · Pydub |
| Stem Separation | Demucs (Meta) · Spleeter (Deezer) |
| Music Generation | Meta MusicGen · Stable Audio · Riffusion |
| NLP / Transformers | HuggingFace Transformers · Diffusers |
| Mood Classification | scikit-learn · XGBoost · MFCC features |
| Audio I/O | FFmpeg · Pydub · SoundFile |

### Frontend — Next.js 14

| Category | Technology |
|---|---|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS 3.x |
| UI Components | Shadcn UI + Lucide React |
| Animation | Framer Motion |
| State Management | Zustand + Redux Toolkit |
| Audio Visualization | WaveSurfer.js |
| Audio Synthesis | Tone.js |
| Charts | Chart.js + react-chartjs-2 |
| HTTP Client | Axios |
| File Upload | React-Dropzone |

### Infrastructure & DevOps

| Area | Technology |
|---|---|
| CI/CD | GitHub Actions |
| Frontend Deploy | Vercel |
| Backend Deploy | Render (Web Service) |
| Background Workers | Render (Background Worker) |
| Database | Render Managed PostgreSQL |
| Cache/Broker | Render Managed Redis |
| Storage | AWS S3 (or Backblaze B2 / Cloudflare R2) |
| Secrets | Render Environment Variables / GitHub Secrets |
| Monitoring | Render built-in metrics + Sentry (optional) |

---

## 📊 Database Schema

```sql
-- Users
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    username    VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active   BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    plan        VARCHAR(20) DEFAULT 'free',   -- free | pro | enterprise
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Audio Files
CREATE TABLE audio_files (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_name VARCHAR(255) NOT NULL,
    s3_key       VARCHAR(1000) NOT NULL,       -- S3 object key
    s3_bucket    VARCHAR(255) NOT NULL,
    file_size    BIGINT NOT NULL,              -- bytes
    duration     FLOAT,                        -- seconds
    format       VARCHAR(20),                  -- mp3, wav, flac, ogg
    sample_rate  INTEGER,
    channels     INTEGER,
    bit_depth    INTEGER,
    waveform_data JSONB,                       -- cached waveform peaks
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Processing Jobs
CREATE TABLE processing_jobs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    audio_file_id UUID REFERENCES audio_files(id) ON DELETE SET NULL,
    celery_task_id VARCHAR(255),               -- Celery task UUID
    job_type      VARCHAR(50) NOT NULL,        -- stem_separation | mood_classify | remix | generate | pitch | tempo
    status        VARCHAR(20) DEFAULT 'pending', -- pending | processing | completed | failed | cancelled
    progress      SMALLINT DEFAULT 0,          -- 0-100
    input_params  JSONB,                       -- job-specific parameters
    result_s3_keys JSONB,                      -- output file locations
    error_message TEXT,
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Mood Classifications
CREATE TABLE mood_classifications (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    audio_file_id UUID NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    job_id        UUID REFERENCES processing_jobs(id) ON DELETE SET NULL,
    primary_mood  VARCHAR(50) NOT NULL,        -- happy | sad | energetic | calm | angry | melancholic | romantic
    confidence    FLOAT NOT NULL,              -- 0.0 - 1.0
    mood_scores   JSONB NOT NULL,              -- {happy: 0.8, sad: 0.1, ...}
    audio_features JSONB,                      -- MFCC, tempo, spectral features
    model_version VARCHAR(50),
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Stems (separated tracks)
CREATE TABLE stems (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id         UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    audio_file_id  UUID NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    stem_type      VARCHAR(20) NOT NULL,       -- vocals | drums | bass | other | guitar | piano
    s3_key         VARCHAR(1000) NOT NULL,
    duration       FLOAT,
    model_used     VARCHAR(50),                -- demucs | spleeter
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Remixes
CREATE TABLE remixes (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id           UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    original_file_id UUID NOT NULL REFERENCES audio_files(id),
    result_file_id   UUID REFERENCES audio_files(id),
    genre            VARCHAR(50),              -- edm | jazz | hiphop | rock | classical | lofi
    tempo_change     FLOAT DEFAULT 0,          -- percentage change (-50 to +100)
    pitch_change     FLOAT DEFAULT 0,          -- semitones (-12 to +12)
    effects_applied  JSONB,                    -- {reverb: true, eq: {...}}
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Generated Music
CREATE TABLE generated_tracks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id       UUID NOT NULL REFERENCES processing_jobs(id) ON DELETE CASCADE,
    prompt       TEXT NOT NULL,
    duration     FLOAT NOT NULL,
    model_used   VARCHAR(50) NOT NULL,         -- musicgen | stable-audio | riffusion
    result_s3_key VARCHAR(1000),
    seed         BIGINT,
    model_params JSONB,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_audio_files_user_id ON audio_files(user_id);
CREATE INDEX idx_processing_jobs_user_id ON processing_jobs(user_id);
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_celery_task ON processing_jobs(celery_task_id);
CREATE INDEX idx_stems_job_id ON stems(job_id);
CREATE INDEX idx_mood_classifications_audio_file ON mood_classifications(audio_file_id);
CREATE INDEX idx_remixes_user_id ON remixes(user_id);
```

---

## 🔌 REST API Endpoints

> Full interactive docs available at `/docs` (Swagger UI) and `/redoc` (ReDoc) after startup.

### Authentication
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/auth/register` | Register new user | — |
| POST | `/api/v1/auth/login` | Login → JWT tokens | — |
| POST | `/api/v1/auth/refresh` | Refresh access token | — |
| POST | `/api/v1/auth/logout` | Invalidate refresh token | ✅ |
| GET | `/api/v1/auth/me` | Get current user profile | ✅ |
| PATCH | `/api/v1/auth/me` | Update user profile | ✅ |

### Audio Files
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/audio/upload` | Upload audio (multipart/form-data) | ✅ |
| GET | `/api/v1/audio` | List user's audio files (paginated) | ✅ |
| GET | `/api/v1/audio/{id}` | Get file metadata | ✅ |
| DELETE | `/api/v1/audio/{id}` | Delete file + S3 object | ✅ |
| GET | `/api/v1/audio/{id}/download` | Presigned S3 download URL | ✅ |
| GET | `/api/v1/audio/{id}/waveform` | Waveform peak data | ✅ |
| GET | `/api/v1/audio/{id}/spectrogram` | Spectrogram image (base64) | ✅ |
| GET | `/api/v1/audio/{id}/features` | Audio feature analysis | ✅ |

### Stem Separation
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/stems/separate` | Start stem separation job | ✅ |
| GET | `/api/v1/stems/jobs/{jobId}` | Job status + progress | ✅ |
| GET | `/api/v1/stems/jobs/{jobId}/results` | Get stem download URLs | ✅ |
| GET | `/api/v1/stems/audio/{audioId}` | List all stems for file | ✅ |

### Mood Classification
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/mood/classify` | Classify audio mood | ✅ |
| GET | `/api/v1/mood/jobs/{jobId}` | Classification job status | ✅ |
| GET | `/api/v1/mood/audio/{audioId}` | Get mood results for file | ✅ |
| GET | `/api/v1/mood/history` | User's mood classification history | ✅ |

### Music Generation (Text-to-Music)
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/generate/text-to-music` | Generate from text prompt | ✅ |
| GET | `/api/v1/generate/jobs/{jobId}` | Generation job status | ✅ |
| GET | `/api/v1/generate/jobs/{jobId}/download` | Presigned URL for result | ✅ |
| GET | `/api/v1/generate/history` | User's generation history | ✅ |
| GET | `/api/v1/generate/models` | List available AI models | ✅ |

### Audio Remixing
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| POST | `/api/v1/remix/genre` | Genre transformation | ✅ |
| POST | `/api/v1/remix/tempo` | Tempo adjustment | ✅ |
| POST | `/api/v1/remix/pitch` | Pitch shifting | ✅ |
| POST | `/api/v1/remix/effects` | Apply audio effects | ✅ |
| GET | `/api/v1/remix/jobs/{jobId}` | Remix job status | ✅ |
| GET | `/api/v1/remix/history` | User's remix history | ✅ |

### Jobs (Generic)
| Method | Endpoint | Description | Auth |
|---|---|---|---|
| GET | `/api/v1/jobs/{jobId}` | Universal job status lookup | ✅ |
| DELETE | `/api/v1/jobs/{jobId}` | Cancel a pending job | ✅ |
| GET | `/api/v1/jobs` | List all user jobs | ✅ |

### WebSocket
| Channel | Description |
|---|---|
| `ws://host/ws/jobs/{jobId}` | Real-time job progress updates |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm/pnpm
- PostgreSQL 16+ (local dev) or a Render PostgreSQL instance
- Redis 7+ (local dev) or a Render Redis instance
- AWS account for S3 (or Backblaze B2 / Cloudflare R2)
- GPU recommended for AI workloads (NVIDIA CUDA 11.8+)

---

### 1. Clone the Repository

```bash
git clone https://github.com/Gaurav11oo/AI-Music-Remix-Mood-Generator.git
cd AI-Music-Remix-Mood-Generator
```

### 2. Environment Configuration

```bash
# Backend
cp backend/.env.example backend/.env

# Frontend
cp frontend/.env.local.example frontend/.env.local
```

Edit `backend/.env`:

```env
# ── Server ────────────────────────────────────────────────
APP_ENV=development
PORT=8000
API_BASE_URL=http://localhost:8000
DEBUG=true
LOG_LEVEL=debug                               # debug | info | warning | error

# ── Database (PostgreSQL) ─────────────────────────────────
# Constructed from individual parts for local dev:
DB_HOST=localhost
DB_PORT=5432
DB_NAME=music_remix_db
DB_USER=postgres
DB_PASSWORD=your_secure_password
DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# ── Redis ─────────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/0
CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/1
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/2

# ── JWT ───────────────────────────────────────────────────
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY=7d
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# ── File Storage ──────────────────────────────────────────
UPLOAD_DIR=./uploads
TEMP_DIR=./temp
MAX_FILE_SIZE=104857600                       # bytes (100 MB)
MAX_UPLOAD_SIZE_MB=100
ALLOWED_AUDIO_FORMATS=mp3,wav,flac,ogg,m4a,aac
MAX_AUDIO_DURATION_SECONDS=600

# ── AWS S3 ────────────────────────────────────────────────
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=music-remix-storage
S3_PRESIGNED_URL_EXPIRY=3600

# ── AI Model Settings ─────────────────────────────────────
HUGGINGFACE_TOKEN=hf_your_token_here
MUSICGEN_MODEL=facebook/musicgen-small        # or musicgen-medium / musicgen-large
RIFFUSION_MODEL=riffusion/riffusion-model-v1
DEMUCS_MODEL=htdemucs_ft
DEVICE=cuda                                   # cuda | cpu
TORCH_DTYPE=float16                           # float16 | float32

# ── CORS ──────────────────────────────────────────────────
CORS_ORIGIN=http://localhost:3000
CORS_ORIGINS=http://localhost:3000,https://your-vercel-domain.vercel.app

# ── Rate Limiting ─────────────────────────────────────────
RATE_LIMIT_WINDOW_MS=900000                   # 15 minutes
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_PER_MINUTE=60

# ── Audio Processing / Job Queue ──────────────────────────
MAX_CONCURRENT_JOBS=3
JOB_TIMEOUT_MS=600000                         # 10 minutes
```

Edit `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_APP_NAME=AI Music Remix Studio
```

---

### 3. Local Development Setup

**Backend:**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run DB migrations
alembic upgrade head

# Start FastAPI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (separate terminal)
celery -A app.celery_app worker --loglevel=info --concurrency=2 -Q default,ai_heavy

# Start Celery Beat scheduler (separate terminal)
celery -A app.celery_app beat --loglevel=info
```

**Frontend:**

```bash
cd frontend
npm install   # or: pnpm install
npm run dev
```

Access the app at `http://localhost:3000`:
- API Docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

### 4. Production-like Local Run

Point `DATABASE_URL` and `REDIS_URL` at your Render managed instances, then run the same commands as step 3. This lets you develop locally against production-equivalent managed services without any extra tooling.

---

## 📁 Project Structure

```
AI-Music-Remix-Mood-Generator/
│
├── backend/                          # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── celery_app.py             # Celery configuration
│   │   ├── config.py                 # Pydantic Settings
│   │   │
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── router.py         # Aggregate router
│   │   │   │   ├── auth.py
│   │   │   │   ├── audio.py
│   │   │   │   ├── stems.py
│   │   │   │   ├── mood.py
│   │   │   │   ├── generate.py
│   │   │   │   ├── remix.py
│   │   │   │   └── jobs.py
│   │   │
│   │   ├── core/
│   │   │   ├── security.py           # JWT, password hashing
│   │   │   ├── dependencies.py       # FastAPI DI (DB session, current user)
│   │   │   ├── exceptions.py
│   │   │   └── middleware.py         # Rate limiting, logging
│   │   │
│   │   ├── db/
│   │   │   ├── base.py               # SQLAlchemy async engine
│   │   │   ├── session.py
│   │   │   └── models/
│   │   │       ├── user.py
│   │   │       ├── audio_file.py
│   │   │       ├── processing_job.py
│   │   │       ├── mood_classification.py
│   │   │       ├── stem.py
│   │   │       ├── remix.py
│   │   │       └── generated_track.py
│   │   │
│   │   ├── schemas/                  # Pydantic v2 schemas
│   │   │   ├── auth.py
│   │   │   ├── audio.py
│   │   │   ├── job.py
│   │   │   ├── mood.py
│   │   │   └── remix.py
│   │   │
│   │   ├── services/
│   │   │   ├── s3_service.py         # AWS S3 operations
│   │   │   ├── audio_service.py      # Upload, metadata extraction
│   │   │   └── job_service.py        # Job creation + status tracking
│   │   │
│   │   ├── tasks/                    # Celery tasks
│   │   │   ├── stem_tasks.py         # Demucs / Spleeter
│   │   │   ├── mood_tasks.py         # ML mood classifier
│   │   │   ├── remix_tasks.py        # Genre, tempo, pitch
│   │   │   ├── generate_tasks.py     # MusicGen / Riffusion
│   │   │   └── audio_tasks.py        # Waveform, spectrogram
│   │   │
│   │   └── ml/                       # AI/ML modules
│   │       ├── stem_separator.py     # Demucs wrapper
│   │       ├── mood_classifier.py    # XGBoost + scikit-learn
│   │       ├── music_generator.py    # MusicGen / Stable Audio
│   │       ├── riffusion.py          # Riffusion integration
│   │       └── audio_utils.py        # Librosa, Pydub helpers
│   │
│   ├── alembic/                      # DB migration files
│   │   ├── env.py
│   │   └── versions/
│   │
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_auth.py
│   │   ├── test_audio.py
│   │   ├── test_stems.py
│   │   └── test_mood.py
│   │
│   ├── Dockerfile
│   ├── Dockerfile.worker             # Celery worker image
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example
│
├── frontend/                         # Next.js 14 frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx
│   │   │   ├── page.tsx              # Landing page
│   │   │   ├── (auth)/
│   │   │   │   ├── login/page.tsx
│   │   │   │   └── register/page.tsx
│   │   │   ├── dashboard/page.tsx
│   │   │   ├── upload/page.tsx
│   │   │   ├── stems/page.tsx
│   │   │   ├── mood/page.tsx
│   │   │   ├── remix/page.tsx
│   │   │   └── generate/page.tsx
│   │   │
│   │   ├── components/
│   │   │   ├── audio/
│   │   │   │   ├── WaveformPlayer.tsx
│   │   │   │   ├── Spectrogram.tsx
│   │   │   │   ├── StemMixer.tsx
│   │   │   │   └── AudioDropzone.tsx
│   │   │   ├── remix/
│   │   │   │   ├── GenreSelector.tsx
│   │   │   │   ├── TempoSlider.tsx
│   │   │   │   └── PitchControl.tsx
│   │   │   ├── mood/
│   │   │   │   ├── MoodRadar.tsx
│   │   │   │   └── FeatureChart.tsx
│   │   │   ├── generate/
│   │   │   │   ├── PromptBuilder.tsx
│   │   │   │   └── ModelSelector.tsx
│   │   │   ├── jobs/
│   │   │   │   └── JobProgressCard.tsx
│   │   │   └── ui/                   # Shadcn UI components
│   │   │
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts       # Real-time job updates
│   │   │   ├── useWaveSurfer.ts
│   │   │   └── useAudioAnalyser.ts
│   │   │
│   │   ├── lib/
│   │   │   ├── api.ts                # Axios instance + interceptors
│   │   │   ├── store/
│   │   │   │   ├── authStore.ts      # Zustand auth slice
│   │   │   │   ├── audioStore.ts
│   │   │   │   └── jobStore.ts
│   │   │   └── utils.ts
│   │   │
│   │   └── types/
│   │       └── index.ts
│   │
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── next.config.ts
│   └── .env.local.example
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Test + lint on PR
│       ├── cd-backend.yml            # Deploy backend to Render
│       └── cd-frontend.yml           # Deploy frontend to Vercel
│
├── render.yaml                       # Render Blueprint (IaC)
├── Makefile                          # Helper commands
└── README.md
```

---

## ⚙️ GitHub Actions CI/CD

### CI Pipeline (`.github/workflows/ci.yml`)

Triggers on every pull request to `main` or `develop`:

```yaml
name: CI

on:
  pull_request:
    branches: [main, develop]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
      redis:
        image: redis:7
        options: --health-cmd "redis-cli ping"

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r backend/requirements-dev.txt
      - run: cd backend && pytest tests/ --cov=app --cov-report=xml
      - uses: codecov/codecov-action@v4

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "18"
          cache: "npm"
          cache-dependency-path: frontend/package-lock.json
      - run: cd frontend && npm ci && npm run lint && npm run type-check && npm test
```

### CD — Backend to Render (`cd-backend.yml`)

Render auto-deploys on push to `main` via its GitHub integration. The workflow below triggers a manual deploy via the Render API as a fallback or for explicit control:

```yaml
name: CD Backend

on:
  push:
    branches: [main]
    paths: ["backend/**"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    needs: []   # runs after CI passes on main
    steps:
      - name: Trigger Render deploy — API service
        run: |
          curl -X POST \
            "https://api.render.com/v1/services/${{ secrets.RENDER_API_SERVICE_ID }}/deploys" \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{"clearCache": false}'

      - name: Trigger Render deploy — Celery worker service
        run: |
          curl -X POST \
            "https://api.render.com/v1/services/${{ secrets.RENDER_WORKER_SERVICE_ID }}/deploys" \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{"clearCache": false}'

      - name: Wait and verify deploy status
        run: |
          echo "Deploy triggered. Monitor at https://dashboard.render.com"
```

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `RENDER_API_KEY` | Render API key (from Account → API Keys) |
| `RENDER_API_SERVICE_ID` | Service ID of the FastAPI web service |
| `RENDER_WORKER_SERVICE_ID` | Service ID of the Celery background worker |

### CD — Frontend to Vercel (`cd-frontend.yml`)

```yaml
name: CD Frontend

on:
  push:
    branches: [main]
    paths: ["frontend/**"]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: ./frontend
          vercel-args: "--prod"
```

---

## ☁️ Deployment Guide

### Backend → Render

Render is a fully managed cloud platform that handles SSL, scaling, and zero-downtime deploys — no Docker or server config required.

#### Option A — Render Blueprint (recommended)

Define all services declaratively in `render.yaml` at the repo root:

```yaml
# render.yaml
services:
  # ── FastAPI Web Service ──────────────────────────────────
  - type: web
    name: music-remix-api
    runtime: python
    region: oregon
    plan: standard
    buildCommand: pip install -r backend/requirements.txt && cd backend && alembic upgrade head
    startCommand: cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: APP_ENV
        value: production
      - key: DATABASE_URL
        fromDatabase:
          name: music-remix-db
          property: connectionString
      - key: REDIS_URL
        fromService:
          name: music-remix-redis
          type: redis
          property: connectionString
      - key: JWT_SECRET
        sync: false          # prompt on first deploy
      - key: AWS_ACCESS_KEY_ID
        sync: false
      - key: AWS_SECRET_ACCESS_KEY
        sync: false
      - key: S3_BUCKET_NAME
        sync: false
      - key: HUGGINGFACE_TOKEN
        sync: false

  # ── Celery Background Worker ─────────────────────────────
  - type: worker
    name: music-remix-celery-worker
    runtime: python
    region: oregon
    plan: standard-plus       # more RAM for AI models
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && celery -A app.celery_app worker --loglevel=info --concurrency=2 -Q default,ai_heavy
    envVars:
      - fromGroup: music-remix-env   # shared env group

  # ── Celery Beat Scheduler ────────────────────────────────
  - type: worker
    name: music-remix-celery-beat
    runtime: python
    region: oregon
    plan: starter
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && celery -A app.celery_app beat --loglevel=info
    envVars:
      - fromGroup: music-remix-env

# ── Managed PostgreSQL ──────────────────────────────────────
databases:
  - name: music-remix-db
    region: oregon
    plan: standard
    databaseName: music_remix_db
    user: music_remix_user

# ── Managed Redis ───────────────────────────────────────────
  - name: music-remix-redis
    type: redis
    region: oregon
    plan: starter
    maxmemoryPolicy: allkeys-lru
```

Deploy with:

```bash
# Install Render CLI
npm install -g @render/cli

# Login
render login

# Deploy all services from render.yaml
render deploy
```

Or push `render.yaml` to your repo and connect it in the Render dashboard under **Blueprints**.

---

#### Option B — Manual Dashboard Setup

1. Go to [render.com](https://render.com) → **New +**
2. Create services in this order:

**1. PostgreSQL Database**
- Type: PostgreSQL
- Name: `music-remix-db`
- Plan: Standard
- Copy the **Internal Database URL** for use in other services

**2. Redis**
- Type: Redis
- Name: `music-remix-redis`
- Plan: Starter
- Max Memory Policy: `allkeys-lru`
- Copy the **Internal Redis URL**

**3. Web Service (FastAPI)**
- Type: Web Service
- Connect your GitHub repo
- Root Directory: `backend`
- Runtime: Python 3.11
- Build Command:
  ```
  pip install -r requirements.txt && alembic upgrade head
  ```
- Start Command:
  ```
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
  ```
- Health Check Path: `/health`
- Add all environment variables from your `.env` file

**4. Background Worker (Celery)**
- Type: Background Worker
- Connect same repo, Root Directory: `backend`
- Runtime: Python 3.11
- Build Command: `pip install -r requirements.txt`
- Start Command:
  ```
  celery -A app.celery_app worker --loglevel=info --concurrency=2 -Q default,ai_heavy
  ```
- Plan: Standard Plus (for AI model memory)

**5. Background Worker (Celery Beat)**
- Same as above but Start Command:
  ```
  celery -A app.celery_app beat --loglevel=info
  ```
- Plan: Starter

---

#### Environment Variables on Render

Set these in the Render dashboard (or in `render.yaml` with `sync: false`):

| Variable | Description |
|---|---|
| `APP_ENV` | `production` |
| `PORT` | `8000` (set automatically by Render) |
| `LOG_LEVEL` | `info` |
| `DB_HOST` | Auto-filled from Render Postgres |
| `DB_PORT` | `5432` |
| `DB_NAME` | `music_remix_db` |
| `DB_USER` | Render Postgres username |
| `DB_PASSWORD` | Render Postgres password |
| `DATABASE_URL` | Full asyncpg connection string (auto-filled) |
| `REDIS_HOST` | Auto-filled from Render Redis |
| `REDIS_PORT` | `6379` |
| `REDIS_PASSWORD` | Render Redis auth password |
| `REDIS_URL` | Full Redis connection string (auto-filled) |
| `CELERY_BROKER_URL` | Redis URL for Celery broker |
| `CELERY_RESULT_BACKEND` | Redis URL for Celery results |
| `JWT_SECRET` | Strong random string (32+ chars) |
| `JWT_EXPIRY` | `7d` |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` |
| `MAX_FILE_SIZE` | `104857600` (100 MB in bytes) |
| `ALLOWED_AUDIO_FORMATS` | `mp3,wav,flac,ogg,m4a,aac` |
| `MAX_CONCURRENT_JOBS` | `3` |
| `JOB_TIMEOUT_MS` | `600000` |
| `RATE_LIMIT_WINDOW_MS` | `900000` |
| `RATE_LIMIT_MAX_REQUESTS` | `100` |
| `AWS_ACCESS_KEY_ID` | S3 access key |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `S3_BUCKET_NAME` | Your S3 bucket name |
| `HUGGINGFACE_TOKEN` | HuggingFace API token |
| `MUSICGEN_MODEL` | `facebook/musicgen-small` |
| `DEMUCS_MODEL` | `htdemucs_ft` |
| `CORS_ORIGIN` | `https://your-app.vercel.app` |
| `CORS_ORIGINS` | Comma-separated allowed origins |

---

#### Run Migrations on Render

Migrations run automatically as part of the **build command** for the web service. To run them manually:

```bash
# Via Render Shell (Dashboard → Service → Shell tab)
alembic upgrade head

# Or via Render CLI
render ssh music-remix-api --command "cd backend && alembic upgrade head"
```

---

### Frontend → Vercel

```bash
cd frontend
npx vercel login
npx vercel --prod
```

Set these environment variables in the Vercel dashboard (Project → Settings → Environment Variables):

```
NEXT_PUBLIC_API_URL=https://music-remix-api.onrender.com/api/v1
NEXT_PUBLIC_WS_URL=wss://music-remix-api.onrender.com
NEXT_PUBLIC_APP_NAME=AI Music Remix Studio
```

Auto-deploys on every push to `main` once the Vercel GitHub integration is connected.

---

## 🧪 Testing

### Backend

```bash
cd backend
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing

# Specific test module
pytest tests/test_stems.py -v

# Run with async support
pytest tests/ -v --asyncio-mode=auto
```

### Frontend

```bash
cd frontend

# Unit tests
npm test

# Watch mode
npm run test:watch

# E2E tests (Playwright)
npm run test:e2e

# Type checking
npm run type-check

# Lint
npm run lint
```

---

## 📈 Performance Considerations

- **Async FastAPI** with asyncpg driver eliminates blocking I/O at the API layer
- **Celery workers** with dedicated `ai_heavy` queue isolate GPU-intensive tasks from lighter jobs
- **S3 presigned URLs** offload file transfers directly to AWS, bypassing the API server
- **Redis caching** stores waveform data and job status to avoid redundant DB queries
- **Database connection pooling** via SQLAlchemy async engine (pool size tunable via env)
- **Render's built-in load balancer** handles SSL termination and distributes traffic across API instances
- **Chunked audio uploads** with multipart support for files up to 200 MB
- **WebSocket** push for real-time job progress (no polling)
- **Model warm-up** — AI models are loaded once per worker process and kept in GPU VRAM

---

## 🔒 Security

- **JWT access tokens** (30 min) + **refresh tokens** (7 days, rotated on use)
- **bcrypt** password hashing with configurable cost factor
- **S3 presigned URLs** for time-limited, authenticated file access (no public buckets)
- **File validation** — MIME type + magic bytes checked on upload, not just extension
- **Rate limiting** — per-IP and per-user via slowapi middleware
- **CORS** — strict origin allowlist configured per environment
- **SQL injection prevention** — SQLAlchemy ORM with parameterized queries
- **Secrets** managed via Render's encrypted environment variables in production, `.env` locally
- **Input sanitization** on all audio processing parameters (clamp ranges, validate types)
- **HTTPS enforced** — Render provides automatic TLS certificates on all web services

---

## 📚 API Documentation

Auto-generated OpenAPI docs are available at runtime:

| URL | Interface |
|---|---|
| `http://localhost:8000/docs` | Swagger UI (interactive) |
| `http://localhost:8000/redoc` | ReDoc (readable) |
| `http://localhost:8000/openapi.json` | Raw OpenAPI 3.1 JSON |

The schema is auto-generated from FastAPI route definitions and Pydantic models — always in sync with the codebase.

---

## 🎨 Frontend Features

### Audio Upload
- Drag-and-drop via React-Dropzone
- Real-time upload progress bar
- Format validation (MP3, WAV, FLAC, OGG, M4A)
- Automatic waveform generation post-upload

### Waveform Player (WaveSurfer.js)
- Interactive zoomable waveform
- Play / pause / seek / loop
- Region selection for partial processing
- Spectrogram toggle overlay

### Stem Mixer
- Individual volume sliders per stem
- Solo / mute controls
- Export individual stems or mix
- Animated level meters via Web Audio API

### Mood Dashboard
- Radar chart of mood dimensions (Chart.js)
- MFCC feature heatmap
- Historical mood timeline

### Remix Studio
- Genre selector with audio preview samples
- Tempo BPM slider with tap-tempo
- Pitch shifter in semitones
- Effects panel (reverb, EQ, compression)
- Before/after waveform comparison

### Text-to-Music Generator
- Natural language prompt input
- Model selector (MusicGen Small / Medium / Large, Riffusion)
- Duration, seed, and guidance scale controls
- Generated track gallery

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with tests
4. Run the full test suite: `make test`
5. Commit: `git commit -m 'feat: add your feature description'`
6. Push: `git push origin feature/your-feature-name`
7. Open a Pull Request against `develop`


---
