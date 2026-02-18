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
music-remix-app/
├── backend/
│   ├── src/
│   │   ├── config/
│   │   │   ├── database.js
│   │   │   └── redis.js
│   │   ├── controllers/
│   │   │   ├── authController.js
│   │   │   ├── audioController.js
│   │   │   ├── stemController.js
│   │   │   ├── moodController.js
│   │   │   └── remixController.js
│   │   ├── middleware/
│   │   │   ├── auth.js
│   │   │   ├── upload.js
│   │   │   └── errorHandler.js
│   │   ├── models/
│   │   │   ├── User.js
│   │   │   ├── AudioFile.js
│   │   │   ├── ProcessingJob.js
│   │   │   └── MoodClassification.js
│   │   ├── routes/
│   │   │   ├── auth.js
│   │   │   ├── audio.js
│   │   │   ├── stems.js
│   │   │   ├── mood.js
│   │   │   └── remix.js
│   │   ├── services/
│   │   │   ├── audioService.js
│   │   │   ├── queueService.js
│   │   │   └── pythonAIService.js
│   │   ├── workers/
│   │   │   └── audioWorker.js
│   │   └── app.js
│   ├── python-ai/
│   │   ├── models/
│   │   │   ├── stem_separator.py
│   │   │   ├── mood_classifier.py
│   │   │   └── music_generator.py
│   │   ├── utils/
│   │   │   └── audio_utils.py
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── package.json
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── (auth)/
│   │   │   ├── dashboard/
│   │   │   ├── upload/
│   │   │   ├── remix/
│   │   │   └── layout.tsx
│   │   ├── components/
│   │   │   ├── audio/
│   │   │   ├── ui/
│   │   │   └── layout/
│   │   ├── lib/
│   │   │   ├── api.ts
│   │   │   └── store.ts
│   │   └── styles/
│   ├── package.json
│   └── .env.local.example
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── ARCHITECTURE.md
└── README.md
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

## 📈 Testing

1. **Install Dependencies**:
   ```bash
   cd backend && npm install
   cd frontend && npm install
   cd backend/python-ai && pip install -r requirements.txt
   ```
2. **Configure Environment**:
   - Set up PostgreSQL database
   - Configure Redis
   - Update .env files

3. **Implement AI Models**:
   - Add Demucs model code
   - Implement mood classifier
   - Add music generation

4. **Build Frontend Pages**:
   - Create auth pages
   - Build dashboard
   - Add upload interface

5. **Deploy**:
   - Frontend to Vercel
   - Backend to AWS/Render
   - Configure domain and SSL

## 🎉 You're Ready!

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
