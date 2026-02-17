# 🎵 AI Music Remix & Mood Generator

A full-stack AI-powered music application that enables users to upload audio files, separate stems, classify moods, remix songs into different genres, generate new music from text prompts, and manipulate audio with advanced processing.

## 🏗️ Architecture Overview

### Technology Stack

**Backend (Node.js/Express)**

- **Runtime**: Node.js 18+ with Express.js
- **Queue System**: Bull + Redis for background job processing
- **Database**: PostgreSQL 14+ with Sequelize ORM
- **Authentication**: JWT tokens with bcrypt
- **File Storage**: Local filesystem with multer
- **AI/ML Libraries**:
  - Python microservice for AI models (subprocess integration)
  - Tone.js for audio manipulation
  - FFmpeg for audio processing
  - Web Audio API utilities

**Frontend (Next.js 14)**
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS + Framer Motion
- **UI Components**: Shadcn UI + Lucide React icons
- **State Management**: Zustand
- **Audio Visualization**: WaveSurfer.js + Tone.js
- **Charts**: Chart.js
- **HTTP Client**: Axios
- **File Upload**: React-Dropzone

**AI/ML Python Microservice**
- **Framework**: Flask (lightweight API)
- **Libraries**: 
  - Demucs for stem separation
  - Librosa for audio analysis
  - Scikit-learn for mood classification
  - Transformers for text-to-music
  - PyTorch + Torchaudio
  - Pydub, SoundFile for audio I/O

### System Architecture

```
┌─────────────────┐
│   Next.js App   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐      ┌──────────────┐
│  Express API    │◄────►│  PostgreSQL  │
│   (Backend)     │      │   Database   │
└────────┬────────┘      └──────────────┘
         │
         ├──────────────┐
         │              │
         ▼              ▼
┌─────────────┐  ┌─────────────┐
│ Bull Queue  │  │   Python    │
│  + Redis    │  │ AI Service  │
└─────────────┘  └─────────────┘
         │              │
         └──────┬───────┘
                ▼
        ┌──────────────┐
        │ File Storage │
        │   (Local)    │
        └──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.9+
- PostgreSQL 14+
- Redis 6+
- FFmpeg

### Installation

1. **Clone and setup:**
```bash
git clone <repository>
cd music-remix-app
```

2. **Setup Backend:**
```bash
cd backend
npm install
cp .env.example .env
# Edit .env with your configuration
npm run migrate
```

3. **Setup Python AI Service:**
```bash
cd backend/python-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. **Setup Frontend:**
```bash
cd frontend
npm install
cp .env.local.example .env.local
# Edit .env.local with API URL
```

5. **Start Services:**

Terminal 1 - PostgreSQL & Redis:
```bash
# Start PostgreSQL (varies by OS)
# Start Redis
redis-server
```

Terminal 2 - Backend:
```bash
cd backend
npm run dev
```

Terminal 3 - Python AI Service:
```bash
cd backend/python-ai
source venv/bin/activate
python app.py
```

Terminal 4 - Frontend:
```bash
cd frontend
npm run dev
```

Access the application at `http://localhost:3000`

## 📊 Database Schema

### Users Table
```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  username VARCHAR(100) UNIQUE NOT NULL,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### Audio Files Table
```sql
CREATE TABLE audio_files (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  filename VARCHAR(255) NOT NULL,
  original_name VARCHAR(255) NOT NULL,
  file_path VARCHAR(500) NOT NULL,
  file_size INTEGER NOT NULL,
  duration FLOAT,
  format VARCHAR(20),
  sample_rate INTEGER,
  channels INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Processing Jobs Table
```sql
CREATE TABLE processing_jobs (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
  job_type VARCHAR(50) NOT NULL,
  status VARCHAR(50) DEFAULT 'pending',
  progress INTEGER DEFAULT 0,
  result_path VARCHAR(500),
  metadata JSONB,
  error_message TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  completed_at TIMESTAMP
);
```

### Mood Classifications Table
```sql
CREATE TABLE mood_classifications (
  id SERIAL PRIMARY KEY,
  audio_file_id INTEGER REFERENCES audio_files(id) ON DELETE CASCADE,
  mood VARCHAR(50) NOT NULL,
  confidence FLOAT NOT NULL,
  features JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
```

### Remixes Table
```sql
CREATE TABLE remixes (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  original_file_id INTEGER REFERENCES audio_files(id),
  remix_file_id INTEGER REFERENCES audio_files(id),
  genre VARCHAR(50),
  tempo_change FLOAT,
  pitch_change FLOAT,
  effects JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔌 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and receive JWT
- `GET /api/auth/me` - Get current user (protected)

### Audio Upload & Management
- `POST /api/audio/upload` - Upload audio file
- `GET /api/audio` - List user's audio files
- `GET /api/audio/:id` - Get audio file details
- `DELETE /api/audio/:id` - Delete audio file
- `GET /api/audio/:id/download` - Download audio file

### Stem Separation
- `POST /api/stems/separate` - Separate audio into stems
- `GET /api/stems/:jobId/status` - Check separation status
- `GET /api/stems/:jobId/download` - Download separated stems

### Mood Classification
- `POST /api/mood/classify` - Classify audio mood
- `GET /api/mood/:audioId` - Get mood classification results

### Music Generation
- `POST /api/generate/text-to-music` - Generate music from text
- `GET /api/generate/:jobId/status` - Check generation status
- `GET /api/generate/:jobId/download` - Download generated music

### Audio Remixing
- `POST /api/remix/genre` - Remix to different genre
- `POST /api/remix/tempo` - Change tempo
- `POST /api/remix/pitch` - Change pitch
- `GET /api/remix/:jobId/status` - Check remix status

### Waveform & Analysis
- `GET /api/audio/:id/waveform` - Get waveform data
- `GET /api/audio/:id/spectrogram` - Get spectrogram data
- `GET /api/audio/:id/features` - Get audio features

## 🎨 Features

### Core Features
1. **Audio Upload & Management**
   - Drag-and-drop upload
   - Multiple format support (MP3, WAV, FLAC, OGG)
   - Audio preview with WaveSurfer.js
   - File organization and history

2. **Stem Separation**
   - Separate vocals, drums, bass, and other instruments
   - Uses Demucs for high-quality separation
   - Download individual stems
   - Preview and mix stems

3. **Mood Classification**
   - AI-powered mood detection
   - Categories: Happy, Sad, Energetic, Calm, Angry, etc.
   - Confidence scores
   - Feature visualization

4. **Music Generation**
   - Text-to-music using AI models
   - Style and genre control
   - Duration and quality settings
   - Prompt engineering support

5. **Audio Remixing**
   - Genre transformation
   - Tempo adjustment (-50% to +100%)
   - Pitch shifting (-12 to +12 semitones)
   - Real-time preview

6. **Waveform Visualization**
   - Interactive waveform display
   - Zoom and pan controls
   - Spectrogram view
   - Timeline markers

## 🔧 Configuration

### Backend Environment Variables (.env)
```env
# Server
NODE_ENV=development
PORT=5000
API_BASE_URL=http://localhost:5000

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=music_remix_db
DB_USER=postgres
DB_PASSWORD=Gaur@1av

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# JWT
JWT_SECRET=qwertyuiop
JWT_EXPIRY=7d

# File Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=104857600

# Python AI Service
PYTHON_AI_URL=http://localhost:5001

# CORS
CORS_ORIGIN=http://localhost:3000
```

### Frontend Environment Variables (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:5000/api
NEXT_PUBLIC_WS_URL=ws://localhost:5000
```

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

## 🧪 Testing

### Backend Tests
```bash
cd backend
npm test
npm run test:coverage
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:e2e
```

## 📈 Performance Optimization

- Background job processing with Bull queues
- Audio streaming for large files
- Database indexing on frequently queried fields
- Redis caching for job status
- Lazy loading in frontend
- Code splitting in Next.js

## 🔒 Security

- JWT authentication with httpOnly cookies
- Password hashing with bcrypt (10 rounds)
- File upload validation and sanitization
- CORS configuration
- Rate limiting on API endpoints
- SQL injection prevention with Sequelize ORM

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📧 Support

For issues and questions, please open a GitHub issue or contact support@musicremix.app

---

Built with ❤️ using Next.js, Express, and AI
