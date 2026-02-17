# System Architecture Documentation

## Overview

The Music Remix & Mood Generator is a full-stack application that leverages AI/ML technologies to provide advanced audio processing capabilities. The system is designed with scalability, maintainability, and performance in mind.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Next.js 14 Frontend                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │   Pages  │  │Components│  │  Hooks   │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │ Zustand  │  │   API    │  │  Utils   │            │    │
│  │  │  Store   │  │  Client  │  │          │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTPS/REST
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                     API GATEWAY LAYER                           │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Express.js Server                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │  Routes  │  │Middleware│  │Controllers│           │    │
│  │  └──────────┘  └──────────┘  └──────────┘            │    │
│  │     │              │              │                   │    │
│  │     └──────────────┴──────────────┘                   │    │
│  │                    │                                  │    │
│  └────────────────────┼──────────────────────────────────┘    │
└────────────────────────┼─────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Service    │  │   Service    │  │   Service    │
│    Layer     │  │    Layer     │  │    Layer     │
│              │  │              │  │              │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │  Audio   │ │  │ │  Queue   │ │  │ │  Python  │ │
│ │ Service  │ │  │ │ Service  │ │  │ │   AI     │ │
│ └──────────┘ │  │ └──────────┘ │  │ │ Service  │ │
│              │  │              │  │ └──────────┘ │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌──────────────────────────────────────────────────┐
│              BACKGROUND PROCESSING               │
│                                                  │
│  ┌────────────────┐  ┌────────────────┐        │
│  │  Bull Queues   │  │  AI Workers    │        │
│  │                │  │                │        │
│  │ ┌────────────┐ │  │ ┌────────────┐ │        │
│  │ │   Stems    │ │  │ │  Demucs    │ │        │
│  │ ├────────────┤ │  │ ├────────────┤ │        │
│  │ │   Mood     │ │  │ │  Librosa   │ │        │
│  │ ├────────────┤ │  │ ├────────────┤ │        │
│  │ │  Generate  │ │  │ │ Transforms │ │        │
│  │ ├────────────┤ │  │ ├────────────┤ │        │
│  │ │   Remix    │ │  │ │  MusicGen  │ │        │
│  │ └────────────┘ │  │ └────────────┘ │        │
│  └────────────────┘  └────────────────┘        │
└──────────────────────────────────────────────────┘
       │                                    │
       ▼                                    ▼
┌──────────────┐                    ┌──────────────┐
│    Redis     │                    │  PostgreSQL  │
│              │                    │              │
│ ┌──────────┐ │                    │ ┌──────────┐ │
│ │  Queue   │ │                    │ │  Users   │ │
│ │  Jobs    │ │                    │ ├──────────┤ │
│ │          │ │                    │ │  Audio   │ │
│ │  Cache   │ │                    │ ├──────────┤ │
│ │          │ │                    │ │   Jobs   │ │
│ │ Sessions │ │                    │ ├──────────┤ │
│ └──────────┘ │                    │ │  Moods   │ │
└──────────────┘                    │ ├──────────┤ │
                                    │ │ Remixes  │ │
                                    │ └──────────┘ │
                                    └──────────────┘
       │                                    │
       └────────────────┬───────────────────┘
                        ▼
                ┌──────────────┐
                │     File     │
                │   Storage    │
                │              │
                │ ┌──────────┐ │
                │ │ /uploads │ │
                │ ├──────────┤ │
                │ │  /temp   │ │
                │ ├──────────┤ │
                │ │ /outputs │ │
                │ └──────────┘ │
                └──────────────┘
```

## Technology Stack

### Frontend
- **Framework**: Next.js 14 (React 18)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Animation**: Framer Motion
- **Audio**: WaveSurfer.js, Tone.js
- **Charts**: Chart.js
- **HTTP Client**: Axios
- **UI Components**: Shadcn UI (Radix UI)

### Backend
- **Runtime**: Node.js 18+
- **Framework**: Express.js
- **Language**: JavaScript (ES6+)
- **ORM**: Sequelize
- **Queue**: Bull (Redis-based)
- **Authentication**: JWT (jsonwebtoken)
- **File Upload**: Multer
- **Audio Processing**: Fluent-FFmpeg

### AI/ML Service
- **Language**: Python 3.9+
- **Framework**: Flask
- **ML Libraries**:
  - PyTorch & Torchaudio
  - Demucs (stem separation)
  - Librosa (audio analysis)
  - Scikit-learn (mood classification)
  - Transformers (text-to-music)

### Databases
- **Primary**: PostgreSQL 14+
- **Cache/Queue**: Redis 6+

### Infrastructure
- **Frontend Hosting**: Vercel
- **Backend Hosting**: AWS EC2 / GCP Compute
- **Database**: AWS RDS / GCP Cloud SQL
- **File Storage**: Local / NFS / S3-compatible

## Data Flow

### 1. Audio Upload Flow
```
User → Frontend → API Gateway → Multer → File System
                     ↓
                 Metadata Extraction (music-metadata)
                     ↓
                 Database (AudioFile record)
                     ↓
                 Return metadata to user
```

### 2. Stem Separation Flow
```
User Request → API Gateway → Queue Service (Bull)
                                   ↓
                            Background Worker
                                   ↓
                         Python AI Service (Demucs)
                                   ↓
                         Generate stem files
                                   ↓
                         Update job status in DB
                                   ↓
                         Store results in file system
                                   ↓
                         Notify user (polling/WebSocket)
```

### 3. Mood Classification Flow
```
Audio File → Queue → Worker → Python AI
                                  ↓
                         Extract features (Librosa)
                                  ↓
                         ML Classification
                                  ↓
                         Store in MoodClassification table
                                  ↓
                         Return results
```

### 4. Music Generation Flow
```
Text Prompt → API → Queue → Worker → Python AI (MusicGen)
                                          ↓
                                  Generate audio
                                          ↓
                                  Save to file system
                                          ↓
                                  Update job status
                                          ↓
                                  Return audio file
```

## Security Architecture

### Authentication & Authorization
```
┌─────────────────────────────────────────┐
│         Authentication Flow             │
│                                         │
│  User Credentials                       │
│         ↓                               │
│  bcrypt.compare()                       │
│         ↓                               │
│  JWT Sign (HS256)                       │
│         ↓                               │
│  Return token to client                 │
│         ↓                               │
│  Store in localStorage                  │
│         ↓                               │
│  Include in Authorization header        │
│         ↓                               │
│  Middleware validates JWT               │
│         ↓                               │
│  Attach user to req.user                │
│         ↓                               │
│  Process request                        │
└─────────────────────────────────────────┘
```

### Security Measures
1. **Password Hashing**: bcrypt with 10 rounds
2. **JWT Tokens**: HS256 algorithm, 7-day expiry
3. **CORS**: Configured for specific origins
4. **Rate Limiting**: 100 requests per 15 minutes
5. **Input Validation**: File type, size checks
6. **SQL Injection Prevention**: Sequelize ORM
7. **XSS Protection**: Helmet middleware
8. **HTTPS**: SSL/TLS in production

## Database Schema

### Entity Relationship Diagram
```
┌────────────┐       ┌─────────────┐       ┌──────────────┐
│   Users    │───┐   │ AudioFiles  │───┐   │ProcessingJobs│
├────────────┤   │   ├─────────────┤   │   ├──────────────┤
│ id (PK)    │   └──→│ user_id(FK) │   └──→│ audio_id(FK) │
│ email      │       │ filename    │       │ job_type     │
│ username   │       │ file_path   │       │ status       │
│ password   │       │ duration    │       │ progress     │
│ created_at │       │ format      │       │ result_path  │
└────────────┘       └─────────────┘       └──────────────┘
                            │                      │
                            │                      │
                            ▼                      ▼
                   ┌──────────────┐       ┌──────────────┐
                   │    Moods     │       │   Remixes    │
                   ├──────────────┤       ├──────────────┤
                   │ audio_id(FK) │       │ user_id (FK) │
                   │ mood         │       │ original_id  │
                   │ confidence   │       │ remix_id     │
                   │ features     │       │ genre        │
                   └──────────────┘       │ effects      │
                                         └──────────────┘
```

## Queue System Architecture

### Bull Queue Configuration
```javascript
Queue: audio-processing
├── Processors: 3 concurrent
├── Retry: 3 attempts with exponential backoff
├── Timeout: 10 minutes
└── Jobs:
    ├── stem-separation
    ├── mood-classification
    ├── music-generation
    └── remix-processing

Events:
├── completed → Update DB status
├── failed → Log error, update DB
├── progress → Emit to WebSocket
└── stalled → Retry with backoff
```

## API Design Principles

1. **RESTful**: Standard HTTP methods (GET, POST, DELETE)
2. **Consistent Response**: `{ success, message, data, error }`
3. **Status Codes**: Proper HTTP status codes
4. **Pagination**: For list endpoints
5. **Filtering**: Query parameters for filtering
6. **Documentation**: OpenAPI/Swagger specs
7. **Versioning**: URL versioning (`/api/v1/`)

## Performance Optimizations

### Backend
- **Connection Pooling**: PostgreSQL pool (10 connections)
- **Redis Caching**: Job status, user sessions
- **File Streaming**: Large file downloads
- **Async Processing**: Background jobs
- **Clustering**: PM2 cluster mode
- **Compression**: gzip responses

### Frontend
- **Code Splitting**: Next.js automatic splitting
- **Lazy Loading**: Components, images
- **Memoization**: React.memo, useMemo
- **Debouncing**: Search, API calls
- **CDN**: Static assets
- **Image Optimization**: Next.js Image component

### Database
- **Indexing**: 
  - users(email, username)
  - audio_files(user_id, created_at)
  - processing_jobs(job_id, user_id, status)
- **Query Optimization**: Select only needed fields
- **Connection Management**: Pool configuration

## Monitoring & Logging

### Application Logs
```
Backend:
├── Access logs (Morgan)
├── Error logs
├── Job logs
└── Custom application logs

Python AI:
├── Flask logs
├── Model inference logs
└── Error tracking
```

### Metrics to Monitor
- API response times
- Queue job processing times
- Database query performance
- Memory usage
- CPU usage
- Disk I/O
- Failed job count
- User activity

## Scalability Considerations

### Horizontal Scaling
1. **Multiple API Servers**: Load balancer distribution
2. **Shared File Storage**: NFS or S3
3. **Redis Cluster**: Distributed queue
4. **Read Replicas**: PostgreSQL read replicas

### Vertical Scaling
1. **Larger Instances**: More CPU/RAM
2. **GPU Instances**: For AI processing
3. **SSD Storage**: Faster I/O

### Optimization Strategies
1. **Caching**: Redis for frequent queries
2. **CDN**: Static assets, audio previews
3. **Database Indexing**: Optimize queries
4. **Async Processing**: Background jobs
5. **Rate Limiting**: Prevent abuse

## Disaster Recovery

### Backup Strategy
1. **Database**: Daily automated backups
2. **Files**: Daily file system backups
3. **Retention**: 30-day retention period

### Recovery Plan
1. **Database Restore**: From latest snapshot
2. **File Restore**: From backup storage
3. **Service Restart**: PM2 automatic restart
4. **Health Checks**: Automated monitoring

## Future Enhancements

1. **Real-time Updates**: WebSocket for job progress
2. **Collaborative Features**: Share and collaborate on remixes
3. **Social Features**: Public profile, sharing
4. **Advanced AI Models**: Better quality, more features
5. **Mobile Apps**: iOS and Android
6. **Advanced Analytics**: Usage metrics, insights
7. **Microservices**: Break into smaller services
8. **Kubernetes**: Container orchestration
9. **GraphQL API**: Alternative to REST
10. **Machine Learning Pipeline**: Model training, versioning

## Conclusion

This architecture provides a solid foundation for a scalable, maintainable music processing application. The separation of concerns between frontend, API, and AI services allows for independent scaling and updates. The use of background job processing ensures responsive user experience even for long-running operations.
