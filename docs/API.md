# API Documentation

## Base URL
```
http://localhost:5000/api
```

## Authentication
All protected endpoints require a JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

---

## Authentication Endpoints

### Register User
**POST** `/auth/register`

Create a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "johndoe",
  "password": "securepassword123"
}
```

**Response:** `201 Created`
```json
{
  "success": true,
  "message": "User registered successfully",
  "data": {
    "user": {
      "id": 1,
      "email": "user@example.com",
      "username": "johndoe",
      "created_at": "2024-01-01T00:00:00.000Z"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

### Login
**POST** `/auth/login`

Authenticate and receive JWT token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Login successful",
  "data": {
    "user": { ... },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

### Get Current User
**GET** `/auth/me`

Get authenticated user details.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "user": {
      "id": 1,
      "email": "user@example.com",
      "username": "johndoe",
      "created_at": "2024-01-01T00:00:00.000Z"
    }
  }
}
```

---

## Audio Management Endpoints

### Upload Audio
**POST** `/audio/upload`

Upload an audio file for processing.

**Headers:**
```
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Form Data:**
- `audio` (file): Audio file (MP3, WAV, FLAC, OGG, M4A)

**Response:** `201 Created`
```json
{
  "success": true,
  "message": "Audio uploaded successfully",
  "data": {
    "audioFile": {
      "id": 1,
      "filename": "uuid-timestamp.mp3",
      "original_name": "my-song.mp3",
      "file_size": 5242880,
      "duration": 180.5,
      "format": "mp3",
      "sample_rate": 44100,
      "channels": 2,
      "created_at": "2024-01-01T00:00:00.000Z"
    }
  }
}
```

### List Audio Files
**GET** `/audio`

Get all audio files for authenticated user.

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Items per page (default: 20)

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "audioFiles": [...],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 50,
      "pages": 3
    }
  }
}
```

### Get Audio File
**GET** `/audio/:id`

Get specific audio file details.

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "audioFile": { ... }
  }
}
```

### Download Audio
**GET** `/audio/:id/download`

Download audio file.

**Response:** Audio file stream

### Delete Audio
**DELETE** `/audio/:id`

Delete an audio file.

**Response:** `200 OK`
```json
{
  "success": true,
  "message": "Audio file deleted successfully"
}
```

---

## Stem Separation Endpoints

### Separate Stems
**POST** `/stems/separate`

Separate audio into individual stems (vocals, drums, bass, other).

**Request Body:**
```json
{
  "audioFileId": 1,
  "model": "htdemucs",
  "stems": ["vocals", "drums", "bass", "other"]
}
```

**Response:** `202 Accepted`
```json
{
  "success": true,
  "message": "Stem separation job queued",
  "data": {
    "jobId": "123e4567-e89b-12d3-a456-426614174000",
    "status": "pending"
  }
}
```

### Get Separation Status
**GET** `/stems/:jobId/status`

Check stem separation job status.

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "jobId": "123e4567...",
    "status": "completed",
    "progress": 100,
    "result": {
      "stems": {
        "vocals": "/path/to/vocals.wav",
        "drums": "/path/to/drums.wav",
        "bass": "/path/to/bass.wav",
        "other": "/path/to/other.wav"
      }
    }
  }
}
```

### Download Stems
**GET** `/stems/:jobId/download`

Download separated stems as ZIP file.

**Response:** ZIP file stream

---

## Mood Classification Endpoints

### Classify Mood
**POST** `/mood/classify`

Classify the mood/emotion of an audio file.

**Request Body:**
```json
{
  "audioFileId": 1
}
```

**Response:** `202 Accepted`
```json
{
  "success": true,
  "message": "Mood classification job queued",
  "data": {
    "jobId": "123e4567...",
    "status": "pending"
  }
}
```

### Get Mood Classification
**GET** `/mood/:audioId`

Get mood classification results for an audio file.

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "mood": "energetic",
    "confidence": 0.89,
    "mood_scores": {
      "happy": 0.12,
      "sad": 0.05,
      "energetic": 0.89,
      "calm": 0.08,
      "angry": 0.15
    },
    "features": {
      "tempo": 128.5,
      "energy": 0.85,
      "valence": 0.72,
      "danceability": 0.78
    }
  }
}
```

---

## Music Generation Endpoints

### Generate Music from Text
**POST** `/generate/text-to-music`

Generate music from text prompt using AI.

**Request Body:**
```json
{
  "prompt": "upbeat electronic dance music with energetic drums",
  "duration": 10,
  "temperature": 1.0,
  "top_k": 250
}
```

**Response:** `202 Accepted`
```json
{
  "success": true,
  "message": "Music generation job queued",
  "data": {
    "jobId": "123e4567...",
    "status": "pending"
  }
}
```

### Get Generation Status
**GET** `/generate/:jobId/status`

Check music generation status.

**Response:** `200 OK` (when completed)
```json
{
  "success": true,
  "data": {
    "jobId": "123e4567...",
    "status": "completed",
    "progress": 100,
    "result": {
      "audio_path": "/path/to/generated.wav",
      "duration": 10.0
    }
  }
}
```

### Download Generated Music
**GET** `/generate/:jobId/download`

Download generated music file.

**Response:** Audio file stream

---

## Remix Endpoints

### Remix to Different Genre
**POST** `/remix/genre`

Transform audio to a different genre.

**Request Body:**
```json
{
  "audioFileId": 1,
  "targetGenre": "edm",
  "intensity": 0.8
}
```

**Response:** `202 Accepted`

### Change Tempo
**POST** `/remix/tempo`

Adjust audio tempo/speed.

**Request Body:**
```json
{
  "audioFileId": 1,
  "tempoChange": 1.2,
  "preservePitch": true
}
```

**Response:** `202 Accepted`

### Change Pitch
**POST** `/remix/pitch`

Adjust audio pitch.

**Request Body:**
```json
{
  "audioFileId": 1,
  "pitchChange": 2.0
}
```

**Response:** `202 Accepted`

### Get Remix Status
**GET** `/remix/:jobId/status`

Check remix job status.

**Response:** Similar to other status endpoints

---

## Audio Analysis Endpoints

### Get Waveform Data
**GET** `/audio/:id/waveform`

Get waveform data for visualization.

**Query Parameters:**
- `samples` (optional): Number of samples (default: 1000)

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "waveform": [0.1, -0.2, 0.5, ...],
    "samples": 1000,
    "duration": 180.5
  }
}
```

### Get Spectrogram
**GET** `/audio/:id/spectrogram`

Get spectrogram image.

**Response:** PNG image

### Extract Audio Features
**GET** `/audio/:id/features`

Extract audio features for analysis.

**Response:** `200 OK`
```json
{
  "success": true,
  "data": {
    "tempo": 120.5,
    "key": "C major",
    "time_signature": "4/4",
    "energy": 0.75,
    "loudness": -5.2,
    "spectral_centroid": 2500.0,
    "zero_crossing_rate": 0.05
  }
}
```

---

## Error Responses

All endpoints may return error responses in the following format:

### 400 Bad Request
```json
{
  "success": false,
  "message": "Invalid request parameters"
}
```

### 401 Unauthorized
```json
{
  "success": false,
  "message": "No token provided, authorization denied"
}
```

### 404 Not Found
```json
{
  "success": false,
  "message": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "message": "Server error",
  "error": "Detailed error message (development only)"
}
```

---

## Rate Limiting

API endpoints are rate limited to:
- **100 requests per 15 minutes** per IP address
- **50 requests per 15 minutes** for resource-intensive operations (stem separation, generation)

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

---

## WebSocket Events (Future)

Real-time job progress updates via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:5000');

ws.on('message', (data) => {
  const { type, jobId, progress, status } = JSON.parse(data);
  // Handle progress updates
});
```

---

## Best Practices

1. **Always include Authorization header** for protected endpoints
2. **Poll job status endpoints** every 2-5 seconds for long-running operations
3. **Implement exponential backoff** for failed requests
4. **Cache audio metadata** to reduce API calls
5. **Handle file uploads** with proper multipart/form-data encoding
6. **Validate file sizes** before upload (max 100MB)
7. **Use pagination** for list endpoints with many results
