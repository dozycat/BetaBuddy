# BetaBuddy

Climbing video analysis system with AI-powered pose estimation and beta suggestions.

## Tech Stack

**Backend:** Python 3.10+, FastAPI, SQLAlchemy, SQLite, MediaPipe, OpenCV, NumPy/SciPy
**Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Recharts, Video.js
**LLM:** Ollama with configurable models (Qwen2.5, Llama3, Mistral, etc.)
**Deployment:** Docker, Docker Compose, Nginx

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings management
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   │   ├── videos.py    # Video upload, list, annotate
│   │   │   ├── analysis.py  # Start analysis, get results
│   │   │   ├── beta.py      # AI beta suggestions
│   │   │   └── llm.py       # LLM configuration
│   │   └── websocket.py     # WebSocket for real-time metrics streaming
│   ├── core/
│   │   ├── pose_estimator.py    # MediaPipe 33-keypoint detection
│   │   ├── physics_engine.py    # Center of mass, joint angles, kinematics
│   │   ├── metrics.py           # Stability, efficiency, dyno detection
│   │   ├── annotator.py         # Video annotation with keypoints/skeleton
│   │   └── llm_client.py        # Ollama integration
│   └── models/
│       ├── database.py      # SQLAlchemy models
│       └── schemas.py       # Pydantic schemas
├── uploads/                 # Video storage (original + annotated)
└── betabuddy.db            # SQLite database

frontend/
├── src/
│   ├── components/          # React components (VideoUploader, MetricsPanel, etc.)
│   ├── pages/               # Home, Analysis pages
│   ├── hooks/               # Custom hooks (useWebSocket)
│   ├── api/                 # API client
│   └── types/               # TypeScript types
```

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker
```bash
docker-compose up --build
```

## API Endpoints

### Videos
- `POST /api/v1/videos/upload` - Upload video
- `GET /api/v1/videos` - List videos
- `GET /api/v1/videos/{id}` - Get video details
- `DELETE /api/v1/videos/{id}` - Delete video
- `POST /api/v1/videos/{id}/annotate` - Generate annotated video with keypoints
- `GET /api/v1/videos/{id}/annotated` - Get annotated video URL

### Analysis
- `GET /api/v1/videos/{id}/analyze` - Start analysis
- `GET /api/v1/videos/{id}/results` - Get analysis results
- `GET /api/v1/tasks/{task_id}` - Get task status

### Beta Suggestions
- `POST /api/v1/beta/suggest` - Generate AI beta suggestions

### LLM Configuration
- `GET /api/v1/llm/models` - List available Ollama models
- `GET /api/v1/llm/config` - Get current LLM configuration
- `PUT /api/v1/llm/config` - Update LLM model/parameters
- `GET /api/v1/llm/status` - Check Ollama service status
- `POST /api/v1/llm/test` - Test LLM connection

### WebSocket
- `WS /ws/analysis/{task_id}` - Real-time analysis updates

#### WebSocket Message Types
- `progress` - Analysis progress (percentage, current frame)
- `keypoints` - Per-frame 33 keypoints with positions
- `metrics` - Per-frame CoM, joint angles, velocity, acceleration
- `complete` - Analysis finished with summary
- `error` - Error occurred

## Key Algorithms

- **Pose Estimation:** MediaPipe 33-keypoint detection (Tasks API)
- **Center of Mass:** Weighted 14-segment body model calculation
- **Joint Angles:** 8 key angles (shoulders, elbows, hips, knees)
- **Stability:** Convex hull method for support polygon
- **Kinematics:** Real-time velocity/acceleration calculation
- **Video Annotation:** Skeleton overlay, CoM trajectory, metrics display

## Environment Variables

Copy `backend/.env.example` to `backend/.env`:
- `DEBUG` - Enable debug mode
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL` - Default LLM model (default: qwen2.5:7b)

## Supported LLM Models

The LLM model can be changed at runtime via the API. Popular options:
- `qwen2.5:7b` - Default, good balance of quality and speed
- `qwen2.5:14b` - Higher quality, slower
- `llama3.2:3b` - Fast, lightweight
- `llama3.1:8b` - Good general purpose
- `mistral:7b` - Fast inference
- `deepseek-coder:6.7b` - Code-focused

## Ports

- Backend API: 8000
- Frontend dev: 5173 (Vite) / 3000 (Docker)
- Ollama: 11434
