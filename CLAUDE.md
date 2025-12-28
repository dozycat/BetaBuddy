# BetaBuddy

Climbing video analysis system with AI-powered pose estimation and beta suggestions.

## Tech Stack

**Backend:** Python 3.10+, FastAPI, SQLAlchemy, SQLite, MediaPipe, OpenCV, NumPy/SciPy
**Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Recharts, Video.js
**LLM:** Ollama with Qwen2.5 (local deployment)
**Deployment:** Docker, Docker Compose, Nginx

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Settings management
│   ├── api/
│   │   ├── routes/          # API endpoints (videos, analysis, beta)
│   │   └── websocket.py     # WebSocket for real-time analysis
│   ├── core/
│   │   ├── pose_estimator.py    # MediaPipe 33-keypoint detection
│   │   ├── physics_engine.py    # Center of mass, joint angles, kinematics
│   │   ├── metrics.py           # Stability, efficiency, dyno detection
│   │   └── llm_client.py        # Ollama integration
│   └── models/
│       ├── database.py      # SQLAlchemy models
│       └── schemas.py       # Pydantic schemas
├── uploads/                 # Video storage
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

- `POST /api/v1/videos/upload` - Upload video
- `GET /api/v1/videos` - List videos
- `GET /api/v1/videos/{id}/analyze` - Start analysis
- `GET /api/v1/videos/{id}/results` - Get results
- `POST /api/v1/beta/suggest` - Generate AI beta suggestions
- `WS /ws/analysis/{task_id}` - Real-time analysis updates

## Key Algorithms

- **Pose Estimation:** MediaPipe 33-keypoint detection
- **Center of Mass:** Weighted body segment calculation
- **Joint Angles:** 8 key angles (shoulders, elbows, hips, knees)
- **Stability:** Convex hull method for support polygon
- **Kinematics:** Velocity/acceleration with Savitzky-Golay smoothing

## Environment Variables

Copy `backend/.env.example` to `backend/.env`:
- `DEBUG` - Enable debug mode
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)

## Ports

- Backend API: 8000
- Frontend dev: 5173 (Vite) / 3000 (Docker)
- Ollama: 11434
