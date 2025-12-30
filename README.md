# BetaBuddy

AI-powered climbing video analysis system with real-time pose estimation, biomechanical metrics, and beta suggestions.

## Demo

<video src="docs/introduction.mp4" controls width="100%"></video>

*Introduction video showing BetaBuddy in action. If the video doesn't play, [download it here](docs/introduction.mp4).*

## Features

- **Real-Time Pose Detection**: MediaPipe 33-keypoint detection for accurate body tracking
- **Biomechanical Analysis**: Center of mass, joint angles, velocity, and acceleration calculations
- **Movement Detection**: Automatic identification of climbing techniques (side pull, undercling, heel hook, flag, drop knee, etc.)
- **Video Annotation**: Generate annotated videos with skeleton overlay, metrics display, and movement labels
- **AI Beta Suggestions**: LLM-powered climbing technique recommendations
- **Configurable LLM**: Runtime model selection (Qwen2.5, Llama3, Mistral, etc.)
- **WebSocket Streaming**: Real-time metrics during video analysis

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama (for AI suggestions)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Ollama Setup (Required for AI Beta Suggestions)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the default model (4.7GB download)
ollama pull qwen2.5:7b

# Start Ollama server (run in background or separate terminal)
ollama serve
```

**Note:** Without Ollama running, the app will use fallback rule-based suggestions instead of AI-generated ones.

## API Overview

### Video Management

```bash
# Upload a video
curl -X POST http://localhost:8000/api/v1/videos/upload \
  -F "file=@climbing_video.mp4"

# List all videos
curl http://localhost:8000/api/v1/videos

# Start analysis
curl http://localhost:8000/api/v1/videos/{video_id}/analyze

# Get analysis results
curl http://localhost:8000/api/v1/videos/{video_id}/results

# Generate annotated video
curl -X POST http://localhost:8000/api/v1/videos/{video_id}/annotate
```

### Movement Detection

```bash
# Detect climbing techniques
curl -X POST http://localhost:8000/api/v1/videos/{video_id}/detect-movements \
  -H "Content-Type: application/json" \
  -d '{"generate_descriptions": true}'

# Get cached movement data
curl http://localhost:8000/api/v1/videos/{video_id}/movements
```

### LLM Configuration

```bash
# List available models
curl http://localhost:8000/api/v1/llm/models

# Get current config
curl http://localhost:8000/api/v1/llm/config

# Change model
curl -X PUT http://localhost:8000/api/v1/llm/config \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.1:8b"}'

# Test LLM connection
curl -X POST http://localhost:8000/api/v1/llm/test
```

### WebSocket Real-Time Metrics

Connect to `ws://localhost:8000/ws/analysis/{task_id}` to receive:
- `keypoints`: 33 body keypoints per frame
- `metrics`: CoM, joint angles, velocity, acceleration per frame
- `progress`: Analysis progress percentage
- `complete`: Final summary when done

## Analysis Metrics

| Metric | Description |
|--------|-------------|
| Center of Mass | Weighted position of body's mass center |
| Joint Angles | 8 angles: elbows, shoulders, hips, knees (degrees) |
| Stability Score | 0-1 based on CoM within support polygon |
| Velocity | CoM movement speed (pixels/frame) |
| Acceleration | CoM movement acceleration (pixels/frame^2) |
| Efficiency | Direct distance / actual path length |
| Dyno Detection | Dynamic move detection via acceleration peaks |

## Movement Detection

Automatically identifies common climbing techniques from pose and angle data:

| Movement | Chinese | Detection Criteria |
|----------|---------|-------------------|
| Side Pull | 侧拉 | Arm pulling sideways, body leaning opposite |
| Undercling | 反提 | Hand below shoulder, elbow bent pulling upward |
| Gaston | 推撑 | Elbow pointed outward, pushing motion |
| Heel Hook | 跟勾 | Heel at or above hip level |
| Toe Hook | 趾勾 | Toe elevated with foot rotated |
| Flag | 旗杆 | Leg extended for balance |
| Drop Knee | 埃及步 | Hip rotation with knee dropped inward |
| Dyno | 动态跳跃 | High acceleration dynamic moves |

**Features:**
- Timestamps and duration for each detected movement
- Confidence scores and challenging move indicators
- LLM-generated Chinese descriptions (with fallback when Ollama unavailable)
- Movement labels overlay on annotated videos
- Interactive timeline in the frontend

## Video Annotation

The annotated video includes:
- Colored keypoints (green=high confidence, yellow=medium, red=low)
- Skeleton connections between joints
- Center of mass point with trajectory trail
- Real-time metrics overlay (velocity, acceleration)
- Joint angles panel
- Movement technique labels (when detected)

**Frontend Features:**
- Annotated videos are automatically generated after analysis completes
- Toggle between original and annotated video views
- Progress bar displays during analysis with real-time frame updates
- Interactive movement timeline with expandable details
- Click movements to seek video to that timestamp

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│   Ollama    │
│  (React)    │◀────│  (FastAPI)  │◀────│   (LLM)     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌──────────┐  ┌──────────┐
              │ MediaPipe│  │  SQLite  │
              │  (Pose)  │  │   (DB)   │
              └──────────┘  └──────────┘
```

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, MediaPipe, OpenCV, NumPy/SciPy
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **LLM**: Ollama (local deployment)
- **Database**: SQLite (async with aiosqlite)

## Docker Deployment

```bash
docker-compose up --build
```

Services:
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- Ollama: http://localhost:11434

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DEBUG | true | Enable debug mode |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| OLLAMA_MODEL | qwen2.5:7b | Default LLM model |
| DATABASE_URL | sqlite+aiosqlite:///./betabuddy.db | Database connection |
| MAX_UPLOAD_SIZE | 524288000 | Max upload size (500MB) |

## License

MIT
