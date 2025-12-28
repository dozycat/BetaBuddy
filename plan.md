# BetaBuddy Implementation Plan

## Phase 1: Backend Foundation

### 1.1 Project Setup
- [ ] Initialize Python project with Poetry/pip
- [ ] Create FastAPI application structure
- [ ] Set up configuration management (config.py)
- [ ] Configure CORS and middleware
- [ ] Set up SQLite database with SQLAlchemy

### 1.2 Video Management
- [ ] Create video upload endpoint (POST /api/v1/videos/upload)
- [ ] Implement local file storage system
- [ ] Create video listing endpoint (GET /api/v1/videos)
- [ ] Add video metadata extraction (duration, fps, resolution)
- [ ] Implement video deletion endpoint

### 1.3 Pose Estimation Integration
- [ ] Install and configure MediaPipe
- [ ] Create PoseEstimator class (core/pose_estimator.py)
- [ ] Implement 33-keypoint detection
- [ ] Add keypoint visibility filtering
- [ ] Test with sample climbing videos

---

## Phase 2: Core Analysis Algorithms

### 2.1 Physics Engine
- [ ] Create physics_engine.py module
- [ ] Implement center of mass calculation with body segment weights
- [ ] Implement joint angle calculation (8 key angles)
- [ ] Implement velocity calculation from position sequences
- [ ] Implement acceleration calculation
- [ ] Add Savitzky-Golay smoothing filter

### 2.2 Climbing Metrics
- [ ] Create metrics.py module
- [ ] Implement ClimbingMetrics class
- [ ] Calculate stability score (convex hull method)
- [ ] Calculate movement efficiency
- [ ] Implement dynamic move (Dyno) detection
- [ ] Track center of mass trajectory

### 2.3 Video Processing Pipeline
- [ ] Create frame-by-frame analysis pipeline
- [ ] Store FrameAnalysis results
- [ ] Generate analysis summary statistics
- [ ] Optimize for batch processing

---

## Phase 3: API & Real-time Communication

### 3.1 Analysis API
- [ ] Create analysis endpoint (GET /api/v1/videos/{video_id}/analyze)
- [ ] Implement background task processing
- [ ] Create results endpoint (GET /api/v1/videos/{video_id}/results)
- [ ] Add export endpoint (GET /api/v1/videos/{video_id}/export)

### 3.2 WebSocket Implementation
- [ ] Set up WebSocket handler (/ws/analysis/{task_id})
- [ ] Implement progress updates
- [ ] Stream keypoint data in real-time
- [ ] Send metrics updates
- [ ] Handle connection lifecycle

### 3.3 Data Models
- [ ] Define Pydantic schemas (Keypoint, FrameAnalysis, VideoAnalysisResult)
- [ ] Create database models
- [ ] Implement data validation

---

## Phase 4: LLM Integration

### 4.1 Ollama Setup
- [ ] Document Ollama installation steps
- [ ] Configure Qwen2.5:7b model
- [ ] Create llm_client.py module

### 4.2 Beta Suggestion Generation
- [ ] Implement BETA_PROMPT_TEMPLATE
- [ ] Create beta suggestion endpoint (POST /api/v1/beta/suggest)
- [ ] Parse metrics summary for prompt
- [ ] Handle LLM response formatting
- [ ] Add error handling and fallbacks

---

## Phase 5: Frontend Development

### 5.1 Project Setup
- [ ] Initialize React 18 + TypeScript project
- [ ] Configure Tailwind CSS
- [ ] Set up project structure
- [ ] Configure API client

### 5.2 Video Components
- [ ] Create VideoUploader component
- [ ] Create VideoPlayer component with Video.js
- [ ] Implement SkeletonOverlay for keypoint visualization
- [ ] Add playback controls

### 5.3 Analysis UI
- [ ] Create MetricsPanel component
- [ ] Implement real-time data display with Recharts
- [ ] Create BetaSuggestion display component
- [ ] Build Analysis page layout

### 5.4 Real-time Communication
- [ ] Implement useWebSocket hook
- [ ] Handle analysis progress updates
- [ ] Display live metrics during analysis

### 5.5 Pages
- [ ] Create Home page (video list, upload)
- [ ] Create Analysis page (video player, metrics, suggestions)

---

## Phase 6: Integration & Deployment

### 6.1 Docker Setup
- [ ] Create backend Dockerfile
- [ ] Create frontend Dockerfile
- [ ] Write docker-compose.yml
- [ ] Configure nginx reverse proxy

### 6.2 Testing
- [ ] Write backend unit tests
- [ ] Test API endpoints
- [ ] Test WebSocket connections
- [ ] End-to-end testing

### 6.3 Performance Optimization
- [ ] Profile pose estimation performance
- [ ] Optimize video processing pipeline
- [ ] Add Redis caching (optional)
- [ ] Ensure target metrics: >= 25 FPS, < 40ms detection latency

---

## Key Technical Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Pose Model | MediaPipe Pose | 33 keypoints, 3D support, lightweight |
| Backend | FastAPI | Async support, WebSocket, auto-docs |
| Frontend | React + TypeScript | Type safety, component ecosystem |
| Database | SQLite (dev) | Simple setup, migrate to PostgreSQL later |
| LLM | Ollama + Qwen2.5 | Local deployment, Chinese support |

---

## File Structure to Create

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── routes/videos.py
│   │   ├── routes/analysis.py
│   │   ├── routes/beta.py
│   │   └── websocket.py
│   ├── core/
│   │   ├── pose_estimator.py
│   │   ├── physics_engine.py
│   │   ├── metrics.py
│   │   └── llm_client.py
│   ├── models/
│   │   ├── database.py
│   │   └── schemas.py
│   └── utils/
│       ├── video_utils.py
│       └── visualization.py
├── requirements.txt
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── hooks/
│   ├── pages/
│   └── App.tsx
└── package.json
```
