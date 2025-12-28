# BetaBuddy Implementation Plan

## Objective

Implement three key features:
1. **Real-time metrics streaming** - Stream CoM, joint angles, velocity, and acceleration during video analysis
2. **Video annotation** - Annotate keypoints on video frames and generate annotated output video
3. **Configurable LLM** - Allow runtime selection of different open-source LLM models

---

## Current State Analysis

### What Already Exists
| Feature | Status | Location |
|---------|--------|----------|
| Center of Mass calculation | ✅ Complete | `core/physics_engine.py:calculate_center_of_mass()` |
| Joint Angles (8 joints) | ✅ Complete | `core/physics_engine.py:calculate_all_joint_angles()` |
| Velocity calculation | ✅ Complete | `core/physics_engine.py:calculate_velocity()` |
| Acceleration calculation | ✅ Complete | `core/physics_engine.py:calculate_acceleration()` |
| Frame metrics processing | ✅ Complete | `core/metrics.py:ClimbingMetrics.process_frame()` |
| WebSocket infrastructure | ✅ Complete | `api/websocket.py` (only sends progress) |
| Ollama LLM client | ✅ Complete | `core/llm_client.py` (hardcoded model) |
| Video annotation | ❌ Missing | Needs new module |

### What Needs to Be Built
1. Extend WebSocket to stream per-frame metrics (CoM, angles, velocity, acceleration)
2. Create `core/annotator.py` for drawing keypoints/skeleton on frames
3. Add annotated video generation during/after analysis
4. Implement LLM model discovery and runtime selection API

---

## Implementation Plan

### Phase 1: Real-Time Metrics Streaming

#### 1.1 Extend WebSocket Message Broadcasting
**File:** `backend/app/api/routes/analysis.py`

Modify `run_analysis()` to broadcast metrics for each processed frame:
- Send `WSKeypointsMessage` with 33 keypoints per frame
- Send `WSMetricsMessage` with CoM, joint angles, velocity, acceleration
- Throttle to every N frames if needed for performance

```python
# Current: Only sends progress
await broadcast_progress(task_id, progress, current_frame)

# Add: Send full metrics
await broadcast_metrics(task_id, frame_metrics)
await broadcast_keypoints(task_id, keypoints)
```

#### 1.2 Add Temporal Buffer for Real-Time Kinematics
**File:** `backend/app/core/metrics.py`

Current velocity/acceleration requires frame history. Implement:
- Circular buffer (last 30 frames) for CoM positions
- Online exponential moving average (EMA) smoothing
- Immediate velocity/acceleration from buffer without full history

```python
class RealTimeKinematics:
    def __init__(self, buffer_size=30, fps=30.0, alpha=0.3):
        self.position_buffer = deque(maxlen=buffer_size)
        self.alpha = alpha  # EMA smoothing factor

    def update(self, position):
        # Add position, compute velocity/acceleration in O(1)
```

#### 1.3 Update WebSocket Handlers
**File:** `backend/app/api/websocket.py`

Add handlers for new message types:
- `send_keypoints(task_id, keypoints)`
- `send_metrics(task_id, metrics)`

---

### Phase 2: Video Annotation

#### 2.1 Create Annotation Module
**New File:** `backend/app/core/annotator.py`

```python
class VideoAnnotator:
    # MediaPipe skeleton connections (33 keypoints)
    SKELETON_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Face
        (0, 4), (4, 5), (5, 6), (6, 8),  # Face
        (11, 12), (11, 13), (13, 15),    # Left arm
        (12, 14), (14, 16),              # Right arm
        (11, 23), (12, 24), (23, 24),    # Torso
        (23, 25), (25, 27), (27, 29),    # Left leg
        (24, 26), (26, 28), (28, 30),    # Right leg
        ...
    ]

    def draw_keypoints(self, frame, keypoints, visibility_threshold=0.5):
        """Draw colored circles at keypoint positions"""

    def draw_skeleton(self, frame, keypoints, visibility_threshold=0.5):
        """Draw bone connections between keypoints"""

    def draw_center_of_mass(self, frame, com, trajectory=None):
        """Draw CoM point and optional trajectory line"""

    def draw_metrics_overlay(self, frame, metrics):
        """Draw text overlay with stability, angles, velocity"""

    def annotate_frame(self, frame, keypoints, metrics=None):
        """Full annotation: keypoints + skeleton + metrics"""
```

#### 2.2 Color Scheme for Visibility
- High confidence (>0.8): Green (#00FF00)
- Medium confidence (0.5-0.8): Yellow (#FFFF00)
- Low confidence (<0.5): Red (#FF0000) or skip

#### 2.3 Annotated Video Generator
**File:** `backend/app/core/annotator.py`

```python
class AnnotatedVideoGenerator:
    def __init__(self, input_path, output_path, fps=30):
        self.cap = cv2.VideoCapture(input_path)
        self.writer = cv2.VideoWriter(output_path, ...)
        self.annotator = VideoAnnotator()

    def process(self, frame_data):
        """Generate annotated video from analysis results"""
        for frame_num, data in frame_data.items():
            ret, frame = self.cap.read()
            annotated = self.annotator.annotate_frame(frame, data.keypoints, data.metrics)
            self.writer.write(annotated)
```

#### 2.4 Integration Options

**Option A: Annotate During Analysis (Real-time)**
Modify `run_analysis()` to write annotated frames as they're processed:
- Pros: Single pass, immediate results
- Cons: Slower analysis, larger memory usage

**Option B: Post-Processing (Recommended)**
Add new API endpoint for annotation after analysis:
```
POST /api/v1/videos/{video_id}/annotate
```
- Pros: Analysis stays fast, annotation is optional
- Cons: Requires re-reading video

#### 2.5 API Endpoint for Annotated Video
**File:** `backend/app/api/routes/videos.py`

```python
@router.post("/{video_id}/annotate")
async def generate_annotated_video(video_id: int):
    # Fetch analysis results
    # Generate annotated video
    # Return path/URL to annotated video
```

#### 2.6 Update Database Schema
**File:** `backend/app/models/database.py`

Add field to track annotated video:
```python
class AnalysisResult(Base):
    ...
    annotated_video_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
```

---

### Phase 3: Configurable LLM Model

#### 3.1 Add Model Configuration Schema
**File:** `backend/app/models/schemas.py`

```python
class LLMModelInfo(BaseModel):
    name: str
    size: str
    parameter_count: Optional[str]

class LLMConfig(BaseModel):
    model: str = "qwen2.5:7b"
    temperature: float = 0.7
    max_tokens: int = 2048

class LLMConfigUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
```

#### 3.2 Add LLM Configuration Endpoints
**New File:** `backend/app/api/routes/llm.py`

```python
@router.get("/models")
async def list_available_models():
    """Query Ollama for installed models via /api/tags"""

@router.get("/config")
async def get_llm_config():
    """Return current LLM configuration"""

@router.put("/config")
async def update_llm_config(config: LLMConfigUpdate):
    """Update LLM model/parameters at runtime"""

@router.post("/test")
async def test_llm_connection():
    """Test connectivity and model availability"""
```

#### 3.3 Extend LLM Client
**File:** `backend/app/core/llm_client.py`

```python
class OllamaClient:
    def __init__(self, base_url, model, timeout):
        self.model = model  # Make configurable

    async def list_models(self) -> List[LLMModelInfo]:
        """GET /api/tags to list installed models"""
        response = await self.client.get(f"{self.base_url}/api/tags")
        return [LLMModelInfo(**m) for m in response.json()["models"]]

    async def switch_model(self, model_name: str) -> bool:
        """Validate model exists and switch"""
        models = await self.list_models()
        if model_name in [m.name for m in models]:
            self.model = model_name
            return True
        return False
```

#### 3.4 Persist Configuration
**Options:**
1. **Environment variable** - Restart required
2. **Database table** - Runtime persistence (recommended)
3. **Config file** - Manual editing

Add `Settings` table:
```python
class Settings(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)
```

#### 3.5 Register Routes
**File:** `backend/app/main.py`

```python
from app.api.routes import llm

app.include_router(llm.router, prefix="/api/v1/llm", tags=["llm"])
```

#### 3.6 Popular Open-Source Models to Support
| Model | Size | Use Case |
|-------|------|----------|
| qwen2.5:7b | 4.4GB | Current default, good balance |
| qwen2.5:14b | 8.9GB | Better quality, slower |
| qwen2.5:32b | 19GB | High quality, requires GPU |
| llama3.2:3b | 2GB | Fast, lightweight |
| llama3.1:8b | 4.7GB | Good general purpose |
| deepseek-coder:6.7b | 3.8GB | Code-focused |
| mistral:7b | 4.1GB | Fast inference |

---

## File Changes Summary

### New Files
```
backend/app/core/annotator.py          # Video annotation module
backend/app/api/routes/llm.py          # LLM configuration endpoints
```

### Modified Files
```
backend/app/api/routes/analysis.py     # Add metrics broadcasting
backend/app/api/routes/videos.py       # Add annotate endpoint
backend/app/api/websocket.py           # Add metrics/keypoints handlers
backend/app/core/metrics.py            # Add RealTimeKinematics class
backend/app/core/llm_client.py         # Add model listing/switching
backend/app/models/database.py         # Add annotated_video_path, Settings table
backend/app/models/schemas.py          # Add LLM config schemas
backend/app/main.py                    # Register LLM routes
backend/app/config.py                  # Add LLM config fields
```

---

## API Endpoints Summary

### New Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/llm/models` | List available Ollama models |
| GET | `/api/v1/llm/config` | Get current LLM configuration |
| PUT | `/api/v1/llm/config` | Update LLM model/parameters |
| POST | `/api/v1/llm/test` | Test LLM connection |
| POST | `/api/v1/videos/{id}/annotate` | Generate annotated video |
| GET | `/api/v1/videos/{id}/annotated` | Download annotated video |

### Enhanced WebSocket Messages
| Message Type | Content |
|--------------|---------|
| `keypoints` | 33 keypoints with x, y, z, visibility per frame |
| `metrics` | CoM, joint angles, velocity, acceleration per frame |
| `progress` | Current frame, percentage (existing) |
| `complete` | Analysis finished (existing) |

---

## Implementation Order

1. **Phase 1.1-1.3:** Real-time metrics streaming (WebSocket enhancement)
2. **Phase 2.1-2.3:** Video annotation module
3. **Phase 2.4-2.6:** Annotate endpoint and database changes
4. **Phase 3.1-3.6:** LLM configuration system

---

## Testing Checklist

### Real-Time Metrics
- [ ] WebSocket receives keypoints for each frame
- [ ] WebSocket receives metrics (CoM, angles, velocity, acceleration)
- [ ] Velocity/acceleration calculations are smooth (no jitter)
- [ ] Performance: <100ms per frame total processing time

### Video Annotation
- [ ] Keypoints drawn at correct positions
- [ ] Skeleton connections drawn correctly
- [ ] CoM trajectory displayed
- [ ] Metrics overlay readable
- [ ] Annotated video plays correctly
- [ ] Output video quality matches input

### LLM Configuration
- [ ] List models returns installed Ollama models
- [ ] Model switching persists across requests
- [ ] Invalid model name returns error
- [ ] Beta suggestions use selected model
- [ ] Configuration survives server restart

---

## Notes

- All physics calculations (CoM, angles, velocity, acceleration) already exist and work correctly
- Focus is on surfacing existing calculations in real-time and adding visualization
- Video annotation uses OpenCV drawing functions (already a dependency)
- LLM selection queries Ollama's `/api/tags` endpoint dynamically
