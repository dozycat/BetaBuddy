from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.models.database import init_db
from app.api.routes import videos, analysis, beta
from app.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    yield
    # Shutdown


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Climbing video analysis system with AI-powered beta suggestions",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for video access
app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")

# Include routers
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])
app.include_router(beta.router, prefix="/api/v1/beta", tags=["beta"])
app.include_router(ws_router, tags=["websocket"])


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
