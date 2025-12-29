import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import enum

from app.config import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class AnalysisStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    total_frames = Column(Integer, nullable=True)
    thumbnail_path = Column(String, nullable=True)  # GIF thumbnail path
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    analyses = relationship("AnalysisTask", back_populates="video", cascade="all, delete-orphan")


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    video_id = Column(String, ForeignKey("videos.id"), nullable=False)
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING)
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    video = relationship("Video", back_populates="analyses")
    results = relationship("AnalysisResult", back_populates="task", cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String, ForeignKey("analysis_tasks.id"), nullable=False)

    # Summary statistics
    total_frames_analyzed = Column(Integer, default=0)
    avg_efficiency = Column(Float, nullable=True)
    max_acceleration = Column(Float, nullable=True)
    dyno_detected = Column(Integer, default=0)

    # Detailed data stored as JSON
    frame_data = Column(JSON, nullable=True)  # List of FrameAnalysis
    summary_stats = Column(JSON, nullable=True)  # Aggregated metrics
    joint_angle_stats = Column(JSON, nullable=True)  # Angle statistics
    com_trajectory = Column(JSON, nullable=True)  # Center of mass trajectory

    # Beta suggestion
    beta_suggestion = Column(Text, nullable=True)

    # Annotated video
    annotated_video_path = Column(String, nullable=True)

    # Unit conversion factor (normalized units to meters)
    meters_per_unit = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    task = relationship("AnalysisTask", back_populates="results")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # Run migrations for existing databases
        if await _column_missing(conn, 'analysis_results', 'annotated_video_path'):
            try:
                await conn.execute(
                    text("ALTER TABLE analysis_results ADD COLUMN annotated_video_path TEXT")
                )
            except Exception:
                pass  # Column might already exist

        if await _column_missing(conn, 'analysis_results', 'meters_per_unit'):
            try:
                await conn.execute(
                    text("ALTER TABLE analysis_results ADD COLUMN meters_per_unit FLOAT")
                )
            except Exception:
                pass  # Column might already exist

        if await _column_missing(conn, 'videos', 'thumbnail_path'):
            try:
                await conn.execute(
                    text("ALTER TABLE videos ADD COLUMN thumbnail_path TEXT")
                )
            except Exception:
                pass  # Column might already exist


async def _column_missing(conn, table: str, column: str) -> bool:
    """Check if a column is missing from a table (SQLite specific)."""
    try:
        result = await conn.execute(text(f"PRAGMA table_info({table})"))
        columns = [row[1] for row in result.fetchall()]
        return column not in columns
    except Exception:
        return False


async def get_db():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
