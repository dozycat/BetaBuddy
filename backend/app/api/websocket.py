import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.models.database import async_session, AnalysisTask, AnalysisStatus
from app.api.routes.analysis import active_tasks

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]

    async def send_message(self, message: dict, task_id: str):
        if task_id in self.active_connections:
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

    async def broadcast(self, message: dict):
        for task_id in self.active_connections:
            await self.send_message(message, task_id)


manager = ConnectionManager()


@router.websocket("/ws/analysis/{task_id}")
async def analysis_websocket(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time analysis updates."""
    await manager.connect(websocket, task_id)

    try:
        # Check if task exists
        async with async_session() as db:
            result = await db.execute(
                select(AnalysisTask).where(AnalysisTask.id == task_id)
            )
            task = result.scalar_one_or_none()

            if not task:
                await websocket.send_json({
                    "type": "error",
                    "message": "Task not found",
                })
                await websocket.close()
                return

            # Send initial status
            await websocket.send_json({
                "type": "status",
                "data": {
                    "status": task.status.value,
                    "progress": task.progress,
                },
            })

            # If already completed, send final status
            if task.status == AnalysisStatus.COMPLETED:
                await websocket.send_json({
                    "type": "complete",
                    "data": {
                        "task_id": task_id,
                        "status": "completed",
                    },
                })
                return

            if task.status == AnalysisStatus.FAILED:
                await websocket.send_json({
                    "type": "error",
                    "message": task.error_message or "Analysis failed",
                })
                return

        # Monitor for updates
        last_progress = 0
        while True:
            try:
                # Check for client messages (with timeout)
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.5,
                    )
                    # Handle client messages if needed
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass

                # Check task status
                if task_id in active_tasks:
                    task_info = active_tasks[task_id]
                    current_progress = task_info.get("progress", 0)

                    # Send progress update if changed
                    if current_progress != last_progress:
                        await websocket.send_json({
                            "type": "progress",
                            "data": {
                                "progress": current_progress,
                                "current_frame": task_info.get("current_frame", 0),
                                "status": task_info.get("status", "processing"),
                            },
                        })
                        last_progress = current_progress

                    # Check if completed or failed
                    if task_info.get("status") == "completed":
                        await websocket.send_json({
                            "type": "complete",
                            "data": {
                                "task_id": task_id,
                                "result_id": task_info.get("result_id"),
                                "status": "completed",
                            },
                        })
                        break

                    if task_info.get("status") == "failed":
                        await websocket.send_json({
                            "type": "error",
                            "message": task_info.get("error", "Analysis failed"),
                        })
                        break

                await asyncio.sleep(0.1)

            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, task_id)
