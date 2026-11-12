"""
LLM configuration and management routes.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
import httpx

from app.config import settings
from app.core.llm_client import OllamaClient


router = APIRouter()


# Runtime configuration (can be updated without restart)
_runtime_config = {
    "model": settings.ollama_model,
    "temperature": 0.7,
    "max_tokens": 2048,
}


class LLMModelInfo(BaseModel):
    """Information about an available LLM model."""
    name: str
    size: Optional[str] = None
    parameter_size: Optional[str] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None


class LLMConfig(BaseModel):
    """Current LLM configuration."""
    model: str
    temperature: float
    max_tokens: int
    ollama_base_url: str


class LLMConfigUpdate(BaseModel):
    """Update LLM configuration."""
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)


class LLMModelsResponse(BaseModel):
    """Response containing available models."""
    models: list[LLMModelInfo]
    current_model: str


class LLMStatusResponse(BaseModel):
    """LLM service status."""
    available: bool
    current_model: str
    ollama_url: str
    message: str


class LLMTestResponse(BaseModel):
    """LLM test result."""
    success: bool
    model: str
    response: Optional[str] = None
    error: Optional[str] = None


@router.get("/models", response_model=LLMModelsResponse)
async def list_available_models():
    """
    List all available models from Ollama.

    Returns installed models with their metadata.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get("models", []):
                models.append(LLMModelInfo(
                    name=model.get("name", ""),
                    size=_format_size(model.get("size", 0)),
                    parameter_size=model.get("details", {}).get("parameter_size", ""),
                    modified_at=model.get("modified_at", ""),
                    digest=model.get("digest", "")[:12] if model.get("digest") else None,
                ))

            return LLMModelsResponse(
                models=models,
                current_model=_runtime_config["model"],
            )

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Ollama: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get("/config", response_model=LLMConfig)
async def get_llm_config():
    """Get current LLM configuration."""
    return LLMConfig(
        model=_runtime_config["model"],
        temperature=_runtime_config["temperature"],
        max_tokens=_runtime_config["max_tokens"],
        ollama_base_url=settings.ollama_base_url,
    )


@router.put("/config", response_model=LLMConfig)
async def update_llm_config(config: LLMConfigUpdate):
    """
    Update LLM configuration at runtime.

    Changes take effect immediately for new requests.
    """
    if config.model is not None:
        # Verify model exists
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{settings.ollama_base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

                available_models = [m.get("name", "") for m in data.get("models", [])]

                if config.model not in available_models:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{config.model}' not found. Available: {available_models}",
                    )

                _runtime_config["model"] = config.model

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to verify model: {str(e)}",
            )

    if config.temperature is not None:
        _runtime_config["temperature"] = config.temperature

    if config.max_tokens is not None:
        _runtime_config["max_tokens"] = config.max_tokens

    return LLMConfig(
        model=_runtime_config["model"],
        temperature=_runtime_config["temperature"],
        max_tokens=_runtime_config["max_tokens"],
        ollama_base_url=settings.ollama_base_url,
    )


@router.get("/status", response_model=LLMStatusResponse)
async def get_llm_status():
    """Check if Ollama service is available and responding."""
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=_runtime_config["model"],
    )

    available = await client.is_available()

    if available:
        return LLMStatusResponse(
            available=True,
            current_model=_runtime_config["model"],
            ollama_url=settings.ollama_base_url,
            message="Ollama is running and accessible",
        )
    else:
        return LLMStatusResponse(
            available=False,
            current_model=_runtime_config["model"],
            ollama_url=settings.ollama_base_url,
            message="Ollama is not accessible. Make sure the service is running.",
        )


@router.post("/test", response_model=LLMTestResponse)
async def test_llm():
    """
    Test the current LLM configuration with a simple prompt.

    Useful for verifying the model is working correctly.
    """
    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=_runtime_config["model"],
        timeout=30,
    )

    if not await client.is_available():
        return LLMTestResponse(
            success=False,
            model=_runtime_config["model"],
            error="Ollama service is not available",
        )

    try:
        response = await client.generate("Say 'Hello, BetaBuddy!' in one sentence.")

        if response:
            return LLMTestResponse(
                success=True,
                model=_runtime_config["model"],
                response=response[:500],  # Truncate long responses
            )
        else:
            return LLMTestResponse(
                success=False,
                model=_runtime_config["model"],
                error="No response from model",
            )

    except Exception as e:
        return LLMTestResponse(
            success=False,
            model=_runtime_config["model"],
            error=str(e),
        )


def get_current_model() -> str:
    """Get the currently configured model name."""
    return _runtime_config["model"]


def get_current_config() -> dict:
    """Get the current runtime configuration."""
    return _runtime_config.copy()


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes == 0:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
