import json
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

try:
    from aether_scientist.core import AetherConfig, AetherScientist

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

    class AetherConfig:
        pass

    class AetherScientist:
        def __init__(self, config=None):
            pass

        def analyze(self, query):
            return {"result": "mock"}


API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


class AnalyzeRequest(BaseModel):
    query: str
    papers: list[str] | None = Field(default_factory=list)


class SynthesizeRequest(BaseModel):
    domain: str
    paper_ids: list[str]


class HypothesizeRequest(BaseModel):
    observations: list[str]
    context: str | None = None


class ExperimentRequest(BaseModel):
    hypothesis: str
    resources: list[str] | None = Field(default_factory=list)


class APIResponse(BaseModel):
    status: str
    data: dict[str, Any]
    error: str | None = None


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.clients: dict[str, list[float]] = {}

    def check(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.clients:
            self.clients[client_id] = []

        self.clients[client_id] = [t for t in self.clients[client_id] if now - t < 60]

        if len(self.clients[client_id]) >= self.requests_per_minute:
            return False

        self.clients[client_id].append(now)
        return True


rate_limiter = RateLimiter()


async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    _api_key = os.environ.get("AETHER_API_KEY", "dev-aether-key")
    if api_key_header != _api_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key_header


async def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")


def create_app(config: AetherConfig | None = None) -> FastAPI:
    app = FastAPI(title="AetherScientist API", version="1.0.0")
    _scientist = AetherScientist(config=config) if CORE_AVAILABLE else None

    @app.post(
        "/analyze",
        response_model=APIResponse,
        dependencies=[Depends(get_api_key), Depends(check_rate_limit)],
    )
    async def analyze(req: AnalyzeRequest) -> APIResponse:
        return APIResponse(status="success", data={"analysis": f"Analyzed: {req.query}"})

    @app.post(
        "/analyze/stream",
        dependencies=[Depends(get_api_key), Depends(check_rate_limit)],
    )
    async def analyze_stream(req: AnalyzeRequest) -> StreamingResponse:
        def event_generator():
            from aether_scientist.core.inference import InferenceEngine

            prompt = f"Question: {req.query}\nAnswer:"
            tokens = []
            for token in InferenceEngine.stream(prompt):
                tokens.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            yield f"event: sources\ndata: {json.dumps({'sources': []})}\n\n"
            yield (
                "event: done\ndata: "
                f"{json.dumps({'status': 'completed', 'total_tokens': len(tokens)})}\n\n"
            )

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post(
        "/synthesize",
        response_model=APIResponse,
        dependencies=[Depends(get_api_key), Depends(check_rate_limit)],
    )
    async def synthesize(req: SynthesizeRequest) -> APIResponse:
        return APIResponse(
            status="success", data={"synthesis": f"Synthesized for domain: {req.domain}"}
        )

    @app.post(
        "/hypothesize",
        response_model=APIResponse,
        dependencies=[Depends(get_api_key), Depends(check_rate_limit)],
    )
    async def hypothesize(req: HypothesizeRequest) -> APIResponse:
        return APIResponse(
            status="success", data={"hypothesis": "Generated hypothesis from observations."}
        )

    @app.post(
        "/experiment",
        response_model=APIResponse,
        dependencies=[Depends(get_api_key), Depends(check_rate_limit)],
    )
    async def experiment(req: ExperimentRequest) -> APIResponse:
        return APIResponse(
            status="success", data={"experiment_design": "Detailed experiment plan."}
        )

    @app.get(
        "/domains",
        response_model=APIResponse,
        dependencies=[Depends(get_api_key), Depends(check_rate_limit)],
    )
    async def get_domains() -> APIResponse:
        return APIResponse(status="success", data={"domains": ["physics", "biology", "chemistry"]})

    @app.get("/health", response_model=APIResponse)
    async def health_check() -> APIResponse:
        return APIResponse(
            status="success", data={"status": "healthy", "core_available": CORE_AVAILABLE}
        )

    return app

# Module-level app instance for uvicorn
app = create_app()
