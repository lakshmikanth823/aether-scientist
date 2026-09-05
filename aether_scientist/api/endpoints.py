import json
import os
import tempfile
import time
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

try:
    from aether_scientist.core import AetherConfig, AetherScientist
    from aether_scientist.core.inference import InferenceEngine
    from aether_scientist.core.profiles import get_profile, list_profiles

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

    class AetherConfig:
        profile: str = "offline"

    class AetherScientist:
        def __init__(self, config=None):
            self.config = config or AetherConfig()

        def analyze(self, query, papers=None, use_rag=False):
            return {"result": "mock"}

    class InferenceEngine:
        device: str = "cpu"

    def get_profile(name):
        raise ValueError(f"Unknown profile '{name}'")

    def list_profiles():
        return []


API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


class AnalyzeRequest(BaseModel):
    query: str
    papers: list[str] | None = Field(default_factory=list)
    profile: str | None = None
    use_web: bool = False


class ResearchRequest(BaseModel):
    question: str
    profile: str | None = None
    k: int = 3
    use_llm_planner: bool = False
    use_web: bool = False


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
        active = [t for t in self.clients.get(client_id, []) if now - t < 60]
        if len(active) >= self.requests_per_minute:
            return False
        active.append(now)
        self.clients[client_id] = active
        return True


rate_limiter = RateLimiter()


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    _api_key = os.environ.get("AETHER_API_KEY", "dev-aether-key")
    if api_key != _api_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key


async def check_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests")


def _validate_profile(profile_name: str | None, default_profile: str = "offline") -> str:
    if not profile_name:
        return default_profile
    try:
        return get_profile(profile_name).name
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def create_app(config: AetherConfig | None = None) -> FastAPI:
    app = FastAPI(title="AetherScientist API", version="1.0.0")
    _scientist = AetherScientist(config=config) if CORE_AVAILABLE else None
    auth = [Depends(get_api_key), Depends(check_rate_limit)]

    def _prof(name: str | None) -> str:
        default = getattr(_scientist.config, "profile", "offline") if _scientist else "offline"
        return _validate_profile(name, default)

    @app.get("/profiles", response_model=APIResponse)
    async def get_profiles_list() -> APIResponse:
        return APIResponse(status="success", data={"profiles": list_profiles()})

    @app.post("/analyze", response_model=APIResponse, dependencies=auth)
    async def analyze(req: AnalyzeRequest) -> APIResponse:
        target_prof = _prof(req.profile)

        if _scientist:
            from aether_scientist.core.profiles import resolve

            original_prof = _scientist.config.profile
            _scientist.config.profile = target_prof
            _scientist.profile = resolve(target_prof)
            try:
                res = _scientist.analyze(req.query, papers=req.papers, use_web=req.use_web)
            finally:
                _scientist.config.profile = original_prof
                _scientist.profile = resolve(original_prof)
            return APIResponse(
                status="success",
                data={
                    "analysis": f"Analyzed: {req.query}",
                    "result": res,
                    "profile": target_prof,
                    "device": InferenceEngine.device,
                },
            )
        return APIResponse(
            status="success",
            data={
                "analysis": f"Analyzed: {req.query}",
                "profile": target_prof,
                "device": getattr(InferenceEngine, "device", "cpu"),
            },
        )

    @app.post("/analyze/stream", dependencies=auth)
    async def analyze_stream(req: AnalyzeRequest) -> StreamingResponse:
        target_prof = _prof(req.profile)

        def event_generator():
            from aether_scientist.core.format import clean_output, get_stop_markers

            tokens = []
            for token in InferenceEngine.stream(query=req.query, profile=target_prof):
                tokens.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            yield f"event: sources\ndata: {json.dumps({'sources': []})}\n\n"
            assembled = "".join(tokens)
            pipe = getattr(InferenceEngine, "_pipe", None)
            tokenizer = getattr(pipe, "tokenizer", None)
            cleaned = clean_output(assembled, get_stop_markers(tokenizer))
            done_payload = {
                "status": "completed",
                "total_tokens": len(tokens),
                "text": cleaned,
                "profile": target_prof,
                "device": InferenceEngine.device,
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/research", response_model=APIResponse, dependencies=auth)
    async def run_research(req: ResearchRequest) -> APIResponse:
        target_prof = _prof(req.profile)
        from aether_scientist.agent.pipeline import ResearchAgent
        from aether_scientist.core.config import AetherConfig

        agent = ResearchAgent(AetherConfig(profile=target_prof))
        if len(agent.rag_engine.store) == 0 and not req.use_web:
            raise HTTPException(
                status_code=409, detail="No documents indexed. Run: aether ingest <paths>"
            )
        try:
            res = agent.run(
                question=req.question,
                k=req.k,
                use_llm_planner=req.use_llm_planner,
                profile=target_prof,
                use_web=req.use_web,
            )
            return APIResponse(status="success", data=res.to_dict())
        except IndexError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.post("/research/stream", dependencies=auth)
    async def run_research_stream(req: ResearchRequest) -> StreamingResponse:
        target_prof = _prof(req.profile)
        from aether_scientist.agent.pipeline import ResearchAgent
        from aether_scientist.core.config import AetherConfig

        agent = ResearchAgent(AetherConfig(profile=target_prof))
        if len(agent.rag_engine.store) == 0 and not req.use_web:
            raise HTTPException(
                status_code=409, detail="No documents indexed. Run: aether ingest <paths>"
            )

        def event_generator():
            for event in agent.stream(
                question=req.question,
                k=req.k,
                use_llm_planner=req.use_llm_planner,
                profile=target_prof,
                use_web=req.use_web,
            ):
                event_type = event.get("event", "step")
                clean_event = {k: v for k, v in event.items() if k != "result_object"}
                yield f"event: {event_type}\ndata: {json.dumps(clean_event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/synthesize", response_model=APIResponse, dependencies=auth)
    async def synthesize(req: SynthesizeRequest) -> APIResponse:
        d = {"synthesis": f"Synthesized for domain: {req.domain}"}
        return APIResponse(status="success", data=d)

    @app.post("/hypothesize", response_model=APIResponse, dependencies=auth)
    async def hypothesize(req: HypothesizeRequest) -> APIResponse:
        d = {"hypothesis": "Generated hypothesis from observations."}
        return APIResponse(status="success", data=d)

    @app.post("/experiment", response_model=APIResponse, dependencies=auth)
    async def experiment(req: ExperimentRequest) -> APIResponse:
        d = {"experiment_design": "Detailed experiment plan."}
        return APIResponse(status="success", data=d)

    @app.get("/domains", response_model=APIResponse, dependencies=auth)
    async def get_domains() -> APIResponse:
        return APIResponse(status="success", data={"domains": ["physics", "biology", "chemistry"]})

    @app.post("/vision", response_model=APIResponse, dependencies=auth)
    async def vision_caption(
        file: UploadFile | None = File(None), image_path: str = ""  # noqa: B008
    ) -> APIResponse:
        from aether_scientist.multimodal.captioner import CaptionEngine

        engine = CaptionEngine()
        if file and file.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(await file.read())
                p = tmp.name
            try:
                cap = engine.caption(p)
            finally:
                os.remove(p)
            return APIResponse(status="success", data={"caption": cap, "filename": file.filename})
        if image_path:
            return APIResponse(
                status="success",
                data={"caption": engine.caption(image_path), "image_path": image_path},
            )
        raise HTTPException(status_code=400, detail="Image file or image_path required.")

    @app.get("/health", response_model=APIResponse)
    async def health_check() -> APIResponse:
        d = {"status": "healthy", "core_available": CORE_AVAILABLE}
        return APIResponse(status="success", data=d)

    return app


# Module-level app instance for uvicorn
app = create_app()
