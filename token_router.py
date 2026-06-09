import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

URL_STANDARD = "http://localhost:8001/v1/chat/completions"
URL_SPECULATIVE = "http://localhost:8002/v1/chat/completions"

std_work = 0.0
spec_work = 0.0

# Completion tokens matter more because decode is iterative.
DECODE_WEIGHT = 2.0


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=120.0)
    yield
    await client.aclose()


app = FastAPI(title="Token-Aware Router", lifespan=lifespan)


def extract_prompt(payload: dict) -> str:
    messages = payload.get("messages", [])
    parts = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))

    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    # Rough approximation: 1 token ≈ 4 chars.
    return max(1, len(text) // 4)


def estimate_work(payload: dict) -> float:
    prompt = extract_prompt(payload)
    prompt_tokens = estimate_tokens(prompt)

    max_tokens = payload.get("max_tokens", 256)
    if max_tokens is None:
        max_tokens = 256

    return prompt_tokens + DECODE_WEIGHT * max_tokens


def choose_backend(work: float) -> tuple[str, str]:
    # Choose backend with less active estimated work.
    if std_work <= spec_work:
        return "standard", URL_STANDARD
    else:
        return "speculative", URL_SPECULATIVE


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "std_work": std_work,
        "spec_work": spec_work,
    }


@app.post("/v1/chat/completions")
async def route(request: Request):
    global std_work, spec_work

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    work = estimate_work(payload)
    backend_name, target_url = choose_backend(work)

    if backend_name == "standard":
        std_work += work
    else:
        spec_work += work

    start = time.perf_counter()

    try:
        response = await client.post(target_url, json=payload)
        elapsed = time.perf_counter() - start

        data = response.json()

        if isinstance(data, dict):
            data["_router"] = {
                "router": "token_aware",
                "backend": backend_name,
                "estimated_work": work,
                "latency_s": elapsed,
                "std_work_after_route": std_work,
                "spec_work_after_route": spec_work,
            }

        return JSONResponse(data, status_code=response.status_code)

    except httpx.RequestError as exc:
        return JSONResponse(
            {
                "error": f"Backend failed: {exc}",
                "backend": backend_name,
                "target_url": target_url,
            },
            status_code=502,
        )

    finally:
        if backend_name == "standard":
            std_work -= work
        else:
            spec_work -= work


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("token_router:app", host="0.0.0.0", port=8000, reload=True)
