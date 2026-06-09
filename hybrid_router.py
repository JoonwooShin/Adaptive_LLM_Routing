import re
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Match your current server setup:
# 8001 = standard / regular decoding
# 8002 = speculative decoding
URL_STANDARD = "http://localhost:8001/v1/chat/completions"
URL_SPECULATIVE = "http://localhost:8002/v1/chat/completions"

# Allows speculative to be slightly preferred, but not overloaded.
SPEC_LOAD_TOLERANCE = 2

std_count = 0
spec_count = 0

# Simple keyword rules for prompts likely to benefit from speculative decoding.
SPEC_KEYWORDS = [
    r"\bjson\b",
    r"\bcsv\b",
    r"\byaml\b",
    r"\btable\b",
    r"\bschema\b",
    r"\blog\b",
    r"\bnginx\b",
    r"\bsql\b",
    r"\binsert\b",
    r"\bapi\b",
    r"\bendpoint\b",
    r"\bcron\b",
    r"\bconfig\b",
    r"\bcode\b",
    r"\brepeat\b",
    r"\blist\b",
    r"\bcount\b",
    r"\bcount down\b",
    r"\bone per line\b",
    r"\bmultiples?\b",
    r"\bevery integer\b",
    r"\bgenerate \d+ lines\b",
    r"\bgenerate \d+ .*entries\b",
    r"\bexactly \d+ times\b",
]

# Prompts likely to be creative / high entropy, so standard decoding may be safer.
STANDARD_KEYWORDS = [
    r"\bhaiku\b",
    r"\bpoem\b",
    r"\bstory\b",
    r"\bmetaphor\b",
    r"\bcreative\b",
    r"\binvent\b",
    r"\bimagine\b",
    r"\bwhat color\b",
    r"\bwhat sound\b",
    r"\bone sentence\b",
    r"\bexplain .* to\b",
    r"\byelp review\b",
    r"\bconcept of\b",
]


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=120.0)
    yield
    await client.aclose()


app = FastAPI(title="Hybrid Prompt-Aware Router", lifespan=lifespan)


def extract_prompt(payload: dict) -> str:
    """Extract user-visible text from OpenAI chat-completions payload."""
    messages = payload.get("messages", [])
    parts = []

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Handles multimodal/chat content format, just in case.
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))

    return "\n".join(parts)


def looks_speculative_friendly(prompt: str) -> bool:
    p = prompt.lower()

    spec_hits = sum(1 for pattern in SPEC_KEYWORDS if re.search(pattern, p))
    std_hits = sum(1 for pattern in STANDARD_KEYWORDS if re.search(pattern, p))

    # Explicit creative/open-ended signals override weak structured signals.
    if std_hits > spec_hits:
        return False

    return spec_hits > 0


def choose_backend(payload: dict) -> tuple[str, str]:
    """
    Returns:
        (backend_name, target_url)
    """
    prompt = extract_prompt(payload)
    spec_friendly = looks_speculative_friendly(prompt)

    # If prompt is spec-friendly, prefer speculative as long as it is not overloaded.
    if spec_friendly and spec_count <= std_count + SPEC_LOAD_TOLERANCE:
        return "speculative", URL_SPECULATIVE

    # Otherwise use least-active-request balancing.
    if std_count <= spec_count:
        return "standard", URL_STANDARD
    else:
        return "speculative", URL_SPECULATIVE


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "std_count": std_count,
        "spec_count": spec_count,
    }


@app.post("/v1/chat/completions")
async def route(request: Request):
    global std_count, spec_count

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    backend_name, target_url = choose_backend(payload)

    if backend_name == "standard":
        std_count += 1
    else:
        spec_count += 1

    start = time.perf_counter()

    try:
        response = await client.post(target_url, json=payload)
        elapsed = time.perf_counter() - start

        data = response.json()

        # Add routing metadata if response is JSON object.
        if isinstance(data, dict):
            data["_router"] = {
                "router": "hybrid_prompt_aware",
                "backend": backend_name,
                "latency_s": elapsed,
                "std_count_after_route": std_count,
                "spec_count_after_route": spec_count,
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
            std_count -= 1
        else:
            spec_count -= 1


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("hybrid_router:app", host="0.0.0.0", port=8000, reload=True)
