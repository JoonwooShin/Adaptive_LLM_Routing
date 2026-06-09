import re
import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Match current setup:
# 8001 = standard / regular decoding
# 8002 = speculative decoding
URL_STANDARD = "http://localhost:8001/v1/chat/completions"
URL_SPECULATIVE = "http://localhost:8002/v1/chat/completions"

std_count = 0
spec_count = 0

# Only obvious repetitive / structured prompts should prefer speculative.
STRONG_SPEC_PATTERNS = [
    r"\bjson\b",
    r"\bcsv\b",
    r"\byaml\b",
    r"\bsql\b",
    r"\bnginx\b",
    r"\blog entries\b",
    r"\berror log\b",
    r"\bone per line\b",
    r"\beach on its own line\b",
    r"\brepeat\b",
    r"\bexactly \d+ times\b",
    r"\bevery integer\b",
    r"\bmultiples? of\b",
    r"\bcount down from\b",
    r"\bmultiplication table\b",
    r"\bfirst \d+ (even|odd)\b",
    r"\bsquares? of integers\b",
    r"\bpowers? of\b",
    r"\bmarkdown table\b",
    r"\b\.env\b",
    r"\b/etc/hosts\b",
    r"\bpython\b",
    r"\bassert statements\b",
    r"\bbash for loop\b",
    r"\bjavascript\b",
    r"\bconsole\.log\b",
    r"\bhtml list items\b",
    r"\bgit commit messages\b",
    r"\bticket IDs?\b",
    r"\bcron entries\b",
    r"\bapi endpoints\b",
    r"\btest function signatures\b",
    r"\bserver hostnames\b",
    r"\bgenerate \d+ lines\b",
    r"\bgenerate \d+ .*entries\b",
]

# Prompts likely to be creative / high-entropy.
STANDARD_PATTERNS = [
    r"\bhaiku\b",
    r"\bpoem\b",
    r"\bstory\b",
    r"\bmetaphor\b",
    r"\binvent\b",
    r"\bimagine\b",
    r"\bwhat color\b",
    r"\bwhat sound\b",
    r"\bconcept of\b",
    r"\byelp review\b",
    r"\bone sentence\b",
    r"\btwo sentences\b",
    r"\btagline\b",
    r"\bstartup\b",
    r"\bshop\b",
    r"\bfortune cookie\b",
    r"\bsuperpower\b",
    r"\bphobia\b",
    r"\blegal disclaimer\b",
    r"\btelenovela\b",
    r"\bcocktail\b",
    r"\bnature documentary\b",
    r"\blife advice\b",
    r"\bweather forecast\b",
    r"\blinkedin\b",
    r"\bmission statement\b",
    r"\bperformance review\b",
]


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=120.0)
    yield
    await client.aclose()


app = FastAPI(title="Conservative Speculative Router", lifespan=lifespan)


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


def match_patterns(prompt: str, patterns: list[str]) -> list[str]:
    p = prompt.lower()
    return [pattern for pattern in patterns if re.search(pattern, p)]


def classify_prompt(prompt: str) -> tuple[str, list[str]]:
    spec_matches = match_patterns(prompt, STRONG_SPEC_PATTERNS)
    std_matches = match_patterns(prompt, STANDARD_PATTERNS)

    # Creative/high-entropy signals override weak spec signals.
    if std_matches:
        return "standard_candidate", std_matches

    if spec_matches:
        return "spec_candidate", spec_matches

    return "unknown", []


def choose_backend(payload: dict) -> tuple[str, str, str, list[str]]:
    prompt = extract_prompt(payload)
    prompt_class, matches = classify_prompt(prompt)

    # Conservative rule:
    # Use speculative only for obvious structured prompts and only if speculative
    # is not already busier than standard.
    if prompt_class == "spec_candidate" and spec_count <= std_count:
        return "speculative", URL_SPECULATIVE, prompt_class, matches

    # Otherwise prefer standard, unless standard is clearly more loaded.
    if std_count <= spec_count + 2:
        return "standard", URL_STANDARD, prompt_class, matches

    return "speculative", URL_SPECULATIVE, prompt_class, matches


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

    backend_name, target_url, prompt_class, matches = choose_backend(payload)

    if backend_name == "standard":
        std_count += 1
    else:
        spec_count += 1

    start = time.perf_counter()

    try:
        response = await client.post(target_url, json=payload)
        elapsed = time.perf_counter() - start
        data = response.json()

        if isinstance(data, dict):
            data["_router"] = {
                "router": "conservative_speculative_v2",
                "backend": backend_name,
                "prompt_class": prompt_class,
                "matched_patterns": matches,
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

    uvicorn.run("conservative_router:app", host="0.0.0.0", port=8000, reload=True)
