import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 8002 = speculative decoding
URL_SPECULATIVE = "http://localhost:8002/v1/chat/completions"

spec_count = 0


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=120.0)
    yield
    await client.aclose()


app = FastAPI(title="Speculative Only Router", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "router": "speculative_only",
        "spec_count": spec_count,
    }


@app.post("/v1/chat/completions")
async def route(request: Request):
    global spec_count

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    spec_count += 1
    start = time.perf_counter()

    try:
        response = await client.post(URL_SPECULATIVE, json=payload)
        elapsed = time.perf_counter() - start
        data = response.json()

        if isinstance(data, dict):
            data["_router"] = {
                "router": "speculative_only",
                "backend": "speculative",
                "latency_s": elapsed,
                "spec_count_after_route": spec_count,
            }

        return JSONResponse(data, status_code=response.status_code)

    except httpx.RequestError as exc:
        return JSONResponse(
            {
                "error": f"Backend failed: {exc}",
                "backend": "speculative",
                "target_url": URL_SPECULATIVE,
            },
            status_code=502,
        )

    finally:
        spec_count -= 1


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("speculative_only_router:app", host="0.0.0.0", port=8000, reload=True)
