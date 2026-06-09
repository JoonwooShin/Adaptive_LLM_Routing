import time
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# 8001 = standard / regular decoding
URL_STANDARD = "http://localhost:8001/v1/chat/completions"

std_count = 0


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=120.0)
    yield
    await client.aclose()


app = FastAPI(title="Standard Only Router", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "router": "standard_only",
        "std_count": std_count,
    }


@app.post("/v1/chat/completions")
async def route(request: Request):
    global std_count

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    std_count += 1
    start = time.perf_counter()

    try:
        response = await client.post(URL_STANDARD, json=payload)
        elapsed = time.perf_counter() - start

        data = response.json()

        if isinstance(data, dict):
            data["_router"] = {
                "router": "standard_only",
                "backend": "standard",
                "latency_s": elapsed,
                "std_count_after_route": std_count,
            }

        return JSONResponse(data, status_code=response.status_code)

    except httpx.RequestError as exc:
        return JSONResponse(
            {
                "error": f"Backend failed: {exc}",
                "backend": "standard",
                "target_url": URL_STANDARD,
            },
            status_code=502,
        )

    finally:
        std_count -= 1


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("standard_router:app", host="0.0.0.0", port=8000, reload=True)
