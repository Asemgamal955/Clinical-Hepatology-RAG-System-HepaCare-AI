from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.api.routes import router, frontend_router

# Resolves to src/
BASE_DIR = Path(__file__).resolve().parent.parent

# Resolves to src/fornt and src/fornt/assets
FRONTEND_DIR = BASE_DIR / "fornt"
ASSETS_DIR = FRONTEND_DIR / "assets"

app = FastAPI(
    title="Hepatology & Liver Disease RAG API",
    description="REST API for clinical LLM generation, vector retrieval, and evaluation metrics benchmark.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(frontend_router)

# Mount static files to /assets
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
else:
    print(f"⚠️ WARNING: Static assets directory not found at: {ASSETS_DIR}")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/")
async def serve_frontend():
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(str(frontend_index), media_type="text/html")
    return {"message": f"Frontend index.html not found at {frontend_index}"}