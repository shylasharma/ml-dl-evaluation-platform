from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import CORS_ORIGINS
from app.database import init_db
from app.routers import dataset, models, experiments, reports
from app.utils.errors import friendly_message

app = FastAPI(
    title="ML/DL Model Evaluation & Comparison Platform",
    description="Research-oriented platform for comparing classifiers on imbalanced datasets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # This handler runs OUTSIDE CORSMiddleware, so without these headers the browser
    # reports every server error as a misleading "blocked by CORS policy" message.
    headers = {}
    origin = request.headers.get("origin")
    if origin and origin in CORS_ORIGINS:
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        }
    return JSONResponse(status_code=500, content={"detail": friendly_message(exc)}, headers=headers)


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(dataset.router)
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(reports.router)