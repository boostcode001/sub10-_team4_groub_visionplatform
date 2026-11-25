"""FastAPI AI 추론 서버 진입점.

책임: 영상 → 사람별 Action JSON 반환. DB·회원·ORM은 다루지 않는다 (저장은 Celery Worker + Django ORM).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers.predictions import router as predictions_router
from services.runtime import runtime
from services.settings import settings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 1회만 로드하고 모든 요청에서 재사용
    if not runtime.ready:
        runtime.load(settings)
    yield


app = FastAPI(
    title="Vision Action AI Service",
    description="6일차 Random Forest + YOLO Pose/ByteTrack 기반 사람별 동작 추론",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(predictions_router)


@app.get("/health", tags=["health"])
def health():
    """서버 응답 여부와 모델 준비 상태. Compose healthcheck와 Worker 사전 점검에 사용."""
    body = {
        "status": "ok" if runtime.ready else "degraded",
        "model_ready": runtime.ready,
        "error": runtime.load_error,
    }
    if runtime.bundle is not None:
        body["model"] = runtime.bundle.summary()
    return body
