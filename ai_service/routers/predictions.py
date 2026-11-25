"""POST /predict/video — Celery Worker가 보낸 영상 1건을 사람별 동작으로 추론한다.

응답 규칙 (Worker와의 API 계약)
- 200: 정상 처리. 사람이 없으면 predictions=[] + person_detected=false + message
- 400: 파일 없음·빈 파일·허용하지 않는 확장자          → Worker는 FAILURE(재시도 안 함)
- 413: 업로드 용량 초과                                 → FAILURE
- 422: 영상 손상·코덱 문제·길이 초과·너무 짧음          → FAILURE
- 503: 모델 미준비                                      → Worker가 잠시 뒤 재시도
오류 본문: {"detail": {"code": "...", "message": "..."}}
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.action_inference import analyze_video
from services.pose_pipeline import VideoReadError, VideoTooLongError, VideoTooShortError
from services.runtime import runtime
from services.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["predictions"])

CHUNK = 1024 * 1024


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _save_upload(upload: UploadFile, target: Path) -> int:
    """업로드를 Chunk 단위로 디스크에 저장하면서 용량 상한을 검사한다."""
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with target.open("wb") as out:
        while chunk := upload.file.read(CHUNK):
            written += len(chunk)
            if written > limit:
                raise _error(413, "FILE_TOO_LARGE", f"영상은 {settings.max_upload_mb}MB 이하만 허용합니다.")
            out.write(chunk)
    return written


# 과제 문서 주소(/predict/video)와 9~10일차 개발 가이드 권장 주소(/api/v1/predict/)를 모두 지원
@router.post("/predict/video")
@router.post("/api/v1/predict/")
def predict_video(file: UploadFile = File(..., description="분석할 영상 (mp4/mov/avi/mkv)")):
    # async def가 아닌 def: OpenCV·YOLO 연산이 이벤트 루프를 막지 않도록 스레드풀에서 실행된다.
    if not runtime.ready:
        raise _error(503, "MODEL_NOT_READY", runtime.load_error or "모델을 준비하는 중입니다.")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise _error(400, "UNSUPPORTED_FORMAT", f"지원하지 않는 형식입니다: '{suffix or '없음'}'")

    tmp_dir = Path(tempfile.mkdtemp(prefix="predict_"))
    video_path = tmp_dir / f"input{suffix}"
    try:
        size = _save_upload(file, video_path)
        if size == 0:
            raise _error(400, "EMPTY_FILE", "빈 파일입니다.")

        with runtime.lock:  # YOLO·ByteTrack 객체는 한 번에 한 영상만
            try:
                result = analyze_video(video_path, runtime.bundle, runtime.tracker, settings)
            except VideoTooLongError as exc:
                raise _error(422, "VIDEO_TOO_LONG", str(exc)) from exc
            except VideoTooShortError as exc:
                raise _error(422, "VIDEO_TOO_SHORT", str(exc)) from exc
            except VideoReadError as exc:
                raise _error(422, "VIDEO_UNREADABLE", str(exc)) from exc

        logger.info(
            "predict ok file=%s persons=%d time=%.2fs",
            filename, len(result["predictions"]), result["processing_time_sec"],
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:  # 예상하지 못한 오류도 Worker가 읽을 수 있는 형식으로
        logger.exception("추론 중 예외")
        raise _error(500, "INFERENCE_ERROR", f"{type(exc).__name__}: {exc}") from exc
    finally:
        file.file.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)
