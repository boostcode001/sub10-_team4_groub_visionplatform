"""Celery 작업: 영상 확보 → FastAPI 추론 → Django ORM으로 결과·실패 저장.

상태 전이: PENDING → STARTED → SUCCESS | FAILURE
- 상태 갱신은 QuerySet.update() 로 필요한 컬럼만 바꾼다.
  (사용자가 처리 중에 메모를 PATCH 해도 덮어쓰지 않음)
- 결과 저장은 transaction.atomic 으로 '예측 N건 + 상태 SUCCESS' 를 한 번에 커밋한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .camera_capture import CaptureError, capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError, AIServiceUnavailable, request_prediction

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_SEC = 10


def _elapsed_sec(started_at, finished_at) -> float | None:
    """전체 처리 시간 = completed_at - started_at (재시도·대기 시간 포함).

    started_at 은 첫 시도 때 DB에 한 번만 기록되므로, 재시도로 다시 들어와도 기준이 바뀌지 않는다.
    (time.monotonic()은 시도마다 새로 재서 마지막 시도 시간만 남았다)
    """
    return round((finished_at - started_at).total_seconds(), 3) if started_at else None


def _mark_failure(job_id: int, code: str, message: str, started_at=None) -> None:
    now = timezone.now()
    AnalysisJob.objects.filter(pk=job_id).update(
        status=AnalysisJob.Status.FAILURE,
        error_code=code[:50],
        error_message=message,
        completed_at=now,
        total_time_sec=_elapsed_sec(started_at, now),
    )
    logger.warning("job %s FAILURE %s: %s", job_id, code, message)


def _acquire_video(job: AnalysisJob) -> Path:
    """업로드 영상이면 Media 경로를 그대로, IP Webcam이면 지금 수집해서 Media에 저장."""
    if job.input_type == AnalysisJob.InputType.UPLOAD:
        if not job.video_file:
            raise CaptureError("VIDEO_MISSING", "업로드된 영상이 없습니다.")
        path = Path(job.video_file.path)
        if not path.exists():
            raise CaptureError("VIDEO_MISSING", "업로드된 영상 파일을 찾을 수 없습니다.")
        return path

    # 재시도(AI 서버 미준비)로 다시 들어온 경우: 이미 수집한 영상을 그대로 쓴다.
    # 다시 촬영하면 사용자가 카메라 앞을 떠난 뒤의 영상을 분석하게 된다.
    if job.video_file:
        captured = Path(job.video_file.path)
        if captured.exists():
            return captured

    camera = settings.IP_WEBCAMS.get(job.camera_key)
    if camera is None:  # 생성 후 .env에서 카메라가 빠진 경우
        raise CaptureError("CAMERA_NOT_ALLOWED", "허용 목록에 없는 카메라입니다.")

    relative = Path("captures") / timezone.now().strftime("%Y/%m/%d") / f"job_{job.pk}.mp4"
    output = Path(settings.MEDIA_ROOT) / relative
    capture_clip(
        camera["url"],
        seconds=job.capture_seconds or settings.CAPTURE_SECONDS,
        output_path=output,
        open_timeout_sec=settings.CAPTURE_OPEN_TIMEOUT_SEC,
        max_width=settings.CAPTURE_MAX_WIDTH,
    )
    AnalysisJob.objects.filter(pk=job.pk).update(video_file=str(relative))
    return output


def _build_predictions(job_id: int, predictions: list) -> list[PersonPrediction]:
    """FastAPI 응답을 저장할 객체로 변환. 필수 키가 없거나 타입이 틀리면 KeyError/TypeError/ValueError."""
    return [
        PersonPrediction(
            job_id=job_id,
            track_id=int(p["track_id"]),
            action_label=p["action_label"],
            raw_label=p.get("raw_label", ""),
            confidence=float(p["confidence"]),
            window_count=int(p.get("window_count", 0)),
            vote_ratio=float(p.get("vote_ratio", 0.0)),
            is_uncertain=bool(p.get("is_uncertain", False)),
            first_seen_sec=p.get("first_seen_sec"),
            last_seen_sec=p.get("last_seen_sec"),
        )
        for p in predictions
    ]


@shared_task(bind=True, name="analyses.run_analysis", max_retries=MAX_RETRIES)
def run_analysis(self, job_id: int) -> dict:
    try:
        job = AnalysisJob.objects.get(pk=job_id)
    except AnalysisJob.DoesNotExist:
        logger.warning("job %s 가 삭제되어 건너뜀", job_id)
        return {"job_id": job_id, "status": "SKIPPED"}

    if job.is_done:  # acks_late 재전달 등으로 같은 작업이 두 번 오면 무시 (멱등성)
        return {"job_id": job_id, "status": job.status}

    started = job.started_at or timezone.now()  # 재시도면 첫 시도의 시작 시각을 그대로 쓴다
    AnalysisJob.objects.filter(pk=job_id).update(status=AnalysisJob.Status.STARTED, started_at=started)

    try:
        video_path = _acquire_video(job)
        result = request_prediction(video_path)
        rows = _build_predictions(job_id, result.get("predictions", []))
    except CaptureError as exc:
        _mark_failure(job_id, exc.code, exc.message, started)
        return {"job_id": job_id, "status": "FAILURE"}
    except AIServiceError as exc:
        _mark_failure(job_id, exc.code, exc.message or "AI 서버가 요청을 처리하지 못했습니다.", started)
        return {"job_id": job_id, "status": "FAILURE"}
    except AIServiceUnavailable as exc:
        if self.request.retries < MAX_RETRIES:
            logger.info("job %s AI 서버 미준비 → %ss 후 재시도 (%s)", job_id, RETRY_DELAY_SEC, exc)
            raise self.retry(exc=exc, countdown=RETRY_DELAY_SEC)
        _mark_failure(job_id, "AI_UNAVAILABLE", str(exc), started)
        return {"job_id": job_id, "status": "FAILURE"}
    except (KeyError, TypeError, ValueError) as exc:
        # FastAPI 응답 형식이 계약(TEAM_CONTEXT 4-3)과 다름. 잡지 않으면 작업이 STARTED에 멈춘다.
        logger.exception("job %s AI 응답 형식 오류", job_id)
        _mark_failure(job_id, "AI_BAD_RESPONSE", f"AI 응답 형식이 올바르지 않습니다: {type(exc).__name__}: {exc}", started)
        return {"job_id": job_id, "status": "FAILURE"}
    except SoftTimeLimitExceeded:
        _mark_failure(job_id, "TIME_LIMIT", "작업 제한 시간을 초과했습니다.", started)
        return {"job_id": job_id, "status": "FAILURE"}
    except Exception as exc:  # 예상 못 한 오류도 DB에 남긴다
        logger.exception("job %s 처리 중 예외", job_id)
        _mark_failure(job_id, "INTERNAL_ERROR", f"{type(exc).__name__}: {exc}", started)
        return {"job_id": job_id, "status": "FAILURE"}

    with transaction.atomic():
        # 추론하는 동안 사용자가 작업을 삭제했거나, 제한 시간 초과로 이미 실패 처리됐으면 저장하지 않음
        current = AnalysisJob.objects.select_for_update().filter(pk=job_id).values_list("status", flat=True).first()
        if current != AnalysisJob.Status.STARTED:
            logger.warning("job %s 결과를 버림 (현재 상태: %s)", job_id, current)
            return {"job_id": job_id, "status": "SKIPPED"}
        PersonPrediction.objects.filter(job_id=job_id).delete()  # 재시도 시 중복 방지
        PersonPrediction.objects.bulk_create(rows)
        now = timezone.now()
        AnalysisJob.objects.filter(pk=job_id).update(
            status=AnalysisJob.Status.SUCCESS,
            result_message=str(result.get("message", ""))[:255],
            person_detected=result.get("person_detected"),
            video_duration_sec=(result.get("video") or {}).get("duration_sec"),
            inference_time_sec=result.get("processing_time_sec"),
            total_time_sec=_elapsed_sec(started, now),
            completed_at=now,
            error_code="",
            error_message="",
        )
    logger.info("job %s SUCCESS persons=%d", job_id, len(rows))
    return {"job_id": job_id, "status": "SUCCESS", "persons": len(rows)}
