"""처리 중(STARTED)에서 멈춘 작업을 실패로 정리한다.

Worker가 강제 종료(메모리 부족, 컨테이너 재시작, hard time limit)되면 tasks.py의 실패 기록 코드가
실행되지 못해 작업이 STARTED로 남는다. 그러면 조회 결과가 끝없이 '처리 중'으로 남는다.
조회 시점에 제한 시간을 넘긴 작업을 FAILURE로 바꿔 이 상태를 없앤다.
"""
from __future__ import annotations

from django.conf import settings
from django.utils import timezone

from .models import AnalysisJob

STALE_ERROR_CODE = "WORKER_TIMEOUT"


def expire_if_stale(job: AnalysisJob) -> bool:
    """제한 시간을 넘긴 STARTED 작업이면 FAILURE로 바꾸고 True를 반환."""
    if job.status != AnalysisJob.Status.STARTED or job.started_at is None:
        return False
    elapsed = (timezone.now() - job.started_at).total_seconds()
    if elapsed < settings.JOB_STALE_SECONDS:
        return False
    # status 조건을 붙여 갱신: 그 사이 Worker가 정상 종료했다면 건드리지 않는다
    updated = AnalysisJob.objects.filter(pk=job.pk, status=AnalysisJob.Status.STARTED).update(
        status=AnalysisJob.Status.FAILURE,
        error_code=STALE_ERROR_CODE,
        error_message="처리가 제한 시간 안에 끝나지 않았습니다. 다시 시도해 주세요.",
        completed_at=timezone.now(),
    )
    job.refresh_from_db()
    return bool(updated)
