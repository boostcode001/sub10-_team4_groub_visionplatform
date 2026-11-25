"""ERD: AnalysisJob 1:N PersonPrediction (로그인 없음 → 사용자 테이블과 연결하지 않음)

- 영상 파일 자체는 DB에 넣지 않는다. Media 볼륨에 저장하고 DB에는 경로(FileField)만 기록.
- 프레임별 관절 좌표는 저장하지 않는다. FastAPI 메모리에서 추론하고 '사람별 결과'만 남긴다.
- track_id 는 '그 작업(영상) 안의' 추적 번호이며 사람의 영구 ID가 아니다.
"""
from __future__ import annotations

from django.db import models


class AnalysisJob(models.Model):
    class InputType(models.TextChoices):
        UPLOAD = "UPLOAD", "영상 업로드"
        IP_WEBCAM = "IP_WEBCAM", "IP Webcam"

    class Status(models.TextChoices):
        # Celery 상태 이름과 동일하게 통일
        PENDING = "PENDING", "대기"
        STARTED = "STARTED", "처리 중"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    # 요청 (views.py가 저장)
    input_type = models.CharField("입력 방식", max_length=16, choices=InputType.choices)
    video_file = models.FileField("영상 경로", upload_to="uploads/%Y/%m/%d/", blank=True)
    camera_key = models.CharField("카메라", max_length=50, blank=True)  # URL이 아니라 허용 목록의 key
    capture_seconds = models.FloatField("수집 시간(초)", null=True, blank=True)
    celery_task_id = models.CharField("Celery task ID", max_length=64, blank=True, db_index=True)

    # 처리 결과 (tasks.py가 QuerySet.update()로 저장)
    status = models.CharField(
        "상태", max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    error_code = models.CharField("오류 코드", max_length=50, blank=True)
    error_message = models.TextField("오류 메시지", blank=True)
    result_message = models.CharField("결과 메시지", max_length=255, blank=True)
    person_detected = models.BooleanField("사람 검출", null=True, blank=True)
    video_duration_sec = models.FloatField("영상 길이(초)", null=True, blank=True)
    inference_time_sec = models.FloatField("AI 추론 시간(초)", null=True, blank=True)
    total_time_sec = models.FloatField("전체 처리 시간(초)", null=True, blank=True)

    created_at = models.DateTimeField("생성", auto_now_add=True, db_index=True)
    started_at = models.DateTimeField("시작", null=True, blank=True)
    completed_at = models.DateTimeField("완료", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"#{self.pk} {self.input_type} {self.status}"

    @property
    def is_done(self) -> bool:
        return self.status in (self.Status.SUCCESS, self.Status.FAILURE)


class PersonPrediction(models.Model):
    """FastAPI 응답의 predictions[] 한 항목 = 1행. 필드는 TEAM_CONTEXT.md 4-3 계약과 같다."""

    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name="predictions")
    track_id = models.PositiveIntegerField("Track ID")
    action_label = models.CharField("예측 동작", max_length=40)  # 모델 class 또는 UNCERTAIN
    raw_label = models.CharField("최다 원시 예측", max_length=40, blank=True)
    confidence = models.FloatField("신뢰도")
    window_count = models.PositiveIntegerField("처리 Window 수", default=0)
    vote_ratio = models.FloatField("득표율", default=0.0)
    is_uncertain = models.BooleanField("판단 불확실", default=False)
    first_seen_sec = models.FloatField("첫 등장(초)", null=True, blank=True)
    last_seen_sec = models.FloatField("마지막 등장(초)", null=True, blank=True)

    class Meta:
        ordering = ["job", "track_id"]
        constraints = [
            models.UniqueConstraint(fields=["job", "track_id"], name="uniq_prediction_per_track")
        ]

    def __str__(self) -> str:
        return f"job#{self.job_id} track {self.track_id}: {self.action_label} ({self.confidence:.2f})"
