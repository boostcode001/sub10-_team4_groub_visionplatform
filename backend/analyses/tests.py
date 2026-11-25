"""기능 테스트 (요청·조회 API + Celery 작업 + FastAPI 호출 규칙).

실행: python manage.py test   (DB_ENGINE=sqlite 로 PostgreSQL 없이도 가능)
FastAPI 호출은 mock 으로 대체한다 (FastAPI 자체 테스트는 ai_service/tests).
"""
from __future__ import annotations

import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

import requests
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError, AIServiceUnavailable, request_prediction
from .tasks import run_analysis

TMP_MEDIA = tempfile.mkdtemp(prefix="test_media_")
CAMERAS = {"phone1": {"label": "교실 휴대폰", "url": "http://192.168.0.10:8080/video"}}

FAKE_RESULT = {
    "status": "ok",
    "message": "2명의 동작을 예측했습니다.",
    "person_detected": True,
    "predictions": [
        {"track_id": 1, "action_label": "wave", "confidence": 0.92, "raw_label": "wave",
         "window_count": 9, "vote_ratio": 0.89, "is_uncertain": False,
         "first_seen_sec": 0.0, "last_seen_sec": 5.9},
        {"track_id": 2, "action_label": "sit_stand", "confidence": 0.81, "raw_label": "sit_stand",
         "window_count": 8, "vote_ratio": 0.75, "is_uncertain": False,
         "first_seen_sec": 0.2, "last_seen_sec": 5.9},
    ],
    "video": {"duration_sec": 6.0},
    "processing_time_sec": 3.21,
}


def video_upload(name="clip.mp4", content=b"\x00\x00\x00\x18ftypmp42" + b"0" * 1024):
    return SimpleUploadedFile(name, content, content_type="video/mp4")


@override_settings(MEDIA_ROOT=TMP_MEDIA, IP_WEBCAMS=CAMERAS, CAPTURE_SECONDS=8.0)
class BaseTestCase(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.client = APIClient()  # 로그인 없이 요청

    def _job(self, **fields):
        defaults = {"input_type": AnalysisJob.InputType.UPLOAD, "video_file": "uploads/x.mp4"}
        return AnalysisJob.objects.create(**{**defaults, **fields})

    def _run(self, job, result=FAKE_RESULT):
        with mock.patch("analyses.tasks._acquire_video", return_value=Path("x.mp4")), \
                mock.patch("analyses.tasks.request_prediction", return_value=result):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        return job



class TaskTests(BaseTestCase):
    """Celery 작업: 상태 전이, 결과 저장, 실패 기록, 재시도."""

    def test_t06_task_saves_predictions_with_orm(self):
        job = self._run(self._job())
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        self.assertEqual(job.predictions.count(), 2)
        self.assertTrue(job.person_detected)
        self.assertIsNotNone(job.total_time_sec)
        self.assertIsNotNone(job.completed_at)

    def test_task_is_idempotent_when_rerun(self):
        job = self._run(self._job())
        with mock.patch("analyses.tasks.request_prediction") as req:
            run_analysis.apply(args=[job.pk])
        req.assert_not_called()
        self.assertEqual(PersonPrediction.objects.filter(job=job).count(), 2)

    def test_deleted_job_is_skipped(self):
        job = self._job()
        pk = job.pk
        job.delete()
        result = run_analysis.apply(args=[pk]).get()
        self.assertEqual(result["status"], "SKIPPED")

    def test_no_person_is_success_with_empty_predictions(self):
        empty = {**FAKE_RESULT, "predictions": [], "person_detected": False,
                 "message": "영상에서 사람을 찾지 못했습니다."}
        job = self._run(self._job(), empty)
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        self.assertFalse(job.person_detected)
        self.assertEqual(job.result_message, "영상에서 사람을 찾지 못했습니다.")
        self.assertEqual(job.predictions.count(), 0)

    def test_t09_corrupt_video_marks_failure(self):
        job = self._job()
        error = AIServiceError("VIDEO_UNREADABLE", "영상을 열 수 없습니다.", 422)
        with mock.patch("analyses.tasks._acquire_video", return_value=Path("x.mp4")), \
                mock.patch("analyses.tasks.request_prediction", side_effect=error):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual((job.status, job.error_code), (AnalysisJob.Status.FAILURE, "VIDEO_UNREADABLE"))

    def test_missing_upload_file_marks_failure(self):
        job = self._job(video_file="uploads/없는파일.mp4")
        run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual(job.error_code, "VIDEO_MISSING")

    def test_t09_camera_disconnect_marks_failure(self):
        from .camera_capture import CaptureError

        job = self._job(input_type=AnalysisJob.InputType.IP_WEBCAM, video_file="", camera_key="phone1")
        err = CaptureError("CAMERA_UNAVAILABLE", "카메라에 연결할 수 없습니다.")
        with mock.patch("analyses.tasks.capture_clip", side_effect=err):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual((job.status, job.error_code), (AnalysisJob.Status.FAILURE, "CAMERA_UNAVAILABLE"))

    def test_ai_unavailable_retries_then_fails(self):
        job = self._job()
        with mock.patch("analyses.tasks._acquire_video", return_value=Path("x.mp4")), \
                mock.patch("analyses.tasks.request_prediction", side_effect=AIServiceUnavailable("down")) as req, \
                mock.patch("analyses.tasks.RETRY_DELAY_SEC", 0):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual(req.call_count, 4)  # 최초 1회 + 재시도 3회
        self.assertEqual((job.status, job.error_code), (AnalysisJob.Status.FAILURE, "AI_UNAVAILABLE"))

    def test_total_time_includes_retries(self):
        """total_time_sec = completed_at - started_at (재시도 대기 포함, 마지막 시도만이 아님)."""
        job = self._job()
        calls = {"n": 0}

        def slow_then_ok(path):
            calls["n"] += 1
            if calls["n"] == 1:
                # 첫 시도 시작을 60초 전으로 돌려, 재시도 사이에 시간이 흐른 것처럼 만든다
                AnalysisJob.objects.filter(pk=job.pk).update(started_at=timezone.now() - timedelta(seconds=60))
                raise AIServiceUnavailable("모델 로딩 중")
            return FAKE_RESULT

        with mock.patch("analyses.tasks._acquire_video", return_value=Path("x.mp4")), \
                mock.patch("analyses.tasks.request_prediction", side_effect=slow_then_ok), \
                mock.patch("analyses.tasks.RETRY_DELAY_SEC", 0):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)
        self.assertGreaterEqual(job.total_time_sec, 60)
        self.assertAlmostEqual(job.total_time_sec, (job.completed_at - job.started_at).total_seconds(), places=2)

    def test_retry_reuses_captured_webcam_clip(self):
        job = self._job(input_type=AnalysisJob.InputType.IP_WEBCAM, video_file="", camera_key="phone1")

        def fake_capture(url, seconds, output_path, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"captured")
            return {"frames": 100, "fps": 12.5, "duration_sec": 8.0}

        with mock.patch("analyses.tasks.capture_clip", side_effect=fake_capture) as capture, \
                mock.patch("analyses.tasks.request_prediction",
                           side_effect=[AIServiceUnavailable("모델 로딩 중"), FAKE_RESULT]), \
                mock.patch("analyses.tasks.RETRY_DELAY_SEC", 0):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual(capture.call_count, 1)  # 재시도 때 다시 촬영하지 않음
        self.assertEqual(job.status, AnalysisJob.Status.SUCCESS)

    def test_malformed_prediction_marks_failure(self):
        bad = {**FAKE_RESULT, "predictions": [{"action_label": "wave"}]}  # track_id 없음
        job = self._run(self._job(), bad)
        self.assertEqual((job.status, job.error_code), (AnalysisJob.Status.FAILURE, "AI_BAD_RESPONSE"))

    def test_stale_started_job_becomes_failure_on_read(self):
        job = self._job(status=AnalysisJob.Status.STARTED,
                        started_at=timezone.now() - timedelta(hours=1))
        body = self.client.get(f"/api/analyses/{job.pk}/status/").json()
        self.assertEqual((body["status"], body["error_code"]), ("FAILURE", "WORKER_TIMEOUT"))

    def test_recent_started_job_is_left_alone(self):
        job = self._job(status=AnalysisJob.Status.STARTED, started_at=timezone.now())
        self.assertEqual(self.client.get(f"/api/analyses/{job.pk}/status/").json()["status"], "STARTED")

    def test_worker_does_not_overwrite_expired_job(self):
        job = self._job()

        def expire_then_answer(path):
            AnalysisJob.objects.filter(pk=job.pk).update(status=AnalysisJob.Status.FAILURE,
                                                        error_code="WORKER_TIMEOUT")
            return FAKE_RESULT

        with mock.patch("analyses.tasks._acquire_video", return_value=Path("x.mp4")), \
                mock.patch("analyses.tasks.request_prediction", side_effect=expire_then_answer):
            run_analysis.apply(args=[job.pk])
        job.refresh_from_db()
        self.assertEqual(job.status, AnalysisJob.Status.FAILURE)
        self.assertEqual(job.predictions.count(), 0)


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class ServiceTests(TestCase):
    """FastAPI 호출 규칙 (TEAM_CONTEXT.md 4-3): 무엇을 재시도하고 무엇을 바로 실패시키나."""

    def setUp(self):
        self.video = Path(TMP_MEDIA) / "svc.mp4"
        self.video.parent.mkdir(parents=True, exist_ok=True)
        self.video.write_bytes(b"0")

    @staticmethod
    def _response(status, code):
        fake = mock.Mock(status_code=status)
        fake.json.return_value = {"detail": {"code": code, "message": "m"}}
        return fake

    def test_503_is_retryable(self):
        with mock.patch("analyses.services.requests.post", return_value=self._response(503, "MODEL_NOT_READY")):
            with self.assertRaises(AIServiceUnavailable):
                request_prediction(self.video)

    def test_4xx_and_500_fail_immediately(self):
        for status, code in [(400, "EMPTY_FILE"), (413, "TOO_LARGE"), (422, "VIDEO_TOO_SHORT"), (500, "INFERENCE_ERROR")]:
            with mock.patch("analyses.services.requests.post", return_value=self._response(status, code)):
                with self.assertRaises(AIServiceError) as ctx:
                    request_prediction(self.video)
            self.assertEqual(ctx.exception.code, code)

    def test_connection_error_and_connect_timeout_are_retryable(self):
        for exc in (requests.ConnectionError("refused"), requests.ConnectTimeout("x")):
            with mock.patch("analyses.services.requests.post", side_effect=exc):
                with self.assertRaises(AIServiceUnavailable):
                    request_prediction(self.video)

    def test_read_timeout_fails_immediately(self):
        with mock.patch("analyses.services.requests.post", side_effect=requests.ReadTimeout("slow")):
            with self.assertRaises(AIServiceError) as ctx:
                request_prediction(self.video)
        self.assertEqual(ctx.exception.code, "AI_TIMEOUT")
