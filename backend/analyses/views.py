"""DRF API: 분석 작업 요청과 결과 조회만 제공한다 (로그인 없음).

POST /api/analyses/              작업 생성 → DB 커밋 후 Celery 등록 → 202 + 작업 ID 즉시 반환
GET  /api/analyses/              작업 목록 (최신순, 20개씩)
GET  /api/analyses/{id}/         상세 + 사람별 예측
GET  /api/analyses/{id}/status/  폴링용 가벼운 응답
GET  /api/cameras/               선택 가능한 카메라 목록 (URL은 노출하지 않음)
"""
from __future__ import annotations

import uuid

from django.conf import settings
from django.db import transaction
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response

from .job_state import expire_if_stale
from .models import AnalysisJob
from .serializers import (
    AnalysisJobCreateSerializer,
    AnalysisJobSerializer,
    AnalysisJobStatusSerializer,
)
from .tasks import run_analysis


class AnalysisJobViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    queryset = AnalysisJob.objects.prefetch_related("predictions")

    def get_serializer_class(self):
        if self.action == "create":
            return AnalysisJobCreateSerializer
        if self.action == "status":
            return AnalysisJobStatusSerializer
        return AnalysisJobSerializer

    def get_object(self):
        # 상세·상태 조회는 이 메서드를 거친다 → 멈춘 작업은 여기서 실패로 정리
        job = super().get_object()
        expire_if_stale(job)
        return job

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        task_id = str(uuid.uuid4())  # task ID를 먼저 정해 DB에 함께 저장
        with transaction.atomic():
            job = serializer.save(celery_task_id=task_id)
            # 트랜잭션이 커밋된 '뒤'에 큐에 넣는다.
            # 커밋 전에 넣으면 Worker가 아직 없는 작업을 조회하는 경쟁 상태가 생긴다.
            transaction.on_commit(lambda: run_analysis.apply_async(args=[job.pk], task_id=task_id))

        return Response(
            {
                "id": job.pk,
                "task_id": task_id,
                "status": job.status,
                "status_url": f"/api/analyses/{job.pk}/status/",
                "detail_url": f"/api/analyses/{job.pk}/",
            },
            status=status.HTTP_202_ACCEPTED,  # 접수만 하고 결과는 나중에
        )

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(AnalysisJobStatusSerializer(self.get_object()).data)


@api_view(["GET"])
def camera_list(request):
    cameras = [{"key": key, "label": cam["label"]} for key, cam in settings.IP_WEBCAMS.items()]
    return Response({"cameras": cameras, "capture_seconds": settings.CAPTURE_SECONDS})
