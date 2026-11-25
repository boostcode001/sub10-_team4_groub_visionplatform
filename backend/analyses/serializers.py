"""입력 검증(역직렬화)과 JSON 표현(직렬화).

- 생성: 입력 방식에 맞는 필드만 받는다 (업로드 파일 / 허용된 camera_key).
- 상태·예측 결과는 모두 읽기 전용. AI 처리 결과이므로 요청으로 바꿀 수 없다.
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction


class PersonPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonPrediction
        fields = [
            "track_id", "action_label", "raw_label", "confidence", "window_count",
            "vote_ratio", "is_uncertain", "first_seen_sec", "last_seen_sec",
        ]
        read_only_fields = fields


class AnalysisJobCreateSerializer(serializers.ModelSerializer):
    """POST 전용. input_type + (video_file | camera_key) 만 받는다."""

    class Meta:
        model = AnalysisJob
        fields = ["input_type", "video_file", "camera_key"]
        extra_kwargs = {"video_file": {"required": False}, "camera_key": {"required": False}}

    def validate_video_file(self, file):
        if file is None:
            return file
        suffix = Path(file.name).suffix.lower()
        if suffix not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise serializers.ValidationError(
                f"mp4·mov·avi·mkv 영상만 업로드할 수 있습니다. (받은 형식: {suffix or '없음'})"
            )
        if file.size == 0:
            raise serializers.ValidationError("빈 파일입니다.")
        if file.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(f"영상은 {settings.MAX_UPLOAD_MB}MB 이하만 허용합니다.")
        return file

    def validate(self, attrs):
        input_type = attrs.get("input_type")
        if input_type == AnalysisJob.InputType.UPLOAD:
            if not attrs.get("video_file"):
                raise serializers.ValidationError({"video_file": "분석할 영상을 선택하세요."})
            attrs["camera_key"] = ""
        elif input_type == AnalysisJob.InputType.IP_WEBCAM:
            if attrs.get("camera_key", "") not in settings.IP_WEBCAMS:
                raise serializers.ValidationError({"camera_key": "허용된 카메라만 선택할 수 있습니다."})
            attrs.pop("video_file", None)
            attrs["capture_seconds"] = settings.CAPTURE_SECONDS
        return attrs


class AnalysisJobSerializer(serializers.ModelSerializer):
    """목록·상세 조회용 (전부 읽기 전용)."""

    done = serializers.BooleanField(source="is_done", read_only=True)
    video_name = serializers.SerializerMethodField()
    predictions = PersonPredictionSerializer(many=True, read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "input_type", "video_name", "camera_key", "capture_seconds",
            "status", "done", "error_code", "error_message", "result_message",
            "person_detected", "video_duration_sec", "inference_time_sec", "total_time_sec",
            "created_at", "started_at", "completed_at", "predictions",
        ]
        read_only_fields = fields

    def get_video_name(self, obj) -> str:
        return Path(obj.video_file.name).name if obj.video_file else ""


class AnalysisJobStatusSerializer(serializers.ModelSerializer):
    """폴링용 가벼운 응답."""

    done = serializers.BooleanField(source="is_done", read_only=True)

    class Meta:
        model = AnalysisJob
        fields = ["id", "status", "done", "error_code", "error_message", "result_message"]
        read_only_fields = fields
