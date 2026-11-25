"""FastAPI 추론 서버 호출 (Celery Worker 안에서만 사용).

FastAPI와 볼륨을 공유하지 않고, 영상 파일을 multipart로 전송한다.
→ FastAPI는 '영상을 받아 JSON을 돌려주는' 독립 서비스로 남는다.
"""
from __future__ import annotations

from pathlib import Path

import requests
from django.conf import settings


class AIServiceError(Exception):
    """재시도해도 결과가 같은 오류 (손상 영상, 형식 오류 등)."""

    def __init__(self, code: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class AIServiceUnavailable(Exception):
    """잠시 뒤 재시도할 가치가 있는 오류 (연결 실패, 모델 로딩 중 503)."""


def _parse_error(response: requests.Response) -> tuple[str, str]:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return f"HTTP_{response.status_code}", response.text[:300]
    if isinstance(detail, dict):
        return detail.get("code", f"HTTP_{response.status_code}"), detail.get("message", "")
    return f"HTTP_{response.status_code}", str(detail)[:300]


def request_prediction(video_path: Path) -> dict:
    url = f"{settings.AI_SERVICE_URL.rstrip('/')}/predict/video"
    try:
        with open(video_path, "rb") as file:
            response = requests.post(
                url,
                files={"file": (video_path.name, file, "application/octet-stream")},
                timeout=(settings.AI_CONNECT_TIMEOUT, settings.AI_READ_TIMEOUT),
            )
    except requests.ConnectTimeout as exc:
        # 연결 단계 타임아웃(5초)은 '연결 실패'다. Timeout보다 먼저 잡아야 재시도 대상이 된다.
        # (ConnectTimeout은 ConnectionError이면서 Timeout이기도 해서, 순서를 바꾸면 읽기 타임아웃으로 분류됨)
        raise AIServiceUnavailable(f"AI 서버 연결 시간 초과: {url}") from exc
    except requests.Timeout as exc:
        # 읽기 타임아웃은 재시도해도 같은 영상이면 또 오래 걸리므로 실패로 확정
        raise AIServiceError("AI_TIMEOUT", f"AI 추론이 {settings.AI_READ_TIMEOUT:.0f}초 안에 끝나지 않았습니다.") from exc
    except requests.ConnectionError as exc:
        raise AIServiceUnavailable(f"AI 서버에 연결할 수 없습니다: {url}") from exc

    if response.status_code == 200:
        return response.json()

    code, message = _parse_error(response)
    if response.status_code == 503:
        raise AIServiceUnavailable(f"{code}: {message}")
    # 400·413·422(입력 문제)와 500(추론 중 예외)은 같은 영상으로 다시 해도 결과가 같다 → 재시도하지 않음
    raise AIServiceError(code, message, response.status_code)
