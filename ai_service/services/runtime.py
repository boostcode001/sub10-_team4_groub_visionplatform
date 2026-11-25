"""서버 수명 동안 공유하는 추론 자원 (모델·Tracker·Lock).

YOLO 모델 객체와 ByteTrack 상태는 스레드 안전하지 않으므로, 영상 1건을 처리하는 동안 Lock으로 직렬화한다.
(동시 처리량은 Celery Worker 수가 아니라 이 서버의 Lock이 결정한다 → Worker concurrency=1 로 맞춤)
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from .model_loader import ArtifactError, ModelBundle, load_bundle
from .settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class InferenceRuntime:
    bundle: ModelBundle | None = None
    tracker: object | None = None
    load_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.bundle is not None and self.tracker is not None

    def load(self, settings: Settings, tracker_factory=None) -> None:
        """모델 로드 실패 시에도 서버는 띄우고 /health 로 원인을 보여 준다."""
        try:
            self.bundle = load_bundle(settings)
            if tracker_factory is None:
                from .pose_pipeline import YoloPoseTracker

                tracker_factory = YoloPoseTracker
            self.tracker = tracker_factory(self.bundle)
            self.load_error = None
        except ArtifactError as exc:
            self.load_error = str(exc)
            logger.error("모델 준비 실패: %s", exc)
        except Exception as exc:  # Pose 가중치 로드 실패 등
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("모델 준비 중 예외")


runtime = InferenceRuntime()
