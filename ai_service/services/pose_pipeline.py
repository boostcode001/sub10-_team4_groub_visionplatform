"""영상 → YOLO Pose + ByteTrack → Frame별·사람별 38-D Feature.

8일차 49_ip_webcam_multi_person_action.py 와의 차이 (발표 포인트)
1. 시간축: 49번은 time.perf_counter()(실제 경과 시간)를 썼다.
   파일 추론에서는 서버 속도에 따라 Window 내용이 달라지므로 '영상 시간'(frame_idx / fps)을 쓴다.
2. Tracker 상태: model.track(persist=True)는 Tracker를 모델 객체에 남긴다.
   서버는 모델을 한 번만 로드해 재사용하므로, 영상마다 reset() 하지 않으면 이전 요청의 Track이 이어진다.
3. 처리 FPS: CPU에서 모든 Frame을 추론하지 않도록 target_process_fps 기준으로 Frame을 건너뛴다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

# 8일차에서 작성한 사람별 Feature 함수 (기존 src/ 를 그대로 복사해서 사용)
from src.multi_person_utils import feature_from_keypoints

from .model_loader import ModelBundle

logger = logging.getLogger(__name__)


class VideoReadError(ValueError):
    """영상이 손상되었거나 열 수 없을 때 (사람이 없는 경우와 구분)."""


class VideoTooLongError(ValueError):
    pass


class VideoTooShortError(ValueError):
    """Sliding Window 1개(30 Step Sequence)도 만들 수 없을 만큼 짧은 영상."""


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float
    fps_assumed: bool = False  # 영상에 FPS 정보가 없어 기본값을 쓴 경우

    def as_dict(self) -> dict:
        return {
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 3),
            "fps_assumed": self.fps_assumed,
        }


@dataclass
class TrackedPerson:
    track_id: int
    keypoints_xy: np.ndarray  # (17, 2) pixel 좌표
    feature: np.ndarray | None  # 38-D, 정규화 실패 시 None


@dataclass
class FrameObservation:
    frame_index: int
    timestamp: float  # 영상 시간(초)
    persons: list[TrackedPerson] = field(default_factory=list)


class PoseTrackerProtocol(Protocol):
    def reset(self) -> None: ...

    def track(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]: ...


class YoloPoseTracker:
    """Ultralytics YOLO Pose + ByteTrack 래퍼. 서버 시작 시 1회 생성해 재사용한다."""

    def __init__(self, bundle: ModelBundle, settings=None):
        from ultralytics import YOLO  # 무거운 import는 실제 서버에서만

        from .settings import settings as default_settings

        settings = settings or default_settings
        self.bundle = bundle
        self.model = YOLO(bundle.pose_model_path)
        self.tracker_cfg = str(bundle.tracking_config.get("tracker", "bytetrack.yaml"))
        self.image_size = int(bundle.project_config["image_size"])
        self.person_conf = float(bundle.project_config["person_confidence"])

        # 팀 모델: 관절만 MediaPipe heavy로 다시 추출 (학습 때와 같은 관절)
        self.refiner = None
        if settings.pose_refiner == "mediapipe_heavy":
            from .keypoint_refiner import HeavyKeypointRefiner

            self.refiner = HeavyKeypointRefiner(
                settings.model_dir / settings.heavy_model_file, min_conf=settings.heavy_min_conf
            )

    def reset(self) -> None:
        """이전 영상의 Track 상태와 ID 카운터를 지운다 (Ultralytics 버전과 무관하게 동작)."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            if hasattr(tracker, "reset"):
                tracker.reset()
        if self.refiner is not None:
            self.refiner.reset_counts()

    def stats(self) -> dict:
        if self.refiner is None:
            return {"keypoint_source": "yolo"}
        return {
            "keypoint_source": "mediapipe_heavy",
            "heavy_refined": self.refiner.refined,
            "heavy_fallback_to_yolo": self.refiner.fallback,
        }

    def track(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
        result = self.model.track(
            source=frame,
            persist=True,  # 같은 영상 안에서는 Frame 사이 Tracker 상태 유지
            tracker=self.tracker_cfg,
            imgsz=self.image_size,
            conf=self.person_conf,
            device=self.bundle.device,
            verbose=False,
        )[0]

        boxes, keypoints = result.boxes, result.keypoints
        if boxes is None or boxes.id is None or keypoints is None or keypoints.xy is None:
            return []  # 사람이 없거나 아직 ID가 배정되지 않은 Frame

        track_ids = boxes.id.detach().cpu().numpy().astype(int)
        kp_xy = keypoints.xy.detach().cpu().numpy()
        count = min(len(track_ids), len(kp_xy))
        if self.refiner is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            return [
                (int(track_ids[i]), self.refiner.refine(frame, xyxy[i], kp_xy[i])) for i in range(count)
            ]
        return [(int(track_ids[i]), kp_xy[i]) for i in range(count)]


def probe_video(path: Path, max_video_seconds: float, min_video_seconds: float = 0.0) -> VideoMeta:
    """영상이 실제로 읽히는지 확인. 확장자만 mp4인 손상 파일을 여기서 걸러낸다."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoReadError("영상을 열 수 없습니다. 손상되었거나 지원하지 않는 코덱입니다.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoReadError("영상의 첫 Frame을 읽을 수 없습니다.")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_assumed = False
        if not (1.0 <= fps <= 240.0):  # MJPEG 녹화본 등은 FPS가 0으로 나오는 경우가 있음
            fps, fps_assumed = 30.0, True

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        height, width = frame.shape[:2]
        duration = frame_count / fps if frame_count > 0 else 0.0
        if duration > max_video_seconds:
            raise VideoTooLongError(
                f"영상 길이 {duration:.1f}초가 허용 길이 {max_video_seconds:.0f}초를 초과합니다."
            )
        # frame_count를 알 수 없는 영상(0)은 길이 검사를 건너뛰고 실제 처리 결과로 판단한다
        if frame_count > 0 and duration < min_video_seconds:
            raise VideoTooShortError(
                f"영상 길이 {duration:.1f}초가 너무 짧습니다. "
                f"동작 판단에 최소 {min_video_seconds:.1f}초 이상 필요합니다."
            )
        return VideoMeta(fps, frame_count, width, height, duration, fps_assumed)
    finally:
        capture.release()


def iter_observations(
    path: Path,
    meta: VideoMeta,
    tracker: PoseTrackerProtocol,
    target_process_fps: float,
    max_video_seconds: float,
) -> Iterator[FrameObservation]:
    """영상을 처음부터 읽어 처리 대상 Frame마다 사람별 Feature를 만든다."""
    stride = max(1, int(round(meta.fps / target_process_fps))) if target_process_fps > 0 else 1
    tracker.reset()  # 요청(영상) 단위로 Track ID를 1부터 다시 시작

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoReadError("영상을 열 수 없습니다.")
    frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            timestamp = frame_index / meta.fps
            if timestamp > max_video_seconds:
                break
            if frame_index % stride != 0:
                continue

            persons = []
            for track_id, kp_xy in tracker.track(frame):
                feature = feature_from_keypoints(kp_xy)  # 3일차·8일차와 같은 38-D 계산
                persons.append(TrackedPerson(track_id, np.asarray(kp_xy), feature))
            yield FrameObservation(frame_index, timestamp, persons)
    finally:
        capture.release()
