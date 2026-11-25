"""MediaPipe Pose heavy로 사람 박스 안의 관절을 다시 뽑는다 (팀 모델용).

팀 모델(rf_action_team.joblib)은 MediaPipe heavy 관절(33개 → COCO 17개 변환)로 학습되었다.
서비스는 사람 찾기·Track ID는 YOLO + ByteTrack을 그대로 쓰고, 관절만 학습 때와 같은 방식으로 바꿔 넣는다.
규칙은 팀 저장소(rlaehdms02/mothion)의 src/heavy_pose.py `HeavyTrackYOLO`와 같다.
  - MP_TO_COCO 대응표 동일
  - 박스를 가로·세로 15%씩 넓혀 자른 뒤 IMAGE 모드로 1명만 추출
  - heavy가 그 사람을 못 잡으면 YOLO 관절을 그대로 사용 (fallback)
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe 33점 → COCO 17점 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

HEAVY_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


class HeavyKeypointRefiner:
    def __init__(self, model_path: Path, min_conf: float = 0.3, pad: float = 0.15):
        import mediapipe as mp  # 팀 모델을 쓸 때만 필요
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"MediaPipe heavy 모델이 없습니다: {model_path} (받는 곳: {HEAVY_MODEL_URL})"
            )
        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,  # 사람마다 잘라서 한 장씩 처리
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.pad = float(pad)
        self.refined = 0
        self.fallback = 0

    def reset_counts(self) -> None:
        self.refined = 0
        self.fallback = 0

    def refine(self, frame: np.ndarray, box_xyxy: np.ndarray, yolo_xy: np.ndarray) -> np.ndarray:
        """한 사람의 박스 안에서 heavy 관절 17개를 반환. 실패하면 YOLO 관절 그대로."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        bw, bh = x2 - x1, y2 - y1
        cx1, cy1 = int(max(0, x1 - bw * self.pad)), int(max(0, y1 - bh * self.pad))
        cx2, cy2 = int(min(width, x2 + bw * self.pad)), int(min(height, y2 + bh * self.pad))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            self.fallback += 1
            return yolo_xy
        try:
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB),
            )
            result = self._landmarker.detect(image)
        except Exception:  # MediaPipe 내부 오류도 서비스 전체를 멈추지 않게
            logger.exception("MediaPipe heavy 추출 실패 → YOLO 관절 사용")
            self.fallback += 1
            return yolo_xy
        if not result.pose_landmarks:
            self.fallback += 1
            return yolo_xy
        landmarks = result.pose_landmarks[0]
        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        refined = np.asarray(
            [[cx1 + landmarks[i].x * crop_w, cy1 + landmarks[i].y * crop_h] for i in MP_TO_COCO],
            dtype=np.float32,
        )
        self.refined += 1
        return refined

    def close(self) -> None:
        self._landmarker.close()
