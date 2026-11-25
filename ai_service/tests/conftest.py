"""테스트 환경: 실제 YOLO·6일차 모델 대신
- tests/_fakes/src  : 기존 src/ 의 호출 형태를 맞춘 대역
- 합성 Skeleton 으로 학습한 더미 Random Forest (1,140 Feature, 6 Class)
- FakePoseTracker   : 두 사람(wave / sit_stand)의 Keypoint를 시간에 따라 생성
을 사용해 FastAPI의 전체 처리 순서를 검증한다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parent
APP_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(TESTS_DIR / "_fakes"))  # 'src' 는 대역 패키지로 해석

from src.multi_person_utils import feature_from_keypoints  # noqa: E402

ACTIONS = ["bend_return", "leg_raise_lower", "walk_turn_walk", "drink_return", "sit_stand", "wave"]
SEQUENCE_STEPS = 30
PROCESS_FPS = 15.0

BASE = np.array(
    [
        [100, 40], [95, 35], [105, 35], [90, 38], [110, 38],  # 얼굴
        [80, 70], [120, 70],  # 어깨
        [75, 100], [125, 100],  # 팔꿈치
        [72, 130], [128, 130],  # 손목
        [88, 140], [112, 140],  # 엉덩이
        [86, 190], [114, 190],  # 무릎
        [85, 240], [115, 240],  # 발목
    ],
    dtype=np.float32,
)


def synth_pose(action: str, t: float, offset_x: float = 0.0, rng=None) -> np.ndarray:
    kp = BASE.copy()
    s = math.sin(2 * math.pi * 0.8 * t)
    if action == "wave":
        kp[8] = [140, 60]
        kp[10] = [150 + 25 * math.sin(2 * math.pi * 2 * t), 20]
    elif action == "sit_stand":
        d = 40 * (0.5 + 0.5 * s)
        kp[11:13, 1] += d
        kp[0:11, 1] += d
        kp[13:15, 0] += np.array([-1, 1]) * d * 0.6
    elif action == "bend_return":
        d = 50 * (0.5 + 0.5 * s)
        kp[0:11, 1] += d
        kp[0:11, 0] += d * 0.5
    elif action == "leg_raise_lower":
        d = 50 * (0.5 + 0.5 * s)
        kp[13, 1] -= d
        kp[15, 1] -= d * 1.6
    elif action == "walk_turn_walk":
        kp[13, 0] += 20 * s
        kp[15, 0] += 30 * s
        kp[14, 0] -= 20 * s
        kp[16, 0] -= 30 * s
    elif action == "drink_return":
        d = 0.5 + 0.5 * s
        kp[10] = kp[10] + d * (np.array([105, 45]) - kp[10])
        kp[8] = kp[8] + d * (np.array([130, 80]) - kp[8])
    kp[:, 0] += offset_x
    if rng is not None:
        kp += rng.normal(0, 1.0, kp.shape).astype(np.float32)
    return kp


def build_dummy_artifacts(model_dir: Path, config_dir: Path) -> None:
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    rng = np.random.default_rng(0)
    rows, labels = [], []
    for action in ACTIONS:
        for _ in range(40):
            start = rng.uniform(0, 3)
            times = start + np.linspace(0, 2.0, SEQUENCE_STEPS)
            seq = [feature_from_keypoints(synth_pose(action, t, rng=rng)) for t in times]
            rows.append(np.concatenate(seq))
            labels.append(action)
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(np.array(rows), labels)

    model_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "rf_action_baseline.joblib")
    columns = [f"t{step:02d}_f{i:02d}" for step in range(SEQUENCE_STEPS) for i in range(38)]
    (model_dir / "rf_feature_columns.json").write_text(json.dumps(columns))
    (model_dir / "rf_training_info.json").write_text(
        json.dumps({"classes": ACTIONS, "feature_count": SEQUENCE_STEPS * 38, "sequence_steps": SEQUENCE_STEPS})
    )
    (config_dir / "day07_realtime.json").write_text(
        json.dumps(
            {
                "window_seconds": 2.0,
                "min_window_fill_ratio": 0.8,
                "min_valid_feature_ratio": 0.7,
                "min_valid_features": 15,
                "prediction_interval_sec": 0.5,
                "confidence_threshold": 0.5,
                "stabilization_buffer": 5,
                "stabilization_min_votes": 3,
            }
        )
    )
    (config_dir / "day08_multi_person.json").write_text(
        json.dumps({"tracker": "bytetrack.yaml", "lost_timeout_seconds": 2.0})
    )
    (config_dir / "project.json").write_text(
        json.dumps({"pose_model": "yolo11n-pose.pt", "image_size": 640, "person_confidence": 0.4, "device": "cpu"})
    )


class FakePoseTracker:
    """Frame 호출 순서로 시간을 계산해 두 사람의 Keypoint를 돌려준다.

    scenario="two_people": Track 1 = wave, Track 2 = sit_stand (6초 내내 등장)
    scenario="nobody"    : 사람 없음
    scenario="blip"      : Track 7 이 0.3초만 등장 (ID Switch 조각 흉내)
    """

    def __init__(self, bundle=None, scenario: str = "two_people"):
        self.scenario = scenario
        self.calls = 0
        self.reset_count = 0
        self.rng = np.random.default_rng(1)

    def reset(self):
        self.calls = 0
        self.reset_count += 1

    def track(self, frame):
        t = self.calls / PROCESS_FPS
        self.calls += 1
        if self.scenario == "nobody":
            return []
        people = []
        if self.scenario in ("two_people", "blip"):
            people.append((1, synth_pose("wave", t, rng=self.rng)))
            people.append((2, synth_pose("sit_stand", t, offset_x=200, rng=self.rng)))
        if self.scenario == "blip" and 1.0 <= t < 1.3:
            people.append((7, synth_pose("walk_turn_walk", t, offset_x=400, rng=self.rng)))
        return people


def write_video(path: Path, seconds: float = 6.0, fps: float = 30.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(int(seconds * fps)):
        frame = np.full((240, 320, 3), 40, np.uint8)
        cv2.putText(frame, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def artifact_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    build_dummy_artifacts(root / "models", root / "configs")
    return root / "models", root / "configs"


@pytest.fixture()
def test_settings(artifact_dirs, monkeypatch):
    model_dir, config_dir = artifact_dirs
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("TARGET_PROCESS_FPS", str(PROCESS_FPS))
    from services.settings import Settings

    return Settings()


@pytest.fixture()
def sample_video(tmp_path):
    return write_video(tmp_path / "sample.mp4")
