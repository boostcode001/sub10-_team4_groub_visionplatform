"""AI 추론 서버 환경설정.

모든 경로·한계값은 환경변수로 받는다.
- 기존 subject10 프로젝트는 ROOT = Path(__file__).parents[1] 로 경로를 계산했다.
  컨테이너(/app)로 옮기면 그 계산이 달라지므로, 이 서비스는 환경변수 → 기본값 순서로 경로를 정한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]  # ai_service/


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # 6일차 산출물(joblib·json)과 Pose 가중치(.pt)가 있는 폴더
    model_dir: Path = field(default_factory=lambda: _env_path("MODEL_DIR", APP_ROOT / "models"))
    # 기존 프로젝트의 configs/ (project.json, day07_realtime.json, day08_multi_person.json)
    config_dir: Path = field(default_factory=lambda: _env_path("CONFIG_DIR", APP_ROOT / "configs"))

    # project.json의 pose_model 값을 덮어쓸 때 사용 (예: /app/models/yolo11n-pose.pt)
    pose_model_override: str = field(default_factory=lambda: os.getenv("POSE_MODEL", ""))
    # cpu / cuda:0 / auto. 비우면 project.json의 device 값을 사용
    device_override: str = field(default_factory=lambda: os.getenv("INFERENCE_DEVICE", ""))

    # 업로드 영상 한계
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 100))
    max_video_seconds: float = field(default_factory=lambda: _env_float("MAX_VIDEO_SECONDS", 60.0))

    # CPU에서 모든 Frame을 Pose 추론하면 느리므로 영상 FPS를 이 값 근처로 낮춰 처리한다.
    # 7일차 IP Webcam 실시간 처리 FPS와 비슷한 수준(10~15)으로 두면 Window 구성이 수업 때와 비슷해진다.
    target_process_fps: float = field(default_factory=lambda: _env_float("TARGET_PROCESS_FPS", 15.0))

    # 예측 횟수(Window 수)가 이보다 적은 Track은 ID Switch·잠깐 검출된 조각으로 보고 결과에서 제외
    min_windows_per_track: int = field(default_factory=lambda: _env_int("MIN_WINDOWS_PER_TRACK", 2))
    # 다수결 득표율이 이보다 낮으면 '판단 불확실' 표시
    min_vote_ratio: float = field(default_factory=lambda: _env_float("MIN_VOTE_RATIO", 0.5))

    # 영상 파일 분석용 Sliding Window 길이(초). 비우면 day07_realtime.json 값(5.0초)을 그대로 쓴다.
    # 학습 클립이 2.4~6초로 짧아 5초 Window는 동작 전후 정지 구간까지 섞이므로 서비스에서는 2.5초를 사용
    # (피험자 교차검증 + 서비스 경로 검증: 2.5초 15/18, 3초 14/18, 5초 8/18).
    window_seconds_override: float | None = field(
        default_factory=lambda: float(os.getenv("WINDOW_SECONDS")) if os.getenv("WINDOW_SECONDS") else None
    )

    # 신뢰도 기준 덮어쓰기. 비우면 day07_realtime.json 값(0.55)
    confidence_threshold_override: float | None = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD")) if os.getenv("CONFIDENCE_THRESHOLD") else None
    )

    # 동작 목록 이름 (basic: 과제 6동작 / team: 팀 재활 6동작). 모델 classes_와 맞는지 시작 시 확인
    action_set: str = field(default_factory=lambda: os.getenv("ACTION_SET", "basic").strip().lower())

    # 관절 재추출: none(YOLO 관절 그대로) / mediapipe_heavy(팀 모델: 학습 때와 같은 MediaPipe heavy 관절)
    pose_refiner: str = field(default_factory=lambda: os.getenv("POSE_REFINER", "none").strip().lower())
    heavy_model_file: str = field(default_factory=lambda: os.getenv("HEAVY_MODEL_FILE", "pose_landmarker_heavy.task"))
    heavy_min_conf: float = field(default_factory=lambda: _env_float("HEAVY_MIN_CONF", 0.3))

    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

    # 파일 이름 (6일차 35_train_rf_baseline.py 산출물 이름 그대로)
    # 팀 모델은 이름이 rf_action_team.joblib 이므로 MODEL_FILE로 지정할 수 있게 함
    model_file: str = field(default_factory=lambda: os.getenv("MODEL_FILE", "rf_action_baseline.joblib"))
    feature_columns_file: str = "rf_feature_columns.json"
    training_info_file: str = "rf_training_info.json"


settings = Settings()
