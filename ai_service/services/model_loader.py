"""6일차 Random Forest 산출물과 7·8일차 설정을 한 번만 로드해 재사용한다.

핵심 검증 (테스트 T01)
- model.n_features_in_ == len(rf_feature_columns.json) == sequence_steps × 38
- rf_training_info.json의 classes 와 model.classes_ 가 같은 집합
- 6개 Action(A001~A006)과 일치하는지 확인 (다르면 경고만 남기고 서비스는 띄운다)

주의: joblib.load()는 pickle 기반이므로 본인이 학습한 신뢰 가능한 파일에만 사용한다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from .settings import Settings

logger = logging.getLogger(__name__)

FEATURES_PER_FRAME = 38  # 17 Keypoint × (x, y) + 관절각 4개

# Django backend/analyses/action_catalog.py 와 같은 목록 (ACTION_SET 으로 선택)
ACTION_SETS: dict[str, dict[str, str]] = {
    "basic": {
        "bend_return": "A001", "leg_raise_lower": "A002", "walk_turn_walk": "A003",
        "drink_return": "A004", "sit_stand": "A005", "wave": "A006",
    },
    "team": {  # 팀 공식 3동작 모델
        "seated_knee_extension": "A001", "shoulder_reach_up_down": "A002", "side_lying_leg_raise": "A006",
    },
    "team6": {  # 확장 실험 6동작 모델
        "seated_knee_extension": "A001", "shoulder_reach_up_down": "A002", "foam_roller_hip_hinge": "A003",
        "side_lying_leg_raise": "A006", "adductor_bridge": "A009", "bird_dog": "A010",
    },
}

REQUIRED_TRAINING_KEYS = ("classes", "feature_count", "sequence_steps")
REQUIRED_RUNTIME_KEYS = (
    "window_seconds",
    "min_window_fill_ratio",
    "min_valid_feature_ratio",
    "min_valid_features",
    "prediction_interval_sec",
    "confidence_threshold",
    "stabilization_buffer",
    "stabilization_min_votes",
)


class ArtifactError(RuntimeError):
    """모델·설정 파일이 없거나 서로 맞지 않을 때 발생."""


@dataclass
class ModelBundle:
    action_model: Any
    feature_columns: list[str]
    training_info: dict
    runtime_config: dict  # day07_realtime.json
    tracking_config: dict  # day08_multi_person.json
    project_config: dict  # project.json
    pose_model_path: str
    device: str
    model_file: str = "rf_action_baseline.joblib"
    action_set: str = "basic"
    pose_refiner: str = "none"
    warnings: list[str] = field(default_factory=list)

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.action_model.classes_]

    @property
    def sequence_steps(self) -> int:
        return int(self.training_info["sequence_steps"])

    @property
    def feature_count(self) -> int:
        return int(self.action_model.n_features_in_)

    @property
    def confidence_threshold(self) -> float:
        return float(self.runtime_config["confidence_threshold"])

    def summary(self) -> dict:
        return {
            "model_file": self.model_file,
            "action_set": self.action_set,
            "pose_refiner": self.pose_refiner,
            "classes": self.classes,
            "sequence_steps": self.sequence_steps,
            "feature_count": self.feature_count,
            "model_version": self.training_info.get("model_version", "day06_baseline"),
            "train_samples": self.training_info.get("train_samples", self.training_info.get("train_clips")),
            "features_per_frame": FEATURES_PER_FRAME,
            "confidence_threshold": self.confidence_threshold,
            "window_seconds": float(self.runtime_config["window_seconds"]),
            "tracker": self.tracking_config.get("tracker"),
            "pose_model": Path(self.pose_model_path).name,
            "device": self.device,
            "warnings": self.warnings,
        }


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ArtifactError(f"파일이 없습니다: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"JSON 형식 오류: {path} ({exc})") from exc


def _find_project_config(settings: Settings) -> Path:
    candidates = [
        settings.config_dir / "project.json",
        settings.config_dir.parent / "project.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ArtifactError(
        "project.json을 찾지 못했습니다. 기존 프로젝트의 configs/project.json을 "
        f"{settings.config_dir} 에 복사하세요."
    )


def resolve_device(value: str) -> str:
    """'auto'면 GPU 사용 가능 여부로 결정. torch가 없으면 cpu."""
    value = (value or "cpu").strip()
    if value != "auto":
        return value
    try:
        import torch  # noqa: WPS433 (지연 import: 테스트 환경에는 torch가 없을 수 있음)

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _resolve_pose_model_path(raw: str, settings: Settings) -> str:
    """project.json의 pose_model 값을 컨테이너 경로로 바꾼다.

    1) models/ 폴더에 같은 파일명이 있으면 그것을 사용
    2) 절대경로이고 존재하면 그대로
    3) 그 외에는 이름 그대로 넘긴다 (Ultralytics가 공식 가중치를 내려받을 수 있음)
    """
    if settings.pose_model_override:
        return settings.pose_model_override
    candidate = settings.model_dir / Path(raw).name
    if candidate.exists():
        return str(candidate)
    if Path(raw).is_absolute() and Path(raw).exists():
        return raw
    return raw


def load_bundle(settings: Settings) -> ModelBundle:
    model_dir = settings.model_dir
    # MODEL_FILE 은 하위 폴더를 포함할 수 있다 (예: team/rf_action_team.joblib).
    # feature_columns·training_info 는 모델 파일과 '같은 폴더'에서 읽어 모델끼리 섞이지 않게 한다.
    model_path = model_dir / settings.model_file
    meta_dir = model_path.parent
    if not model_path.exists():
        raise ArtifactError(
            f"학습 완료 모델이 없습니다: {model_path} "
            "(기존 artifacts/day06/rf_action_baseline.joblib 을 복사하세요)"
        )

    feature_columns = _read_json(meta_dir / settings.feature_columns_file)
    if isinstance(feature_columns, dict):  # {"feature_columns": [...]} 형태도 허용
        feature_columns = feature_columns.get("feature_columns") or feature_columns.get("columns")
    if not isinstance(feature_columns, list):
        raise ArtifactError("rf_feature_columns.json은 Feature 이름 목록이어야 합니다.")

    training_info = _read_json(meta_dir / settings.training_info_file)
    missing = [key for key in REQUIRED_TRAINING_KEYS if key not in training_info]
    if missing:
        raise ArtifactError(f"rf_training_info.json에 필수 항목이 없습니다: {missing}")

    runtime_config = _read_json(settings.config_dir / "day07_realtime.json")
    missing = [key for key in REQUIRED_RUNTIME_KEYS if key not in runtime_config]
    if missing:
        raise ArtifactError(f"day07_realtime.json에 필수 항목이 없습니다: {missing}")

    if settings.window_seconds_override:
        logger.info(
            "Window 길이 덮어쓰기: day07 %.2fs → 서비스 %.2fs",
            float(runtime_config["window_seconds"]), settings.window_seconds_override,
        )
        runtime_config = {**runtime_config, "window_seconds": float(settings.window_seconds_override)}
    if settings.confidence_threshold_override is not None:
        runtime_config = {**runtime_config, "confidence_threshold": float(settings.confidence_threshold_override)}

    tracking_config = _read_json(settings.config_dir / "day08_multi_person.json")
    tracking_config.setdefault("tracker", "bytetrack.yaml")
    tracking_config.setdefault("lost_timeout_seconds", 2.0)

    project_config = _read_json(_find_project_config(settings))
    for key in ("pose_model", "image_size", "person_confidence"):
        if key not in project_config:
            raise ArtifactError(f"project.json에 '{key}' 항목이 없습니다.")

    action_model = joblib.load(model_path)  # 본인이 학습한 파일만 로드
    if not hasattr(action_model, "predict_proba"):
        raise ArtifactError("로드한 모델에 predict_proba가 없습니다. Random Forest 분류기인지 확인하세요.")

    sequence_steps = int(training_info["sequence_steps"])
    expected = sequence_steps * FEATURES_PER_FRAME
    n_in = int(getattr(action_model, "n_features_in_", -1))

    # 순서·개수가 하나라도 어긋나면 RF는 오류 없이 '틀린 값'을 낸다 → 시작 시점에 차단
    if n_in != expected:
        raise ArtifactError(
            f"모델 입력 차원({n_in})과 sequence_steps×38({expected})이 다릅니다."
        )
    if len(feature_columns) != n_in:
        raise ArtifactError(
            f"Feature 목록 길이({len(feature_columns)})와 모델 입력 차원({n_in})이 다릅니다."
        )
    if int(training_info["feature_count"]) != n_in:
        raise ArtifactError(
            f"training_info.feature_count({training_info['feature_count']})와 모델 입력 차원({n_in})이 다릅니다."
        )
    if hasattr(action_model, "feature_names_in_"):
        names = [str(n) for n in action_model.feature_names_in_]
        if names != [str(c) for c in feature_columns]:
            raise ArtifactError("모델의 feature_names_in_ 순서가 rf_feature_columns.json과 다릅니다.")

    warnings: list[str] = []
    model_classes = {str(c) for c in action_model.classes_}
    info_classes = {str(c) for c in training_info["classes"]}
    if model_classes != info_classes:
        raise ArtifactError(
            f"model.classes_ {sorted(model_classes)} 와 training_info.classes {sorted(info_classes)} 가 다릅니다."
        )
    expected = ACTION_SETS.get(settings.action_set)
    if expected is None:
        warnings.append(f"알 수 없는 ACTION_SET='{settings.action_set}' (basic · team · team6)")
    elif model_classes != set(expected):
        warnings.append(
            f"모델 Class {sorted(model_classes)} 가 ACTION_SET='{settings.action_set}' 목록 "
            f"{sorted(expected)} 과 다릅니다. .env의 ACTION_SET을 확인하세요."
        )

    device = resolve_device(settings.device_override or str(project_config.get("device", "cpu")))
    pose_model_path = _resolve_pose_model_path(str(project_config["pose_model"]), settings)

    bundle = ModelBundle(
        action_model=action_model,
        feature_columns=[str(c) for c in feature_columns],
        training_info=training_info,
        runtime_config=runtime_config,
        tracking_config=tracking_config,
        project_config=project_config,
        pose_model_path=pose_model_path,
        device=device,
        model_file=settings.model_file,
        action_set=settings.action_set,
        pose_refiner=settings.pose_refiner,
        warnings=warnings,
    )
    for message in warnings:
        logger.warning(message)
    logger.info("ModelBundle loaded: %s", bundle.summary())
    return bundle
