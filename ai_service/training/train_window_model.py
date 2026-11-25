"""서비스용 동작 인식 모델(v2) 학습 스크립트.

6일차 35번 스크립트와 같은 Feature 규격(30 Step × 38 = 1,140, 같은 이름·순서)을 쓰되,
학습 샘플을 만드는 방법만 바꾼다.

| 구분 | 6일차 baseline | v2 (이 스크립트) |
|---|---|---|
| 사용 데이터 | S01 클립 6개 | S01~S03 클립 18개 (사람 검수 PASS) |
| 샘플 만드는 법 | 클립 전체 → 30 Step 1개 | 클립 안에서 2·3·4·5초 Window를 0.25초씩 이동하며 여러 개 |
| 샘플 수 | 6 | 655 |
| 이유 | — | 서비스는 영상 전체가 아니라 '몇 초 Window'를 30 Step으로 만들어 예측한다. 학습도 같은 단위로 맞춘다. |

실행 (WSL, subject10 가상환경):
    cd ~/ai_vision/vision_action_platform/ai_service
    ~/ai_vision/subject10_pose_action/.venv/bin/python training/train_window_model.py \
        --project ~/ai_vision/subject10_pose_action --out training/model_v2

출력: rf_action_baseline.joblib, rf_feature_columns.json, rf_training_info.json, cv_report.json
(파일 이름은 서비스가 읽는 이름 그대로 → models/ 에 복사만 하면 교체 완료)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TRAIN_WINDOWS = (2.0, 3.0, 4.0, 5.0)  # 서비스 Window(2.5초)·7일차 실시간 Window(5초)를 모두 포함하도록 섞음
TRAIN_STRIDE = 0.25
EVAL_WINDOW = 2.5  # 서비스 기본 WINDOW_SECONDS
SEQUENCE_STEPS = 30
MIN_FRAMES_PER_WINDOW = 10
RF_PARAMS = dict(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)  # 6일차와 같은 설정


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="subject10_pose_action 경로 (src/, data/day05/ 사용)")
    p.add_argument("--out", default="training/model_v2")
    p.add_argument("--skip-cv", action="store_true", help="피험자 교차검증 생략")
    return p.parse_args()


def load_clips(project: Path) -> dict:
    """사람 검수를 통과한 Train/Val/Test 분할 CSV를 모두 읽어 클립별 Frame Feature를 만든다."""
    from src.sequence_utils import build_frame_feature_matrix

    frames = [pd.read_csv(project / "data/day05/splits" / f"{name}.csv") for name in ("train", "val", "test")]
    manifest = pd.concat(frames, ignore_index=True)
    clips = {}
    for _, row in manifest.iterrows():
        df = pd.read_csv(project / row["keypoint_csv"])
        matrix, frame_ids = build_frame_feature_matrix(df)  # 6일차와 같은 38-D 계산
        times = df.set_index("frame_id").loc[frame_ids, "time_sec"].to_numpy(dtype=float)
        clips[row["clip_id"]] = {
            "subject": row["subject"],
            "label": row["action_label"],
            "matrix": np.asarray(matrix),
            "frame_ids": np.asarray(frame_ids),
            "times": times,
        }
        print(f"[LOAD] {row['clip_id']} {row['action_label']:<16} valid={len(frame_ids):4d} "
              f"duration={times[-1] - times[0]:.2f}s")
    return clips


def window_samples(clip: dict, window: float, stride: float) -> list[np.ndarray]:
    """클립 안에서 window 초 구간을 stride 초씩 옮기며 30 Step Sequence를 만든다.
    클립이 window보다 짧으면 클립 전체로 1개를 만든다 (6일차 방식과 동일)."""
    from src.sequence_utils import flatten_sequence, resample_sequence

    times = clip["times"]
    if times[-1] - times[0] <= window:
        seq = resample_sequence(clip["matrix"], clip["frame_ids"], target_steps=SEQUENCE_STEPS)
        return [flatten_sequence(seq)]
    samples, start = [], times[0]
    while start + window <= times[-1] + 1e-6:
        mask = (times >= start) & (times <= start + window)
        if mask.sum() >= MIN_FRAMES_PER_WINDOW:
            seq = resample_sequence(clip["matrix"][mask], clip["frame_ids"][mask], target_steps=SEQUENCE_STEPS)
            samples.append(flatten_sequence(seq))
        start += stride
    return samples


def training_samples(clip: dict) -> list[np.ndarray]:
    out = []
    for window in TRAIN_WINDOWS:
        out += window_samples(clip, window, TRAIN_STRIDE)
    return out


def build_xy(clips: dict, subjects: set[str]):
    X, y = [], []
    for clip in clips.values():
        if clip["subject"] in subjects:
            samples = training_samples(clip)
            X += samples
            y += [clip["label"]] * len(samples)
    return np.asarray(X, dtype=np.float32), y


def leave_one_subject_out(clips: dict) -> dict:
    """한 명을 빼고 학습 → 그 사람으로 시험. 처음 보는 사람에게 얼마나 통하는지 확인."""
    subjects = sorted({c["subject"] for c in clips.values()})
    report = {"window_seconds": EVAL_WINDOW, "folds": [], "misclassified": []}
    all_hits, all_conf, clip_hits = [], [], []
    for hold in subjects:
        X, y = build_xy(clips, set(subjects) - {hold})
        model = RandomForestClassifier(**RF_PARAMS).fit(X, y)
        fold_hits = []
        for clip_id, clip in clips.items():
            if clip["subject"] != hold:
                continue
            proba = model.predict_proba(np.asarray(window_samples(clip, EVAL_WINDOW, 0.5)))
            pred = model.classes_[proba.argmax(1)]
            all_hits += list(pred == clip["label"])
            all_conf += list(proba.max(1))
            labels, counts = np.unique(pred, return_counts=True)
            top = str(labels[counts.argmax()])
            fold_hits.append(top == clip["label"])
            if top != clip["label"]:
                report["misclassified"].append({"clip_id": clip_id, "true": clip["label"], "pred": top})
        clip_hits += fold_hits
        report["folds"].append({"test_subject": hold, "clip_accuracy": round(float(np.mean(fold_hits)), 4)})
        print(f"[CV] test={hold} clip_accuracy={np.mean(fold_hits):.3f}")
    conf = np.asarray(all_conf)
    report.update(
        clip_accuracy=round(float(np.mean(clip_hits)), 4),
        window_accuracy=round(float(np.mean(all_hits)), 4),
        mean_confidence=round(float(conf.mean()), 4),
        ratio_confidence_ge_055=round(float((conf >= 0.55).mean()), 4),
    )
    print(f"[CV] clip={report['clip_accuracy']:.3f} window={report['window_accuracy']:.3f} "
          f"conf={report['mean_confidence']:.2f} conf>=0.55={report['ratio_confidence_ge_055']:.2f}")
    return report


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    sys.path.insert(0, str(project))  # 기존 src/ 를 그대로 사용 (Feature 규격 보장)
    from src.sequence_utils import sequence_feature_names

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = load_clips(project)

    cv = None if args.skip_cv else leave_one_subject_out(clips)

    subjects = sorted({c["subject"] for c in clips.values()})
    X, y = build_xy(clips, set(subjects))
    model = RandomForestClassifier(**RF_PARAMS).fit(X, y)
    names = list(sequence_feature_names(SEQUENCE_STEPS))
    if len(names) != model.n_features_in_:
        raise RuntimeError("Feature 이름 수와 모델 입력 차원이 다릅니다.")

    joblib.dump(model, out / "rf_action_baseline.joblib")
    (out / "rf_feature_columns.json").write_text(json.dumps(names, ensure_ascii=False))
    info = {
        "model_version": "v2_window_augmented",
        "classes": [str(c) for c in model.classes_],
        "feature_count": int(model.n_features_in_),
        "sequence_steps": SEQUENCE_STEPS,
        "min_valid_frames": MIN_FRAMES_PER_WINDOW,
        **{k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "subjects": subjects,
        "train_clips": len(clips),
        "train_samples": int(len(X)),
        "samples_per_class": dict(Counter(y)),
        "window_seconds_used": list(TRAIN_WINDOWS),
        "window_stride_seconds": TRAIN_STRIDE,
        "cv_leave_one_subject_out": cv,
    }
    (out / "rf_training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    if cv:
        (out / "cv_report.json").write_text(json.dumps(cv, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out} samples={len(X)} classes={info['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
