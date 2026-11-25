"""[테스트 전용 대역] 실제 파일은 subject10_pose_action/src/realtime_utils.py 를 복사해 사용한다.
8일차 49번 스크립트가 호출하는 형태(TimedFeatureBuffer, PredictionStabilizer, predict_from_window,
validate_runtime_values)만 맞춘 최소 구현.
"""
from collections import Counter, deque

import numpy as np


class TimedFeatureBuffer:
    def __init__(self, window_seconds):
        self.window_seconds = float(window_seconds)
        self.items = deque()

    def append(self, timestamp, feature):
        self.items.append((float(timestamp), feature))
        while self.items and timestamp - self.items[0][0] > self.window_seconds:
            self.items.popleft()

    def duration(self):
        return 0.0 if len(self.items) < 2 else self.items[-1][0] - self.items[0][0]

    def valid_features(self):
        return [f for _, f in self.items if f is not None]

    def valid_ratio(self):
        return 0.0 if not self.items else len(self.valid_features()) / len(self.items)

    def is_ready(self, min_window_fill_ratio, min_valid_feature_ratio, min_valid_features):
        return (
            self.duration() >= self.window_seconds * min_window_fill_ratio
            and self.valid_ratio() >= min_valid_feature_ratio
            and len(self.valid_features()) >= min_valid_features
        )


class PredictionStabilizer:
    def __init__(self, buffer_size, min_votes):
        self.labels = deque(maxlen=int(buffer_size))
        self.min_votes = int(min_votes)

    def update(self, label):
        self.labels.append(label)
        top, votes = Counter(self.labels).most_common(1)[0]
        return top if votes >= self.min_votes else "UNKNOWN"


def predict_from_window(model, feature_buffer, sequence_steps, confidence_threshold):
    feats = np.asarray(feature_buffer.valid_features(), dtype=np.float32)
    idx = np.linspace(0, len(feats) - 1, sequence_steps).round().astype(int)
    x = feats[idx].reshape(1, -1)
    proba = model.predict_proba(x)[0]
    best = int(np.argmax(proba))
    raw = str(model.classes_[best])
    conf = float(proba[best])
    return (raw if conf >= confidence_threshold else "UNKNOWN"), raw, conf, proba


def validate_runtime_values(window_seconds, confidence_threshold, stabilization_buffer, stabilization_min_votes):
    if window_seconds <= 0 or not 0 <= confidence_threshold <= 1:
        raise ValueError("invalid runtime values")
    if stabilization_min_votes > stabilization_buffer:
        raise ValueError("min_votes > buffer")
