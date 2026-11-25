"""[테스트 전용 대역] 실제 파일은 subject10_pose_action/src/feature_utils.py 를 복사해 사용한다.
호출 형태(normalize_by_shoulder_width → (normalized, center, scale), selected_joint_angles → dict)만 맞춘 최소 구현.
"""
import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16


def normalize_by_shoulder_width(keypoints):
    kp = np.asarray(keypoints, dtype=np.float32)
    width = float(np.linalg.norm(kp[L_SH] - kp[R_SH]))
    if width < 1e-6:
        raise ValueError("shoulder width too small")
    center = (kp[L_HIP] + kp[R_HIP]) / 2.0
    return (kp - center) / width, center, width


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1, 1))))


def selected_joint_angles(keypoints):
    kp = np.asarray(keypoints, dtype=np.float32)
    return {
        "left_elbow_angle": _angle(kp[L_SH], kp[L_EL], kp[L_WR]),
        "right_elbow_angle": _angle(kp[R_SH], kp[R_EL], kp[R_WR]),
        "left_knee_angle": _angle(kp[L_HIP], kp[L_KN], kp[L_AN]),
        "right_knee_angle": _angle(kp[R_HIP], kp[R_KN], kp[R_AN]),
    }
