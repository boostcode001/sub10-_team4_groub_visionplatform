# src/ — 기존 subject10_pose_action/src 복사본

`scripts/import_from_subject10.sh` 가 이 폴더를 기존 `src/` 전체로 채운다.
이 서비스가 직접 import 하는 함수 (8일차 49번 스크립트와 동일한 호출 형태):

- `src.multi_person_utils`: `MultiPersonStateManager`, `feature_from_keypoints`
- `src.realtime_utils`: `predict_from_window`, `validate_runtime_values` (내부에서 `TimedFeatureBuffer`, `PredictionStabilizer`)
- 위 모듈이 다시 import 하는 `feature_utils`, `pose_utils`, `common` 등도 함께 복사된다.
