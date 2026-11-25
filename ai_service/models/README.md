# models/ (Git 제외)

`bash scripts/import_from_subject10.sh ~/ai_vision/subject10_pose_action` 로 아래 파일을 복사한다.

| 파일 | 출처 | 역할 |
|---|---|---|
| `rf_action_baseline.joblib` | `artifacts/day06/` | 6개 Action Random Forest (재학습 없음) |
| `rf_feature_columns.json` | `artifacts/day06/` | 1,140개 Feature 이름·순서 |
| `rf_training_info.json` | `artifacts/day06/` | classes, feature_count, sequence_steps |
| `<pose 가중치>.pt` | `project.json`의 `pose_model` | YOLO Pose |

용량이 크고 재생성 가능한 산출물이므로 Git에는 올리지 않고, compose에서 `/app/models`로 읽기 전용 마운트한다.
