# 10일차 개별 시연·발표 가이드

발표는 **README를 열고 위에서부터** 진행한다 (아키텍처 → ERD → 버튼 흐름 → 시연). 슬라이드는 필요 없다.
과제 문서가 요구하는 발표 항목 6개와 평가 기준(100점)에 맞춰 대본을 짰다.

| 평가 항목 | 배점 | 이 프로젝트에서 보여 줄 근거 |
|---|---:|---|
| 핵심 기능 실행·6개 Action 모델 재사용 | 30 | `check_artifacts.py` 출력, 두 사람 영상 시연 |
| Django·DRF·ORM·권한·ERD | 20 | README ERD, 다른 계정으로 404 시연, PATCH 제한 |
| Docker·Redis·Celery·서비스 분리 | 20 | `docker compose ps` 5개, 202 즉시 응답, Celery 로그 |
| 입력 UI·결과 표시·관리자 대시보드 | 10 | 탭 2개, 상태 단계, `/dashboard/` 차트 |
| 발표·코드 이해도·문제 해결 | 20 | 49번 코드와의 차이 4가지, DB 재시작 오류 해결 |

---

## 시연 전 준비 (10분)

```bash
grep ACTION_SET .env                                # basic 이어야 함 (아니면 bash scripts/switch_model.sh basic)
docker compose up -d && docker compose ps          # 5개 모두 Up, fastapi healthy
docker compose exec fastapi python scripts/check_artifacts.py --with-pose   # T01 화면을 미리 띄워 둠 (basic 모드)
docker compose logs -f celery                       # 별도 터미널: 발표 중 Worker 로그 보여 주기
```

- 브라우저 탭 3개: 사용자 화면(일반 계정 로그인), `/dashboard/`(관리자), README
- 시연 영상: 강사 제공 과제 영상 `S01_A00x_T01.mp4`(2~6초, 짧아서 10초 안팎에 끝남), 손상 파일(`broken.mp4`: 아무 텍스트 파일의 확장자만 변경)
- IP Webcam 시연 동작(basic): 손 흔들기(wave)나 앉았다 일어나기(sit_stand). 전신이 정면으로 나오게, 8초 내내 반복. 리허설로 정답이 나오는지 먼저 확인
- 팀 모델을 보여 줄 때: `bash scripts/switch_model.sh team`(약 1분) → 팀 영상 A001·A002·A006 업로드 → 끝나면 `basic`으로 복귀
- 휴대폰 IP Webcam 앱 실행, PC와 같은 Wi-Fi, `.env`의 `IP_WEBCAMS` 주소 확인
- 캡처에 카메라 IP·`.env`가 보이지 않는지 확인

---

## 발표 대본 (약 8분)

### 1. 기존 실습에서 무엇을 가져왔나요? (1분)
- 가져온 것: 6일차 `rf_action_baseline.joblib`·`rf_feature_columns.json`·`rf_training_info.json`의 **형식**, `src/` 전체(3일차 38-D Feature, 6일차 Sequence, 7일차 `realtime_utils`, 8일차 Tracking), 7·8일차 설정 JSON.
- 솔직하게: 6일차 모델을 그대로 연결하면(`switch_model.sh day06`) 샘플이 6개뿐이라 **모든 결과가 판단 불확실**(신뢰도 0.21~0.32). 그래서 Feature 규격·서비스 코드는 그대로 두고 학습 샘플만 바꾼 v2, 팀 재활 동작 모델(team)을 같은 규격으로 만들어 **파일 교체만으로** 바꿔 끼웠다.
- 핵심 문장: "joblib 모델은 영상이 아니라 **1,140개 숫자**를 입력으로 받습니다. 그래서 영상 → YOLO Pose → 38-D Feature → 30 Step Sequence로, 학습 때와 똑같은 입력을 만드는 과정을 FastAPI 안에 다시 구성했습니다."
- `check_artifacts.py` 출력 보여 주기: 6 Class, `n_features_in_ = 1140 = 30 × 38`
- 서버 시작 시 Feature 수·순서·Class가 하나라도 어긋나면 **모델을 '준비 안 됨'으로 두고 예측을 503으로 거절한다** (/health에 원인 표시): Random Forest는 순서가 틀려도 오류 없이 틀린 값을 내기 때문

### 2. 서비스를 어떻게 나눴나요? (1분 30초)
- README 3번 그림. "Django는 요청 접수와 저장, FastAPI는 추론만, Celery는 둘을 잇는 일꾼입니다."
- FastAPI는 DB를 모른다 → 추론 서버를 GPU 서버로 옮기거나 여러 대로 늘려도 Django 쪽 코드는 그대로
- 컨테이너끼리는 `localhost`가 아니라 서비스 이름(`http://fastapi:8000`)으로 통신
- FastAPI는 영상을 볼륨이 아니라 multipart로 받는다 → 파일 저장소에도 의존하지 않음

### 3. 어떤 데이터를 저장하나요? (1분)
- README 4번 ERD. "작업 1건에 사람별 결과 N건. 두 사람이 나오면 Job 1행 + Prediction 2행입니다."
- **관절 시퀀스는 저장하지 않고 결과만** 저장한다. 영상 파일도 DB가 아니라 Media 볼륨에 두고 경로만 저장한다.
- `track_id`는 그 영상 안에서의 번호일 뿐 사람의 고유 ID가 아니다.
- `expected_action`은 사용자가 '하려고 한' 동작일 뿐 검증된 정답이 아니다.

### 4. AI 분석 버튼을 누르면 무슨 일이 일어나나요? (2분, 시연 포함)
- README 5번 시퀀스 다이어그램을 따라 설명한 뒤 **영상 업로드 시연**
- 강조할 점 3가지
  1. `POST`가 **즉시 202 + 작업 ID**를 돌려준다 (브라우저 개발자도구 Network 탭에서 응답 시간 보여 주기)
  2. Celery 등록은 `transaction.on_commit` 안에서 한다. 커밋 전에 등록하면 Worker가 아직 저장되지 않은 Job을 조회할 수 있다.
  3. 결과 저장은 `transaction.atomic`으로 "예측 N건 + SUCCESS"를 한 번에 커밋한다. 중간에 실패해도 반쯤 저장된 상태가 남지 않는다.
- 두 사람이 잡힌 작업 #13(IP Webcam, 예측 2행) 상세 화면으로 T04 설명

### 5. 두 가지 영상 입력은 어떻게 다른가요? (1분 30초, 시연 포함)
- 업로드: 이미 촬영한 파일. IP Webcam: **Celery Worker가** 카메라에 연결해 **8초** 동안 들어오는 Frame을 MP4로 저장한 뒤, 업로드와 같은 경로로 FastAPI에 보낸다.
- 8초는 입력 영상의 길이이며, 연결 시간은 포함하지 않는다 (첫 Frame을 받은 뒤부터 측정).
- 저장 FPS: MJPEG 스트림은 FPS를 0으로 알려 주는 경우가 많아 **실제로 받은 Frame 수 ÷ 수집 시간**으로 계산해 저장한다. FastAPI가 `frame_idx / fps`로 시간을 계산하기 때문.
- 보안: 사용자는 URL을 입력하지 않고 **허용 목록의 카메라 key만** 고른다 (임의 주소 호출 = SSRF 방지).
- IP Webcam 시연 → 처리 중 단계에서 카메라 앞에서 동작
- 비교 시 주의: 두 입력은 거리·길이·조명이 달라서 Confidence 크기로 성능 우열을 단정하지 않는다.

### 6. 어떤 오류를 만났고 어떻게 해결했나요? (1분)
README 7번 "해결한 오류 사례"를 **증상 → 확인 → 수정 → 검증** 순서로 읽는다. 추천: libEGL.
- 증상: `/health`가 `model_ready: false`, `OSError: libEGL.so.1`
- 확인: `curl /health`의 error 필드 → MediaPipe가 쓰는 그래픽 라이브러리가 slim 이미지에 없음. Dockerfile을 고친 뒤에도 `up -d`만 해서 예전 이미지로 실행 중이었음
- 수정: Dockerfile apt에 `libegl1 libgles2` 추가 + 빌드 때 네이티브 라이브러리 로드 검사, `docker compose build fastapi`
- 검증: `/health` healthy, `check_artifacts.py --with-pose`에서 `[OK] MediaPipe heavy 로드`
- 교훈: Dockerfile·requirements를 바꾸면 `build`, `.env`만 바꾸면 `up -d`(재생성)

직접 겪은 오류가 더 있으면 (Tracking ID Switch, 버전 불일치 경고 등) 같은 형식으로 바꿔도 된다.

마지막으로 손상 파일을 업로드해 **실패 상태와 사유가 저장되는 것**(T09)과 `/dashboard/` 차트 2개(T08)를 보여 주고 마친다.

---

## 예상 질문과 답변

| 질문 | 답변 요지 |
|---|---|
| 왜 joblib 하나로는 MP4를 분류할 수 없나요? | joblib은 1,140차원 숫자 벡터를 받는 분류기일 뿐, 영상에서 사람·관절을 찾는 기능이 없다. Pose → Feature → Sequence를 학습 때와 똑같이 재현해야 한다. |
| 30 × 38 = 1,140은 무엇인가요? | 한 Frame을 38개 숫자(정규화 좌표 34 + 관절각 4)로 표현하고, 시간축으로 30 Step을 이어 붙인 길이. |
| Feature 순서를 바꾸면? | 오류 없이 **틀린 예측**이 나온다. 그래서 시작 시 `rf_feature_columns.json` 길이·순서와 `n_features_in_`을 검사한다. |
| `model.classes_`의 역할은? | `predict_proba` 출력의 각 열이 어떤 동작인지 알려 주는 순서표. 최댓값 위치를 Label로 바꿀 때 쓴다. |
| FastAPI가 반환하는 것과 Django가 저장하는 것을 왜 나누나요? | 추론 서버를 DB와 분리하면 따로 확장·교체할 수 있고, 저장 규칙(권한·트랜잭션·상태)은 ORM 한 곳에서 관리된다. |
| 요청 함수에서 분석을 기다리면 안 되나요? | 수십 초 동안 HTTP 연결과 웹 Worker를 붙잡아 타임아웃·동시 접속 문제가 생긴다. 작업 ID를 즉시 주고 상태를 조회하게 했다. |
| 사용자가 SUCCESS나 Label을 수정하면 안 되는 이유는? | AI 처리 결과의 무결성. 통계·대시보드가 조작된다. PATCH는 제목·메모만 허용하고 나머지는 400. |
| 대시보드는 ORM과 FastAPI 중 어디서 집계하나요? | Django ORM. FastAPI는 DB에 접근하지 않는 추론 전용 서비스다. |
| 사람이 없는 영상과 손상 영상은 어떻게 구분하나요? | 사람 없음은 200 + 빈 목록(SUCCESS + 메시지), 손상은 422(FAILURE + 사유). |
| Confidence가 낮으면? | Window마다 `confidence_threshold`(7일차 설정) 미만이면 UNKNOWN. 전부 미만이면 `판단 불확실`, 득표율이 50% 미만이어도 불확실 표시. |
| 두 사람이 교차해서 ID가 바뀌면? | 새 ID는 Buffer가 비어 있어 WARMING_UP부터 시작한다. Window 2개 미만으로 끝난 조각 Track은 결과에서 빼고 `dropped_tracks`에 기록한다. |
| 실시간(49번)과 무엇이 다른가요? | 시간축(영상 시간), 요청마다 Tracker reset, Track당 1행 집계, Lock 직렬화. |
| Redis에 결과를 저장하지 않는 이유는? | Redis는 큐·임시 상태용. 영구 보존할 업무 데이터는 PostgreSQL에 저장한다. |
| 예측이 틀리면? | 모델 교체부터 하지 않고 입력 품질 → Pose → Feature → Window → 모델 → 저장 순서로 원인을 찾는다. 학습 데이터가 3명 × 6동작뿐이라 새 사람·환경에서는 틀릴 수 있다. |
