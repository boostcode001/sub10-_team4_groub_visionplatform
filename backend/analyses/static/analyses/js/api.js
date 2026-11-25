// =====================================================================
// api.js — 서버(Django · DRF)와 통신하는 함수만 모아 둔 파일
// 화면을 그리는 코드는 analyses.js에 있고, 여기 함수를 import 해서 씀
// =====================================================================

// [연결] config/urls.py · analyses/urls.py (A 담당)
//        tests.py가 /api/analyses/{id}/status/ 를 쓰고 있어서 같은 앞부분으로 맞춤
const API = "/api/analyses/";

// 1) 분석 작업 만들기
// [연결] analyses/views.py      → 작업 생성 (A 담당, 성공 시 202)
//        analyses/serializers.py → input_type · video_file 검사 (확장자·빈 파일·용량 초과면 400)
//        analyses/tasks.py       → on_commit 안에서 run_analysis.apply_async(args=[job.pk]) (B 담당)
// 응답 예 (202): { "id": 12, "task_id": "...", "status": "PENDING", "status_url": "...", "detail_url": "..." }
// fetch의 res.ok 는 200~299면 true 라서 202도 성공으로 처리됨
export async function createJob(formData) {
  const res = await fetch(API, { method: "POST", body: formData });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // serializers.py가 돌려준 오류 이유 예: { "video_file": ["..."] }
    throw new Error(Object.values(body).flat().join(" / ") || `요청 실패 (${res.status})`);
  }
  return body;
}

// 2) 작업 기록 목록 가져오기
// [연결] analyses/views.py → 목록 (list)
// settings.py에 PageNumberPagination이 켜져 있어서 응답이 { count, next, previous, results } 모양
// → results 배열만 꺼내서 돌려줌 (페이지 기능이 꺼져 있으면 배열 그대로)
export async function fetchJobs() {
  const res = await fetch(API);
  if (!res.ok) throw new Error("작업 기록을 불러오지 못했습니다");
  const body = await res.json();
  return Array.isArray(body) ? body : body.results;
}

// 3) 작업 하나의 상태만 확인 (필요할 때 사용)
// [연결] analyses/views.py → status 액션 (A 담당)
//        멈춘 작업은 job_state.py expire_if_stale() 가 FAILURE로 바꿔 줌
// 응답 예: { "id": 12, "status": "STARTED", "done": false, "error_code": "", "error_message": "", "result_message": "" }
export async function fetchStatus(jobId) {
  const res = await fetch(`${API}${jobId}/status/`);
  if (!res.ok) throw new Error("상태를 불러오지 못했습니다");
  return res.json();
}
