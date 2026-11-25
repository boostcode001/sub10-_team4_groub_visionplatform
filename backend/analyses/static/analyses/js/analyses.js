// =====================================================================
// analyses.js — 화면 동작 (버튼 클릭, 작업 기록 표 그리기, 2초마다 새로고침)
// 서버 통신은 직접 하지 않고 api.js의 함수를 불러와서 씀
// =====================================================================

// [연결] 같은 폴더의 api.js 에서 함수 불러오기
import { createJob, fetchJobs } from "./api.js";

const form = document.getElementById("analysis-form");
const videoInput = document.getElementById("video-input");
const startBtn = document.getElementById("start-btn");
const message = document.getElementById("message");
const tbody = document.querySelector("#history-table tbody");

const POLL_MS = 2000;                            // 2초마다 새로고침
const RUNNING = ["PENDING", "STARTED"];          // [연결] models.py AnalysisJob.Status
let timer = null;

// ---------------------------------------------------------------------
// 1) AI 분석 시작 버튼
// ---------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();                            // 페이지 새로고침 막기

  if (videoInput.files.length === 0) {
    message.textContent = "영상 파일을 선택하세요.";
    return;
  }

  startBtn.disabled = true;
  message.textContent = "요청 중...";

  try {
    // [연결] api.js → createJob()  (서버가 PostgreSQL에 저장하고 id, PENDING 반환)
    const job = await createJob(new FormData(form));
    message.textContent = `작업 #${job.id} 등록됨 (${job.status})`;
    form.reset();
    await renderHistory();
    startPolling();
  } catch (err) {
    message.textContent = `요청 실패: ${err.message}`;
  } finally {
    startBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------
// 2) 작업 기록 표 그리기
// ---------------------------------------------------------------------
async function renderHistory() {
  // [연결] api.js → fetchJobs()
  const jobs = await fetchJobs();

  tbody.innerHTML = "";
  if (jobs.length === 0) {
    addRow(["-", "기록 없음", "-", "-", "-", "-"]);
    return jobs;
  }

  jobs.forEach((job) => {
    // [확인 필요] serializers.py (A 담당)에서 정할 필드 이름
    //   - predictions : 사람별 결과 목록 (PersonPrediction: action_label, confidence)
    //                   기초 화면이라 첫 번째 사람만 표시
    //   - total_time_sec : 처리 시간 (tasks.py가 저장. AI 추론만이면 inference_time_sec)
    const first = (job.predictions || [])[0];
    addRow([
      job.id,                                                      // 번호
      first ? first.action_label : "-",                            // 자세
      first ? Number(first.confidence).toFixed(2) : "-",           // 신뢰도
      fileType(job.video_file),                                    // 파일 유형
      job.status,                                                  // 결과
      job.total_time_sec != null ? `${Number(job.total_time_sec).toFixed(1)}초` : "-", // 처리 시간
    ]);
  });
  return jobs;
}

// 파일 유형: 서버에 따로 필드가 없어서 video_file 경로의 확장자로 표시
// 예: "uploads/2026/11/25/wave.mp4" → "mp4"
function fileType(path) {
  if (!path) return "-";
  const name = String(path).split("?")[0];
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "-" : name.slice(dot + 1).toLowerCase();
}

function addRow(values) {
  const tr = document.createElement("tr");
  values.forEach((v) => {
    const td = document.createElement("td");
    td.textContent = v;            // 글자로만 넣기 (HTML로 해석되지 않게)
    tr.appendChild(td);
  });
  tbody.appendChild(tr);
}

// ---------------------------------------------------------------------
// 3) 2초마다 새로고침 → PENDING·STARTED 작업이 없으면 멈춤
// ---------------------------------------------------------------------
function startPolling() {
  if (timer) return;
  timer = setInterval(async () => {
    try {
      const jobs = await renderHistory();
      if (!jobs.some((j) => RUNNING.includes(j.status))) stopPolling();
    } catch (err) {
      message.textContent = err.message;
      stopPolling();
    }
  }, POLL_MS);
}

function stopPolling() {
  clearInterval(timer);
  timer = null;
}

// ---------------------------------------------------------------------
// 처음 화면이 열릴 때 기록 불러오기 (진행 중 작업이 있으면 자동 새로고침 시작)
// ---------------------------------------------------------------------
renderHistory()
  .then((jobs) => {
    if (jobs.some((j) => RUNNING.includes(j.status))) startPolling();
  })
  .catch((err) => { message.textContent = err.message; });
