"""사용자 화면(HTML)을 돌려주는 view — 프론트 담당.

API(JSON)는 views.py(A 담당), 화면(HTML)은 이 파일로 나눈다.
로그인 기능이 없으므로 @login_required 를 붙이지 않는다.
"""
from django.shortcuts import render


def index(request):
    """분석 화면: 영상 업로드 + 작업 기록 표.

    [연결] templates: analyses/templates/analyses/index.html
    [연결] 주소: config/urls.py 에 path("", index, name="index") 추가 필요 (A 담당 파일)
    """
    return render(request, "analyses/index.html")
