#!/usr/bin/env bash
# ============================================================
# FICL Chatbot - Linux/Mac 실행 스크립트
# ============================================================
set -e
cd "$(dirname "$0")"

# --- Python 확인 -------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3가 설치되어 있지 않습니다. Python 3.10 이상이 필요합니다."
    exit 1
fi

echo "[INFO] $(python3 --version)"

# --- 가상환경 ----------------------------------------------
if [ ! -d ".venv" ]; then
    echo "[INFO] 가상환경 생성 중..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "[INFO] 가상환경 활성화됨."

# --- 로컬/API 실행에 필요한 패키지 선택 ---------------------
shopt -s nullglob
GGUF_FILES=(models/*.gguf)
REQ_FILE=requirements.txt
if [ ${#GGUF_FILES[@]} -eq 0 ] && [ -f api_models.json ]; then
    REQ_FILE=requirements-api.txt
fi

# --- 패키지 설치 확인 --------------------------------------
CHECK_IMPORTS="import fastapi, httpx, llama_cpp, sentence_transformers"
if [ "$REQ_FILE" = "requirements-api.txt" ]; then
    CHECK_IMPORTS="import fastapi, httpx, sentence_transformers"
fi
if ! python -c "$CHECK_IMPORTS" 2>/dev/null; then
    echo "[INFO] 패키지 설치 중... (5~10분 소요)"
    pip install --upgrade pip
    pip install -r "$REQ_FILE"
fi

# --- 모델 파일 확인 (any *.gguf in models/) ----------------
if [ ${#GGUF_FILES[@]} -eq 0 ]; then
    if [ "$REQ_FILE" = "requirements-api.txt" ]; then
        echo "[INFO] 로컬 GGUF 없이 API 모델 설정으로 실행합니다."
    else
        echo ""
        echo "============================================================"
        echo "[경고] models/ 폴더에 GGUF 모델 파일이 없습니다."
        echo "       README.md '모델 다운로드' 섹션을 참고해 받아주세요."
        echo "============================================================"
        echo ""
    fi
else
    echo "[INFO] GGUF 모델 파일 ${#GGUF_FILES[@]}개 감지됨."
fi

# --- 서버 실행 ---------------------------------------------
echo ""
echo "============================================================"
echo "  FICL Chatbot"
echo "  URL: http://localhost:7861"
echo "  종료: Ctrl+C"
echo "============================================================"
echo ""

python chatbot.py
