# 🤖 FICL Chatbot — RAG + LLM 챗봇 학습 예제

GGUF 양자화된 한국어 LLM을 로컬 PC에서 실행하거나 서버에 등록된 API 모델을 사용하고, 업로드한 문서를 참고해 답변하는 RAG(검색 증강 생성) 챗봇 예제입니다.

> **학습 목표**
> 1. GGUF 양자화 모델을 일반 PC에서 어떻게 돌리는가
> 2. RAG(Retrieval-Augmented Generation)가 무엇이며 어떻게 구현되는가
> 3. LLM 응답을 SSE로 어떻게 실시간 스트리밍하는가

---

## 📦 폴더 구성

```
FICL_Chatbot/
├── chatbot.py         # FastAPI 백엔드 (LLM 로딩 + RAG + 채팅 스트림)
├── ui.html            # 중국어·한국어 웹 UI (외부 CDN 없이 동작)
├── guide.html         # 다국어 사용 안내서 (/guide)
├── requirements.txt   # Python 패키지 목록
├── requirements-api.txt # API 전용 실행 시 패키지 목록
├── api_models.example.json # 여러 API 서비스 모델 등록 예시
├── api_models.json    # 서버 전용 통합 모델·키 설정 (Git 제외)
├── start.bat          # Windows 실행 스크립트 (자동 설치 + 실행)
├── start.sh           # Linux/Mac 실행 스크립트
├── models/            # 🚨 GGUF 모델을 직접 다운로드해서 넣는 폴더
├── rag_index/         # 업로드한 자료의 벡터 인덱스 (자동 생성)
├── uploads/           # 업로드한 원본 파일 (자동 생성, Git 제외)
└── README.md          # 이 파일
```

---

## 🚀 5분 만에 시작하기 (Windows)

### ① Python 설치 확인
- **필요 버전**: Python 3.10 이상
- 미설치 시: [python.org/downloads](https://www.python.org/downloads/) 에서 다운로드
- 설치 시 **"Add Python to PATH"** 체크 필수

### ② 모델 다운로드 (셋 중 하나 선택)

`models/` 폴더 안에 GGUF 파일을 받아 두세요. 파일명이 정확히 일치해야 합니다.

#### A) Qwen3-8B (다국어 균형형) — 권장
- **다운로드 URL**: https://huggingface.co/bartowski/Qwen3-8B-GGUF
- **받을 파일**: `Qwen3-8B-Q4_K_M.gguf`
- **용량**: 약 5.0GB

#### B) EEVE-Korean-Instruct-10.8B (한국어 특화)
- **다운로드 URL**: https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF
- **받을 파일**: `ggml-model-Q4_K_M.gguf` → 다운로드 후 **`EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf`로 이름 변경**
- **용량**: 약 6.8GB

#### C) EXAONE-3.5-7.8B (LG AI · 한국어 특화)
- **다운로드 URL**: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF
- **받을 파일**: 페이지에서 `EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf` 선택
- **용량**: 약 4.9GB

> 💡 **TIP** · 처음이라면 가장 가벼운 EXAONE 7.8B부터 시도하세요. PC 사양이 충분하면 EEVE 10.8B가 한국어 품질 최상입니다.

> ⚠ **HuggingFace 가입** · 일부 모델은 다운로드 시 무료 가입이 필요합니다.

### ③ 실행

```
start.bat 더블클릭
```

첫 실행 시 자동으로 진행되는 작업:
1. `.venv` 가상환경 생성
2. `requirements.txt`의 패키지 설치 (5~10분 소요)
3. 모델 파일 존재 확인
4. 서버 시작 → http://localhost:7861

### ④ 접속

브라우저에서 **http://localhost:7861** 열기

- 상단 드롭다운에서 다운로드한 모델 선택
- 📁 파일 업로드로 PDF·DOCX 등을 인덱싱 (선택)
- 메시지 입력 → Enter

---

## 🛠 수동 설치 (가상환경 + 라이브러리)

`start.bat` 없이 직접 설치하는 방법:

### Windows

```bat
REM 1) 가상환경 생성
python -m venv .venv

REM 2) 가상환경 활성화
.venv\Scripts\activate

REM 3) pip 업그레이드
python -m pip install --upgrade pip

REM 4) 패키지 설치
pip install -r requirements.txt

REM 5) 실행
python chatbot.py
```

### Linux / Mac

```bash
# 1) 가상환경 생성
python3 -m venv .venv

# 2) 가상환경 활성화
source .venv/bin/activate

# 3) pip 업그레이드
pip install --upgrade pip

# 4) 패키지 설치
pip install -r requirements.txt

# 5) 실행 권한 부여 후 실행
chmod +x start.sh
./start.sh
# 또는
python chatbot.py
```

---

## ⚡ GPU 가속 (선택)

CPU만으로도 동작하지만, NVIDIA GPU가 있으면 응답 속도가 5~10배 빨라집니다.

### Windows + NVIDIA GPU

기본 `llama-cpp-python`은 CPU 전용입니다. CUDA 빌드로 교체하세요.

```bat
REM 가상환경 활성화 후
.venv\Scripts\activate

REM CPU 버전 제거
pip uninstall llama-cpp-python -y

REM CUDA 12.1 미리 빌드된 휠 설치
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

CUDA 11.8을 사용한다면 URL 끝 `cu121` → `cu118`로 변경.

> **CUDA 확인 방법** · 명령 프롬프트에서 `nvidia-smi` 입력. 표시되는 CUDA Version 확인.

### GPU 레이어 조절

기본값은 모든 레이어를 GPU에 올립니다 (`N_GPU_LAYERS=-1`).
**GPU 메모리 부족 시** 일부만 올리세요:

```bat
REM CMD
set N_GPU_LAYERS=20
python chatbot.py

REM PowerShell
$env:N_GPU_LAYERS=20
python chatbot.py
```

| VRAM | 권장 N_GPU_LAYERS |
|---|---|
| 24GB (RTX 3090/4090) | -1 (전체) |
| 12GB (RTX 3060/4070) | 25~30 |
| 8GB | 15~20 |
| 4GB 또는 GPU 없음 | 0 (CPU만) |

---

## 💡 사용 방법

### 1. 모델 선택
화면 상단의 언어 선택에서 **中文 / 한국어**를 고르면 UI 문구와 이후 답변의 언어가 바뀝니다. 선택은 브라우저에 저장되며, 기존 대화 내용은 자동 번역되지 않습니다. 이어서 모델 드롭다운에서 서버가 제공하는 모델을 선택합니다. 로컬 GGUF와 서버에 등록한 API 모델이 한 목록에 표시됩니다.

- 로컬 GGUF: 다운로드된 모델만 선택 가능.
- API 모델: 서버에 등록되고 API Key가 설정된 모델만 선택 가능.

브라우저는 모델 ID만 서버에 보냅니다. API Key, Base URL, 실제 API 모델 이름은 서버 설정에서만 관리합니다.

#### API 모델을 서버에 등록하기

[api_models.example.json](api_models.example.json)을 `api_models.json`으로 복사한 뒤, 각 provider의 `base_url`, `models`, `api_key`를 설정합니다. 서버는 `api_key`를 파일에서 직접 읽으며 환경변수로 키를 조회하지 않습니다. 한 provider 아래에 여러 모델을 둘 수 있습니다.

```json
[{"id":"example","name":"Example API","base_url":"https://api.example.com/v1","api_key":"YOUR_KEY","models":["example-small","example-large"]}]
```

`api_models.json`은 Git에서 제외되며 브라우저 API에서 키 값을 반환하지 않습니다. 파일 접근 권한을 서버 관리자에게만 부여하고 키를 공유·커밋하지 마세요. `enable_thinking`은 이를 지원하는 API 모델(예: Qwen)에만 설정하세요. 설정 변경 후 `/status`의 모델 목록을 다시 확인하세요. API 모델의 `available`은 키가 설정되었다는 뜻이며, 실제 인증 성공 여부는 요청 전까지 확인되지 않습니다.

로컬/API 모델은 같은 목록에서 선택합니다. `/status`의 `default_model`은 사용 가능한 `DEFAULT_MODEL`을 우선하며, 그렇지 않으면 첫 사용 가능 모델입니다. 명시적으로 선택한 로컬 모델의 파일이 없으면 다른 모델로 몰래 바꾸지 않고 오류를 반환합니다. API 모델을 선택하면 **테스트 API 연결** 버튼으로 인증과 모델 ID를 직접 확인할 수 있습니다. 이 버튼은 짧은 실제 요청을 보내므로 소량의 API 요금이 발생할 수 있으며 자동으로 실행되지 않습니다.

GGUF 파일이 없고 API 모델이 설정되어 있으면 `start.bat` / `start.sh`는 로컬 추론 라이브러리를 제외한 `requirements-api.txt`를 설치합니다. 수동 설치도 `pip install -r requirements-api.txt`를 사용할 수 있습니다.

외부 API에는 HTTPS를 사용하며, HTTP는 localhost에서만 허용됩니다. API 모델을 선택한 경우 질문과 검색된 RAG 자료가 해당 API 서비스로 전송됩니다. `HOST=0.0.0.0` 등으로 LAN에 공개하려면 16자 이상의 `APP_ACCESS_TOKEN`을 반드시 설정해야 서버가 시작됩니다. 브라우저는 처음 접속할 때 토큰을 입력하며, 토큰은 해당 브라우저 세션에만 보관됩니다. HTTPS가 없는 LAN에서는 토큰이 평문 HTTP로 전송되므로 신뢰할 수 있는 네트워크에서만 사용하거나 HTTPS 역방향 프록시를 설치하세요.

### 2. 파일 업로드 (선택)
**📁 파일 업로드** → PDF·DOCX·XLSX·TXT·MD 선택.
- 자동으로 텍스트 추출 → 청크 분할 → 임베딩 → 인덱스 저장
- 다른 이름의 파일은 기존 인덱스에 추가되고, 같은 이름의 파일을 다시 업로드하면 해당 파일의 기존 내용만 교체
- 파일당 20MB, 한 번에 최대 10개·총 50MB
- 원본 파일은 `uploads/`에 저장되며 웹의 **업로드 파일 목록**에서 파일별로 삭제할 수 있음. 삭제 시 해당 파일의 검색 청크도 함께 제거
- 인덱스는 `rag_index/index.npz`에 영구 저장되어 재시작 후에도 유지 (기존 `docs.json` + `embeddings.npy` 형식은 읽기 지원)
- 과거 인덱스에만 있고 원본이 저장되지 않은 파일은 목록에 **원본 파일 미보관**으로 표시되며 검색 청크를 삭제할 수 있음

### 3. 질문하기
입력창에 질문 → Enter (Shift+Enter는 줄바꿈).
업로드한 자료를 자동 검색해 참고해서 답변하며, 답변 아래에 **📚 참고 자료: 파일명** 표시됩니다.

### 4. 파일 삭제 또는 전체 초기화
웹의 파일 목록에서 **삭제**를 누르면 해당 원본 파일과 인덱스 청크가 함께 삭제됩니다. **전체 자료 비우기**는 모든 원본 파일과 인덱스를 삭제하므로 확인 후 사용하세요.

---

## ⚙️ 환경변수 (선택)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DEFAULT_MODEL` | `qwen3-8b` | 기본 모델 키 (로컬 키 또는 `api:provider:model-id`; 사용 불가 시 첫 가용 모델) |
| `HOST` | `127.0.0.1` | 바인딩 주소 (LAN 공유 시 `0.0.0.0`) |
| `APP_ACCESS_TOKEN` | 빈 값 | LAN 공개 시 필수인 웹 접근 토큰. 서버 API Key와 별도 |
| `PORT` | `7861` | 포트 번호 |
| `N_GPU_LAYERS` | `-1` | GPU에 올릴 레이어 수 (`-1`: 전체, `0`: CPU만) |
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | RAG용 임베딩 모델 |

설정 예시:

```bat
REM Windows
set DEFAULT_MODEL=eeve-10.8b
set N_GPU_LAYERS=25
python chatbot.py
```

```bash
# Linux/Mac
DEFAULT_MODEL=exaone-7.8b N_GPU_LAYERS=0 python chatbot.py
```

---

## 🧠 코드 구조 이해하기 (학습용)

### 백엔드 ([chatbot.py](chatbot.py))

| 영역 | 함수 | 역할 |
|---|---|---|
| LLM 로딩 | `get_llm()` | GGUF 파일을 `Llama()`로 로딩. 락 보호 + fast-path |
| 텍스트 청킹 | `chunk_text()` | 긴 문서를 500자 단위로 자르기 (overlap 80자) |
| 텍스트 추출 | `extract_text()` | PDF/DOCX/XLSX/TXT/MD에서 본문 추출 |
| 인덱스 구축 | `build_index()` | 청크 → 임베딩 → 단일 NPZ 인덱스 및 원본 파일 저장 |
| 모델 선택 | `get_model_catalog()` / `select_model()` | 상태 조회·채팅 공통 모델 목록과 선택 검증 |
| 검색 | `search_rag_per_doc()` | 문서별 코사인 유사도 검색 |
| 프롬프트 | `build_messages()` | 시스템 + 참고 자료 + 대화 이력 + 현재 질문 |
| 스트리밍 | `/chat` 엔드포인트 | SSE로 토큰 단위 yield |

### 프론트엔드 ([ui.html](ui.html))

- 프레임워크·CDN 없는 단일 HTML/JavaScript 페이지 (설치 후 오프라인 UI 사용 가능)
- `fetch()` + `ReadableStream` 으로 SSE 직접 파싱
- 중국어·한국어 UI 및 답변 언어 선택 / 모델 선택 드롭다운 / 파일 업로드·목록·개별 삭제 / 전체 자료 비우기 / 채팅

### RAG 동작 흐름

```
[파일 업로드]
    ↓
1. extract_text(): 문서 → 평문 텍스트
    ↓
2. chunk_text(): 평문 → 500자 청크 리스트
    ↓
3. embedder.encode(): 청크 → 768차원 임베딩 벡터
    ↓
4. np.save(): 디스크에 영구 저장

[채팅]
    ↓
1. 사용자 메시지 → embedder.encode() → 질문 벡터
    ↓
2. (저장된 임베딩) @ (질문 벡터) → 코사인 유사도
    ↓
3. top-k 청크를 시스템 프롬프트에 추가
    ↓
4. LLM 호출 → 토큰 단위 스트리밍 응답
```

---

## ❓ 문제 해결

### "Python이 설치되어 있지 않습니다"
[python.org](https://www.python.org/downloads/) 에서 설치. **PATH 추가** 옵션 체크.

### `llama-cpp-python` 설치 실패 (Windows)
빌드 도구가 필요합니다. 둘 중 하나 시도:

```bat
REM 옵션 A: 미리 빌드된 휠로 설치
pip install llama-cpp-python --prefer-binary

REM 옵션 B: Visual Studio Build Tools 2022 설치 후 재시도
REM https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### "모델 파일을 찾을 수 없습니다" 오류
- `models/` 폴더에 GGUF 파일이 있는지 확인
- 파일명이 정확히 일치하는지 확인 (대소문자 포함)
- 파일 다운로드 중간에 끊겼다면 다시 받기

### `sentence-transformers` 첫 실행이 느림
- 약 500MB의 임베딩 모델이 한 번 자동 다운로드됩니다 (인터넷 필요)
- 한 번 받은 뒤로는 빠릅니다

### 응답이 너무 느림
1. GPU 가속 활성화 (위 'GPU 가속' 섹션)
2. 더 작은 모델 사용 (EXAONE 7.8B가 가장 가벼움)
3. CPU만 쓴다면 `n_threads`를 코어 수에 맞춰 조정 (`chatbot.py` 수정)

### 포트 7861 충돌
환경변수로 변경: `set PORT=8000`

---

## 📚 외부 모델 라이선스 안내

| 모델 | 라이선스 | 상업 사용 |
|---|---|---|
| Qwen3 | Apache 2.0 | 가능 |
| EEVE-Korean | Apache 2.0 (베이스 SOLAR는 별도) | 베이스 라이선스 확인 필요 |
| EXAONE | LG AI Research 라이선스 | 상업 사용 시 확인 필요 |

학술·교육 목적으로는 모두 자유롭게 사용 가능합니다.

---

## 🎯 다음 단계 (학습 확장)

이 예제를 이해했다면 다음을 시도해 보세요:

1. **다른 임베딩 모델 적용** — `intfloat/multilingual-e5-base` 대신 `BAAI/bge-m3` 등으로 교체
2. **재순위(reranking) 추가** — 1차 검색 결과를 cross-encoder로 재정렬
3. **하이브리드 검색** — BM25(키워드) + 임베딩(의미) 결합
4. **대화 메모리 요약** — 긴 대화를 LLM이 압축해 컨텍스트 유지
5. **함수 호출(Function Calling)** — 날씨 API 등 외부 도구 연결

---

문의/이슈가 있으면 콘솔 로그와 `/status` 응답을 함께 첨부해 주세요.
