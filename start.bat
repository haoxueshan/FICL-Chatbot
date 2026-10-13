@echo off
REM ============================================================
REM FICL Chatbot - Windows 실행 스크립트
REM 첫 실행: 가상환경 생성 + 패키지 설치 + 서버 시작
REM
REM 중요: 이 파일은 UTF-8로 저장되어 있어 cmd 기본 cp949와 충돌.
REM        chcp 65001로 cmd 코드페이지를 UTF-8로 전환해야
REM        한글 echo 문구와 IF 블록 안의 괄호 매칭이 정상 동작합니다.
REM ============================================================

chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- Python 확인 -------------------------------------------
if exist ".venv\Scripts\activate.bat" goto :activate_venv
where python >nul 2>&1
if errorlevel 1 goto :no_python

REM --- 가상환경 ----------------------------------------------
if not exist ".venv\Scripts\activate.bat" goto :create_venv
goto :activate_venv

:create_venv
echo [INFO] 가상환경 생성 중 — 처음 한 번만 실행됩니다
python -m venv .venv
if errorlevel 1 goto :venv_fail

:activate_venv
call ".venv\Scripts\activate.bat"
echo [INFO] 가상환경 활성화됨
for /f "tokens=*" %%i in ('python --version') do echo [INFO] %%i

REM --- 로컬/API 실행에 필요한 패키지 선택 --------------------
set HAS_MODEL=0
for %%f in (models\*.gguf) do set HAS_MODEL=1
set REQ_FILE=requirements.txt
if "%HAS_MODEL%"=="0" if exist "api_models.json" set REQ_FILE=requirements-api.txt

REM --- 패키지 설치 확인 --------------------------------------
if "%REQ_FILE%"=="requirements-api.txt" goto :check_api_deps
python -c "import fastapi, httpx, llama_cpp, sentence_transformers" >nul 2>&1
if errorlevel 1 goto :install_deps
goto :check_model

:check_api_deps
python -c "import fastapi, httpx, sentence_transformers" >nul 2>&1
if errorlevel 1 goto :install_deps
goto :check_model

:install_deps
echo [INFO] 패키지 설치 중 — 처음에는 5~10분 소요됩니다
python -m pip install --upgrade pip
python -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :pip_fail

:check_model
REM --- 모델 파일 확인 (any *.gguf in models/) ---------------
if "%HAS_MODEL%"=="0" goto :no_model
echo [INFO] GGUF 모델 파일 감지됨
goto :check_gpu

:no_model
echo.
if "%API_MODE%"=="1" goto :api_only
echo ============================================================
echo [경고] models 폴더에 GGUF 모델 파일이 없습니다
echo        서버는 켜지지만 채팅하면 오류가 납니다
echo        README.md의 모델 다운로드 섹션을 참고해 받아주세요
echo ============================================================
echo.
goto :check_gpu

:api_only
echo [INFO] 로컬 GGUF 없이 API 모델 설정으로 실행합니다
echo.

:check_gpu
REM --- GPU 안내 -----------------------------------------------
where nvidia-smi >nul 2>&1
if not errorlevel 1 echo [INFO] NVIDIA GPU 감지됨 — GPU 가속은 README의 GPU 가속 섹션 참고

REM --- 서버 실행 ---------------------------------------------
echo.
echo ============================================================
echo  FICL Chatbot
echo  URL: http://localhost:7861
echo  종료: Ctrl+C
echo ============================================================
echo.

python chatbot.py
pause
exit /b 0

:no_python
echo [ERROR] Python이 설치되어 있지 않습니다
echo         Python 3.10 이상을 https://www.python.org 에서 설치하고 PATH에 추가해주세요
pause
exit /b 1

:venv_fail
echo [ERROR] 가상환경 생성 실패
pause
exit /b 1

:pip_fail
echo [ERROR] 패키지 설치 실패 — README의 문제 해결 섹션을 참고하세요
pause
exit /b 1
