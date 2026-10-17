# Cinderella

Cinderella is a local multimodal runtime project with an opt-in camera/microphone AI mode. The live mode currently depends on separately maintained local/external backends; this repository does not contain self-trained speech, vision, or language models and does not establish human-like feelings or consciousness.

## Current implementation status

Implemented:
- typed runtime configuration loader with validation
- event schema and JSON serialization
- thread-safe bounded event bus
- lifecycle state management for long-running components
- deterministic simulation components for audio, vision, and hardware
- runtime orchestrator with cleanup on partial initialization failure
- optional laptop camera/microphone preview
- opt-in local Whisper transcription, MediaPipe object/pose/face-movement inference, local image-aware Ollama conversation, and offline Korean speech output, with external-backend provenance adapters; concrete model implementations are not yet fully separated from the live pipeline
- validated object observations that distinguish invalid, unsupported, low-confidence, and unstable results; normalized boxes are clamped only within a small frame-edge tolerance and stable labels require repeated detections. This improves result handling and consistency, not detector accuracy.
- JSON-safe typed observations plus short-lived world-state tracking with object regions, motion and position history; uncertain labels persist briefly across empty frames before expiring
- bounded microphone buffering with overflow, drop, queue-pressure, and segment diagnostics
- typed capability provenance/status records; the former 54% estimate is retained only as a developer helper and is no longer shown in the live overlay
- opt-in provider contracts, conservative policy validation, HTTPS allowlisting, redirect checks, response-size limits, bounded retries, and TTL caching
- offline-first provider scaffolding for MediaWiki, Open-Meteo, local status, approved static records, citations, and single-tool decision plans
- local SQLite persistence APIs for identity, relationship, and explicitly consented memories; not automatically connected to chat or initialized by the CLI
- typed identity-change review, evidence-bound relationship changes, approval-gated goals, and source-aware response plans
- face movement alone does not create a state hypothesis; the optional heuristic requires independent non-facial evidence and is not connected to the live dialogue path
- object-detection annotation validation and a basic class-aware IoU evaluation helper
- quiet-pause and stable visual-cue conversation prompts, with basic speech-interruption detection
- pytest coverage for the runtime foundation

Not implemented yet:
- self-developed or self-trained speech, vision, language, or speech-synthesis models; the live mode still relies on the external projects/models listed in the development guide
- a training pipeline, checkpoints, data collection, or full benchmark/evaluation suite
- robust general action recognition beyond simple pose cues
- emotion diagnosis (appearance and voice cues cannot establish a person's true emotional state)
- hardware device drivers
- voice cloning or character voice profiles
- automatic provider requests from chat or the CLI
- an operational Wikidata provider, face identity enrollment, automatic memory learning, autonomous goal execution, or integrated consent UI

New information providers remain offline and disabled by default. MediaWiki is blocked until query-data retention is verified. Open-Meteo is blocked by the required policy because its service offers paid/commercial and paid-data tiers; there is no override. The provider classes are not connected to the live conversation pipeline. See [docs/integrations.md](docs/integrations.md), [docs/provider-policy.md](docs/provider-policy.md), and [docs/offline-operation.md](docs/offline-operation.md).

## Runtime foundation

The simulated runtime is intentionally deterministic. The separate live preview and AI modes are opt-in and use local hardware only while running. World-state freshness is checked by the simulation safety gate; this is not a certified hardware safety controller. See [docs/model-migration-plan.md](docs/model-migration-plan.md) for the staged replacement plan, [docs/dependency-audit.md](docs/dependency-audit.md) for current backend provenance, and [docs/state-understanding.md](docs/state-understanding.md) for observation/belief boundaries.

To inspect the simulation-only physical-AI action/safety flow without opening hardware:

```powershell
.\.venv314\Scripts\python.exe main.py --physical-ai-demo
```

The demo emits JSON proposal/decision/simulated-command events; it does not actuate devices. Research boundaries and limitations are documented in [docs/physical-ai.md](docs/physical-ai.md).

The live-window 54% estimate was removed. It was a rough developer-status calculation, not an accuracy or human-similarity measure.

## 빠른 시작

```bash
python -m pytest
```

위 명령은 테스트를 실행합니다. 카메라나 AI를 시작하는 명령은 아닙니다.

## 노트북 카메라·마이크 및 실시간 AI 실행 (Windows)

### 1. 필요한 프로그램 설치

Python 3.14가 필요합니다. Ollama도 별도로 설치하고 실행해야 합니다. 실시간 AI 모드는 카메라·마이크 장치와 로컬 AI 모델을 사용합니다.

아래 Git Bash 명령은 저장소 폴더에서 실행합니다. 백슬래시(`\`)는 Bash에서 Windows 경로 구분자로 쓸 수 없으므로 `./`와 슬래시(`/`)를 사용합니다. 가상환경은 처음 한 번만 만들면 됩니다.

```bash
py -3.14 -m venv .venv314
./.venv314/Scripts/python.exe --version
./.venv314/Scripts/python.exe -m pip install -e '.[dev,hardware,ai]'
```

버전 확인 결과가 `Python 3.14.x`인지 확인하세요. 설치와 실행 명령에는 반드시 `.venv314` 안의 `python.exe`를 사용해야 합니다. 터미널에 이미 활성화된 `.venv`의 `python.exe`를 사용하면 Python 3.15일 수 있으며, 이 경우 `ctranslate2` 같은 AI 패키지를 설치할 수 없습니다.

PowerShell을 사용한다면 위 설치 명령은 다음과 같습니다.

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe --version
.\.venv314\Scripts\python.exe -m pip install -e ".[dev,hardware,ai]"
```

### 2. 카메라와 마이크 확인

연결된 장치를 확인합니다.

```bash
./.venv314/Scripts/python.exe main.py --devices
```

카메라 영상과 마이크 입력 레벨만 확인하려면 다음 명령을 실행합니다. 이 미리보기는 녹화하거나 AI 분석을 하지 않습니다.

```bash
./.venv314/Scripts/python.exe main.py --preview
```

창을 닫으려면 카메라 미리보기 창을 클릭한 뒤 `Q` 또는 `Esc`를 누릅니다. 노트북 내장 마이크 배열을 직접 지정하려면 `--microphone-device 32`를 추가합니다.

### 3. 로컬 AI 모델 준비

Ollama를 설치하고 실행한 다음, Git Bash 또는 PowerShell에서 대화·비전 모델을 내려받습니다. 모델 다운로드에는 인터넷이 필요합니다. Git Bash에서 `ollama: command not found`가 나오면 Ollama 실행 파일 폴더를 현재 터미널의 `PATH`에 추가하고 설치 여부를 확인합니다.

```bash
export PATH="$(cygpath -u "$LOCALAPPDATA")/Programs/Ollama:$PATH"
ollama --version
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

위 `export`는 현재 Git Bash 창에만 적용됩니다. Windows에 Ollama가 기본 경로가 아닌 곳에 설치되어 있으면 설치 폴더에 맞게 경로를 바꾸세요. PowerShell에서는 보통 `ollama --version`과 `ollama pull 모델이름`을 바로 실행할 수 있습니다.

### 4. 실시간 AI 모드 실행

다음 명령은 노트북 카메라와 기본 마이크를 사용해 영상 분석, 음성 인식, 대화를 시작합니다.

```bash
./.venv314/Scripts/python.exe main.py --live
```

노트북 내장 마이크 배열(장치 32번)을 사용하려면 다음 명령을 실행합니다.

```bash
./.venv314/Scripts/python.exe main.py --live --microphone-device 32
```

실행 중에는 터미널에 글을 입력하고 `Enter`를 눌러 대화할 수도 있습니다. 답변은 소리로도 출력됩니다. `/text-off`를 입력하면 터미널 입력만 중지되고 마이크 대화는 계속됩니다. 종료하려면 카메라 창에서 `Q` 또는 `Esc`를 누릅니다.

명령이 실패하면 먼저 Ollama가 실행 중인지, 모델 다운로드가 끝났는지, Windows 설정에서 카메라·마이크 사용 권한이 허용되어 있는지 확인하세요. 다른 마이크를 쓰려면 `--devices` 명령으로 표시된 번호를 `--microphone-device 번호`에 넣습니다.

첫 AI 실행 때 Whisper 음성 인식 모델과 MediaPipe 영상 분석 파일이 사용자 캐시에 다운로드됩니다. 다운로드 후 추론은 로컬에서 실행됩니다. 더 작고 빠르지만 인식 정확도가 낮은 음성 모델을 사용하려면 명령 끝에 `--speech-model base`를 추가합니다.

실시간 모드는 사용자가 직접 실행해야 합니다. 원본 오디오와 카메라 프레임은 파일로 저장하지 않습니다. 화면에 대해 음성이나 글로 질문하면 현재 프레임의 임시 축소 이미지가 내 컴퓨터의 Ollama에 전달됩니다. 대화에는 호스팅된 외부 AI API를 사용하지 않습니다. CPU만 있는 컴퓨터에서는 화면 질문 답변이 느릴 수 있습니다. Windows에 한국어 음성 출력이 설치되어 있어야 답변을 소리로 들을 수 있습니다.

얼굴 분석 결과는 화면에서 관찰된 움직임 단서일 뿐 감정·의도·신원·성격을 판정하지 않습니다. 사물 인식은 일반 사물 데이터셋에 기반하므로 사물을 놓치거나 잘못 표시할 수 있습니다. 헤드폰을 사용하면 스피커 소리가 마이크에 다시 들어가는 현상을 줄일 수 있습니다.

선택적 AI 패키지 호환성을 위해 Python 3.14 사용을 권장합니다. Python 3.15에서는 일부 패키지가 설치되지 않을 수 있습니다.
