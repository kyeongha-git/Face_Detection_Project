# RESULTS: 프로젝트 분석 및 AWS 배포 계획

> 작성일: 2026-04-05

---

## 1. our_model 폴더 전체 분석

### 1-1. 프로젝트 개요

RetinaFace 기반의 경량 Face Detection 모델로, MobileNet 0.25 백본 위에 다음 커스텀 모듈들을 결합한 아키텍처입니다:

| 모듈 | 역할 | 위치 |
|------|------|------|
| **ECA-CBAM** | ECA 기반 Channel Attention + CBAM Spatial Attention 통합 | `src/layers/eca_cbam.py` |
| **WFPN** | Learnable Weighted FPN + Deformable Conv | `src/layers/wfpn.py` |
| **SCM** | Shuffle Context Module + Deformable Conv | `src/layers/scm.py` |
| **DeformableConv2d** | 오프셋 기반 변형 합성곱 | `src/layers/conv_block.py` |

**모델 파이프라인**: Backbone → ECA-CBAM → WFPN → ECA-CBAM → SCM → ECA-CBAM → SCM → ECA-CBAM → Detection Head

### 1-2. 디렉토리 구조

```
our_model/
├── Dockerfile                    # Multi-stage Docker 빌드 (평가용)
├── docker-entrypoint.sh          # 컨테이너 엔트리포인트 (evaluate/detect/shell)
├── .dockerignore                 # Docker 빌드 컨텍스트 제외 규칙
├── requirements.txt              # 로컬 개발용 의존성
├── requirements.docker.txt       # Docker 전용 의존성 (headless OpenCV 등)
├── README.md                     # 프로젝트 설명 및 Docker 사용법
├── TODO.md                       # Docker Hub 푸시 및 VM 테스트 체크리스트
├── TODO_AWS.md                   # AWS 기반 CI/CD 및 K8s 배포 전체 로드맵
│
├── src/                          # 소스 코드
│   ├── __init__.py
│   ├── backbone/                 # MobileNetV1 백본
│   ├── layers/                   # 커스텀 레이어 (ECA-CBAM, WFPN, SCM, DeformableConv)
│   ├── models/                   # OurModel (메인 모델 아키텍처)
│   ├── loss/                     # MultiBoxLoss
│   ├── data/                     # WiderFace 데이터 로더, 데이터 증강
│   └── utils/                    # config.py, config_docker.py, prior_box, box_utils, NMS 등
│
├── tools/                        # 실행 스크립트
│   ├── train.py                  # 학습 스크립트
│   ├── test_widerface.py         # WiderFace 벤치마크 추론
│   ├── test_fddb.py              # FDDB 벤치마크 추론
│   ├── detect.py                 # 단일 이미지 검출
│   └── calculate_flops.py        # FLOPs/파라미터 측정
│
├── widerface_evaluate/           # WiderFace AP 평가 도구 (Cython bbox 모듈 포함)
├── weights/                      # 학습된 가중치 (MobileNet ~2.5MB, ResNet50 ~107MB 각 에포크)
├── data/                         # 데이터셋 (git에서 제외)
├── eval/                         # 평가 결과 저장
├── logs/                         # 학습 로그
├── visualization/                # 시각화 도구
├── assets/                       # README 이미지 에셋
└── utils/                        # 최상위 유틸리티 (현재 __init__.py만 존재)
```

### 1-3. 모델 가중치 현황

| 파일 | 크기 | 용도 |
|------|------|------|
| `mobilenet0.25_eca_cbam_Final.pth` | **~2.5 MB** | ✅ 서빙 대상 (MobileNet) |
| `mobilenet0.25_eca_cbam_epoch_*.pth` | ~2.5 MB × 30+ | 에포크별 체크포인트 |
| `Resnet50_Final.pth` | ~107 MB | ResNet50 최종 모델 |
| `Resnet50_epoch_*.pth` | ~107 MB × 13 | ResNet50 체크포인트 |
| `mobilenetV1X0.25_pretrain.tar` | ~3.8 MB | 프리트레인 가중치 |

### 1-4. WiderFace 평가 결과

| Case | AP |
|---|---|
| Easy |0.9188 |
| Medium | 0.8974 |
| Hard | 0.7689 |

---

## 2. 현재 Docker 설정 평가

### 2-1. 현재 Dockerfile 분석

현재 `Dockerfile`은 **WiderFace 평가 전용 이미지**로 설계되어 있습니다.

#### ✅ 잘된 점

| 항목 | 세부 |
|------|------|
| **Multi-stage 빌드** | Cython 빌드 도구를 builder 스테이지에 분리하여 런타임 이미지 경량화 |
| **Docker-specific config** | `config_docker.py`로 `pretrain=False` 설정하여 불필요한 프리트레인 가중치 로드 방지 |
| **최적화된 .dockerignore** | 불필요한 가중치, 데이터, 로그 등 제외 |
| **환경변수 분리** | `PYTHONUNBUFFERED`, `PYTHONPATH` 설정 |
| **Entrypoint 패턴** | `evaluate`, `detect`, `shell` 모드를 깔끔하게 분리 |
| **Volume mount 설계** | 데이터셋을 이미지에 포함하지 않고 런타임에 마운트 |
| **requirements.docker.txt 분리** | `opencv-python-headless` 등 Docker 환경에 최적화된 의존성 관리 |

#### ⚠️ TODO_AWS.md와의 Gap (개선 필요 사항)

| 항목 | 현재 상태 | TODO_AWS.md 목표 | Gap |
|------|-----------|-----------------|-----|
| **모델 가중치** | 이미지에 `COPY`로 포함 (Line 59) | S3에서 런타임 다운로드 | 🔴 **구조 변경 필요** |
| **서빙 서버** | 없음 (배치 평가 전용) | FastAPI 서빙 (`/health`, `/detect`) | 🔴 **신규 개발 필요** |
| **의존성** | 평가 도구만 포함 | `fastapi`, `uvicorn`, `boto3` 추가 필요 | 🟡 **추가 필요** |
| **빌드 대상** | 평가 파이프라인만 | 서빙 + 평가 모두 지원 | 🟡 **확장 필요** |
| **레지스트리** | Docker Hub (`kyeonghah/our-model-eval`) | AWS ECR으로 이전 | 🟡 **이전 필요** |
| **CI/CD** | 수동 빌드/푸시 | GitHub Actions 자동화 | 🔴 **신규 구축 필요** |
| **테스트** | 없음 | pytest 기반 유닛/통합 테스트 | 🔴 **신규 구축 필요** |
| **K8s 배포** | 없음 | AWS EKS Deployment + HPA | 🔴 **신규 구축 필요** |

### 2-2. docker-entrypoint.sh 평가

현재 entrypoint는 **평가 파이프라인에 잘 최적화**되어 있습니다:
- `evaluate`: `test_widerface.py` → `evaluation.py` 2단계 자동 실행
- `detect`: 단일 이미지 검출
- `shell`: 디버깅용 인터랙티브 셸
- `USE_CPU` 환경변수로 CPU/GPU 전환

하지만 **TODO_AWS.md의 서빙 모드와는 완전히 다른 목적**입니다:
- 현재: 배치 평가 → 결과 출력 후 컨테이너 종료
- 목표: FastAPI 서버 상시 가동 (`/health`, `/detect` 엔드포인트)

### 2-3. 종합 평가

> **현재 Docker 설정은 "평가/벤치마크 재현" 목적으로는 완성도가 높습니다.**
> 그러나 **TODO_AWS.md가 목표로 하는 "프로덕션 서빙 환경"과는 아키텍처적으로 다른 목적의 이미지**입니다.
>
> Phase 1부터 진행할 때, 기존 평가용 Dockerfile을 유지하면서 **서빙용 Dockerfile을 별도로 구성**하거나,
> 혹은 **하나의 Dockerfile에서 multi-target 빌드**로 통합하는 방향을 결정해야 합니다.

---

## 3. 요약

### 현재 위치 (As-Is)
- ✅ Docker 이미지 빌드 완료 (평가용)
- ✅ Docker Hub에 push 완료 (`kyeonghah/our-model-eval:latest`)
- ✅ CPU/GPU 양쪽 모드 동작 확인
- ✅ WiderFace AP 벤치마크 재현 성공

### 목표 (To-Be, TODO_AWS.md 완료 시)
- FastAPI 기반 실시간 추론 API 서버
- 모델 가중치 S3 관리
- GitHub Actions CI/CD 파이프라인
- AWS EKS 기반 K8s 배포 (Auto-scaling)

### 현재 Gap
- Phase 0 (AWS 계정/인프라): **미착수** — 선행 필수
- Phase 1, 2: **완료**
- Phase 3~6: **미착수**

---

## 4. 현황 요약 (업데이트)

| Phase | 항목 | 상태 |
|-------|------|------|
| Phase 0 | AWS 계정/인프라 준비 | ⬜ 미착수 (선행 필수) |
| Phase 1 | 기반 환경 정비 | ✅ **완료** |
| Phase 2 | pytest 테스트 코드 작성 | ✅ **완료** (99 passed) |
| Phase 3 | FastAPI 서빙 서버 구현 | ⬜ 다음 단계 |
| Phase 4 | Docker 이미지 빌드 (서빙용) | ⬜ 미착수 |
| Phase 5 | GitHub Actions CI/CD | ✅ **초기 구축 완료** (자동 테스트 및 Linting) |
| Phase 6 | Kubernetes 배포 (AWS EKS) | ⬜ 미착수 |

### Phase 5 진행 내역
- ✅ `.github/workflows/pytest.yml` 생성 (자동 테스트 파이프라인)
- ✅ `black` 포맷팅 일괄 적용 (코딩 스타일 통일)
- ✅ `.flake8` 설정 파일 생성 (코드 품질 관리 및 예외 처리)
- ✅ CI 패스 확인 (로컬 드라이런 및 구문 검증 완료)
