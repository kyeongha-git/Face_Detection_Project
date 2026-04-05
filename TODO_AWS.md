# Face Detection Project — CI/CD 및 배포 TODO

> 본 문서는 프로젝트의 테스트, CI/CD, Docker 빌드, K8s 배포, FastAPI 서빙 통합까지의 전체 작업 목록입니다.
>
> **확정된 결정 사항**
> - 서빙 모델: **MobileNet 0.25** (`mobilenet0.25_Final.pth`) 우선 사용, ResNet50는 추후 고려
> - 모델 가중치 저장: **AWS S3**에 업로드 후 컨테이너 시작 시 다운로드 (Docker 이미지에 미포함)
> - AWS 인프라: **현재 AWS 계정 없음** → Phase 0에서 계정 생성 및 인프라 구축부터 진행

---

## Phase 0: AWS 계정 및 인프라 준비 ⚠️ (선행 필수)

> AWS에 가입되어 있지 않으므로, Phase 1~6 진행 전 아래 항목을 완료해야 합니다.

### 0-1. AWS 계정 생성
- [ ] AWS 루트 계정 생성 (https://aws.amazon.com)
- [ ] 루트 계정 MFA(다중 인증) 활성화
- [ ] 결제 알림(Billing Alert) 설정 (예: 월 $10 초과 시 이메일 알림)

### 0-2. IAM 사용자 및 권한 설정
- [ ] IAM 사용자 생성 (루트 계정 대신 일상 작업용)
- [ ] 필요 권한 부여: `AmazonECR_FullAccess`, `AmazonEKSFullAccess`, `AmazonS3FullAccess`
- [ ] IAM 사용자의 액세스 키(Access Key ID + Secret Access Key) 발급 및 안전한 곳에 보관
- [ ] AWS CLI 설치 및 `aws configure`로 자격증명 설정

### 0-3. AWS S3 버킷 생성 (모델 가중치 저장용)
- [ ] S3 버킷 생성 (예: `face-detection-model-weights`)
  - [ ] 리전 선택 (권장: `ap-northeast-2` — 서울)
  - [ ] 퍼블릭 액세스 차단 설정 유지 (기본값)
- [ ] `mobilenet0.25_Final.pth` 파일을 S3에 업로드
  - [ ] 업로드 경로 예: `s3://face-detection-model-weights/mobilenet0.25_Final.pth`
- [ ] 컨테이너가 S3에서 파일을 다운로드할 수 있는 IAM 역할/정책 설정

### 0-4. AWS ECR 리포지토리 생성 (Docker 이미지 저장용)
- [ ] ECR 리포지토리 생성 (예: `face-detection`)
  - [ ] 리전: `ap-northeast-2`
- [ ] ECR URI 메모 (예: `123456789.dkr.ecr.ap-northeast-2.amazonaws.com/face-detection`)

### 0-5. AWS EKS 클러스터 생성 (Kubernetes 배포용)
- [ ] `eksctl` CLI 도구 설치
- [ ] EKS 클러스터 생성 (예: `face-detection-cluster`)
  - [ ] 노드 그룹 설정: CPU 노드, t3.medium 이상 권장
  - [ ] 최소 2개 노드
- [ ] `kubectl` 설정: `aws eks update-kubeconfig --name face-detection-cluster`
- [ ] 클러스터 정상 동작 확인: `kubectl get nodes`

---

## Phase 1: 기반 환경 정비

- [ ] `requirements.txt` 생성 (torch, torchvision, numpy, opencv, thop, fastapi, uvicorn, boto3 등)
  - `boto3` 포함: 컨테이너 시작 시 S3에서 가중치 다운로드에 필요
- [ ] `requirements-dev.txt` 생성 (pytest, pytest-cov, flake8, black)
- [ ] `pytest.ini` 설정 (testpaths, markers, addopts)
- [ ] `.gitignore` 업데이트 (테스트 관련 캐시, Docker 관련 파일 등)

---

## Phase 2: pytest 테스트 코드 작성

### 공통
- [ ] `tests/conftest.py` 작성 (공통 fixture: `cfg_mnet`, 더미 입력 텐서 등)

### 레이어 단위 테스트
- [ ] `tests/test_layers/test_eca_cbam.py` — ECA_CBAM 입출력 shape, gradient 흐름
- [ ] `tests/test_layers/test_wfpn.py` — WFPN 멀티스케일 입출력 shape
- [ ] `tests/test_layers/test_scm.py` — SCM 입출력 shape
- [ ] `tests/test_layers/test_conv_block.py` — DeformableConv2d forward pass

### 모델 테스트
- [ ] `tests/test_models/test_our_model.py` — OurModel forward (train/test 모드), 출력 shape
- [ ] `tests/test_models/test_heads.py` — ClassHead, BboxHead, LandmarkHead 출력 shape

### 유틸리티 테스트
- [ ] `tests/test_utils/test_prior_box.py` — PriorBox 앵커 개수, 값 범위
- [ ] `tests/test_utils/test_box_utils.py` — encode/decode 역변환 일관성, point_form
- [ ] `tests/test_utils/test_nms.py` — py_cpu_nms 결과 검증

### Loss 테스트
- [ ] `tests/test_loss/test_multibox_loss.py` — MultiBoxLoss 더미 데이터로 정상 동작

### 테스트 실행 검증
- [ ] 전체 unit 테스트 통과 확인 (`pytest -m unit -v`)
- [ ] 코드 커버리지 확인 (`pytest --cov=src`)

---

## Phase 3: FastAPI 서빙 서버 구현

- [ ] `app/__init__.py` 생성
- [ ] `app/config.py` — 서버 설정 (모델 경로, S3 버킷명, 임계값, 디바이스 등)
- [ ] `app/schemas.py` — Pydantic 요청/응답 스키마 정의
- [ ] `app/model_loader.py` — **S3에서 가중치 다운로드** 후 모델 로드 로직
  - 환경 변수로 S3 버킷명 및 키 경로를 주입받는 방식
  - 이미 로컬에 가중치가 있으면 다운로드 생략 (캐싱)
- [ ] `app/inference.py` — 전처리 → MobileNet 모델 추론 → 후처리 파이프라인
- [ ] `app/main.py` — FastAPI 앱 (엔드포인트: `/health`, `/detect`, `/detect/base64`)

### API 테스트
- [ ] `tests/test_api/test_fastapi_app.py` — TestClient 기반 엔드포인트 테스트
  - 테스트 시에는 S3 다운로드 대신 mock 사용

### 통합 테스트
- [ ] `tests/test_integration/test_inference_pipeline.py` — 이미지 → 추론 → 결과 전체 파이프라인 검증

---

## Phase 4: Docker 이미지 빌드

- [ ] `Dockerfile` 작성 (Multi-stage 빌드, Python 3.10 기반)
  - 모델 가중치는 이미지에 포함하지 않음 (S3에서 런타임 다운로드)
  - 컨테이너 시작 시 S3 다운로드 스크립트 실행
- [ ] `.dockerignore` 작성 (weights/, data/, experiments_weights/ 등 제외)
- [ ] 로컬 Docker 빌드 테스트 (`docker build -t face-detection:test .`)
- [ ] 로컬 Docker 실행 테스트 (AWS 자격증명 환경 변수 주입 필요)
  ```bash
  docker run -p 8000:8000 \
    -e AWS_ACCESS_KEY_ID=... \
    -e AWS_SECRET_ACCESS_KEY=... \
    -e AWS_REGION=ap-northeast-2 \
    -e S3_BUCKET=face-detection-model-weights \
    -e MODEL_KEY=mobilenet0.25_Final.pth \
    face-detection:test
  ```
- [ ] 컨테이너 내부 `/health` 엔드포인트 응답 확인

---

## Phase 5: GitHub Actions CI/CD

- [ ] `.github/workflows/ci-cd.yml` 작성
  - [ ] `test` 잡: Python 설치 → 의존성 설치 → pytest 실행
  - [ ] `build-and-push` 잡: Docker 빌드 → AWS ECR 로그인 → 이미지 푸시
    - `main` 브랜치 push 시에만 실행 (PR은 테스트만)
- [ ] GitHub Repository Secrets 등록 (Phase 0 완료 후 진행 가능)
  - [ ] `AWS_ACCESS_KEY_ID` — IAM 사용자 액세스 키
  - [ ] `AWS_SECRET_ACCESS_KEY` — IAM 사용자 시크릿 키
  - [ ] `AWS_REGION` — 예: `ap-northeast-2`
  - [ ] `ECR_REGISTRY` — ECR 레지스트리 주소 (Phase 0-4에서 메모한 URI)
  - [ ] `ECR_REPOSITORY` — ECR 리포지토리 이름 (예: `face-detection`)

---

## Phase 6: Kubernetes 배포 (AWS EKS)

> Phase 0-5(EKS 클러스터 생성) 완료 후 진행 가능

- [ ] `k8s/namespace.yaml` — 네임스페이스 정의 (예: `face-detection`)
- [ ] `k8s/configmap.yaml` — 환경 설정 (S3 버킷명, 모델 키, 임계값 등)
- [ ] `k8s/secret.yaml` — AWS 자격증명 (Secret 리소스 또는 IAM Role for Service Account 방식)
- [ ] `k8s/deployment.yaml` — Deployment 정의
  - [ ] `livenessProbe` 설정 (`/health`) — Self-healing
  - [ ] `readinessProbe` 설정 (`/health`)
  - [ ] 리소스 requests/limits 설정 (MobileNet 기준: CPU 500m, Memory 1Gi)
  - [ ] S3 관련 환경 변수 주입 (ConfigMap/Secret 참조)
- [ ] `k8s/service.yaml` — Service (LoadBalancer 타입으로 외부 접근)
- [ ] `k8s/hpa.yaml` — HorizontalPodAutoscaler (Auto-scaling)
  - CPU 사용률 70% 기준, min: 2, max: 10
- [ ] EKS에 매니페스트 배포: `kubectl apply -f k8s/`
- [ ] 배포 상태 확인: `kubectl get pods -n face-detection`
- [ ] HPA 동작 확인: `kubectl get hpa -n face-detection`
- [ ] 외부 접근 URL 확인: `kubectl get svc -n face-detection`

---

## 미결정 사항 (추후 논의)

- [ ] **GPU 추론**: K8s GPU 노드를 사용할 것인지? (현재는 CPU-only로 진행)
- [ ] **도메인/HTTPS**: Ingress + TLS 인증서 필요 여부?
- [ ] **모니터링 스택**: Prometheus/Grafana 포함 여부?
- [ ] **ResNet50 서빙 추가**: MobileNet 안정화 후 ResNet50 모델도 S3에 업로드하여 추가 서빙

---

## 참고: 전체 파일 생성 목록

```
[신규 생성]
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── Dockerfile
├── .dockerignore
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── model_loader.py      ← S3 다운로드 로직 포함
│   ├── inference.py
│   └── schemas.py
├── tests/
│   ├── conftest.py
│   ├── test_layers/
│   │   ├── test_eca_cbam.py
│   │   ├── test_wfpn.py
│   │   ├── test_scm.py
│   │   └── test_conv_block.py
│   ├── test_models/
│   │   ├── test_our_model.py
│   │   └── test_heads.py
│   ├── test_utils/
│   │   ├── test_prior_box.py
│   │   ├── test_box_utils.py
│   │   └── test_nms.py
│   ├── test_loss/
│   │   └── test_multibox_loss.py
│   ├── test_api/
│   │   └── test_fastapi_app.py
│   └── test_integration/
│       └── test_inference_pipeline.py
├── .github/workflows/
│   └── ci-cd.yml
└── k8s/
    ├── namespace.yaml
    ├── configmap.yaml
    ├── secret.yaml           ← AWS 자격증명 (신규)
    ├── deployment.yaml
    ├── service.yaml
    ├── hpa.yaml
    └── ingress.yaml          ← 선택사항
```
