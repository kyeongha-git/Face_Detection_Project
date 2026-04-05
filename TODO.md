# TODO: 모델 평가 및 배포 잔여 작업

> **목표**: Docker 이미지의 로컬 평가를 완료하고, Docker Hub에 푸시하여 VM(AWS 등) 환경에서 원격 테스트를 수행한다.

---

## 📋 현재 진행 상황 및 다음 단계

### 1. 로컬 평가 완료 대기 (CPU 모드)
- [ ] **1.1 추론 로그 모니터링**
  - 현재 상태: `im_detect: 461/3226` 진행 중 (CPU 모드는 안정적이나 속도가 느림).
  - 예상 소요 시간: 약 25~30분 남음.
- [ ] **1.2 AP 결과 검증**
  - 추론이 끝나면 `evaluation.py`가 자동 실행됨.
  - 출력된 점수가 기대값과 일치하는지 확인:
    - `Easy Val AP: ~0.9039`
    - `Medium Val AP: ~0.8810`
    - `Hard Val AP: ~0.7808`

### 2. Docker Hub 푸시 (Push)
- [ ] **2.1 이미지 태깅**
  - 로컬 `our-model-eval:latest` 이미지를 레포지토리 이름(`kyeonghah/face_detection`)에 맞춰 변경.
- [ ] **2.2 이미지 푸시**
  - `docker push` 명령어로 Docker Hub에 업로드.

### 3. VM(원격 서버) 테스트
- [ ] **3.1 VM 접속 및 이미지 다운로드**
  - AWS 등의 VM에 SSH 접속 후 `docker pull` 실행.
- [ ] **3.2 GPU 추론 확인** (VM에 NVIDIA GPU가 있는 경우)
  - `nvidia-container-toolkit` 설치 확인.
  - `--gpus all` 옵션으로 실행하여 비약적인 속도 향상 확인.

---

## 🚀 Docker Hub 푸시 방법

작성하신 `face_detection` 레포지토리에 업로드하기 위해 다음 명령어를 순서대로 실행하세요:

```bash
# 1. 이미지 이름 변경 (Tagging)
docker tag our-model-eval:latest kyeonghah/face_detection:latest

# 2. Docker Hub로 업로드 (Push)
docker push kyeonghah/face_detection:latest
```

> **참고**: 현재 `kyeonghah` 계정으로 로그인되어 있으므로 추가 인증 없이 바로 업로드 가능합니다.

---

## 🖥️ VM(원격 서버)에서 실행하는 방법

Docker Hub에 푸시가 완료된 후, VM 환경에서 다음과 같이 실행하면 됩니다.

### 1. 이미지 가져오기 (Pull)
```bash
docker pull kyeonghah/face_detection:latest
```

### 2. 데이터셋 준비
VM 내부의 적절한 경로에 WiderFace 검증 데이터셋(images 폴더, wider_val.txt 파일)을 준비합니다. (예: `/home/ubuntu/data/widerface/val`)

### 3. 컨테이너 실행

**A. GPU를 사용하는 경우 (권장)**:
*전제조건: `nvidia-container-toolkit`이 설치되어 있어야 합니다.*
```bash
docker run --rm --gpus all \
  -v /절대/경로/to/data:/app/data/widerface/val:ro \
  kyeonghah/face_detection:latest evaluate
```

**B. CPU만 사용하는 경우**:
```bash
docker run --rm \
  -e USE_CPU=1 \
  -v /절대/경로/to/data:/app/data/widerface/val:ro \
  kyeonghah/face_detection:latest evaluate
```

---

## ⚠️ 주의사항
- **이미지 용량**: 현재 이미지는 약 13.4GB입니다. `push` 및 `pull` 시 네트워크 대역폭에 따라 시간이 소요될 수 있습니다.
- **데이터 마운트**: `-v` 옵션 뒤의 호스트 경로는 반드시 **절대 경로**로 작성해야 데이터 로딩 오류가 발생하지 않습니다.
