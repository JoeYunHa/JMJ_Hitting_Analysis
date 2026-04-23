# ⚾ Baseball Swing Analysis Pipeline (Draft)

# ⚾ 야구 타격폼 분석 파이프라인 (초안)

---

## 1. Overview

## 1. 개요

This project aims to build a lightweight pipeline that analyzes baseball batting mechanics from broadcast highlight videos.
본 프로젝트는 야구 경기 하이라이트 영상을 기반으로 타격 메커니즘을 분석하는 경량 파이프라인을 구축하는 것을 목표로 한다.

The core idea is:
핵심 아이디어는 다음과 같다:

* Extract key frames from highlight videos
  → 하이라이트 영상에서 핵심 프레임 추출
* Detect body pose
  → 신체 포즈 추정
* Derive simplified swing metrics
  → 단순화된 타격 지표 생성
* Visualize results for comparison
  → 결과 시각화 및 비교

This is an **early-stage prototype** focused on feasibility rather than accuracy.
본 프로젝트는 정확도보다는 가능성 검증에 초점을 둔 **초기 프로토타입 단계**이다.

---

## 2. Goals

## 2. 목표

* Build a reproducible analysis pipeline using public video sources
  → 공개 영상 기반 재현 가능한 분석 파이프라인 구축

* Extract pose-based features from batting scenes
  → 타격 장면에서 포즈 기반 특징 추출

* Generate simple, explainable metrics (timing, rotation, balance)
  → 타이밍, 회전, 균형 등 해석 가능한 지표 생성

* Minimize dependency on raw video by converting to structured data
  → 원본 영상 의존도를 줄이고 구조화된 데이터로 변환

---

## 3. Scope (Current)

## 3. 현재 범위

**Input**

* Game highlight videos (YouTube-based, manually selected)
  → 유튜브 기반 경기 하이라이트 영상 (수동 선택)

**Output**

* Pose keypoints (JSON)
  → 포즈 좌표 데이터 (JSON)
* Derived metrics (CSV)
  → 계산된 지표 데이터 (CSV)
* Visualization images (PNG)
  → 시각화 이미지 (PNG)

**Not included**

* Real-time processing
  → 실시간 처리
* Full automation
  → 완전 자동화
* High-precision biomechanics modeling
  → 정밀 생체역학 모델링

---

## 4. High-Level Architecture

## 4. 전체 구조

```id="p8x21a"
[Video Input / 영상 입력]
        ↓
[Frame Extraction / 프레임 추출]
        ↓
[Pose Estimation / 포즈 추정]
        ↓
[Event Detection / 타격 이벤트 탐지]
        ↓
[Metric Calculation / 지표 계산]
        ↓
[Visualization / 시각화 및 결과 출력]
```

---

## 5. Tech Stack (Planned)

## 5. 기술 스택 (예정)

* Python
* OpenCV (video processing / 영상 처리)
* MediaPipe (pose estimation / 포즈 추정)
* NumPy / Pandas (data processing / 데이터 처리)

**Environment**

* Google Colab (primary execution / 주요 실행 환경)
* Local machine (development only / 개발용)

---

## 6. Workflow

## 6. 작업 흐름

1. Select highlight video manually
   → 하이라이트 영상 수동 선택

2. Load project code from GitHub in Colab
   → Colab에서 GitHub 코드 로드

3. Run analysis notebook
   → 분석 노트북 실행

4. Export results (Drive / local)
   → 결과 저장 (Drive 또는 로컬)

5. Review and iterate
   → 결과 검토 및 반복 개선

---

## 7. Design Principles

## 7. 설계 원칙

* **Minimal dependency on raw video**
  → 원본 영상 의존 최소화
  (frames → pose → structured data)

* **Explainable metrics**
  → 해석 가능한 지표 중심

* **Loose coupling**
  → 모듈 간 결합도 최소화

* **Iterative development**
  → 점진적 개발 방식

---

## 8. Limitations

## 8. 한계

* Broadcast footage is not optimized for analysis
  → 중계 영상은 분석용으로 최적화되어 있지 않음

* Camera angle inconsistency affects accuracy
  → 카메라 각도 차이로 정확도 저하

* Bat tracking is not fully supported yet
  → 배트 추적 기능 미완성

* Metrics are approximations, not ground truth
  → 지표는 근사값이며 실제 물리량이 아님

---

## 9. Future Work (Tentative)

## 9. 향후 계획 (초안)

* Improve swing event detection
  → 타격 이벤트 탐지 정확도 개선

* Add bat tracking or proxy estimation
  → 배트 추적 또는 근사 모델 추가

* Build dataset for repeated analysis
  → 반복 분석을 위한 데이터셋 구축

* Compare swings across players/seasons
  → 선수/시즌 간 비교 분석

* Optional API or visualization dashboard
  → API 또는 시각화 대시보드 구축

---

## 10. Status

## 10. 진행 상태

* [ ] Project structure setup
  → 프로젝트 구조 설정

* [ ] Basic frame extraction
  → 프레임 추출 구현

* [ ] Pose estimation test
  → 포즈 추정 테스트

* [ ] Metric definition (draft)
  → 지표 정의 (초안)

* [ ] Visualization prototype
  → 시각화 프로토타입

---

## 11. Notes

## 11. 참고 사항

* This project is for **personal analysis purposes**
  → 본 프로젝트는 **개인 분석 목적**

* No original broadcast content is redistributed
  → 원본 중계 영상은 재배포하지 않음

* Only derived data and visualizations are used
  → 가공된 데이터 및 시각화 결과만 사용

---

> This project is in an early exploration stage.
> 본 프로젝트는 초기 탐색 단계이며 향후 구조가 변경될 수 있음.

---
