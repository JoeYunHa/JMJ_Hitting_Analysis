# ⚾ JMJ Hitting Analysis

전민재 선수의 **홈런 스윙 폼**을 방송 하이라이트 영상에서 자동으로 분석하는 파이프라인.

---

## 분석 전략

방송 하이라이트 영상에서 안타 장면은 대부분 **정면샷**으로 촬영되어 힙 회전·배트 래그 등 핵심 폼 지표를 측정하기 어렵다.
반면 **홈런 리플레이**에는 측면 슬로우 영상과 함께 발사각·타구속도 등 TV 그래픽이 포함되므로, 홈런 영상에 집중하여 정밀 폼 분석을 수행한다.

```
수동 수집 홈런 링크 (homerun_links.csv)
    ↓ yt-dlp 다운로드 → raw_data/
    ↓ 측면샷 구간 탐지 + TV 그래픽 OCR (발사각·타구속도)
    ↓ YOLO-Pose 포즈 추정 (측면샷 집중)
    ↓ 홈런 유형 분류 (발사각·방향·카운트)
    ↓ 스윙 폼 클러스터링 (DTW)
    ↓ React 시각화
```

---

## 홈런 유형 분류

| 유형 | 기준 |
|---|---|
| 발사각별 | line_drive (10~25°) / optimal (25~35°) / fly_ball (35°+) |
| 방향별 | pull / center / oppo |
| 카운트별 | early_count (B≥S) / two_strike (??-2) |

---

## 기술 스택

| 영역 | 도구 |
|---|---|
| 영상 수집 | yt-dlp |
| 영상 처리 | OpenCV |
| OCR | PaddleOCR |
| 포즈 추정 | YOLOv8-Pose |
| 데이터 처리 | NumPy, Pandas |
| 클러스터링 | tslearn (DTW) |

---

## 개발 현황

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 환경 구성, 채널 config 등록 | 완료 |
| 2 | 홈런 링크 수동 수집 + yt-dlp 다운로드 | 구현 예정 |
| 3 | 측면샷 탐지 + 타구정보 OCR + 스윙 탐지 | 부분 구현 |
| 4 | YOLO-Pose 포즈 추정 + 홈런 유형 분류 + 지표 계산 | 미구현 |
| 5 | DTW 클러스터링 + 홈런 유형별 폼 비교 | 미구현 |
| 6 | React 시각화 연결 | 미구현 |

---

## 빠른 시작

```bash
# 홈런 영상 다운로드
python scripts/download_homeruns.py   # raw_data/homerun_links.csv 기반

# Phase 3 실행 (단일 영상)
python -c "from src.detection.segment_pipeline import run_phase3; run_phase3('raw_data/xxx.mp4')"

# ROI 설정용 프레임 추출
python scripts/extract_frame.py --video raw_data/xxx.mp4 --timestamp 63.5 --out raw_data/sample.png
```

---

## 참고

- 원본 중계 영상은 재배포하지 않으며, 가공된 포즈 데이터·지표·시각화만 사용
- 개인 분석 목적 프로젝트
