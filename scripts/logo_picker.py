"""로고 영역을 마우스로 선택해 template 이미지로 저장하는 도구.

Usage:
    python scripts/logo_picker.py --frame data/frames/kbsn_sample.png --channel kbsn
"""
import cv2
import argparse
from pathlib import Path

drawing = False
start_x, start_y = -1, -1
rect = None
scale_factor = 1.0
img_orig = None
img_display = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, rect, img_display, img_orig

    ox, oy = int(x / scale_factor), int(y / scale_factor)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = ox, oy
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = cv2.resize(img_orig, None, fx=scale_factor, fy=scale_factor)
        sx, sy = int(start_x * scale_factor), int(start_y * scale_factor)
        cv2.rectangle(img_display, (sx, sy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, ox), min(start_y, oy)
        x2, y2 = max(start_x, ox), max(start_y, oy)
        rect = (x1, y1, x2, y2)
        sx, sy = int(start_x * scale_factor), int(start_y * scale_factor)
        cv2.rectangle(img_display, (sx, sy), (x, y), (0, 255, 0), 2)
        print(f"  선택 영역: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")


def pick_logo(frame_path: str, out_path: str, display_width: int = 1280) -> None:
    global img_orig, img_display, rect, scale_factor

    img_orig = cv2.imread(frame_path)
    if img_orig is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {frame_path}")

    h, w = img_orig.shape[:2]
    scale_factor = min(display_width / w, 1.0)
    img_display = cv2.resize(img_orig, None, fx=scale_factor, fy=scale_factor)

    print(f"프레임: {w}x{h}  |  표시 스케일: {scale_factor:.2f}")
    print("로고 영역을 드래그하세요.")
    print("  ENTER : 저장  |  r : 다시 선택  |  q : 취소")

    cv2.namedWindow("Logo Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Logo Picker", int(w * scale_factor), int(h * scale_factor))
    cv2.setMouseCallback("Logo Picker", mouse_callback)

    rect = None
    while True:
        cv2.imshow("Logo Picker", img_display)
        key = cv2.waitKey(1) & 0xFF

        if key in (10, 13):  # ENTER
            if rect is None:
                print("  영역을 먼저 선택하세요.")
                continue
            x1, y1, x2, y2 = rect
            crop = img_orig[y1:y2, x1:x2]
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(out_path, crop)
            print(f"저장 완료 -> {out_path}  ({x2-x1}x{y2-y1}px)")
            break

        elif key == ord("r"):
            img_display = cv2.resize(img_orig, None, fx=scale_factor, fy=scale_factor)
            rect = None
            print("  다시 선택하세요.")

        elif key == ord("q"):
            print("  취소되었습니다.")
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="채널 로고 템플릿 추출 도구")
    parser.add_argument("--frame", required=True, help="샘플 프레임 이미지 경로")
    parser.add_argument(
        "--channel", required=True,
        help="채널명 (예: kbsn, mbcsports, spotv, tving)"
    )
    parser.add_argument(
        "--out_dir", default="configs/channels/templates",
        help="템플릿 저장 디렉토리 (기본: configs/channels/templates)"
    )
    parser.add_argument("--display_width", default=1280, type=int)
    args = parser.parse_args()

    out_path = str(Path(args.out_dir) / f"{args.channel}_logo.png")
    pick_logo(args.frame, out_path, display_width=args.display_width)


if __name__ == "__main__":
    main()
