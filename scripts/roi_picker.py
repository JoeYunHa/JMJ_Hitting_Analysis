# scripts/roi_picker.py
# Interactive ROI coordinate picker for channel config registration

import cv2
import json
import argparse
from pathlib import Path

rois = {}
current_label = ""
drawing = False
start_x, start_y = -1, -1
rect = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, rect, img_display, img_orig

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = img_orig.copy()
        cv2.rectangle(img_display, (start_x, start_y), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, x), min(start_y, y)
        w, h = abs(x - start_x), abs(y - start_y)
        rect = {"x": x1, "y": y1, "w": w, "h": h}
        cv2.rectangle(img_display, (start_x, start_y), (x, y), (0, 255, 0), 2)
        print(f"  [{current_label}] -> {rect}")


def pick_rois(frame_path: str, channel: str, labels: list[str]) -> dict:
    global img_orig, img_display, current_label, rois

    img_orig = cv2.imread(frame_path)
    assert img_orig is not None, f"Cannot read: {frame_path}"
    img_display = img_orig.copy()

    cv2.namedWindow("ROI Picker")
    cv2.setMouseCallback("ROI Picker", mouse_callback)

    result = {"channel": channel}

    for label in labels:
        current_label = label
        print(f"\nDraw ROI for [{label}], then press ENTER. Press 'r' to redo.")

        while True:
            cv2.imshow("ROI Picker", img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and rect:  # ENTER: confirm
                result[label] = rect.copy()
                img_orig = img_display.copy()  # keep drawn rect on canvas
                break
            elif key == ord("r"):  # r: redo current label
                img_display = img_orig.copy()
                print(f"  Redo [{label}]")

    cv2.destroyAllWindows()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Sample frame image path")
    parser.add_argument(
        "--channel", required=True, help="Channel name (kbs/mbc/sbs/spotv)"
    )
    parser.add_argument("--out_dir", default="configs/channels")
    args = parser.parse_args()

    # Define ROI labels to register per channel
    labels = ["batter_name_roi", "score_roi"]

    data = pick_rois(args.frame, args.channel.upper(), labels)
    data["logo_template"] = f"configs/channels/templates/{args.channel}_logo.png"
    data["notes"] = "Coordinates based on captured frame resolution"

    out_path = Path(args.out_dir) / f"{args.channel.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {out_path}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
