# scripts/roi_picker.py
# Interactive ROI picker - supports platform/channel/year combinations

import cv2
import json
import argparse
from pathlib import Path

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
        print(f"  DEBUG rect set: {rect}")  # debug


def pick_rois(frame_path: str, labels: list[str], meta: dict) -> dict:
    global img_orig, img_display, current_label, rect

    img_orig = cv2.imread(frame_path)
    assert img_orig is not None, f"Cannot read: {frame_path}"
    img_display = img_orig.copy()

    h, w = img_orig.shape[:2]
    cv2.namedWindow("ROI Picker")
    cv2.setMouseCallback("ROI Picker", mouse_callback)

    result = {**meta, "frame_resolution": {"w": w, "h": h}}

    for label in labels:
        current_label = label
        rect = None
        print(f"\nDraw ROI for [{label}], then press ENTER. Press 'r' to redo.")

        while True:
            cv2.imshow("ROI Picker", img_display)
            key = cv2.waitKey(1) & 0xFF

            if key != 255:
                print(f" DEBUG key={key}, rect={rect}")
            if key in (10, 13) and rect:
                result[label] = rect.copy()
                img_orig = img_display.copy()
                break
            elif key == ord("r"):
                img_display = img_orig.copy()
                rect = None
                print(f"  Redo [{label}]")

    cv2.destroyAllWindows()
    return result


# Valid platforms and their allowed content types
VALID_PLATFORMS = {
    "tving": ["game_highlight"],
    "kbo": ["game_highlight"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Sample frame image path")
    parser.add_argument(
        "--platform",
        required=True,
        choices=list(VALID_PLATFORMS.keys()),
        help="Platform (tving/kbo)",
    )
    parser.add_argument(
        "--year", required=True, type=int, help="Broadcast year (e.g. 2025)"
    )
    parser.add_argument(
        "--content",
        default="game_highlight",
        help="Content type (default: game_highlight)",
    )
    parser.add_argument(
        "--version", default=None, help="Optional sub-version tag (e.g. v2, widescreen)"
    )
    parser.add_argument("--out_dir", default="configs/channels")
    args = parser.parse_args()

    # Validate content type for platform
    if args.content not in VALID_PLATFORMS[args.platform]:
        raise ValueError(
            f"'{args.content}' is not valid for platform '{args.platform}'. "
            f"Allowed: {VALID_PLATFORMS[args.platform]}"
        )

    labels = ["batter_name_roi", "score_roi"]

    # e.g. tving_2025_game_highlight or tving_2025_game_highlight_v2
    config_id = f"{args.platform}_{args.year}_{args.content}"
    if args.version:
        config_id += f"_{args.version}"

    meta = {
        "config_id": config_id,
        "platform": args.platform,
        "year": args.year,
        "content": args.content,
        "version": args.version,
        "logo_template": f"configs/channels/templates/{args.platform}_logo.png",
        "notes": f"ROI for {config_id}",
    }

    data = pick_rois(args.frame, labels, meta)

    # Save to configs/channels/{platform}/{year}_{content}.json
    filename = f"{args.year}_{args.content}"
    if args.version:
        filename += f"_{args.version}"
    out_path = Path(args.out_dir) / args.platform / f"{filename}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {out_path}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
