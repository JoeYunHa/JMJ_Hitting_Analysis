# scripts/roi_picker.py
import cv2
import json
import argparse
from pathlib import Path

current_label = ""
drawing = False
start_x, start_y = -1, -1
rect = None
scale_factor = 1.0

VALID_PLATFORMS = {
    "tving": {"default": ["game_highlight"]},
    "kbo": {
        "kbsn": ["game_highlight"],
        "mbcsports": ["game_highlight"],
        "spotv": ["game_highlight"],
        "spotv2": ["game_highlight"],
        "sbs": ["game_highlight"],
        "kbslife": ["game_highlight"],
    },
}

ROI_LABELS = [
    "batter_name_roi",
    "pitcher_name_roi",
    "score_roi",
    "count_roi",
    "base_roi",
    "inning_roi",
]


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, rect, img_display, img_orig

    # Convert display coords back to original resolution
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
        w, h = abs(ox - start_x), abs(oy - start_y)
        rect = {"x": x1, "y": y1, "w": w, "h": h}
        sx, sy = int(start_x * scale_factor), int(start_y * scale_factor)
        cv2.rectangle(img_display, (sx, sy), (x, y), (0, 255, 0), 2)
        print(f"  [{current_label}] -> {rect}  (original resolution)")


def pick_rois(
    frame_path: str, labels: list[str], meta: dict, display_width: int = 1280
) -> dict:
    global img_orig, img_display, current_label, rect, scale_factor

    img_orig = cv2.imread(frame_path)
    assert img_orig is not None, f"Cannot read: {frame_path}"

    h, w = img_orig.shape[:2]
    scale_factor = min(display_width / w, 1.0)  # never upscale
    img_display = cv2.resize(img_orig, None, fx=scale_factor, fy=scale_factor)

    print(
        f"Original: {w}x{h} | Display scale: {scale_factor:.2f} | Display: {int(w*scale_factor)}x{int(h*scale_factor)}"
    )

    cv2.namedWindow("ROI Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Picker", int(w * scale_factor), int(h * scale_factor))
    cv2.setMouseCallback("ROI Picker", mouse_callback)

    result = {**meta, "frame_resolution": {"w": w, "h": h}}

    for label in labels:
        current_label = label
        rect = None
        print(f"\nDraw ROI for [{label}], then press ENTER. Press 'r' to redo.")

        while True:
            cv2.imshow("ROI Picker", img_display)
            key = cv2.waitKey(1) & 0xFF

            if key in (10, 13) and rect:
                result[label] = rect.copy()
                img_orig = (
                    cv2.resize(img_display, (w, h))
                    if scale_factor != 1.0
                    else img_display.copy()
                )
                img_orig = cv2.imread(
                    frame_path
                )  # re-read clean base; keep drawn rects on display only
                break
            elif key == ord("r"):
                img_display = cv2.resize(
                    img_orig, None, fx=scale_factor, fy=scale_factor
                )
                rect = None
                print(f"  Redo [{label}]")

    cv2.destroyAllWindows()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True)
    parser.add_argument(
        "--platform", required=True, choices=list(VALID_PLATFORMS.keys())
    )
    parser.add_argument("--channel", default=None)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--content", default="game_highlight")
    parser.add_argument("--version", default=None)
    parser.add_argument("--out_dir", default="configs/channels")
    parser.add_argument(
        "--display_width", default=1280, type=int, help="Max display width in pixels"
    )
    args = parser.parse_args()

    channels = VALID_PLATFORMS[args.platform]

    if args.platform == "tving":
        channel_key = "default"
    else:
        if not args.channel:
            raise ValueError(f"--channel is required for platform '{args.platform}'")
        if args.channel not in channels:
            raise ValueError(
                f"'{args.channel}' not valid. Allowed: {list(channels.keys())}"
            )
        channel_key = args.channel

    if args.content not in channels[channel_key]:
        raise ValueError(f"'{args.content}' not valid for channel '{channel_key}'.")

    config_id = f"{args.platform}_{channel_key}_{args.year}_{args.content}"
    if args.version:
        config_id += f"_{args.version}"

    meta = {
        "config_id": config_id,
        "platform": args.platform,
        "channel": channel_key,
        "year": args.year,
        "content": args.content,
        "version": args.version,
        "logo_template": f"configs/channels/templates/{channel_key}_logo.png",
        "notes": f"ROI for {config_id}",
    }

    data = pick_rois(args.frame, ROI_LABELS, meta, display_width=args.display_width)

    filename = f"{args.year}_{args.content}"
    if args.version:
        filename += f"_{args.version}"

    if args.platform == "tving":
        out_path = Path(args.out_dir) / args.platform / f"{filename}.json"
    else:
        out_path = Path(args.out_dir) / args.platform / channel_key / f"{filename}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {out_path}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
