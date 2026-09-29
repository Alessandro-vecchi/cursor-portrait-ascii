"""Capture one webcam photo per gaze target on a GxG screen grid -> raw/r{row}_c{col}.jpg,
plus reaction shots -> raw/click.jpg, raw/hover.jpg"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

WIN = "capture"
REMINDER = ("Turn your whole HEAD toward the dot (exaggerate it!) and let your eyes follow. Eye-only movement "
            "is only 1-2 characters wide once converted to ASCII. Keep the same distance from the screen the "
            "whole time. Use bright, soft, frontal light and a plain background.")
EXTRAS = {"click": "Look SURPRISED!", "hover": "Big GRIN!"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=5, help="grid size G")
    p.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--settle", type=int, default=700, help="ms the ring takes to shrink onto the dot")
    p.add_argument("--burst", type=int, default=5, help="frames per target; the sharpest is kept")
    p.add_argument("--screen", default="1920x1080", help="WxH fallback if the window size can't be read")
    p.add_argument("--only", nargs="+", metavar="R,C", help="re-shoot only these cells (or click / hover)")
    p.add_argument("--extras", default="click,hover", help="reaction shots after the grid ('' for none)")
    p.add_argument("--out", default="raw", help="output directory")
    return p.parse_args()


def check_esc(delay=1):
    if cv2.waitKey(delay) & 0xFF == 27:
        print("Aborted; files already saved are kept.")
        sys.exit(0)


def sharpness(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def shoot(cap, burst):
    """Grab a burst over ~250 ms and return the sharpest frame (this also tends to reject blinks)."""
    best, best_s = None, -1.0
    for _ in range(burst):
        ok, frame = cap.read()
        if ok and (s := sharpness(frame)) > best_s:
            best, best_s = frame, s
        check_esc(max(1, 250 // burst))
    return best


def text_screen(sw, sh, lines, color=(255, 255, 255)):
    canvas = np.zeros((sh, sw, 3), np.uint8)
    for i, (txt, scale) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        y = sh // 2 + int((i - (len(lines) - 1) / 2) * 110)
        cv2.putText(canvas, txt, ((sw - tw) // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3, cv2.LINE_AA)
    return canvas


def targets(g, sw, sh, only):
    mx, my = 0.04 * sw, 0.04 * sh
    step = lambda i, size, m: int(round(m + i * (size - 2 * m) / max(g - 1, 1)))
    out = []
    for r in range(g):
        cols = range(g) if r % 2 == 0 else reversed(range(g))
        for c in cols:
            if only is None or (r, c) in only:
                out.append((r, c, step(c, sw, mx), step(r, sh, my)))
    return out


def main():
    args = parse_args()
    only, extras = None, [e for e in args.extras.split(",") if e]
    if args.only:
        only = {tuple(int(v) for v in s.split(",")) for s in args.only if s not in EXTRAS}
        extras = [s for s in args.only if s in EXTRAS]
    print(REMINDER)
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera} (on macOS, grant the terminal Camera permission).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:
            cap.read()

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        ok, frame = cap.read()
        cv2.imshow(WIN, frame if ok else np.zeros((720, 1280, 3), np.uint8))
        cv2.waitKey(100)
        _, _, sw, sh = cv2.getWindowImageRect(WIN)
        if sw < 200 or sh < 200:
            sw, sh = (int(v) for v in args.screen.lower().split("x"))
        print(f"Screen: {sw}x{sh}")

        # Framing phase: live preview until SPACE.
        while True:
            ok, frame = cap.read()
            canvas = np.zeros((sh, sw, 3), np.uint8)
            if ok:
                s = min(sw / frame.shape[1], sh / frame.shape[0])
                fw, fh = int(frame.shape[1] * s), int(frame.shape[0] * s)
                x0, y0 = (sw - fw) // 2, (sh - fh) // 2
                canvas[y0:y0 + fh, x0:x0 + fw] = cv2.resize(frame, (fw, fh))
            cv2.putText(canvas, "Center your face, press SPACE", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WIN, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("Aborted.")
                return
            if key == 32:
                break

        todo = targets(args.grid, sw, sh, only)
        for n, (r, c, x, y) in enumerate(todo, 1):
            t0 = time.monotonic()
            while (t := (time.monotonic() - t0) * 1000) < args.settle:
                cap.read()  # keep the buffer fresh so the burst is not stale
                canvas = np.zeros((sh, sw, 3), np.uint8)
                radius = int(12 + (90 - 12) * (1 - t / args.settle))
                cv2.circle(canvas, (x, y), radius, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 12, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, f"{n}/{len(todo)}", (20, sh - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (110, 110, 110), 1, cv2.LINE_AA)
                cv2.imshow(WIN, canvas)
                check_esc()

            best = shoot(cap, args.burst)
            if best is None:
                print(f"No frame for r{r}_c{c}; re-shoot with --only {r},{c}")
                continue
            cv2.imwrite(os.path.join(args.out, f"r{r}_c{c}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])

        for name in extras:  # reaction shots: look at the camera, pull the face, SPACE, 3-2-1
            while True:
                cap.read()
                cv2.imshow(WIN, text_screen(sw, sh, [(EXTRAS[name], 3.0), ("look at the camera, press SPACE", 1.0)]))
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("Aborted; files already saved are kept.")
                    return
                if key == 32:
                    break
            for n in (3, 2, 1):
                t0 = time.monotonic()
                while time.monotonic() - t0 < 0.7:
                    cap.read()
                    cv2.imshow(WIN, text_screen(sw, sh, [(str(n), 6.0)]))
                    check_esc()
            best = shoot(cap, args.burst)
            if best is not None:
                cv2.imwrite(os.path.join(args.out, f"{name}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Done: {len(todo)} grid photos + {len(extras)} reactions in {args.out}/")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
