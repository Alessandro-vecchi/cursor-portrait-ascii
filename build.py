"""Turn raw/r{row}_c{col}.jpg (+ optional raw/click.jpg, raw/hover.jpg) into ASCII frames -> web/frames.js."""
import argparse
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "selfie_segmenter.tflite")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
HAAR = "haarcascade_frontalface_default.xml"
HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + HAAR
EXTRAS = ("click", "hover")
RAMP = " .,:;i1tfLCG08@"  # letters over dashes: fewer horizontal stripes across the face


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, default=5, help="grid size G")
    p.add_argument("--cols", type=int, default=140, help="ASCII width in characters")
    p.add_argument("--aspect", type=float, default=0.5, help="character cell width/height (must match the page)")
    p.add_argument("--ramp", default=RAMP, help="characters from lowest to highest density")
    p.add_argument("--invert", action="store_true", help="black text on white instead of white on black")
    p.add_argument("--crop", help="x,y,w,h in raw pixels, overriding the automatic face crop")
    p.add_argument("--no-reverse-x", action="store_true", help="disable the horizontal mirror mapping")
    p.add_argument("--gamma", type=float, default=0.8, help="tone curve applied after normalization")
    return p.parse_args()


def make_segmenter():
    """Return a function BGR image -> person confidence mask (HxW float32)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        if not os.path.exists(MODEL):
            os.makedirs(os.path.dirname(MODEL), exist_ok=True)
            print(f"Downloading {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, MODEL)
        seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False))

        def run(bgr):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return np.squeeze(seg.segment(img).confidence_masks[0].numpy_view()).astype(np.float32)
    except (ImportError, AttributeError):
        import mediapipe as mp
        print("Note: MediaPipe Tasks API unavailable, using legacy SelfieSegmentation.")
        seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

        def run(bgr):
            return seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).segmentation_mask.astype(np.float32)

    def conf(bgr):
        m = run(bgr)
        return cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return conf


def center_beats_corners(m):
    h, w = m.shape
    center = m[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)].mean()
    ch, cw = int(h * 0.1), int(w * 0.1)
    corners = np.mean([m[:ch, :cw].mean(), m[:ch, -cw:].mean(), m[-ch:, :cw].mean(), m[-ch:, -cw:].mean()])
    return center > corners


def load_cascade():
    path = os.path.join(getattr(cv2, "data", None) and cv2.data.haarcascades or "", HAAR)
    if not os.path.exists(path):  # opencv 5 wheels no longer ship the cascades
        path = os.path.join(ROOT, "models", HAAR)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"Downloading {HAAR_URL}")
            urllib.request.urlretrieve(HAAR_URL, path)
    return cv2.CascadeClassifier(path)


def largest_face(cascade, bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None


def auto_crop(face, shape):
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2 - 0.1 * fh
    w, h = fw * 1.6, fh * 2.0
    H, W = shape[:2]
    x0, y0 = max(0, int(round(cx - w / 2))), max(0, int(round(cy - h / 2)))
    x1, y1 = min(W, int(round(cx + w / 2))), min(H, int(round(cy + h / 2)))
    return x0, y0, x1 - x0, y1 - y0


def clahe_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray).astype(np.float32) / 255


def tone(g, lo, hi, gamma):
    g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    return np.clip(g + 0.6 * (g - cv2.GaussianBlur(g, (0, 0), 1.0)), 0, 1)


def to_ascii(gray, alpha, ramp, invert):
    d = (1 - gray if invert else gray) * alpha
    idx = np.round(d * (len(ramp) - 1)).astype(int)
    idx[alpha >= 0.5] = np.maximum(idx[alpha >= 0.5], 1)  # dark parts of me stay visible as dim dots
    chars = np.array(list(ramp))[idx]
    chars[alpha < 0.5] = " "
    return ["".join(row) for row in chars]


def main():
    args = parse_args()
    G = args.grid
    paths = {(r, c): os.path.join(ROOT, "raw", f"r{r}_c{c}.jpg") for r in range(G) for c in range(G)}
    missing = [f"{r},{c}" for (r, c), p in paths.items() if not os.path.exists(p)]
    if missing:
        sys.exit("Missing raw cells: " + " ".join(missing) + "\nRe-shoot with: python capture.py --only "
                 + " ".join(missing))

    conf = make_segmenter()
    cascade = load_cascade()
    center = cv2.imread(paths[(G // 2, G // 2)])
    flip = not center_beats_corners(conf(center))
    if flip:
        print("Note: mask looked inverted on the center frame; using 1 - mask.")

    def alpha_of(bgr):
        m = conf(bgr)
        m = 1 - m if flip else m
        return cv2.GaussianBlur(np.clip((m - 0.35) / 0.3, 0, 1), (3, 3), 0)

    face0 = largest_face(cascade, center)
    if args.crop:
        x, y, w, h = (int(v) for v in args.crop.split(","))
    elif face0 is None:
        sys.exit("No face found in the center frame; pass --crop x,y,w,h")
    else:
        x, y, w, h = auto_crop(face0, center.shape)
        print(f"Auto crop: --crop {x},{y},{w},{h}")
    rows = round(args.cols * (h / w) * args.aspect)

    # Shared normalization from the center frame only, so swapped frames don't flicker.
    # Measured on the face itself when possible: a bright shirt would otherwise push the face into the darks.
    g0 = clahe_gray(center[y:y + h, x:x + w])
    a0 = alpha_of(center)[y:y + h, x:x + w]
    sel = a0 > 0.5
    if face0 is not None:
        fx, fy, fw, fh = face0
        box = np.zeros_like(sel)
        box[max(0, fy - y):fy - y + fh, max(0, fx - x):fx - x + fw] = True
        sel &= box
    lo, hi = np.percentile(g0[sel], [2, 98])

    extras = {k: os.path.join(ROOT, "raw", f"{k}.jpg") for k in EXTRAS}
    for k, p in list(extras.items()):
        if not os.path.exists(p):
            print(f"Note: raw/{k}.jpg not found, no {k} reaction (shoot it with: python capture.py --only {k})")
            del extras[k]

    ascii_raw, thumbs = {}, {}
    tw, th = 160, round(160 * h / w)
    for key, p in sorted(paths.items()) + list(extras.items()):
        bgr = cv2.imread(p)
        # A face that is noticeably bigger or smaller than in the center frame means I moved.
        face = largest_face(cascade, bgr)
        if face0 is not None and face is not None and abs(face[2] / face0[2] - 1) > 0.10:
            name = key if isinstance(key, str) else "{},{}".format(*key)
            print(f"warning: {name} face size {face[2] / face0[2] - 1:+.0%} vs center; check contact.jpg, "
                  f"re-shoot with --only {name} if you moved")
        a = alpha_of(bgr)[y:y + h, x:x + w]
        g = tone(clahe_gray(bgr[y:y + h, x:x + w]), lo, hi, args.gamma)
        thumbs[key] = cv2.resize((g * a * 255).astype(np.uint8), (tw, th), interpolation=cv2.INTER_AREA)
        gs = cv2.resize(g, (args.cols, rows), interpolation=cv2.INTER_AREA)
        As = cv2.resize(a, (args.cols, rows), interpolation=cv2.INTER_AREA)
        ascii_raw[key] = to_ascii(gs, As, args.ramp, args.invert)

    # Web cell (r, c) shows raw (r, G-1-c): a cursor on the viewer's right needs the face turned to image-right.
    src = lambda r, c: (r, c) if args.no_reverse_x else (r, G - 1 - c)
    frames = [ascii_raw[src(r, c)] for r in range(G) for c in range(G)]
    extra_frames = [ascii_raw[k] for k in extras]

    # Shared trim to the union bounding box of non-space characters, plus 1 char of padding.
    ink = np.array([[[ch != " " for ch in line] for line in f] for f in frames + extra_frames]).any(axis=0)
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        sys.exit("All frames are blank; check --crop and the segmentation.")
    r0, r1 = max(0, int(ys.min()) - 1), min(rows, int(ys.max()) + 2)
    c0, c1 = max(0, int(xs.min()) - 1), min(args.cols, int(xs.max()) + 2)
    trim = lambda f: "\n".join(line[c0:c1] for line in f[r0:r1])

    out = os.path.join(ROOT, "web", "frames.js")
    data = {"grid": G, "cols": c1 - c0, "rows": r1 - r0, "aspect": args.aspect, "invert": args.invert,
            "ramp": args.ramp, "frames": [trim(f) for f in frames]}
    data.update({k: trim(ascii_raw[k]) if k in extras else None for k in EXTRAS})
    with open(out, "w") as f:
        f.write("window.PORTRAIT = " + json.dumps(data) + ";\n")

    os.makedirs(os.path.join(ROOT, "debug"), exist_ok=True)
    sheet = [np.hstack([thumbs[src(r, c)] for c in range(G)]) for r in range(G)]
    if extras:  # last row: the reaction shots, padded with black
        row = [thumbs[k] for k in extras] + [np.zeros_like(thumbs[k])] * (G - len(extras))
        sheet.append(np.hstack(row[:G]))
    cv2.imwrite(os.path.join(ROOT, "debug", "contact.jpg"), np.vstack(sheet))
    with open(os.path.join(ROOT, "debug", "center.txt"), "w") as f:
        f.write("\n".join(frames[(G // 2) * G + G // 2]) + "\n")  # untrimmed, full crop

    print(f"Frames: {G * G} + {len(extras)} extras, {c1 - c0} cols x {r1 - r0} rows "
          f"(before trim {args.cols}x{rows})")
    print(f"Wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
