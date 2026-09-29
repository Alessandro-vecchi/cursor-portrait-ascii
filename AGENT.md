## GoalR

Build a static web page showing a black-and-white ASCII portrait of me whose gaze follows the visitor's cursor. There's no computer vision at runtime. The page is a lookup table of pre-rendered text frames, one per gaze direction on a G×G grid (default G=9, 81 frames), and it picks the frame nearest the pointer.

The pipeline has three parts:

1. `capture.py` shows a dot on a G×G grid on screen and takes a webcam photo while I look at each position. Output: `raw/r{row}_c{col}.jpg`.
2. `build.py` runs offline. It removes the background with MediaPipe Selfie Segmenter, applies one fixed face crop to every frame, sets tone, converts to ASCII, and writes `web/frames.js`.
3. `web/index.html` holds one `<pre>` and swaps its `textContent` to follow the pointer. It uses plain HTML/CSS/JS in a single file, with no framework, no bundler, and no runtime dependencies. It must also work when opened over `file://`.

Keep the whole thing as simple as possible: 3 source files plus `requirements.txt` and `README.md`. Avoid extra abstractions, classes, or config files. Use argparse flags with the defaults given below.

## v2 decisions (these override the original spec below where they conflict)

v1 was built exactly to the spec below, but the result was hard to read and the frames looked identical. After reviewing it, these decisions replace the original ones:

- **Grayscale, not pure black and white.** Each character's brightness follows its position in the ramp (sqrt curve, 20% floor, 12 levels), drawn on a `<canvas>` in bold monospace. `web/frames.js` also carries `ramp`.
- **Legibility defaults:** `--cols 140`, `--ramp " .,:;i1tfLCG08@"` (letters, because `-=~+` draw horizontal stripes), `--gamma 0.8`. The tone stretch percentiles are taken over the face box only, so a bright shirt doesn't push the face into the dark characters. Pixels inside the silhouette use at least the second ramp character. The auto crop is w×1.6, h×2.0.
- **5×5 grid (default) with big head turns**, not 9×9 eye-only gaze.
- **Reaction frames:** `capture.py` shoots `raw/click.jpg` ("Look SURPRISED!") and `raw/hover.jpg` ("Big GRIN!") after the grid (`--extras`, `--only click`). `build.py` adds them as `click`/`hover` in `frames.js` (null if missing).
- **Distance check:** `build.py` warns when the Haar face width in any shot differs more than 10% from the center shot. OpenCV 5 wheels don't ship the Haar XML, so it is downloaded to `models/` when missing.
- **Drop-in embed:** `web/portrait.js` turns every `[data-portrait]` element into a canvas that fits the element. It follows the pointer anywhere on the page. It runs an auto-demo loop until the first move and 3 s after the pointer leaves, and holds the center frame under `prefers-reduced-motion`. It shows a `data-hint` caption until the first move, `hover` when the pointer is on the central 40% of the face, and `click` for 900 ms on pointerdown. It sets `data-frame` on the element (used by the tests). `web/index.html` is just a demo page that uses the embed.

## Repository layout (final)

```
capture.py
build.py
requirements.txt        # only: mediapipe   (it pulls numpy + opencv-contrib-python)
README.md               # setup, capture tips, build, deploy
models/                 # selfie_segmenter.tflite (auto-downloaded, gitignored)
raw/                    # captured photos (gitignored)
debug/                  # contact sheets for QA (gitignored)
web/index.html
web/frames.js           # generated, committed (it is the deployable asset)
.gitignore
```

## Environment

- Python 3.10–3.12 in a venv (`python3 -m venv .venv`). Check with pip that a `mediapipe` wheel exists for the interpreter; if it doesn't, use 3.11.
- `requirements.txt` contains only `mediapipe`. Do NOT add `opencv-python` too, because it conflicts with the `opencv-contrib-python` that mediapipe already installs (two `cv2` packages).
- On macOS the terminal needs Camera permission (System Settings → Privacy & Security → Camera).

## 1. `capture.py`

Behavior:

- Open the webcam with `cv2.VideoCapture(args.camera)` and request 1280×720. Warm up for about 1.5 s so auto-exposure settles.
- **Framing phase:** show a fullscreen window with the live camera preview and the text "Center your face, press SPACE". Wait for SPACE, or ESC to quit.
- **Screen size:** make the window fullscreen (`cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)`), show one frame, then read `cv2.getWindowImageRect(win)`. Fall back to the `--screen WxH` flag if that returns nonsense.
- **Target positions:** G×G points spread evenly across the screen with a 4% margin. Visit them in serpentine order (row 0 left→right, row 1 right→left, …) to keep jumps small.
- For each target:
  - Draw a black background. Draw a white dot (radius about 12 px) and a ring that shrinks onto it over `--settle` ms (default 700) so the eyes lock on.
  - Keep calling `cap.read()` during the settle period so OpenCV's frame buffer never delivers a stale frame. Pump `cv2.waitKey(1)` so the window stays responsive.
  - After settling, grab `--burst` frames (default 5) over about 250 ms. Keep the sharpest one, measured by variance of the Laplacian on the central 50% of the image. This also tends to reject blink frames.
  - Save the unmodified frame (no flip, no crop) as `raw/r{r}_c{c}.jpg` at JPEG quality 95. Row 0 is the top of the screen and column 0 is the left.
   Show progress `n/81` in a small grey font in a corner.
- `--only r,c [r,c ...]` re-shoots only those cells, for fixing a blink or a bad frame.
- ESC aborts cleanly and keeps the files already saved.
- Print a reminder at startup: *"Follow the dot with your eyes AND turn your head slightly. Eye-only movement is only 1–2 characters wide once converted to ASCII. Use even, soft, frontal light and a plain background."*

## 2. `build.py`

Flags and defaults:

| Flag | Default | Meaning |
|---|---|---|
| `--grid` | 9 | Grid size G |
| `--cols` | 80 | ASCII width in characters |
| `--aspect` | 0.5 | Character cell width/height used for row calculation (must match the web page) |
| `--ramp` | `" .:-=+*#%@"` | Characters from lowest to highest density |
| `--invert` | off | Off: white text on black. On: black text on white |
| `--crop` | auto | `x,y,w,h` in raw pixels, overriding the automatic face crop |
| `--no-reverse-x` | off | Turns off the horizontal mapping flip (see step 7) |
| `--gamma` | 1.0 | Tone curve applied after normalization |

Steps:

1. **Load** all G×G raw images. If any cell is missing, stop and list the missing `r,c` pairs.
2. **Segmentation (MediaPipe Tasks API)**
   - If `models/selfie_segmenter.tflite` is missing, download it from `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite`.
   - Create `mediapipe.tasks.python.vision.ImageSegmenter` with `running_mode=IMAGE`, `output_confidence_masks=True`, and `output_category_mask=False`. Pass images as `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)` (convert BGR→RGB first). Take `result.confidence_masks[0].numpy_view()`, squeeze it to H×W float32, and resize it to the image size.
   - **Orientation check:** on the center frame, the mean confidence in the central 20% box must be higher than in the four 10% corner boxes. If it isn't, use `1 - mask` and print a note. Do not assume which one it is.
   - Get alpha as `clip((conf - 0.35) / 0.3, 0, 1)` followed by a 3×3 Gaussian blur.
   - If the Tasks API cannot be imported in the installed version, fall back to the legacy `mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)`. Use this only as a fallback.
3. **Fixed crop, computed once and used for all frames.** This matters because a per-frame crop makes the face jitter. Selfie Segmenter keeps the whole person (hair, neck, shoulders), so the crop is what limits the output to the head.
   - If `--crop` is set, use it.
   - Otherwise, run the OpenCV Haar cascade (`cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`) on the center frame `r=c=G//2` and take the largest face. Expand the box: width ×1.8, height ×2.2, with its center shifted up by 0.1×face height to include the hair. Clamp it to the image and print the result as `--crop x,y,w,h` so I can pin or adjust it.
   - If no face is found, stop and ask for `--crop`.
4. **Tone, with shared normalization so frames don't flicker when swapped**
   - Convert to grayscale, then apply CLAHE (clipLimit 2.0, tiles 8×8) to the cropped region.
   - Compute the 2nd and 98th luminance percentiles over pixels with alpha > 0.5 on the center frame only. Apply that same linear stretch to every frame, clip to [0, 1], then apply gamma.
   - Apply a light unsharp mask before downsampling so the eyes and pupils survive (amount 0.6, sigma 1.0).
5. **Resize** gray and alpha to `(cols, rows)` with `cv2.INTER_AREA`, where `rows = round(cols * (crop_h / crop_w) * aspect)`.
6. **Map to characters**
   - Density `d = gray` (or `1 - gray` with `--invert`), then `d *= alpha`. Where alpha < 0.5, force a space so the silhouette edge stays crisp.
   - Character index = `round(d * (len(ramp) - 1))`.
   - Every frame has exactly `rows` lines of exactly `cols` characters. Do not strip trailing spaces, because alignment must stay identical across frames.
7. **Horizontal mapping (mirror fix).** OpenCV frames are not mirrored. When I look at the right side of my screen, my face appears turned toward the image's left. On the web page, a cursor on the viewer's right must show a face turned toward the image's right, which is the frame shot while I looked at screen-left. So by default the frame for web cell `(r, c)` is `raw` cell `(r, G-1-c)`. `--no-reverse-x` turns this off. Rows are not flipped.
8. **Shared trim.** Find the bounding box of non-space characters across all frames. Crop every frame to that same box, plus 1 character of padding.
9. **Write `web/frames.js`:**
   ```js
   window.PORTRAIT = {"grid":9,"cols":..,"rows":..,"aspect":0.5,"invert":false,"frames":[ ...G*G strings, row-major, lines joined by "\n" ... ]};
   ```
   Use `json.dumps` so backslashes and quotes are escaped correctly.
10. **Debug output**
    - `debug/contact.jpg`: a G×G grid of the masked, cropped grayscale images in web order (after the mirror fix), each about 160 px wide.
    - `debug/center.txt`: the ASCII center frame.
    - Print frame dimensions and the size of `frames.js` in KB.

## 3. `web/index.html` (single file, loads `frames.js` via `<script src>`)

- `html, body { margin:0; height:100%; background:#000; }` and `pre { color:#fff; }`. If `PORTRAIT.invert` is set, swap to white background with black text. Pure black and white only: no gray text, no color.
- Center the `<pre id="p">` with flexbox. Font stack: `ui-monospace, Menlo, Consolas, "DejaVu Sans Mono", monospace`. Also set `margin:0; user-select:none; letter-spacing:0; white-space:pre`.
- **Aspect-correct sizing:** render the first frame at `font-size:10px` and measure the character width as `pre.getBoundingClientRect().width / cols`. Set `line-height = charWidth / aspect` in px so each cell's width/height matches the value used in `build.py`. Then scale `font-size` (and recompute line-height) so the portrait fits within 92vw × 92vh. Redo this on `resize`.
- **Pointer → cell:**
  - Take `cx, cy` as the center of the `<pre>`.
  - `nx = (x - cx) / (x >= cx ? innerWidth - cx : cx)`, and likewise for `ny`, clamped to [-1, 1]. This way the viewport edges map to the extreme gaze angles.
  - `targetC = round((nx + 1) / 2 * (G - 1))`, `targetR = round((ny + 1) / 2 * (G - 1))`.
  - Listen to `pointermove` and `pointerdown` on `window`. This covers mouse, pen, and touch drag. Set `touch-action:none` on `body`.
  - On `pointerleave` of the document or on `blur`, set the target back to the center cell.
- **Motion:** use one `requestAnimationFrame` loop. Every 30 ms, move the current `(r, c)` at most one cell toward the target on each axis. Change `pre.textContent = frames[r*G + c]` only when the cell changes. The eyes then sweep through the in-between frames instead of teleporting, which looks much better at no real cost.
- No other code. Total JS should be about 50 lines.

## 4. Verification (do these yourself before reporting done)

1. Create the venv, `pip install -r requirements.txt`, and confirm `python -c "import mediapipe, cv2; print(mediapipe.__version__, cv2.__version__)"` runs.
2. **Build smoke test without me:** write a throwaway script `debug/make_fake_raw.py`. It takes one webcam snapshot if a camera is available, or otherwise any frontal-face photo I provide at `debug/face.jpg`, and writes G×G copies to `raw/` with small horizontal/vertical shifts. Run `build.py` on them and check:
   - `web/frames.js` parses (`node -e "require('./web/frames.js')"` fails because it uses `window`, so check with `python -c` by stripping the prefix and running `json.loads`, or define `global.window={}` in node).
   - There are exactly G×G frames, every frame has the same line count, and every line has the same length.
   - The background is blank: the first and last rows of the center frame before trimming are all spaces.
   - Look at `debug/contact.jpg` and `debug/center.txt` and confirm a face is visible and the background is removed.
   - Then delete the fake `raw/` files.
3. Open `web/index.html` with headless Chromium (Playwright if available) at 1280×800. Move the mouse to the four corners and the center, and check that `pre.textContent` differs between corners and matches `frames[G*G//2]` at the center. Also check that the page has no console errors when opened via `file://`.
4. **STOP and hand over to me.** Tell me to run `python capture.py` myself; it needs my face and gaze. After that, run `python build.py`, open the page, and ask me to confirm that the portrait looks toward the cursor horizontally. If it looks away, rebuild with `--no-reverse-x`.

## 5. README must include

- Setup commands, then `python capture.py`, then `python build.py`, then open `web/index.html`.
- Capture tips:
  - Soft frontal light and a plain background.
  - Follow the dot with eyes plus a slight head turn.
  - Don't blink during the ring animation.
  - Use `--only r,c` to redo bad cells.
- Tuning: `--cols` (60–100), `--ramp`, `--gamma`, `--invert`, `--crop`.
- Deploy: copy the `web/` folder to any static host (GitHub Pages, Netlify). `frames.js` is roughly 350–600 KB raw and roughly 50–100 KB gzipped, and hosts compress it automatically.
- Known limitation: the webcam sits above the screen, so the center frame looks slightly below the viewer. To reduce this, sit farther back, or put the camera at eye level behind the screen's center.

## Constraints

- No runtime dependencies in the browser. No build step for the web part.
- Do not commit `raw/`, `models/`, `debug/`, or `.venv/`.
- Keep the code short and readable. Only add features that are listed here.
