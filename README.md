# cursor-portrait-ascii

A grayscale ASCII portrait that turns its head to look at your cursor. The page is a static lookup table
of pre-rendered text frames, one per head pose on a G×G grid (default 5×5 = 25), plus two reaction faces:
one for a click and one for hovering on the face. There's no computer vision at runtime and no
browser dependencies.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # only mediapipe; it pulls numpy + opencv-contrib-python
python -c "import mediapipe, cv2; print(mediapipe.__version__, cv2.__version__)"
```

Use a Python version that has a `mediapipe` wheel. Don't install `opencv-python` alongside, because it
clashes with the `opencv-contrib-python` that mediapipe already pulls in. On macOS, give your terminal
Camera permission (System Settings → Privacy & Security → Camera).

## Run

```sh
python capture.py        # fullscreen: frame your face, SPACE, follow the dot, then 2 reaction shots
python build.py          # raw/ -> web/frames.js, plus debug/contact.jpg and debug/center.txt
open web/index.html      # works straight from file://
```

`build.py` prints the automatic face crop as `--crop x,y,w,h`. Pass it back in to pin the crop or adjust it.
If the portrait looks *away* from the cursor horizontally, rebuild with `--no-reverse-x`.

## Capture tips

- Use bright, soft, frontal light and a plain background. Light from one side leaves half the face in
  the dark characters.
- **Turn your whole head** toward the dot and exaggerate it. Eye-only movement is just 1–2 characters wide
  in ASCII.
- Keep the same distance from the screen the whole time. `build.py` warns about any shot where your face is
  more than 10% bigger or smaller than in the center shot.
- Don't blink while the ring shrinks onto the dot.
- Reactions: "Look SURPRISED!" is shown on click, "Big GRIN!" on hover. Look at the camera, press SPACE,
  and hold the face through the 3-2-1.
- Redo bad shots with `python capture.py --only 1,3 0,4 click` (row,col, where row 0 = top and col 0 =
  left, or a reaction name).
- Check `debug/contact.jpg` after building. The last row holds the reaction shots.

## Put it on your website

Copy `web/frames.js` and `web/portrait.js` next to your page, then add:

```html
<div data-portrait style="width: 420px; height: 480px"
     data-label="ASCII portrait of me that looks at your cursor"></div>
<script src="frames.js"></script>
<script src="portrait.js"></script>
```

- The portrait fits inside the element's size and follows the cursor anywhere on the page.
- `data-hint` sets the caption shown until the first mouse move (default "psst — move your cursor 👀").
  `data-hint=""` hides it.
- Until someone moves the mouse, and 3 s after the pointer leaves the page, the face looks around on its own.
  With the "reduce motion" system setting it stays still.
- The element's background is up to you. The characters are light gray to white (or dark with `--invert`).
- `frames.js` is roughly 250–350 KB at 5×5 and 140 columns (a lot less gzipped). Static hosts
  (GitHub Pages, Netlify, …) compress it automatically.

## Tuning (`build.py`)

| Flag | Default | Notes |
|---|---|---|
| `--cols` | 140 | More columns = more detail and a bigger `frames.js` |
| `--ramp` | `" .,:;i1tfLCG08@"` | Characters from lowest to highest density. Letters beat `-=~+`, which draw stripes |
| `--gamma` | 0.8 | Values < 1 brighten the midtones, values > 1 darken them |
| `--invert` | off | Dark characters for a light background |
| `--crop` | auto | `x,y,w,h` in raw pixels |
| `--aspect` | 0.5 | Character cell width/height, read by the page from `frames.js` |
| `--grid` | 5 | Must match the `--grid` used for capture |

## Known limitation

The webcam sits above the screen, so the center frame looks slightly below the viewer. To reduce this,
sit farther back, or put the camera at eye level behind the center of the screen.
