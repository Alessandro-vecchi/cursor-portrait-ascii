// ASCII portrait that follows the cursor. Needs frames.js loaded first (window.PORTRAIT).
// Embed: <div data-portrait data-label="..." data-hint="..."></div> sized by its container.
(function () {
  const P = window.PORTRAIT, G = P.grid, mid = Math.floor(G / 2);
  const FONT = 'ui-monospace, Menlo, Consolas, "DejaVu Sans Mono", monospace';
  const LEVELS = 12, STEP_MS = 70, CLICK_MS = 900, DEMO_AFTER_MS = 3000;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Brightness follows the character's position in the ramp: dim dots, bright '@'. The sqrt lifts the
  // midtones, because thin glyphs already look darker than their gray.
  const level = {};
  [...P.ramp].forEach((ch, i) => { level[ch] = Math.round(i / (P.ramp.length - 1) * (LEVELS - 1)); });
  const shade = (l) => {
    const v = Math.round(255 * (0.2 + 0.8 * Math.sqrt(l / (LEVELS - 1))));
    return P.invert ? `rgb(${255 - v},${255 - v},${255 - v})` : `rgb(${v},${v},${v})`;
  };

  function setup(el) {
    const canvas = document.createElement("canvas"), ctx = canvas.getContext("2d");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", el.dataset.label || "ASCII portrait that looks at your cursor");
    canvas.style.cssText = "display:block;margin:auto;touch-action:none;cursor:pointer";
    el.style.position = el.style.position || "relative";
    el.appendChild(canvas);
    const hintText = el.dataset.hint ?? "psst — move your cursor \u{1F440}";
    const hint = document.createElement("span");
    hint.textContent = hintText;
    hint.style.cssText = `position:absolute;left:0;right:0;bottom:4%;text-align:center;font:13px ${FONT};` +
      `color:${P.invert ? "#555" : "#aaa"};transition:opacity .6s;pointer-events:none`;
    if (hintText) el.appendChild(hint);

    let cache = {}, cw = 0, lh = 0, px = 10, dpr = 1, shown = null;
    let r = mid, c = mid, tr = mid, tc = mid, last = 0, leftAt = -1e9, clickUntil = 0, hovering = false;

    function fit() {
      dpr = devicePixelRatio || 1;
      ctx.font = `bold 10px ${FONT}`;
      const w10 = ctx.measureText("M").width;
      const s = Math.min(el.clientWidth / (P.cols * w10), el.clientHeight / (P.rows * w10 / P.aspect));
      px = 10 * s; cw = w10 * s; lh = cw / P.aspect;
      canvas.width = Math.floor(P.cols * cw * dpr); canvas.height = Math.floor(P.rows * lh * dpr);
      canvas.style.width = canvas.width / dpr + "px"; canvas.style.height = canvas.height / dpr + "px";
      canvas.style.marginTop = (el.clientHeight - canvas.height / dpr) / 2 + "px";
      cache = {}; shown = null;
    }

    function render(text) {  // draw a frame once into an offscreen canvas, grouped by gray level
      const off = document.createElement("canvas"), o = off.getContext("2d");
      off.width = canvas.width; off.height = canvas.height;
      o.scale(dpr, dpr);
      o.font = `bold ${px}px ${FONT}`;
      o.textBaseline = "top";
      const lines = text.split("\n"), byLevel = [];
      lines.forEach((line, y) => { for (let x = 0; x < line.length; x++) {
        const l = level[line[x]];
        if (line[x] !== " " && l !== undefined) (byLevel[l] = byLevel[l] || []).push(x, y, line[x]);
      } });
      byLevel.forEach((cells, l) => {
        o.fillStyle = shade(l);
        for (let i = 0; i < cells.length; i += 3) o.fillText(cells[i + 2], cells[i] * cw, cells[i + 1] * lh);
      });
      return off;
    }

    function show(key, text) {
      if (key === shown || !text) return;
      shown = el.dataset.frame = key;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(cache[key] = cache[key] || render(text), 0, 0);
    }

    function aim(e) {
      const b = canvas.getBoundingClientRect(), cx = b.left + b.width / 2, cy = b.top + b.height / 2;
      const n = (v, ctr, size) => Math.max(-1, Math.min(1, (v - ctr) / (v >= ctr ? size - ctr : ctr)));
      tc = Math.round((n(e.clientX, cx, innerWidth) + 1) / 2 * (G - 1));
      tr = Math.round((n(e.clientY, cy, innerHeight) + 1) / 2 * (G - 1));
      hovering = Math.abs(e.clientX - cx) < b.width * 0.2 && Math.abs(e.clientY - cy) < b.height * 0.2;
      leftAt = null;
      hint.style.opacity = 0;
    }

    function tick(t) {
      if (leftAt !== null && t - leftAt > DEMO_AFTER_MS) {  // nobody is steering: look around on a slow loop
        hovering = false;
        tc = still ? mid : Math.round((Math.sin(t / 1400) + 1) / 2 * (G - 1));
        tr = still ? mid : Math.round((Math.sin(t / 2300 + 1) + 1) / 2 * (G - 1));
      }
      if (t - last >= STEP_MS) {
        last = t;
        r += Math.sign(tr - r); c += Math.sign(tc - c);
      }
      if (t < clickUntil && P.click) show("click", P.click);
      else if (hovering && P.hover) show("hover", P.hover);
      else show(r * G + c, P.frames[r * G + c]);
      requestAnimationFrame(tick);
    }

    addEventListener("pointermove", aim);
    addEventListener("pointerdown", aim);
    el.addEventListener("pointerdown", () => { clickUntil = performance.now() + CLICK_MS; });
    document.documentElement.addEventListener("pointerleave", () => { leftAt = performance.now(); });
    addEventListener("blur", () => { leftAt = performance.now(); });
    new ResizeObserver(fit).observe(el);
    fit();
    requestAnimationFrame(tick);
  }

  const start = () => document.querySelectorAll("[data-portrait]").forEach(setup);
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", start); else start();
})();
