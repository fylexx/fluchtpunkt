// ---------- Reveal on scroll ----------
function reveal() {
  const reveals = document.querySelectorAll(".reveal");
  const windowHeight = window.innerHeight;
  const revealPoint = 150;
  for (const el of reveals) {
    const top = el.getBoundingClientRect().top;
    el.classList.toggle("active", top < windowHeight - revealPoint);
  }
}
window.addEventListener("scroll", reveal);

// ---------- Mobile menu ----------
const menuToggle = document.querySelector(".nav-toggle");
const navPanel = document.getElementById("nav-links-sub");

menuToggle?.addEventListener("click", () => {
  const isOpen = navPanel.style.left === "0px";
  navPanel.style.left = isOpen ? "-200px" : "0px";
  document.body.style.overflowY = isOpen ? "visible" : "hidden";
});

// ---------- AI image detector ----------
const uploadBtn = document.getElementById("button-one");
const fileInput = document.getElementById("image-input");
const results = document.getElementById("results");
const vpText = document.getElementById("vp-text");
const canvasLines = document.getElementById("canvas-lines");
const canvasVp = document.getElementById("canvas-vp");

// OpenCV.js loads asynchronously and initializes WASM after the JS file is fetched.
// We need to wait for both: the script onload event AND the runtime to be live.
function whenCvReady(cb) {
  const finish = () => {
    if (cv.Mat) cb();
    else cv.onRuntimeInitialized = cb;
  };
  if (window.__cvLoaded) finish();
  else window.addEventListener("opencv-loaded", finish, { once: true });
}

whenCvReady(() => {
  uploadBtn.disabled = false;
  uploadBtn.textContent = "Upload Image";
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Analyzing…";
  vpText.textContent = "";
  try {
    const img = await loadImage(file);
    runPipeline(img);
  } catch (err) {
    vpText.textContent = `Error: ${err.message}`;
    results.hidden = false;
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload another";
    fileInput.value = "";
  }
});

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Could not decode image")); };
    img.src = url;
  });
}

// Downscale very large images so the in-browser pipeline stays fast
const MAX_DIM = 1600;
function imageToCvMat(img) {
  const scale = Math.min(1, MAX_DIM / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  off.getContext("2d").drawImage(img, 0, 0, w, h);
  return cv.imread(off); // RGBA
}

function runPipeline(img) {
  const src = imageToCvMat(img);
  const gray = new cv.Mat();
  const edges = new cv.Mat();
  const lines = new cv.Mat();

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
    cv.Canny(gray, edges, 80, 200);

    const h = gray.rows, w = gray.cols;
    const minLen = Math.round(0.08 * Math.max(h, w));
    cv.HoughLinesP(edges, lines, 1, Math.PI / 180, 100, minLen, 20);

    const segs = [];
    for (let i = 0; i < lines.rows; i++) {
      const o = i * 4;
      segs.push([
        lines.data32S[o], lines.data32S[o + 1],
        lines.data32S[o + 2], lines.data32S[o + 3],
      ]);
    }

    if (segs.length === 0) {
      vpText.textContent = "No line segments detected.";
      results.hidden = false;
      return;
    }

    drawTopSegments(src, segs, w, h);
    drawVanishingPoint(src, segs);
    results.hidden = false;
  } finally {
    src.delete(); gray.delete(); edges.delete(); lines.delete();
  }
}

// ---- Output 1: top 25 longest segments, extended to image borders ----
function drawTopSegments(src, segs, w, h) {
  const lengths = segs.map(([a, b, c, d]) => Math.hypot(c - a, d - b));
  const order = [...segs.keys()].sort((i, j) => lengths[j] - lengths[i]);
  const N = Math.min(25, segs.length);
  const topIdx = order.slice(0, N);
  const topLens = topIdx.map((i) => lengths[i]);
  const minL = Math.min(...topLens), maxL = Math.max(...topLens);

  const vis = src.clone();
  const blue = new cv.Scalar(0, 0, 255, 255);
  for (const i of topIdx) {
    const ext = extendToBorders(segs[i], w, h);
    if (!ext) continue;
    const t = Math.round(interp(lengths[i], minL, maxL, 3, 12));
    cv.line(vis, new cv.Point(ext[0][0], ext[0][1]),
                  new cv.Point(ext[1][0], ext[1][1]), blue, t);
  }
  cv.imshow(canvasLines, vis);
  vis.delete();
}

// ---- Output 2: vanishing point on padded canvas + VP text ----
function drawVanishingPoint(src, segs) {
  const { vp, support } = fitVanishingPoint(segs);
  if (!vp || support < Math.max(4, 0.25 * segs.length)) {
    vpText.textContent = `No reliable vanishing point (support ${support}/${segs.length}).`;
    const ctx = canvasVp.getContext("2d");
    ctx.clearRect(0, 0, canvasVp.width, canvasVp.height);
    return;
  }
  const xy = vpXY(vp);
  vpText.textContent = xy
    ? `VP at (${xy[0].toFixed(1)}, ${xy[1].toFixed(1)})  |  inlier support ${support}/${segs.length}`
    : `VP at infinity, direction (${vp[0].toFixed(3)}, ${vp[1].toFixed(3)})  |  inlier support ${support}/${segs.length}`;

  const canvas = drawVpHonest(src, vp, segs);
  cv.imshow(canvasVp, canvas);
  canvas.delete();
}

// ---- Geometry helpers ----
function interp(x, a, b, ya, yb) {
  if (b === a) return (ya + yb) / 2;
  return ya + (yb - ya) * (x - a) / (b - a);
}

function extendToBorders(seg, w, h, big = 1e6) {
  let [x1, y1, x2, y2] = seg;
  let dx = x2 - x1, dy = y2 - y1;
  const n = Math.hypot(dx, dy);
  if (n === 0) return null;
  dx /= n; dy /= n;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  return clipLine([cx - dx * big, cy - dy * big],
                  [cx + dx * big, cy + dy * big], w, h);
}

// Liang-Barsky clip of an arbitrary line segment to rect (0,0,w-1,h-1)
function clipLine(p1, p2, w, h) {
  let [x1, y1] = p1, [x2, y2] = p2;
  const dx = x2 - x1, dy = y2 - y1;
  const p = [-dx, dx, -dy, dy];
  const q = [x1, (w - 1) - x1, y1, (h - 1) - y1];
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return null;
    } else {
      const t = q[i] / p[i];
      if (p[i] < 0) { if (t > t1) return null; if (t > t0) t0 = t; }
      else          { if (t < t0) return null; if (t < t1) t1 = t; }
    }
  }
  return [
    [Math.round(x1 + t0 * dx), Math.round(y1 + t0 * dy)],
    [Math.round(x1 + t1 * dx), Math.round(y1 + t1 * dy)],
  ];
}

// ---- Vanishing-point estimation ----
function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
const norm3 = (a) => Math.hypot(a[0], a[1], a[2]);

function lineFromSeg(s) {
  const l = cross3([s[0], s[1], 1], [s[2], s[3], 1]);
  const n = Math.hypot(l[0], l[1]) + 1e-12;
  return [l[0] / n, l[1] / n, l[2] / n];
}

// Perpendicular distance from each segment's endpoint to the line through its
// midpoint and the homogeneous VP. Stable even when v is at infinity.
function vpErrors(segs, v) {
  const errs = new Float64Array(segs.length);
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i];
    const mx = (s[0] + s[2]) / 2, my = (s[1] + s[3]) / 2;
    const lp = cross3([mx, my, 1], v);
    const nrm = Math.hypot(lp[0], lp[1]) + 1e-12;
    errs[i] = Math.abs(lp[0] * s[0] + lp[1] * s[1] + lp[2]) / nrm;
  }
  return errs;
}

const vpXY = (v) => (Math.abs(v[2]) < 1e-9 ? null : [v[0] / v[2], v[1] / v[2]]);

function fitVanishingPoint(segs, iters = 3000, thresh = 3.0) {
  if (segs.length < 2) return { vp: null, support: 0 };
  const L = segs.map(lineFromSeg);

  let bestV = null, bestMask = null, bestInl = -1;
  for (let it = 0; it < iters; it++) {
    const i = Math.floor(Math.random() * L.length);
    let j = Math.floor(Math.random() * (L.length - 1));
    if (j >= i) j++;
    const raw = cross3(L[i], L[j]);
    const n = norm3(raw);
    if (n < 1e-12) continue;
    const v = [raw[0] / n, raw[1] / n, raw[2] / n];
    const errs = vpErrors(segs, v);
    let k = 0;
    for (let m = 0; m < errs.length; m++) if (errs[m] < thresh) k++;
    if (k > bestInl) {
      bestInl = k;
      bestV = v;
      bestMask = Array.from(errs, (e) => e < thresh);
    }
  }
  if (!bestV) return { vp: null, support: 0 };

  // Refine: smallest right singular vector of the inlier line matrix.
  // Equivalent to the smallest-eigenvalue eigenvector of L^T L.
  const inliers = L.filter((_, i) => bestMask[i]);
  const refined = smallestEigVec3x3(buildLTL(inliers));
  const rn = norm3(refined) + 1e-12;
  const refV = [refined[0] / rn, refined[1] / rn, refined[2] / rn];
  const errs2 = vpErrors(segs, refV);
  let support = 0;
  for (let i = 0; i < errs2.length; i++) if (errs2[i] < thresh) support++;
  return { vp: refV, support };
}

function buildLTL(L) {
  const M = new Float64Array(9);
  for (const l of L) {
    M[0] += l[0] * l[0]; M[1] += l[0] * l[1]; M[2] += l[0] * l[2];
    M[4] += l[1] * l[1]; M[5] += l[1] * l[2];
    M[8] += l[2] * l[2];
  }
  M[3] = M[1]; M[6] = M[2]; M[7] = M[5];
  return M;
}

// Jacobi eigendecomposition for a symmetric 3x3 matrix, returning the
// eigenvector with the smallest eigenvalue.
function smallestEigVec3x3(A) {
  const a = Array.from(A);
  const V = [1, 0, 0, 0, 1, 0, 0, 0, 1]; // identity; columns = eigenvectors
  for (let iter = 0; iter < 50; iter++) {
    let p = 0, q = 1, max = Math.abs(a[1]);
    if (Math.abs(a[2]) > max) { p = 0; q = 2; max = Math.abs(a[2]); }
    if (Math.abs(a[5]) > max) { p = 1; q = 2; max = Math.abs(a[5]); }
    if (max < 1e-12) break;

    const app = a[p * 3 + p], aqq = a[q * 3 + q], apq = a[p * 3 + q];
    const theta = (aqq - app) / (2 * apq);
    const sign = theta >= 0 ? 1 : -1;
    const t = sign / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    a[p * 3 + p] = app - t * apq;
    a[q * 3 + q] = aqq + t * apq;
    a[p * 3 + q] = 0; a[q * 3 + p] = 0;

    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue;
      const arp = a[r * 3 + p], arq = a[r * 3 + q];
      a[r * 3 + p] = c * arp - s * arq; a[p * 3 + r] = a[r * 3 + p];
      a[r * 3 + q] = s * arp + c * arq; a[q * 3 + r] = a[r * 3 + q];
    }
    for (let r = 0; r < 3; r++) {
      const vrp = V[r * 3 + p], vrq = V[r * 3 + q];
      V[r * 3 + p] = c * vrp - s * vrq;
      V[r * 3 + q] = s * vrp + c * vrq;
    }
  }
  const eig = [a[0], a[4], a[8]];
  let idx = 0;
  if (eig[1] < eig[idx]) idx = 1;
  if (eig[2] < eig[idx]) idx = 2;
  return [V[idx], V[3 + idx], V[6 + idx]];
}

// ---- Padded VP visualization (port of draw_vp_honest) ----
function drawVpHonest(srcRGBA, v, segs, thresh = 3.0, padFrac = 0.2, maxPadMult = 4.0) {
  const h = srcRGBA.rows, w = srcRGBA.cols;
  const errs = vpErrors(segs, v);
  const cap = Math.round(maxPadMult * Math.max(h, w));
  const xy = vpXY(v);

  let padL, padT, padR, padB;
  if (!xy) {
    padL = padT = padR = padB = Math.round(padFrac * Math.min(h, w));
  } else {
    const [px, py] = xy;
    padL = Math.min(cap, Math.round(Math.max(0, -px)     + padFrac * w));
    padT = Math.min(cap, Math.round(Math.max(0, -py)     + padFrac * h));
    padR = Math.min(cap, Math.round(Math.max(0,  px - w) + padFrac * w));
    padB = Math.min(cap, Math.round(Math.max(0,  py - h) + padFrac * h));
  }

  const canvas = new cv.Mat();
  cv.copyMakeBorder(srcRGBA, canvas, padT, padB, padL, padR,
                    cv.BORDER_CONSTANT, new cv.Scalar(30, 30, 30, 255));
  const ox = padL, oy = padT;
  const W = canvas.cols, H = canvas.rows;

  const green = new cv.Scalar(0, 200, 0, 255);
  const red   = new cv.Scalar(255, 0, 0, 255);
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i];
    let dx = s[2] - s[0], dy = s[3] - s[1];
    const n = Math.hypot(dx, dy) + 1e-9; dx /= n; dy /= n;
    const mx = (s[0] + s[2]) / 2 + ox, my = (s[1] + s[3]) / 2 + oy;
    const clipped = clipLine(
      [mx - dx * 1e7, my - dy * 1e7],
      [mx + dx * 1e7, my + dy * 1e7], W, H
    );
    if (!clipped) continue;
    cv.line(canvas,
      new cv.Point(clipped[0][0], clipped[0][1]),
      new cv.Point(clipped[1][0], clipped[1][1]),
      errs[i] < thresh ? green : red, 2);
  }

  const yellow = new cv.Scalar(255, 255, 0, 255);
  const black  = new cv.Scalar(0, 0, 0, 255);
  if (xy && xy[0] + ox >= 0 && xy[0] + ox < W && xy[1] + oy >= 0 && xy[1] + oy < H) {
    const c = new cv.Point(Math.round(xy[0] + ox), Math.round(xy[1] + oy));
    cv.circle(canvas, c, 36, yellow, -1);
    cv.circle(canvas, c, 36, black, 3);
  } else {
    let d;
    if (!xy) {
      const n = Math.hypot(v[0], v[1]) + 1e-9;
      d = [v[0] / n, v[1] / n];
    } else {
      const dx = (xy[0] + ox) - W / 2, dy = (xy[1] + oy) - H / 2;
      const n = Math.hypot(dx, dy) + 1e-9;
      d = [dx / n, dy / n];
    }
    const c0x = W / 2, c0y = H / 2, r = 0.45 * Math.min(W, H);
    cv.arrowedLine(canvas,
      new cv.Point(Math.round(c0x), Math.round(c0y)),
      new cv.Point(Math.round(c0x + d[0] * r), Math.round(c0y + d[1] * r)),
      yellow, 5, cv.LINE_8, 0, 0.04);
  }
  return canvas;
}
