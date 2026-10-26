// The pixel avatar on the about page, redrawn as one particle per pixel. When
// the page loads, the avatar is built up from low-rank approximations, with
// the rank growing to full over a few seconds. Then the pixels react to the
// mouse, and return. A click switches the reaction: the avatar is redrawn at
// a rank that grows as the mouse moves from the top left corner to the bottom
// right, or the pixels swirl around the mouse, ripple, or crumble and fall.
// Without JavaScript, or with reduced motion, the plain image stays.
(() => {
  const img = document.querySelector(".pixel-avatar img");
  if (!img || matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const CELL = 10;       // size of one pixel of the pixel art, in image pixels
  const RADIUS = 45;     // reach of the swirl and the crumble, in screen pixels
  const PUSH = 14;       // how hard the swirl pushes, in image pixels per frame
  const WAVE = 28;       // wavelength of the ripple, in screen pixels
  const REACH = 70;      // how far the ripple spreads, in screen pixels
  const HEIGHT = 3;      // height of the ripple, in screen pixels
  const GRAVITY = 0.25;  // of the crumble, in screen pixels per frame per frame
  const FALL = 1500;     // ms that a crumbled pixel lies, at least, before it returns
  const BUILD = 3000;    // ms that the avatar takes to build up to full rank
  const EFFECTS = ["rank", "swirl", "ripple", "crumble"];

  // The singular value decomposition A = U S Vᵀ of a square n × n matrix
  // (row-major), by one-sided Jacobi rotations: the columns of A are rotated
  // until they are orthogonal, and then they are the columns of U S. Returns
  // the columns of U S and of V, by decreasing singular value.
  const svd = (A, n) => {
    const US = Float64Array.from(A), V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    for (let sweep = 0; sweep < 50; sweep++) {
      let rotated = false;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          let alpha = 0, beta = 0, gamma = 0;
          for (let i = 0; i < n; i++) {
            const up = US[i * n + p], uq = US[i * n + q];
            alpha += up * up; beta += uq * uq; gamma += up * uq;
          }
          if (Math.abs(gamma) <= 1e-12 * Math.sqrt(alpha * beta)) continue;
          rotated = true;
          const zeta = (beta - alpha) / (2 * gamma);
          const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.hypot(1, zeta));
          const c = 1 / Math.hypot(1, t), s = c * t;
          for (const M of [US, V]) {
            for (let i = 0; i < n; i++) {
              const mp = M[i * n + p], mq = M[i * n + q];
              M[i * n + p] = c * mp - s * mq;
              M[i * n + q] = s * mp + c * mq;
            }
          }
        }
      }
      if (!rotated) break;
    }
    const norm = (j) => Math.hypot(...Array.from({ length: n }, (_, i) => US[i * n + j]));
    const order = Array.from({ length: n }, (_, j) => j).sort((a, b) => norm(b) - norm(a));
    const column = (M, j) => Float64Array.from({ length: n }, (_, i) => M[i * n + j]);
    return { US: order.map((j) => column(US, j)), V: order.map((j) => column(V, j)) };
  };

  const start = () => {
    const size = img.naturalWidth, n = Math.ceil(size / CELL);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", img.alt);
    const ctx = canvas.getContext("2d");

    // One particle per opaque cell, colored by the center of the cell
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, size, size).data;
    const pixels = [];
    for (let y = 0; y < size; y += CELL) {
      for (let x = 0; x < size; x += CELL) {
        const i = 4 * ((y + CELL / 2) * size + x + CELL / 2);
        if (data[i + 3] < 128) continue;
        pixels.push({
          homeX: x, homeY: y, x, y,
          vx: 0, vy: 0, fallen: 0,                   // fallen: until when it lies, in ms
          speed: 0.04 + Math.random() * 0.08,       // share of the way home per frame
          cell: (y / CELL) * n + x / CELL,
          rgb: [data[i], data[i + 1], data[i + 2]],
          color: `rgb(${data[i]}, ${data[i + 1]}, ${data[i + 2]})`,
        });
      }
    }
    // The colors of the avatar at each rank, computed when first shown. The
    // cells around the avatar get its mean color, which blurs least into it.
    let factors = null;
    const ranked = new Map();
    const colorsAt = (rank) => {
      if (!factors) {
        factors = [0, 1, 2].map((c) => {
          const mean = pixels.reduce((sum, p) => sum + p.rgb[c], 0) / pixels.length;
          const A = new Float64Array(n * n).fill(mean);
          for (const p of pixels) A[p.cell] = p.rgb[c];
          return svd(A, n);
        });
      }
      if (!ranked.has(rank)) {
        ranked.set(rank, pixels.map((p) => {
          const row = Math.floor(p.cell / n), col = p.cell % n;
          const rgb = factors.map(({ US, V }) => {
            let value = 0;
            for (let k = 0; k < rank; k++) value += US[k][row] * V[k][col];
            return Math.round(Math.min(255, Math.max(0, value)));
          });
          return `rgb(${rgb.join(", ")})`;
        }));
      }
      return ranked.get(rank);
    };
    // The decomposition takes a moment, so the image stays until it is done
    colorsAt(1);
    img.replaceWith(canvas);

    let effect = 0;     // index into EFFECTS of the reaction to the mouse
    let mouse = null;   // in image pixels
    let center = null;  // the center of the ripple, where the mouse was last
    let strength = 0;   // of the ripple, growing to 1 under the mouse
    let frame = null;
    let built = null;   // when the build-up started, at the first frame
    const draw = (time) => {
      built ??= time;
      ctx.clearRect(0, 0, size, size);
      const scale = size / canvas.getBoundingClientRect().width;
      const hover = mouse && EFFECTS[effect];
      strength += ((hover === "ripple" ? 1 : 0) - strength) * 0.08;
      if (strength < 0.01) strength = 0;
      // The rank grows with the distance from the top left corner, from 1
      // there to n at the bottom right, evenly on a log scale, as the visible
      // change is in the first few
      const reach = hover === "rank" && Math.hypot(Math.max(0, mouse.x), Math.max(0, mouse.y)) / (Math.SQRT2 * size);
      // While the avatar builds up, the rank grows the same way with time
      const building = time - built < BUILD;
      const rank = hover === "rank" ? Math.round(n ** Math.min(1, reach))
        : building ? Math.round(n ** ((time - built) / BUILD)) : 0;
      const colors = rank ? colorsAt(rank) : null;
      let moving = false;
      pixels.forEach((p, index) => {
        if (time < p.fallen) {
          // A crumbled pixel falls, and bounces off the edges of the canvas
          p.vy += GRAVITY * scale;
          p.x += p.vx; p.y += p.vy;
          if (p.y > size - CELL) { p.y = size - CELL; p.vy *= -0.4; p.vx *= 0.8; }
          if (p.x < 0 || p.x > size - CELL) { p.x = Math.min(size - CELL, Math.max(0, p.x)); p.vx *= -0.5; }
        } else {
          p.x += (p.homeX - p.x) * p.speed;
          p.y += (p.homeY - p.y) * p.speed;
        }
        if (hover === "swirl" || hover === "crumble") {
          const dx = p.x + CELL / 2 - mouse.x, dy = p.y + CELL / 2 - mouse.y;
          const d = Math.hypot(dx, dy), radius = RADIUS * scale;
          if (d > 0 && d < radius) {
            if (hover === "swirl") {
              const f = PUSH * (1 - d / radius) / d;
              p.x += (dx - 0.8 * dy) * f;              // outward, with a swirl
              p.y += (dy + 0.8 * dx) * f;
            } else if (time >= p.fallen) {
              // Knocked loose: a small push away from the mouse, and down it goes
              p.vx = (dx / d + Math.random() - 0.5) * 2 * scale;
              p.vy = (dy / d - Math.random()) * 2 * scale;
              p.fallen = time + FALL * (1 + Math.random());
            }
          }
        }
        if (time >= p.fallen && Math.abs(p.x - p.homeX) + Math.abs(p.y - p.homeY) < 0.3) {
          p.x = p.homeX; p.y = p.homeY;
        } else {
          moving = true;
        }
        // The ripple only shifts where a pixel is drawn: a wave running out
        // from the center, fading with the distance
        let x = p.x, y = p.y;
        if (strength > 0) {
          const dx = p.x + CELL / 2 - center.x, dy = p.y + CELL / 2 - center.y;
          const d = Math.hypot(dx, dy) || 1;
          const wave = Math.sin(2 * Math.PI * d / (WAVE * scale) - time / 120);
          const shift = HEIGHT * scale * strength * wave * Math.exp(-d / (REACH * scale));
          x += dx / d * shift;
          y += dy / d * shift;
        }
        ctx.fillStyle = colors ? colors[index] : p.color;
        ctx.fillRect(Math.round(x), Math.round(y), CELL, CELL);
      });
      // The rank, in the empty top left corner, in the color of the text, for
      // as long as the rank is the reaction to the mouse: full rank when the
      // mouse is away
      const label = rank || (EFFECTS[effect] === "rank" && n);
      if (label) {
        const style = getComputedStyle(canvas);
        ctx.fillStyle = style.color;
        ctx.font = `${15 * scale}px ${style.fontFamily}`;
        ctx.textBaseline = "top";
        ctx.fillText(`rank ${label}`, 2 * scale, 2 * scale);
      }
      // Stop once the avatar is built, every pixel is home and the ripple has
      // faded; the mouse starts it again
      frame = building || moving || mouse || strength > 0 ? requestAnimationFrame(draw) : null;
    };
    const run = () => { if (!frame) frame = requestAnimationFrame(draw); };

    const track = (clientX, clientY) => {
      const r = canvas.getBoundingClientRect();
      mouse = center = { x: (clientX - r.left) * size / r.width, y: (clientY - r.top) * size / r.height };
      run();
    };

    // A click switches to the next reaction
    canvas.addEventListener("click", (e) => {
      effect = (effect + 1) % EFFECTS.length;
      track(e.clientX, e.clientY);
    });
    canvas.addEventListener("mousemove", (e) => track(e.clientX, e.clientY));
    canvas.addEventListener("touchmove", (e) => { e.preventDefault(); track(e.touches[0].clientX, e.touches[0].clientY); }, { passive: false });
    for (const type of ["mouseleave", "touchend", "touchcancel"]) {
      canvas.addEventListener(type, () => { mouse = null; });
    }
    run();
  };

  if (img.complete) start(); else img.addEventListener("load", start);
})();
