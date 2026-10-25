// The pixel avatar on the about page, redrawn as one particle per pixel: the
// pixels fly in when the page loads, swirl or ripple around the mouse (a
// click switches between the two), and return.
// Without JavaScript, or with reduced motion, the plain image stays.
(() => {
  const img = document.querySelector(".pixel-avatar img");
  if (!img || matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const CELL = 10;       // size of one pixel of the pixel art, in image pixels
  const RADIUS = 45;     // reach of the swirl, in screen pixels
  const PUSH = 14;       // how hard the swirl pushes, in image pixels per frame
  const WAVE = 28;       // wavelength of the ripple, in screen pixels
  const REACH = 70;      // how far the ripple spreads, in screen pixels
  const HEIGHT = 3;      // height of the ripple, in screen pixels

  const start = () => {
    const size = img.naturalWidth;
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
        const angle = Math.random() * 2 * Math.PI;  // start far outside
        const distance = size * (0.6 + Math.random());
        pixels.push({
          homeX: x, homeY: y,
          x: size / 2 + Math.cos(angle) * distance,
          y: size / 2 + Math.sin(angle) * distance,
          speed: 0.04 + Math.random() * 0.08,       // share of the way home per frame
          color: `rgb(${data[i]}, ${data[i + 1]}, ${data[i + 2]})`,
        });
      }
    }
    img.replaceWith(canvas);

    let hover = "swirl"; // the effect under the mouse, "swirl" or "ripple"
    let mouse = null;   // in image pixels
    let center = null;  // the center of the ripple, where the mouse was last
    let strength = 0;   // of the ripple, growing to 1 under the mouse
    let frame = null;
    const draw = (time) => {
      ctx.clearRect(0, 0, size, size);
      const scale = size / canvas.getBoundingClientRect().width;
      strength += ((mouse && hover === "ripple" ? 1 : 0) - strength) * 0.08;
      if (strength < 0.01) strength = 0;
      let moving = false;
      for (const p of pixels) {
        p.x += (p.homeX - p.x) * p.speed;
        p.y += (p.homeY - p.y) * p.speed;
        if (mouse && hover === "swirl") {
          const dx = p.x + CELL / 2 - mouse.x, dy = p.y + CELL / 2 - mouse.y;
          const d = Math.hypot(dx, dy), radius = RADIUS * scale;
          if (d > 0 && d < radius) {
            const f = PUSH * (1 - d / radius) / d;
            p.x += (dx - 0.8 * dy) * f;                // outward, with a swirl
            p.y += (dy + 0.8 * dx) * f;
          }
        }
        if (Math.abs(p.x - p.homeX) + Math.abs(p.y - p.homeY) < 0.3) {
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
        ctx.fillStyle = p.color;
        ctx.fillRect(Math.round(x), Math.round(y), CELL, CELL);
      }
      // Stop once every pixel is home and the ripple has faded; the mouse
      // starts it again
      frame = moving || mouse || strength > 0 ? requestAnimationFrame(draw) : null;
    };
    const run = () => { if (!frame) frame = requestAnimationFrame(draw); };

    const track = (clientX, clientY) => {
      const r = canvas.getBoundingClientRect();
      mouse = center = { x: (clientX - r.left) * size / r.width, y: (clientY - r.top) * size / r.height };
      run();
    };

    // A click switches between the swirl and the ripple
    canvas.addEventListener("click", (e) => {
      hover = hover === "swirl" ? "ripple" : "swirl";
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
