// On the page for addresses with nothing at them (content/404/index.typ),
// the snail of the drawing in the corner crawls out of it and wanders slowly
// about the window, behind the text, leaving a fading trail with bits of
// math from the blog along it. Where the drawing is not shown, a small snail
// wanders instead. A click on the page starts another small snail, up to MAX
// snails, or now and then puts down a chanterelle, which stays, and which the
// snails crawl around.
// With reduced motion, there is no snail, nor on narrow screens, where the
// text leaves it no room to wander and the page shows a large drawing instead.
(() => {
  if (matchMedia("(prefers-reduced-motion: reduce), (max-width: 760px)").matches) return;

  const SPEED = 26;      // px per second
  const SCALE = 1.4;     // of the snail, drawn in the 28 × 16 units of the reading snail
  const MARGIN = 70;     // px from the edges of the window, where it turns back
  const CLEAR = 24;      // px that it keeps from the text and the drawings
  const LOOK = 45;       // px ahead that it looks for them
  const TRAIL = 25;      // s that the trail takes to fade away
  const TRAIL_STEP = 3;  // px between the points of a trail
  const BANDS = 8;       // batches of segments of about the same age, drawn at once
  const MAX = 100;       // snails
  const TURN = 0.9;      // rad per second at most, to get away from an edge or the text
  const EASE = 2.5;      // per second, how quickly the turning follows where it should go
  const INTRO_LEFT = 90; // px that the big snail first crawls straight to the left
  const INTRO_TURN = 0.3;  // rad per second as it then curves upward
  const CHANTERELLE = 1 / 6;  // chance that a click puts down a chanterelle instead
  const MAX_CHANTERELLES = 30;
  const MATH_EVERY = 150;  // px of trail between bits of math
  const FONT = "italic 17px et-book, Palatino, Georgia, serif";
  const MATH = [
    "J = PL⁻¹Mᵀ", "ε² = 0", "A = UΣVᵀ", "s ∈ 𝒦", "∂u/∂t = Δu", "Ax = b",
    "∇f(x)", "½xᵀPx + qᵀx", "‖x‖₂ ≤ t", "LLᵀ", "det A ≠ 0", "λₘₐₓ",
    "f(x + ε) = f(x) + f′(x)ε", "u ∈ H¹(Ω)", "O(n)", "∑ᵢ λᵢ = tr A",
  ];

  // The snail of assets/reading-snail.js, facing right, with its foot along y = 15
  const body = new Path2D("M 1.5 15 H 23 Q 27 15 26.6 12.2 Q 26 9.9 23.4 10.4 Q 21.6 10.9 20.2 12.6 L 6 13.2 Q 2.2 13.6 1.5 15 Z");
  const spiral = new Path2D("M 12 8.2 a 1 1 0 0 1 1 1 a 2 2 0 0 1 -2 2 a 3 3 0 0 1 -3 -3 a 4 4 0 0 1 4 -4");
  const stalks = new Path2D("M 23.6 10.6 Q 23.4 8 24.2 5.6 M 25.4 10.8 Q 25.8 8.4 27 6.4");
  // A chanterelle, standing on y = 0 around x = 0: a wavy cap whose rim
  // curls down, narrowing into the stem, with ridges running down from the rim
  const cap = new Path2D("M -14 -20 Q -10 -22.8 -6 -21.6 Q -2.5 -20.4 0 -20.8 Q 4 -21.8 8 -22.6 Q 12 -23.2 14.2 -21 Q 15.6 -19.4 13.4 -18.2 Q 7.5 -15.5 4 -10.5 Q 2.8 -5 2.6 0 L -2.6 0 Q -2.9 -5 -4 -10.5 Q -7.5 -15.5 -13.4 -18 Q -15.6 -19.2 -14 -20 Z");
  const gills = new Path2D("M -12 -18.4 Q -6.5 -14.5 -3 -8.5 M -8 -19.6 Q -4.2 -14.5 -1.8 -9 M -3.6 -20.2 Q -1.6 -14.5 -0.6 -9 M 1.2 -20.5 Q 1 -14.5 0.6 -9 M 5.6 -21 Q 3 -14.5 1.8 -9 M 10.2 -20.6 Q 6.4 -14.5 3 -8.5");
  const ground = new Path2D("M -9 0 H 9");

  const canvas = document.createElement("canvas");
  canvas.className = "wandering-snail";
  canvas.setAttribute("aria-hidden", "true");
  document.body.append(canvas);
  document.body.classList.add("has-wandering-snail");
  const ctx = canvas.getContext("2d");

  const dark = matchMedia("(prefers-color-scheme: dark)");
  const colors = () => dark.matches
    ? { ink: "201, 201, 201", paper: "#151515", shell: "#3a3a3a" }
    : { ink: "61, 61, 61", paper: "#fffff8", shell: "#e6e6df" };

  let width, height;
  const resize = () => {
    const ratio = devicePixelRatio || 1;
    width = innerWidth; height = innerHeight;
    canvas.width = width * ratio; canvas.height = height * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  };
  addEventListener("resize", resize);
  resize();

  // What the snails crawl around, in the window: the navigation, the text,
  // the drawings, those that are shown, and the chanterelles put down by
  // clicks. Measured once a frame.
  let obstacles = [];
  const chanterelles = [];  // { x, y, scale, flip }
  const measure = () => {
    obstacles = [...document.querySelectorAll("header nav, article h1, article p, .forest, .lost-snail")]
      .map((e) => e.getBoundingClientRect()).filter((r) => r.width > 0 && r.height > 0)
      .map((r) => ({ left: r.left - CLEAR, top: r.top - CLEAR, right: r.right + CLEAR, bottom: r.bottom + CLEAR }))
      .concat(chanterelles.map((c) => ({ left: c.x - 16 * c.scale - 10, right: c.x + 16 * c.scale + 10, top: c.y - 24 * c.scale - 10, bottom: c.y + 10 })));
  };
  const inside = (x, y, r) => x > r.left && x < r.right && y > r.top && y < r.bottom;

  // Each snail: where it is, where it heads, and how fast it turns, which
  // drifts at random, so that it crawls in smooth, wandering curves; with
  // its trail, a point every TRAIL_STEP px, and its bits of math. The big
  // snail from the drawing has `sprite`, the element that shows it, and
  // keeps further from the edges and looks further ahead, by `reach` px.
  const snails = [];
  // `flip` mirrors it when it crawls left, so that it is never upside down;
  // `intro` is the opening move of the big snail.
  const spawn = (x, y, now, { heading = Math.random() * 2 * Math.PI, sprite = null, reach = 0, intro = null } = {}) => snails.push({
    x, y, heading, turn: 0, sprite, reach, intro, flip: Math.cos(heading) < 0 ? -1 : 1,
    trail: [{ x, y, t: now }], notes: [],
    travelled: MATH_EVERY / 2, nextMath: Math.floor(Math.random() * MATH.length),
  });

  // A small first snail, somewhere free in the window
  const spawnSmall = () => {
    measure();
    let first = { x: width / 2, y: height - MARGIN };
    for (let tries = 0; tries < 100; tries++) {
      const x = MARGIN + Math.random() * (width - 2 * MARGIN), y = MARGIN + Math.random() * (height - 2 * MARGIN);
      if (!obstacles.some((r) => inside(x, y, r))) { first = { x, y }; break; }
    }
    spawn(first.x, first.y, performance.now());
  };

  // The big first snail: a copy of the snail of the drawing in the corner,
  // in its own svg element, at the same place and size, while the snail of
  // the drawing is hidden. The copy has the drawing's styles and filters, and
  // the snail without the drawing's mirroring, so crawling right, in the
  // snail's own units (tools/forest-svg.py); FOOT is the middle of its foot.
  const BOX = { x: 10, y: 100, width: 104, height: 60 };
  const FOOT = { x: 60, y: 153 };
  const spawnBig = (drawing, original) => {
    const unit = drawing.getBoundingClientRect().width / drawing.viewBox.baseVal.width * 1.1;
    const sprite = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    sprite.setAttribute("class", "forest-drawing wandering-big");
    sprite.setAttribute("aria-hidden", "true");
    sprite.setAttribute("viewBox", `${BOX.x} ${BOX.y} ${BOX.width} ${BOX.height}`);
    sprite.setAttribute("width", BOX.width * unit);
    sprite.setAttribute("height", BOX.height * unit);
    sprite.style.transformOrigin = `${(FOOT.x - BOX.x) * unit}px ${(FOOT.y - BOX.y) * unit}px`;
    sprite.anchor = { x: (FOOT.x - BOX.x) * unit, y: (FOOT.y - BOX.y) * unit };
    for (const part of drawing.querySelectorAll(":scope > style, :scope > defs")) sprite.append(part.cloneNode(true));
    const pen = document.createElementNS("http://www.w3.org/2000/svg", "g");
    pen.setAttribute("filter", "url(#forest-pen)");
    for (const child of original.children) pen.append(child.cloneNode(true));
    sprite.append(pen);
    document.body.append(sprite);
    // It starts where the snail of the drawing is, facing the way it does
    const foot = new DOMPoint(FOOT.x, FOOT.y).matrixTransform(original.getScreenCTM());
    original.style.visibility = "hidden";
    document.body.classList.add("big-snail-wandering");
    spawn(foot.x, foot.y, performance.now(), { heading: Math.PI, sprite, reach: (FOOT.x - BOX.x) * unit, intro: { left: INTRO_LEFT } });
  };

  // snail.js puts the drawing into the page once it has loaded it; if the
  // drawing is not shown, or does not come, a small snail wanders instead
  const forest = document.querySelector(".forest");
  if (!forest || getComputedStyle(forest).display === "none") {
    spawnSmall();
  } else {
    const started = performance.now();
    const wait = () => {
      const drawing = forest.querySelector("svg"), original = drawing && drawing.querySelector("#forest-snail");
      if (original) spawnBig(drawing, original);
      else if (performance.now() - started > 5000) spawnSmall();
      else setTimeout(wait, 100);
    };
    wait();
  }

  // A click on the page, other than on a link or the snail in the corner,
  // starts another snail where it was, up to MAX, or now and then puts down
  // a chanterelle there instead
  document.addEventListener("click", (e) => {
    if (e.target.closest("a, button, .forest")) return;
    if (Math.random() < CHANTERELLE && chanterelles.length < MAX_CHANTERELLES) {
      chanterelles.push({ x: e.clientX, y: e.clientY, scale: 1.5 + Math.random() * 0.6, flip: Math.random() < 0.5 ? -1 : 1 });
    } else if (snails.length < MAX) {
      spawn(e.clientX, e.clientY, performance.now());
    }
  });

  const step = (snail, dt, now) => {
    // Near an edge, turn toward the middle of the window; with the text or a
    // drawing ahead, turn away from the nearest point of it; else drift
    let toward = null, rate = TURN;
    const margin = MARGIN + snail.reach, look = LOOK + snail.reach;
    if (snail.intro) {
      // The big snail's opening: straight to the left, out of the drawing,
      // then a slow curve upward, before it wanders like the others
      if (snail.intro.left > 0) {
        snail.intro.left -= SPEED * dt;
        toward = snail.heading;
      } else {
        toward = -Math.PI / 2;
        rate = INTRO_TURN;
        if (Math.abs(Math.atan2(Math.sin(toward - snail.heading), Math.cos(toward - snail.heading))) < 0.05) snail.intro = null;
      }
    } else if (snail.x < margin || snail.x > width - margin || snail.y < margin || snail.y > height - margin) {
      toward = Math.atan2(height / 2 - snail.y, width / 2 - snail.x);
    } else {
      const aheadX = snail.x + Math.cos(snail.heading) * look, aheadY = snail.y + Math.sin(snail.heading) * look;
      const hit = obstacles.find((r) => inside(aheadX, aheadY, r) || inside(snail.x, snail.y, r));
      if (hit) {
        const nearestX = Math.min(Math.max(snail.x, hit.left), hit.right), nearestY = Math.min(Math.max(snail.y, hit.top), hit.bottom);
        toward = nearestX === snail.x && nearestY === snail.y
          ? Math.atan2(snail.y - (hit.top + hit.bottom) / 2, snail.x - (hit.left + hit.right) / 2)
          : Math.atan2(snail.y - nearestY, snail.x - nearestX);
      }
    }
    // The turning eases toward where it should go, so that the snail curves
    // rather than swerves
    if (toward !== null) {
      const off = Math.atan2(Math.sin(toward - snail.heading), Math.cos(toward - snail.heading));
      const target = Math.max(-rate, Math.min(rate, 2 * off));
      snail.turn += (target - snail.turn) * Math.min(1, EASE * dt);
    } else {
      snail.turn += (Math.random() - 0.5) * 1.2 * dt;
      snail.turn *= Math.pow(0.6, dt);
    }
    snail.heading += snail.turn * dt;
    // Mirrored only once clearly crawling the other way, so that it does not
    // flip back and forth while crawling up or down
    const c = Math.cos(snail.heading);
    if (c < -0.25) snail.flip = -1;
    else if (c > 0.25) snail.flip = 1;
    const distance = SPEED * dt;
    snail.x += Math.cos(snail.heading) * distance;
    snail.y += Math.sin(snail.heading) * distance;
    const end = snail.trail[snail.trail.length - 1];
    if (Math.hypot(snail.x - end.x, snail.y - end.y) >= TRAIL_STEP) snail.trail.push({ x: snail.x, y: snail.y, t: now });

    // A bit of math every so often, beside the trail: far enough to the side
    // that its box clears the line, whichever way the trail runs
    snail.travelled += distance;
    if (snail.travelled >= MATH_EVERY) {
      snail.travelled = 0;
      const text = MATH[snail.nextMath];
      ctx.font = FONT;
      const halfWidth = ctx.measureText(text).width / 2, halfHeight = 9;
      const nx = -Math.sin(snail.heading), ny = Math.cos(snail.heading);
      const away = Math.abs(nx) * halfWidth + Math.abs(ny) * halfHeight + 8;
      const side = Math.random() < 0.5 ? -1 : 1;
      // behind the big snail's tail, rather than beside its shell
      const backX = snail.x - Math.cos(snail.heading) * snail.reach, backY = snail.y - Math.sin(snail.heading) * snail.reach;
      snail.notes.push({ x: backX + nx * away * side, y: backY + ny * away * side, t: now, text });
      snail.nextMath = (snail.nextMath + 1 + Math.floor(Math.random() * 3)) % MATH.length;
    }
    while (snail.trail.length > 1 && now - snail.trail[0].t > TRAIL * 1000) snail.trail.shift();
    while (snail.notes.length && now - snail.notes[0].t > TRAIL * 1000) snail.notes.shift();
  };

  // A trail, fading with its age: its segments are stroked in BANDS batches
  // of about the same age, rather than one by one
  const drawTrail = (trail, now, ink) => {
    let band = null;
    for (let i = 1; i < trail.length; i += 1) {
      const age = Math.min(1, (now - trail[i].t) / (TRAIL * 1000));
      const b = Math.min(BANDS - 1, Math.floor(age * BANDS));
      if (b !== band) {
        if (band !== null) ctx.stroke();
        band = b;
        ctx.strokeStyle = `rgba(${ink}, ${0.35 * (1 - (b + 0.5) / BANDS)})`;
        ctx.beginPath();
        ctx.moveTo(trail[i - 1].x, trail[i - 1].y);
      }
      ctx.lineTo(trail[i].x, trail[i].y);
    }
    if (band !== null) ctx.stroke();
  };

  const drawSnail = (snail, ink, paper, shell) => {
    // Turned the way it crawls, and flipped when it crawls left, so that it
    // is never upside down
    ctx.save();
    ctx.translate(snail.x, snail.y);
    ctx.rotate(snail.heading);
    ctx.scale(1, snail.flip);
    ctx.scale(SCALE, SCALE);
    ctx.translate(-20, -15);
    ctx.strokeStyle = `rgb(${ink})`;
    ctx.fillStyle = paper;
    ctx.lineWidth = 1.1;
    ctx.fill(body); ctx.stroke(body);
    ctx.beginPath(); ctx.arc(12, 8.2, 5.6, 0, 2 * Math.PI);
    ctx.fillStyle = shell; ctx.fill(); ctx.stroke();
    ctx.lineWidth = 1;
    ctx.stroke(spiral); ctx.stroke(stalks);
    ctx.fillStyle = `rgb(${ink})`;
    for (const [x, y] of [[24.2, 5.6], [27, 6.4]]) { ctx.beginPath(); ctx.arc(x, y, 0.9, 0, 2 * Math.PI); ctx.fill(); }
    ctx.restore();
  };

  const drawChanterelle = (c, ink, shell) => {
    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.scale(c.flip * c.scale, c.scale);
    ctx.strokeStyle = `rgb(${ink})`;
    ctx.fillStyle = shell;
    ctx.lineWidth = 1.1;
    ctx.fill(cap); ctx.stroke(cap);
    ctx.lineWidth = 0.5;
    ctx.stroke(gills);
    ctx.lineWidth = 1;
    ctx.stroke(ground);
    ctx.restore();
  };

  const draw = (now) => {
    const { ink, paper, shell } = colors();
    ctx.clearRect(0, 0, width, height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 2;
    for (const snail of snails) drawTrail(snail.trail, now, ink);
    ctx.font = FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const snail of snails) {
      for (const note of snail.notes) {
        const age = (now - note.t) / (TRAIL * 1000);
        ctx.fillStyle = `rgba(${ink}, ${0.6 * (1 - age)})`;
        ctx.fillText(note.text, note.x, note.y);
      }
    }
    for (const c of chanterelles) drawChanterelle(c, ink, shell);
    for (const snail of snails) {
      if (!snail.sprite) { drawSnail(snail, ink, paper, shell); continue; }
      // The big snail's element, turned about the middle of its foot
      const { x, y } = snail.sprite.anchor;
      snail.sprite.style.transform = `translate(${snail.x - x}px, ${snail.y - y}px) rotate(${snail.heading}rad) scale(1, ${snail.flip})`;
    }
  };

  // A tab in the background gets no frames; on its return, the snails pick
  // up where they were rather than jumping ahead
  let last = null;
  const frame = (now) => {
    const dt = last === null ? 0 : Math.min(0.1, (now - last) / 1000);
    last = now;
    measure();
    if (dt > 0) for (const snail of snails) step(snail, dt, now);
    draw(now);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
})();
