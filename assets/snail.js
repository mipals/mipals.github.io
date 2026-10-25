// The snail of the drawing in the corner (assets/forest.svg) turns its eye
// stalks toward the cursor, and pulls them in when the cursor comes close.
// Clicked, it flinches, pulling its stalks in for a moment, and then says
// something about the page: one of the page's lines (snail-says in
// config.typ), a random post on the blog index, or else the page's title.
// Once per visit, shortly after the first page loads, it says one of the
// lines unasked, as a hint that it talks.
// The drawing is inlined so that it can move; without JavaScript, or with
// reduced motion, it stays a still image.
(() => {
  const img = document.querySelector(".forest img");
  if (!img || matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const LENGTH = 17;   // of an eye stalk, in drawing units
  const NEAR = 30;     // the stalks shrink as the cursor comes this close
  const EASE = 0.15;   // share of the way to the target per frame
  const FLINCH = 200;  // ms that the stalks stay in after a click, before it speaks
  const GAP = 14;      // px between the drawing and the bubble above it
  const CLEAR = 6;     // px between the line down from the bubble and the
                       // highest the eyes reach
  const SAY = 5000;    // ms that a line stays up
  const GREET = 1200;  // ms after the first page of a visit loads before it speaks

  // Lines for the home page, of which a couple join each round of its own
  const GENERAL = [
    "Some say I'm slow. I say I take small step sizes.",
    "Carrying my whole home around. Talk about memory overhead.",
    "Don't tell anyone where the chanterelles are.",
    "Chanterelle season is August to October. Not that I'm counting.",
    "These mushrooms have ridges, not gills. Ask me how I know.",
    "I converge. Eventually.",
    "Gradient descent: small steps, always downhill. Basically me with a sense of direction.",
    "My favorite matrix is lower triangular.",
    "Sparsity is beautiful.",
    "Some call it standing still. I call it convergence.",
  ];
  const shuffled = (list) => list.map((x) => [Math.random(), x]).sort((a, b) => a[0] - b[0]).map((p) => p[1]);

  // The page's own lines: its snail-says lines, the posts on the blog index,
  // or else its title
  const ownLines = () => {
    const own = [...document.querySelectorAll(".snail-says p")].map((p) => p.innerHTML);
    if (own.length) return own;
    if (location.pathname === "/blog/") {
      const posts = [...document.querySelectorAll("article li > a")];
      const phrases = ["Have you read <a>?", "I liked <a>.", "Try <a> next."];
      return posts.map((a, i) => phrases[i % phrases.length].replace("<a>", a.outerHTML));
    }
    const title = document.querySelector("article h1");
    return [title ? `Enjoying <i>${title.textContent}</i>?` : "hej!"];
  };

  // Lines for the moment: on the about page the reader's time of day and the
  // night in Copenhagen, and anywhere chanterelle season and a reader back
  // for another visit
  let returning = false;
  const momentLines = () => {
    const now = new Date(), hour = now.getHours();
    const copenhagen = +new Intl.DateTimeFormat("en-GB", { hour: "numeric", hourCycle: "h23", timeZone: "Europe/Copenhagen" }).format(now);
    const lines = [];
    if (location.pathname === "/about/") {
      if (hour >= 22 || hour < 5) lines.push("Up late? So am I. Snails love the night.");
      if (hour >= 5 && hour < 9) lines.push("Morning! The dew is perfect today.");
      if (copenhagen < 6) lines.push("It's past midnight in Copenhagen. Mikkel is asleep. Probably.");
    }
    if (now.getMonth() >= 7 && now.getMonth() <= 9) lines.push("It's chanterelle season! Don't tell anyone.");
    if (returning) lines.push("Welcome back! Same snail as last time.");
    return lines;
  };

  // A round of lines in a shuffled order, starting over when used up. On a
  // blog post, only the post's own; elsewhere also those for the moment, and
  // on the home page two of the general ones, with one of the page's own
  // first.
  const onPost = /^\/blog\/[^/]+\/[^/]+\//.test(location.pathname);
  let queue = [];
  const nextLine = () => {
    if (!queue.length) {
      const own = shuffled(ownLines());
      const general = location.pathname === "/" ? shuffled(GENERAL).slice(0, 2) : [];
      queue = onPost ? own : [...shuffled([...own.slice(1), ...momentLines(), ...general]), own[0]];
    }
    return queue.pop();
  };

  fetch(img.src).then((r) => r.text()).then((text) => {
    const svg = new DOMParser().parseFromString(text, "image/svg+xml").documentElement;
    svg.setAttribute("aria-hidden", "true");
    img.replaceWith(svg);

    // The bubble above the drawing, and the line from it down to the eyes,
    // behind the bubble so that it ends at the bubble's edge
    const bubble = document.createElement("div");
    bubble.className = "snail-bubble";
    bubble.setAttribute("role", "status");
    const tail = document.createElement("div");
    tail.className = "snail-tail";
    svg.parentElement.append(tail, bubble);

    // Each stalk grows from a fixed base on the head
    const stalks = [1, 2].map((i) => {
      const path = svg.querySelector(`#forest-stalk-${i}`), eye = svg.querySelector(`#forest-eye-${i}`);
      const [x, y] = path.getAttribute("d").match(/-?[\d.]+/g).map(Number);
      const tipX = +eye.getAttribute("cx"), tipY = +eye.getAttribute("cy");
      const angle = Math.atan2(tipY - y, tipX - x);
      return { path, eye, x, y, angle, length: LENGTH, rest: angle, target: { angle, length: LENGTH } };
    });

    const drawStalk = (s) => {
      const tipX = s.x + Math.cos(s.angle) * s.length, tipY = s.y + Math.sin(s.angle) * s.length;
      // The stalk leaves the head upward and bends toward its tip
      s.path.setAttribute("d", `M ${s.x} ${s.y} Q ${s.x} ${s.y - s.length / 2} ${tipX} ${tipY}`);
      s.eye.setAttribute("cx", tipX);
      s.eye.setAttribute("cy", tipY);
    };

    const show = (shown) => {
      bubble.classList.toggle("shown", shown);
      tail.classList.toggle("shown", shown);
    };
    const say = () => {
      bubble.innerHTML = nextLine();
      // Above the drawing, with a line from its bottom edge, a little in from
      // its left end, straight down toward the eyes: above the stalks' bases,
      // ending above the highest point that the eyes reach on fully grown
      // stalks, wherever the stalks are now
      const box = svg.parentElement.getBoundingClientRect();
      const reach = stalks.map((s) => new DOMPoint(s.x, s.y - LENGTH - 2).matrixTransform(s.path.getScreenCTM()));
      const x = reach.reduce((sum, p) => sum + p.x, 0) / reach.length - box.left;
      const top = Math.min(...reach.map((p) => p.y));
      bubble.style.left = `${Math.max(0, x - 20)}px`;
      bubble.style.bottom = `${box.height + GAP}px`;
      tail.style.left = `${x}px`;
      tail.style.bottom = `${box.bottom - top + CLEAR}px`;
      tail.style.height = `${top - box.top + GAP - CLEAR + 2}px`;
      show(true);
      clearTimeout(quiet);
      quiet = setTimeout(() => show(false), SAY);
    };

    // When the snail was last clicked, and whether it has something to say
    // once its stalks are out again
    let clicked = -Infinity, speaking = false, quiet = null, frame = null;
    const step = (now) => {
      const flinching = now - clicked < FLINCH;
      let moving = flinching;
      for (const s of stalks) {
        const length = flinching ? 0.25 * LENGTH : s.target.length;
        s.angle += (s.target.angle - s.angle) * EASE;
        s.length += (length - s.length) * EASE;
        if (Math.abs(s.target.angle - s.angle) > 0.002 || Math.abs(length - s.length) > 0.02) moving = true;
        drawStalk(s);
      }
      // Done flinching: time to say something
      if (speaking && !flinching) {
        speaking = false;
        say();
      }
      frame = moving ? requestAnimationFrame(step) : null;
    };
    const run = () => { if (!frame) frame = requestAnimationFrame(step); };

    document.addEventListener("mousemove", (e) => {
      // The cursor in the coordinates of the snail, which the drawing mirrors
      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(stalks[0].path.getScreenCTM().inverse());
      for (const s of stalks) {
        let angle = Math.atan2(p.y - s.y, p.x - s.x);
        // Measured from straight up, so that a cursor below the snail turns
        // the stalks to the nearer side, not through the snail; they lean
        // back no further than up and backward, and forward a little below
        // the horizontal
        if (angle > Math.PI / 2) angle -= 2 * Math.PI;
        s.target.angle = Math.min(Math.max(angle, -Math.PI + 0.6), 0.25);
        const d = Math.hypot(p.x - s.x, p.y - s.y);
        s.target.length = LENGTH * Math.min(Math.max(d / NEAR, 0.35), 1);
      }
      run();
    });
    // Off the page, the stalks return to rest
    document.documentElement.addEventListener("mouseleave", () => {
      for (const s of stalks) s.target = { angle: s.rest, length: LENGTH };
      run();
    });
    // A line unasked, once per visit; where the browser keeps no session
    // storage, the snail just stays quiet until clicked
    // A reader is back if an earlier visit left a mark in local storage
    try {
      if (!sessionStorage.getItem("snail-greeted")) {
        sessionStorage.setItem("snail-greeted", "yes");
        sessionStorage.setItem("snail-returning", localStorage.getItem("snail-visited") ? "yes" : "");
        localStorage.setItem("snail-visited", "yes");
        setTimeout(() => { if (!speaking) say(); }, GREET);
      }
      returning = sessionStorage.getItem("snail-returning") === "yes";
    } catch {}
    svg.querySelector("#forest-snail").addEventListener("click", () => {
      show(false);
      clicked = performance.now();
      speaking = true;
      run();
    });
  });
})();
