// A small snail at the top of the window of a blog post, which comes out once
// the reader has scrolled to the title and crawls from left to right as they
// scroll through the post, leaving a trail.
// Included by post-template in content/blog/post.typ.
(() => {
  const WIDTH = 28;  // of the snail, in pixels

  const bar = document.createElement("div");
  bar.className = "reading-snail away";
  bar.setAttribute("aria-hidden", "true");
  bar.innerHTML = `<div class="reading-snail-trail"></div>
    <svg viewBox="0 0 28 16" width="${WIDTH}" height="16">
      <path class="body" d="M 1.5 15 H 23 Q 27 15 26.6 12.2 Q 26 9.9 23.4 10.4 Q 21.6 10.9 20.2 12.6 L 6 13.2 Q 2.2 13.6 1.5 15 Z"/>
      <circle class="shell" cx="12" cy="8.2" r="5.6"/>
      <path class="line" d="M 12 8.2 a 1 1 0 0 1 1 1 a 2 2 0 0 1 -2 2 a 3 3 0 0 1 -3 -3 a 4 4 0 0 1 4 -4"/>
      <path class="line" d="M 23.6 10.6 Q 23.4 8 24.2 5.6 M 25.4 10.8 Q 25.8 8.4 27 6.4"/>
      <circle class="eye" cx="24.2" cy="5.6" r="0.9"/>
      <circle class="eye" cx="27" cy="6.4" r="0.9"/>
    </svg>`;
  document.body.append(bar);
  const trail = bar.querySelector(".reading-snail-trail"), snail = bar.querySelector("svg");

  // How far the reader is through the post, from 0 when the title reaches
  // the top of the window to 1 at the bottom of the page. Above the title,
  // the snail stays away, so that it does not cover the navigation.
  const title = document.querySelector("article h1");
  let frame = null;
  const update = () => {
    frame = null;
    const start = title ? title.getBoundingClientRect().top + scrollY : 0;
    const scrollable = document.documentElement.scrollHeight - innerHeight - start;
    bar.classList.toggle("away", scrollable <= 0 || scrollY < start);
    const progress = Math.min(1, Math.max(0, (scrollY - start) / scrollable));
    const x = progress * (bar.clientWidth - WIDTH);
    trail.style.width = `${x + 4}px`;
    snail.style.transform = `translateX(${x}px)`;
  };
  const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
  addEventListener("scroll", schedule, { passive: true });
  addEventListener("resize", schedule);
  update();
})();
