window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    macros: {
            bA: "{\\mathbf{A}}",
            bB: "{\\mathbf{B}}",
            bC: "{\\mathbf{C}}",
            bF: "{\\mathbf{F}}",
            bF: "{\\mathbf{F}}",
            bG: "{\\mathbf{G}}",
            bH: "{\\mathbf{H}}",
            bI: "{\\mathbf{I}}",
            bK: "{\\mathbf{K}}",
            bL: "{\\mathbf{L}}",
            bM: "{\\mathbf{M}}",
            bN: "{\\mathbf{N}}",
            bP: "{\\mathbf{P}}",
            bQ: "{\\mathbf{Q}}",
            bR: "{\\mathbf{R}}",
            bS: "{\\mathbf{S}}",
            bD: "{\\mathbf{D}}",
            bT: "{\\mathbf{T}}",
            bU: "{\\mathbf{U}}",
            bV: "{\\mathbf{V}}",
            bW: "{\\mathbf{W}}",
            bX: "{\\mathbf{X}}",
            bY: "{\\mathbf{Y}}",
            bZ: "{\\mathbf{Z}}",
            bx: "{\\mathbf{x}}",
            by: "{\\mathbf{y}}",
            bz: "{\\mathbf{z}}",
            bt: "{\\mathbf{t}}",
            bu: "{\\mathbf{u}}",
            bv: "{\\mathbf{v}}",
            bw: "{\\mathbf{w}}",
            bi: "{\\mathbf{i}}",
            bj: "{\\mathbf{j}}",
            bh: "{\\mathbf{h}}",
            bk: "{\\mathbf{k}}",
            bl: "{\\mathbf{l}}",
            bbm: "{\\mathbf{m}}",
            bn: "{\\mathbf{n}}",
            bo: "{\\mathbf{o}}",
            bp: "{\\mathbf{p}}",
            bq: "{\\mathbf{q}}",
            br: "{\\mathbf{r}}",
            bs: "{\\mathbf{s}}",
            bzero: "{\\mathbf{0}}",
            bff: "{\\mathbf{f}}",
            bb: "{\\mathbf{b}}",
            be: "{\\mathbf{e}}",
            ba: "{\\mathbf{a}}",
            bd: "{\\mathbf{d}}",
            bc: "{\\mathbf{c}}",
            dbu: "{\\mathbf{\\dot{u}}}",
            dby: "{\\mathbf{\\dot{y}}}",
            dbx: "{\\mathbf{\\dot{x}}}",
            dbX: "{\\mathbf{\\dot{X}}}",
            dr: "{\\mathrm{d}}",
            tr: "{\\text{tr}}",
            cov: "{\\text{cov}}",
            diag: "{\\text{diag}}",
            det: "{\\text{det}}",
            blkdiag: "{\\text{blkdiag}}",
            tril: "{\\text{tril}}",
            triu: "{\\text{triu}}",
            ind: "{\\mathcal{ind}}"
        }
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
};
