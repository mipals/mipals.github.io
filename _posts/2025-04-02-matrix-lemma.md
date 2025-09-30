---
layout: distill
title: General Sherman-Morrison-Woodbury Identity
description: 
tags: linear-algebra gaussian-processes low-rank-approximation
giscus_comments: true
date: 2024-05-20 12:00:00
featured: false
# citation: true

authors:
  - name: Mikkel Paltorp
    url: "https://mipals.github.io"
    affiliations:
      name: Technical University of Denmark

bibliography: 2025-05-25-continuous-matrix-factorization.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Some identities

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

# General Sherman-Morrison-Woodbury Identity
In this note we derive the general form of the Sherman-Morrison-Woodury identity. In addition, we derive an extension of the formula that is used in Kernel ridge regression, often without any explanation.

## Some identities

We start by introducing three useful identities.

\definition{**Identity 1** The first identity comes from the standard mathematical trick of adding zero as $I + P - P = I$. Using this it follows that

$$
\begin{aligned}
    (I + P)^{-1} 
    &= (I + P)^{-1}(I + P - P)\\
    &= I - (I + P)^{-1}P
\end{aligned}
$$
}
\definition{**Identity 2**
For invertible A, but possible rectangular $B, C$, and $D$ one can show using identity 1 that

$$
\begin{aligned}
    (A + BCD)^{-1} 
    &= (A(I + A^{-1}BCD)^{-1})\quad\quad\quad\quad\quad\quad\quad\quad \text{Using invertible A}\\
    &= (I + A^{-1}BCD)^{-1}A^{-1}\\
    &= \left(I - (I + A^{-1}BCD)^{-1}A^{-1}BCD\right)A^{-1}\quad \text{Using identity 1}
\end{aligned}
$$

This identity will be the first step in derivation of the general Sherman-Morrison-formula. 
}
\definition{**Identity 3** 
First notice that we have

$$
\begin{equation}
    P + PQP = P(I + QP) = (I + PQ)P
\end{equation}
$$

Moving the parentheses to the corresponding other sides of the equation result in

$$
\begin{equation}
    (I + PQ)^{-1}P = P(I + QP)^{-1}
\end{equation}
$$

This identity is extensively used to move matrices from one side of an inverse to the other.
}

For an invertible $A$ and $C$ one can, by repeadly using identity 3, show that the following hold

$$
\begin{aligned}
    (A + BCD)^{-1} BC
    &= (A(I + A^{-1}BCD))^{-1}BC\quad\ \quad\ \ \ \text{Using Invertible A}\\
    &= (I + A^{-1}BCD)^{-1}A^{-1}BC\\
    &= A^{-1}(I + BCDA^{-1})^{-1}BC\quad\ \quad\ \ \ \text{Using Identity 3}\\
    &= A^{-1}B(I + CDA^{-1}B)^{-1}C\quad\ \quad\ \ \ \text{Using Identity 3}\\
    &= A^{-1}B(C(C^{-1} + DA^{-1}B))^{-1}C\quad \text{Using Invertible C}\\
    &= A^{-1}B(C^{-1} + DA^{-1}B)^{-1}C^{-1}C\\
    &= A^{-1}B(C^{-1} + DA^{-1}B)^{-1}
\end{aligned}
$$

This result is use in e.g. [kernel ridge regression](https://web2.qatar.cmu.edu/~gdicaro/10315-Fall19/additional/welling-notes-on-kernel-ridge.pdf) as it makes it possible to possible change the size of the matrix being inverted. In particular if $\Phi \in \mathbb{R}^{d \times n}$ we have that

$$
\begin{aligned}
    w 
    &= (\lambda I_d + \Phi\Phi^\top)^{-1}\Phi y\\
    &= \lambda^{-1}\Phi\left(I_n + \Phi^\top \lambda^{-1} \Phi\right)^{-1}y\\
    &= \Phi\left(\lambda\left(I_n + \Phi^\top \lambda^{-1} \Phi\right)\right)^{-1}y\\
    &= \Phi\left(\lambda I_n + \Phi^\top \Phi\right)^{-1}y.
\end{aligned}
$$

From which it can be seen that can compute $w$ by either inverting a $d\times d$ matrix of a $n\times n$ matrix. 

We are now ready to introduce the general Sherman-Morrison-Woodbury identity as
\definition{**General Sherman-Morrison-Woodbury identity** 
$$
\begin{aligned}
    (A + BCD)^{-1} 
    &= A^{-1} - (I + A^{-1}BCD)^{-1}BCDA^{-1}\quad\quad\ \text{   Using Identity 2}\\
    &= A^{-1} - A^{-1}(I + BCDA^{-1})^{-1}BCDA^{-1}\quad \text{Using Identity 3}\\
    &= A^{-1} - A^{-1}B(I + CDA^{-1}B)^{-1}CDA^{-1}\quad \text{Using Identity 3}\\
    &= A^{-1} - A^{-1}BC(I + DA^{-1}BC)^{-1}DA^{-1}\quad \text{Using Identity 3}\\
    &= A^{-1} - A^{-1}BCD(I + A^{-1}BCD)^{-1}A^{-1}\quad \text{Using Identity 3}\\
    &= A^{-1} - A^{-1}BCDA^{-1}(I + BCDA^{-1})^{-1}\quad \text{Using Identity 3}\\
\end{aligned}
$$
}

Either of the equalities might be of use, depending on the chosen application. In particular in the case of $C$ being invertible we arrive at the more known Sherman-Morrison-Woodbury identity as

$$
\begin{aligned}
(A + BCD)^{-1} 
    &= A^{-1} - A^{-1}B(C(C^{-1} + DA^{-1}B))^{-1}CDA^{-1}\\
    &= A^{-1} - A^{-1}B(C^{-1} + DA^{-1}B)^{-1}C^{-1}CDA^{-1}\\
    &= A^{-1} - A^{-1}B(C^{-1} + DA^{-1}B)^{-1}DA^{-1}\\
\end{aligned}
$$



