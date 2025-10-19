---
layout: distill
title: Polar Express
description: approximating matrix functions using just matrix-matrix products
tags: linear-algebra matrix-functions GPUs
giscus_comments: true
date: 2025-10-03 12:06:00
featured: true
# citation: true

authors:
  - name: Mikkel Paltorp
    url: "https://mipals.github.io"
    affiliations:
      name: Technical University of Denmark

bibliography: 2025-01-01-polar-express.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).

toc:
    - name: Some background
    - name: Polar Express for the matrix sign function
    - name: Polar express for projection onto the semidefinite cone

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


Polar normalization is a generalization of the matrix sign function, where non-square matrices is allowed. In short the idea is to take the singular value decomposition (SVD) of a matrix and set all singular values to one. That is given $M=U\Sigma V^\top$ then the polar normalization is given by

$$
    \text{polar}(M) = U\text{diag}(\text{sign}(\sigma_1), \dots, \text{sign}(\sigma_k))V^\top = UV^\top.
$$

The polar normalization is a key ingredient in the recently proposed Muon algorithm for optimizing large language models <d-cite key="jordan2024muon"></d-cite>. In short the steps of the Muon algorithm is <d-cite key="bernstein2025manifolds"></d-cite>

$$
\begin{aligned}
    M_t     &= \beta M_{t-1} + (1 - \beta) G_t, \\
    W_{t+1} &= W_t - \lambda \text{polar}(M_t),
\end{aligned}
$$

A distinct disadvantage is that the polar step in the naive implementation requires the computation of an SVD which is computationally expensive and not very GPU friendly (it is primarily memory bound rather than compute bound). Recently the polar express algorithm have been proposed as a method for approximating the matrix sign function using just matrix-matrix products <d-cite key="polar-express"></d-cite>. The reason for the interest in this method is that matrix-matrix products are very efficient to compute on modern hardware such as GPUs.


Inspired by the work of polar express a matrix-matrix multiplication based approach to projecting matrices back onto the cone of semidefinite matrices have been proposed <d-cite key="polar-express-sdp"></d-cite>. That is computing the solution to

$$
    \Pi_{\mathbb{S}_+^n}(X) 
    = \arg\min_{Y \in \mathbb{S}_+^n} \|Y - X\|_F 
    = U \text{diag}(\text{ReLu}(\lambda_1), \dots, \text{ReLu}(\lambda_n)) U^T
$$

While the idea here is not new (a similar fixed-point approach was suggested in <d-cite key="fixedpoint"></d-cite>) it does provide a new and faster converging approach.


### Some background
While some of the following is seemingly quite technical it is important to understand that the overall idea behind the methods are quite simple: Approximating a function $f$ using a composition of polynomials e.g.

$$
    f(x) \approx p_k(p_{k-1}(\dots p_2(p_1(x))\dots)).
$$

**As the approximation is simply polynomial it can be evaluated using just additions and multiplications.**

For the matrix polar function what we aim to approximate is the sign function given by

$$
    \text{sign}(x) = 
    \begin{cases}
        1, & x > 0, \\
        0, & x = 0, \\
        -1, & x < 0.
    \end{cases}
$$

Previous methods for approximating the above is to set $p_1=p_2=\dots=p_k$. However, one can instead look for an optimal series of polynomials. The polar express method suggest using polynomials of the form


#### Newton-Schulz
In Newton-Schulz method all the polynomial are the same. For order 3 and 5 they look as follows

$$
    p(x) = \frac{3}{2}x - \frac{1}{2}x^3, \quad p(x) = (15x - 10x^3 + 3x^5)/8.
$$

The Newton-Schulz method converges quickly for values close to $\pm 1$. However, if the values are close to zero the approximation converges slowly poor. The slow convergence is exactly what the Polar Express method aims to fix. 


{% details Newton-Schulz %}
```julia
using LinearAlgebra
```
{% enddetails %}



### The Polar Express algorithm
The basic idea of the polar express algorithm is as follows: Given a set of $T$ polynomials $\{p_i(x) = a_1^ix + a_3^ix^3 + a_5x^5\}_{i=1}^T$, can we find optimal set of coefficients that mimizes the maximum error over a given interval $[\ell,u]$? That is we want to solve

$$
    p^\star = \underset{\substack{p = p_T \circ p_{T-1} \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_d^{\text{odd}}}}{\text{argmin}} \; \max_{\substack{M \in \mathbb{R}^{m \times n} \\ \sigma(M) \subset [\ell, u]}} \; \| \operatorname{polar}(M) - p(M) \|_2.
$$

In <d-cite key="polar-express"></d-cite> they show that the above is solved by a greedy approach, with the caveat is the convergence is only guaranteed if the smallest singular value is larger than $\ell$, which we do not know in advance. A comment here is that the main focus in the paper is on $\texttt{Float16}$. Here the machine precision is $\epsilon_\text{mach} = 2^-7 = 0.0078125$. Therefore, they suggest to set $\ell = 10^{-3}$ and $u=1$. In addition, they provide a few modifications in order to make the method more robust. These additions are skipped here, but can be found in the original paper.

{% details Polar Express for sign computation %}
```julia
using LinearAlgebra
```
{% enddetails %}




<!-- A comment here is that the main focus in the paper is on $\texttt{Float16}$. Here the machine precision is $\epsilon_\text{mach} = 2^-7 = 0.0078125$. Therefore, they suggest to set $\ell = 10^{-3}$ and $u=1$. In addition, in order to avoid numerical issues of they replace the polynomials so that they instead do $p_t(x/1.01)$, which makes singular values converge to 0.999998 instead of to 1. -->


<!-- It turns out the above problem can be solved as follows. Let $d$ be odd and define $\ell_1 = \ell$ and $u_1 = u$. For $t = 1,\ldots,T$ define

$$
    p_t = \arg\min_{\substack{p \in \mathbb{P}_d^{\text{odd}}}} \; \max_{x \in [\ell_t,u_t]} \; |1 - p(x)| \tag{9}
$$

$$
    \ell_{t+1} = \min_{x \in [\ell_t,u_t]} p_t(x), \qquad
    u_{t+1} = \max_{x \in [\ell_t,u_t]} p_t(x)
$$

The resulting composition $p^\star := p_T \circ p_{T-1} \circ \cdots \circ p_1$ is optimal and the error is given by:

$$
    \max_{x \in [\ell,u]} |1 - p^\star(x)| =
    \min_{\substack{p = p_T \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_d^{\text{odd}}}} \; \max_{x \in [\ell,u]} |1 - p(x)| = 1 - \ell_{T+1}. \tag{10}
$$

Furthermore, the new error, lower and upper bounds can be computed through

$$
    \ell_{t+1} = p_t(\ell_t), \qquad
    u_{t+1} = 2 - \ell_{t+1}, 
$$

and

$$
\max_{x \in [\ell_t,u_t]} |1 - p_t(x)| = 1 - \ell_{t+1}. \tag{11}
$$ -->




### Polar Express for projection onto the positive semidefinite cone

The **positive semidefinite cone** is the set of all real symmetric $n\times n$ positive semidefinite matrices. That is

$$
    \mathbb{S}_+^n := \lbrace X \in \mathbb{R}^{n\times n}\ |\ X\succeq 0 \rbrace.
$$

Algorithms that aim to solve semidefinite programming (SDP) off course need to ensure that the solution actually satisfy the positive semidefinite constraint. In interior point methods this is done by only taking steps for which the next iterate remain inside the cone, while first order methods most often rely on projecting back onto the set after taking a step. The ideas of the polar express method can be used to create a similar method of projecting back onto the PSD set by using just matrix-matrix products. 


$$
\begin{aligned}
    &\inf_{f_1,\ldots,f_T} \; \max_{x \in [-1,1]} \; \big| f_T \circ f_{T-1} \circ \cdots \circ f_1(x) - f_{\mathrm{ReLU}}(x) \big| \\
    \quad \text{subject to} \quad &f_t \in \mathbb{R}_{d_t}[x], \; t = 1,\ldots,T.
\end{aligned}
$$

where $T$ is the total number of polynomials in the composition, $d_t$ is the degree of the $t$'th polynomial which are fixed in advance as part of a "matrix-matrix" multiplication budget. Directly solving the above is challenging, and instead a two-stage approach is used. The idea is to realize that we can write the ReLu function 

$$
    \text{ReLu}(x) = \frac{1}{2}x\left(1 + \text{sign}(x)\right)
$$

The first step is then to find a polynomial approximation of the sign function on the interval $[-1,-\epsilon]\cup[\epsilon,1]$ for some small $\epsilon > 0$. That is we solve

$$
\begin{aligned}
&\min_{\substack{f_1,\ldots,f_T}} \; \max_{x \in [-1,-\epsilon] \cup [\epsilon,1]} \; \big| f_T \circ f_{T-1} \circ \cdots \circ f_1(x) - f_{\mathrm{sign}}(x) \big|\\
\quad \text{subject to} \quad & f_t \in \mathbb{R}_{d_t}^{\text{odd}}[x], \; t = 1,\ldots,T.
\end{aligned}
$$

The second step is then two refine the coefficients of the polynomials found in the first step by minimizing the error to the ReLu function on the entire interval $[-1,1]$. 

$$
\ell(f_T,\ldots,f_1) := \max_{x \in [-1,1]} \Big| \frac{1}{2} x \big( 1 + f_T \circ f_{T-1} \circ \cdots \circ f_1(x) \big) - f_{\mathrm{ReLU}}(x) \Big|.
$$

{% details Polar Express for ReLu computation %}
```julia
using LinearAlgebra
```
{% enddetails %}


