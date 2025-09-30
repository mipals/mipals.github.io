---
layout: distill
title: Associative Scan
description: and its state-space models
tags: linear-algebra state-space-models semiseparable-matrices
giscus_comments: true
date: 2025-07-30 12:06:00
featured: true
# citation: true

authors:
  - name: Mikkel Paltorp
    url: "https://mipals.github.io"
    affiliations:
      name: Technical University of Denmark

bibliography: 2018-12-22-distill.bib

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

Recently a plethora of methods within machine learning have made use of state-space models (SSMs) to model sequences<d-cite key="gu2022efficientlymodelinglongsequences,gu2024mambalineartimesequencemodeling,dao2024a"></d-cite>. A big reason for their success is their usage of the associative scan operation to parallelize the state computation on the GPU. In my opinion this is a neat trick that was poorly explained in the original papers (although Appendix B in <d-cite key="dao2024a"></d-cite> does explain it for the scalar case), and I will therefore in this blog post explain it in more detail. The aim is to show that the transition of a (linear) state-space model 

$$
\begin{equation}
    \begin{aligned}
        \bh_i &= \bA_i \bh_{i-1} + \bB_i \bx_i, \\
        \by_i &= \bC_i \bh_i
    \end{aligned},
\end{equation} 
$$

is associative<d-footnote>Associativity simply mean that we can apply the operator (denoted by $\bullet$) in arbitrary order, i.e. that $a \bullet (b \bullet c) = (a \bullet b) \bullet c$. Examples of associative operators are addition and multiplication.</d-footnote>. In short this means that the states of a (linear) state-space model up until time step $T$ can be computed in parallel using an associative scan operation. For details on the associative operation open the example box below.

{% details Example: Cumulative sum using associative scan %}
The (parallel) associative scan, as the name suggest, is a way to apply an associative operator in parallel. The simplest of such operator is addition, which is associative because we can switch around the order of computation (i.e $a + (b + c) = (a + b) + c$). In Jax the associative scan is implemented in the accurately named *lax.associative_scan* function while in Julia it is denoted by the *accumulate* function. If we supply the add operator the resulting computation will be equivalent with the cumulative sum, e.g. in Python

```python
>>> lax.associative_scan(jnp.add, jnp.arange(0, 4))
Array([0, 1, 3, 6], dtype=int32)
```
or in Julia
```julia
julia> accumulate(+, 0:3)
5-element Vector{Int64}:
  0
  1
  3
  6
```

While the above computation is trivial and easily computed by looping from the first to last element in the vector, the main idea of the associative scan is that the operator can be applied pairwise in parallel. That means if enough processors is available the computation can be done in $O(\log n)$ time rather than the trivial implementation of $O(n)$. 
{% enddetails %}


The first step in showing the associativity of the state-space model is to define the transition of the state-space models using matrix multiplication (which is associative) by embedding the transition into a larger matrix $\bs_k$ as follows

$$
\bs_k = 
\begin{bmatrix}
    \bA_k & \bB_k \bx_k \\
    \bzero & 1
\end{bmatrix},
\quad \bs_0 = \begin{bmatrix}
    \bzero & \bh_0 \\
    \bzero & 1
\end{bmatrix}
$$

Using the definition of $\bs_k$ the state transition from state $k-1$ to state $k$ can be computed using matrix multiplication as

$$
\bs_k\bs_{k-1} = \begin{bmatrix}
    \bA_k\bA_{k-1} & \bA_k(\bB_{k-1}\bx_{k-1}) + \bB_k\bx_k \\
    \bzero & 1
\end{bmatrix}.
$$

Using this we can compute the $i$th state of the state-space model as
 <!-- <d-footnote>Th notation for the product here is a bit tricky. We aim to perform the product from the </d-footnote> -->

$$
\begin{equation}
    \bx_i = 
    \begin{bmatrix}\bI & \bzero \end{bmatrix}
    \left(\prod_{k=i}^0 \bs_k\right)
    \begin{bmatrix}\bzero \\ 1\end{bmatrix}.
\end{equation}
$$

Given that the cumulative product can be computed using the associative scan operator the full dynamics can be computed as

$$
\begin{equation}
    \begin{aligned}
        \bp_i &= \text{associative_scan}(\bs_i, \text{init} = \bs_0)\\
        \by_i &= \bC \bh_i = \begin{bmatrix}\bC & \bzero \end{bmatrix} \bp_i \begin{bmatrix}\bzero \\ 1\end{bmatrix}.
    \end{aligned}
\end{equation} 
$$

While the above works, it can be simplified slightly. As we are really only interested in what happens in top block (as the top right block contain $\bx_i$) we can instead define elements by just the top row, i.e. instead define the states as $\bs_k = \begin{bmatrix}\bA_k & \bB_k \bx_k \end{bmatrix}$ and then define the associative operator (denoted by $\bullet$) by how the top row propagates, i.e.

$$
\begin{equation}
    \bs_k \bullet \bs_{k-1} = \begin{bmatrix} \bA_k \bA_{k-1} & \bA_k (\bB_{k-1} \bx_{k-1}) + \bB_k \bx_k\end{bmatrix}.
\end{equation}
$$

The SSM stages can then be computed as $\bp_i = \bs_i \bullet \bp_{i-1}$ with $\bs_0 = \begin{bmatrix} \bzero & \bh_0 \end{bmatrix}$. 

As a final remark note that while the associative scan is parallelizable it performs matrix-matrix products of the form $\bA_k \bA_{k-1}$ which will be computational prohibitive unless $\bA_k$ has structure (e.g. diagonal or low-rank). This is one of the reasons why e.g. Mamba-2 utilizes a scaled identity as its $\bA_k$ <d-cite key="dao2024a"></d-cite>. 
