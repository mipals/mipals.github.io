---
layout: distill
title: Block Tridiagonal Matrices
description: and their relation to semiseparable matrices and state-space models
tags: block-tridiagonal linear-algebra state-space-models semiseparable-matrices
giscus_comments: true
date: 2025-07-28 15:06:00
featured: true
# citation: true

authors:
  - name: Mikkel Paltorp
    url: "https://mipals.github.io"
    affiliations:
      name: Technical University of Denmark

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).


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

We will in this post restrict ourselves to the case of symmetric block tridiagonal matrices. These matrices occur in the context of state space models, since the state transition matrix is lower block bidiagonal resulting in the covariance matrix being symmetric block tridiagonal. It turns out that the inverse of a symmetric block tridiagonal matrix is semiseparable with separability rank equal to the size of the blocks <d-cite key="meurant1992a"></d-cite>. We introduce the following notation for a symmetric block tridiagonal matrix

$$
    \bK = 
    \begin{bmatrix}
        \bD_1   & -\bA_2^\top   &               &           & \\ 
        -\bA_2  & \bD_2         & -\bA_3^\top   &           & \\
                & \ddots        & \ddots        & \ddots    & \\
                &               & -\bA_{N-1}    & \bD_{N-1} & -\bA_{N}^\top\\
                &               &               & -\bA_T    & \bD_T
    \end{bmatrix}.
$$

It turns out that the matrix can be factorized in two ways as follows

$$
    \bK
    = \left(\Delta + \bL\right)\Delta^{-1}\left(\Delta + \bL^\top\right) 
    = \left(\Sigma + \bL^\top\right)\Sigma^{-1}\left(\Sigma + \bL\right) ,
$$

where $\bL$ is the matrix of lower triangular blocks and $\Delta$ and $\Sigma$ are block diagonal matrices with blocks computed as follows

$$
    \begin{cases}
        \Delta_1 = \bD_1, \\
        \Delta_i = \bD_i - \bA_i\Delta_{i-1}^{-1}\bA_i^\top 
    \end{cases},
    \quad
    \begin{cases}
        \Sigma_T = \bD_T, \\
        \Sigma_i = \bD_i - \bA_{i+1}^\top\Sigma_{i+1}^{-1}\bA_{i+1} 
    \end{cases}.
$$

In the case of the of diagonal blocks ($\bA_i$) being invertible, and using $\Delta$ and $\Sigma$, then $\bK^{-1}$ is semiseparable generator representable. In short this means that the inverse of $\bK$ can be written as

$$
    \bK^{-1} = 
    \text{tril}\left(\bU\bV^\top\right) + \text{triu}\left(\bV\bU^\top,1\right).
$$

where the so-called generators $\bU$ and $\bV$ can be computed as

$$\label{eq:generators}
    \begin{cases}
        \bU_1 = \Sigma_1^{-1}\\
        \bU_i = \Sigma_{i}^{-1}\bA_i\bU_{i-1}
    \end{cases}, \quad
    \begin{cases}
        \bV_1 = \bI\\
        \bV_i = \bA_i^{-\top}\Delta_{i-1}\bV_{i-1}
    \end{cases},
$$

The semiseparable representation extremely important computationally. We have that $\bU,\bV\in\mathbb{R}^{nT\times n}$ which mean that it is a data-sparse representation of $\bK^{-1}\in\mathbb{R}^{nT\times nT}$ (we have $D_i \in \mathbb{R}^{n\times n}$ and $N$ being the number of states. In most cases $n \ll N$ meaning that the representation is extremely efficient). At the same time the structure make is possible to compute various important linear algebraic operation in $O(n^2T)$ that in the dense case would have scaled as $O((nT)^3)$ <d-cite key="andersen2020a"></d-cite>. 

Some Julia code implementing the above is given below.

{% details Code %}
```julia
using LinearAlgebra, FillArrays, LinearAlgebra, BlockBandedMatrices, Test

T = 5 # Sequence length
n = 3 # blk sizes

A_blks = [i == 0 ? zeros(n,n) : exp(rand(n,n)) for i in 1:T] 
D_blks = [Matrix(1.0*I(n)) for i in 1:T]
A_blks_transposed = [A_blk' for A_blk in A_blks]
zero_blks = [zeros(n,n) for i in 1:T]

K = BlockTridiagonal(-A_blks[2:end], D_blks, -A_blks_transposed[2:end])
Δ_blks = copy(D_blks)
Σ_blks = copy(D_blks)
for (i,j) in zip(2:length(D_blks),length(D_blks):-1:2)
    Δ_blks[i]   = D_blks[i] - A_blks[i]*(Δ_blks[i-1]\A_blks[i]')
    Σ_blks[j-1] = D_blks[j] - A_blks[j]'*(Σ_blks[j]\A_blks[j])
end

L = BlockTridiagonal(-A_blks[2:end], zero_blks, zero_blks[1:end-1])
Δ = BlockTridiagonal(zero_blks[1:end-1], Δ_blks, zero_blks[1:end-1])
Σ = BlockTridiagonal(zero_blks[1:end-1], Σ_blks, zero_blks[1:end-1])

@test (Δ + L)*(Δ\(Δ + L')) ≈ K
@test (Σ + L')*(Σ\(Σ + L)) ≈ K

# Creating the semiseperable form
V_blk = [Matrix(1.0*I(n)) for _ in 1:T]
U_blk = [Matrix(1.0*I(n)) for _ in 1:T]
U_blk[1] = inv(Σ_blks[1])
for i in 2:T
    U_blk[i] = Σ_blks[i]'\A_blks[i]*U_blk[i-1]
    V_blk[i] = A_blks[i]'\(Δ_blks[i-1]*V_blk[i-1])
end
U = vcat(U_blk...)
V = vcat(V_blk...)
@test tril(U*V') ≈ tril(inv(K))
@test triu(V*U') ≈ triu(inv(K))
```
{% enddetails %}


<!-- Notice, however, that computing $\bU_i$ requires the inversion of $\bA_i^\top$, meaning that the generator representation is only valid in the case of invertible off-diagonal blocks.  -->



<!-- In the case of non-invertible off diagonal blocks it is still possible to compute the diagonal blocks of the inverse as

$$
    \bS_i^\top = \left(\Sigma_i + \Delta_i - \bD_i\right)^{-1} = \Sigma_i^{-1} + \Sigma_i^{-1}\bA_i\bS_{i-1}\bA_i^\top\Sigma_i^{-1}.
$$

This should not come as a surprise as the diagonal blocks are computed for all state space models - even those for which the dynamics are not invertible (also, note that the formula is identical to the covariance update in a state space model). -->
