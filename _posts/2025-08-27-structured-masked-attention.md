---
layout: distill
title: Structured Masked Attention
description: and its relation to semiseparable matrices and state-space models
tags: attention linear-algebra state-space-models semiseparable-matrices
giscus_comments: true
date: 2025-07-28 15:06:00
featured: true
citation: true

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
toc:
  - name: State-space models as structured matrices
  - name: State-space models as SMAs

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


Masked Attention (MA) is given by the relation<d-cite key="dao2024a"></d-cite>

$$
\bM = \bB \odot \bL,
$$

where $\bL$ is a mask applied to the matrix $\bB$ using the elementwise Hadamard product (denoted by $\odot$). When the mask is *structured* one can most often apply multiplication with $\bM$ efficiently, and we refer to the masked attention as Structured Masked Attention (SMA). In the simplest case of $\bL$ being a lower triangular matrix filled with ones the SMA reduces to

$$
\bM = \bB \odot \bL = \tril(\bB),
$$

which can be viewed as a weighted cumulative sum. While the above is a *structured* computation, it is not efficient unless $\bB$ itself have some structure. An example of when $\bB$ is structured is when it is of low-rank (i.e. $\bB = \bU\bV^\top$). The resulting SMA is a semiseparable matrix for which multiplication can be applied in linear time<d-cite key="andersen2020a"></d-cite>. 

In general if $\bL$ is a low-rank matrix (of rank $p$) then multiplication with $\bM$ scales as $p$-times the scaling of multiplication with $\bB$ as

$$
\bM\bx = \left(\bB \odot \bU\bV^\top\right)\bx = \sum_{i=1}^n \diag(\bu_i)\bB \diag(\bv_i)\bx,
$$

for which we see that we need to perform $p$ multiplications with $\bB$ as well as $2p$ diagonal multiplications.

{% details Code %}
```julia
using LinearAlgebra, Test
n, p = 10, 2
U, V = randn(n,p), randn(n,p)
B = rand(n,n)
M = B .* (U*V')
x = randn(n)
@test M*x ≈ sum(i -> Diagonal(U[:,i])*(B*(Diagonal(V[:,i])) * x),1:p)
```
{% enddetails %}


## State-space models as structured matrices
First we recap that a state-space model is given by the equations

$$
\begin{equation}
    \begin{aligned}
        \bh_i &= \bA_i \bh_{i-1} + \bB_i \bx_i, \\
        \by_i &= \bC_i \bh_i
    \end{aligned},
\end{equation} 
$$

An alternative way to view the state-space model is through the lens of structured matrices. In particular a state-space model up until time step $T$ can also be written using block matrices as well as block bidiagonal matrices as


$$
\begin{equation}
    \underbrace{\begin{bmatrix}
        \bI    &        &         & \\
    -\bA_1  &  \bI   &         & \\
            & \ddots & \ddots  & \\
            &        & -\bA_{T-1} & \bI 
    \end{bmatrix}}_{\bA}
    \begin{bmatrix}
        \bh_0 \\ 
        \bh_1 \\
        \vdots   \\
        \bh_{T-1}
    \end{bmatrix}
    = \begin{bmatrix}
        \bB_0\bx_0 \\
        \bB_1\bx_1 \\
        \vdots   \\
        \bB_{T-1}\bx_{T-1}
    \end{bmatrix}
    = 
    \underbrace{\blkdiag\left(\begin{bmatrix} \bB_0 \\ \bB_1 \\ \vdots \\ \bB_{T-1} \end{bmatrix}\right)}_{\bB}
    \begin{bmatrix} 
        \bx_0 \\
        \bx_1 \\
        \vdots   \\
        \bx_{T-1}
    \end{bmatrix}
\end{equation}
$$

We know that by simply iterating forwards in time we can compute the states in linear time. As such it is not a surprise that the inverse of the bidiagonal matrix $\bA$ (i.e. $\bA^{-1}$) can be computed in linear time as 

$$
\begin{equation}
    \bA^{-1} 
    = \tril\left(\underbrace{\begin{bmatrix}\bI \\ \bA_1 \\ \bA_2 \bA_1 \\ \vdots \\ \bA_{T-1}\dots \bA_1\end{bmatrix}}_{\bU}\underbrace{\begin{bmatrix}\bI \\ \bA_1^{-1} \\ (\bA_2 \bA_1)^{-1} \\ \vdots \\ (\bA_{T-1}\dots \bA_1)^{-1}\end{bmatrix}^\top}_{\bV^\top}\right),
\end{equation}
$$

This type of matrix structure is called semiseparable <d-cite key="andersen2020a"></d-cite>. Using the explicit inverse of $\bA$ we can compute the hidden states efficiently as 

$$
\begin{equation}
    \begin{bmatrix}
        \bh_0 \\ 
        \bh_1 \\
        \vdots   \\
        \bh_{T-1}
    \end{bmatrix}   
    =
\underbrace{\tril\left(\begin{bmatrix}\bI \\ \bA_1 \\ \bA_2 \bA_1 \\ \vdots \\ \bA_{T-1}\dots \bA_1\end{bmatrix}\begin{bmatrix}\bI \\ \bA_1^{-1} \\ (\bA_2 \bA_1)^{-1} \\ \vdots \\ (\bA_{T-1}\dots \bA_1)^{-1}\end{bmatrix}^\top\right)}_{\bA^{-1}}
    \blkdiag\left(\begin{bmatrix} \bB_0 \\ \bB_1 \\ \vdots \\ \bB_{T-1} \end{bmatrix}\right)
    \begin{bmatrix}
        \bx_0 \\
        \bx_1 \\ 
        \vdots   \\
        \bx_{T-1}
    \end{bmatrix},
\end{equation}
$$

Similarly, the output $\by$ can be computed by applying the output matrices $\bC_i$ in a blocked fashion to the hidden states, i.e.

$$
\begin{equation}
    \begin{bmatrix}
        \by_1 \\
        \by_2 \\
        \vdots   \\
        \by_T
    \end{bmatrix}
    = 
    \underbrace{
    \overbrace{\blkdiag\left(\begin{bmatrix} \bC_0^\top \\ \bC_1^\top \\ \vdots \\ \bC_{T-1}^\top \end{bmatrix}\right)}^{\bC^\top}
    \begin{bmatrix}
        \bI    &        &         & \\
    -\bA_1  &  \bI   &         & \\
            & \ddots & \ddots  & \\
            &        & -\bA_{T-1} & \bI 
    \end{bmatrix}^{-1}
    \blkdiag\left(\begin{bmatrix} \bB_0 \\ \bB_1 \\ \vdots \\ \bB_{T-1} \end{bmatrix}\right)}_{\bM}
    \begin{bmatrix}
        \bx_1 \\ 
        \bx_2 \\
        \vdots   \\
        \bx_T
    \end{bmatrix}
\end{equation}
$$

{% details Code %}
```julia
using LinearAlgebra, BlockBandedMatrices, Test, SymSemiseparableMatrices

T = 10          # Sequence length
n = 6           # State size
input_dim = 1   # Dimension of forcing term

A_blks = [rand()*I(n) for _ in 1:T-1]       # Diagonal dynamics in Mamba 2
D_blks = [I(n)/1.0 for i in 1:T]            # Diagonal blocks are identity
C_blks = [rand(n,1) for _ in 1:T]           # Measurements are scalars
B_blks = [rand(n,input_dim) for _ in 1:T]   # Inputs are scalars
# Defining zeros blocks for the matrices
A_zero_blks = [zeros(n,n) for b in A_blks]
C_zero_blks = [zeros(n,1) for _ in 1:T-1]
B_zero_blks = [zeros(n,input_dim) for _ in 1:T-1]   
# Defining the block matrices
A = BlockTridiagonal(-A_blks,D_blks,A_zero_blks)
C = BlockTridiagonal(C_zero_blks,C_blks,C_zero_blks)
B = BlockTridiagonal(B_zero_blks,B_blks,B_zero_blks)

# Computing states by iterating forward
x_blks = [randn(input_dim) for _ in 0:T-1]  # input data
h_blks = [B_blks[1]*x_blks[1]]              # initial hidden state
for i in 2:T
    push!(h_blks, A_blks[i-1]*h_blks[i-1] + B_blks[i]*x_blks[i])
end
y_blks = [C'*h for (C,h) in zip(C_blks,h_blks)] 

# Computing states using semiseparable matrices
Ai_blks = [prod(A_blks[1:i-1],init=1.0*I(n)) for i in 1:T]
U = vcat(Ai_blks...)
V = vcat(inv.(Ai_blks)...)
# "SymSemiseparableCholesky" represents the matrix A^{-1} = tril(UV') 
Ai = SymSemiseparableCholesky(U',V')
x = vcat(x_blks...) # Collecting input data

@test Ai*(B*x) ≈ vcat(h_blks...)        # Checking hidden states
@test C'*(Ai*(B*x)) ≈ vcat(y_blks...)   # Checking measurement
```
{% enddetails %}

## State-space models as SMAs
In this section we aim to show derive that the simplified SSM in the mamba-2 paper is a special case of Structured Masked Attention (SMA) <d-cite key="dao2024a"></d-cite>. That is that the multiplication 

$$
    \bM\bx = \blkdiag(\bC_0, \dots, \bC_{T-1})^\top (\bA^{-1} (\blkdiag(\bB_0, \dots, \bB_{T-1}) \bx)),
$$

can be written differently as

$$
  \bM\bx = \left(\bB\odot \bL\right) \bx,
$$

The SSM in the Mamba-2 paper restricts the dynamics of $\bA$ to be scalar-times-identity dynamics in order for the masked to be structured <d-cite key="dao2024a"></d-cite>. In short this restriction mean that the dynamics for all hidden states are independent but equal. 

$$
  \begin{aligned}
      \bh_i &= \left(a_i\bI\right) \bh_{i-1} + \bb_i x_i, \\
      \by_i &= \bc_i^\top \bh_i
  \end{aligned}.
$$

From a practical point-of-view this mean that we can collect each index of the dynamics and treat them separately. The resulting $\bA$-matrix can be described by a Kronecker product (depending on how we organize the states its either $\bA = \ba \otimes \bI_n$ or $\bA = \bI_n \otimes \ba$). In the following we choose to separate the states, resulting in $\bA$ having the form

$$
    \bA = \bI_n \otimes \ba, \quad 
    \ba = 
    \begin{bmatrix}
            1    &          &           & \\
          -a_1    &      1   &           & \\
                  & \ddots   & \ddots    & \\
                  &          & -a_{T-1}  & 1
    \end{bmatrix},
$$

Using that the inverse of a Kronecker product is the Kronecker product of the inverses it follows that

$$
    \bA^{-1} = \bI_n \otimes \ba^{-1}, \quad 
    \ba^{-1} 
    = \tril\left(
        \underbrace{\begin{bmatrix}1 \\ a_1      \\ \vdots \\  a_{T-1}\dots a_1\end{bmatrix}}_{\bu}
        \underbrace{\begin{bmatrix}1 \\ a_1^{-1} \\ \vdots \\ (a_{T-1}\dots a_1)^{-1}\end{bmatrix}^\top}_{\bv^\top}\right),
$$

where we further used that the inverse of a bidiagonal matrix is semiseparable. Furthermore, we have to re-arrange $\bB$ and $\bC$ which result in

$$
        \bB = \begin{bmatrix} \diag\left(\bb_1\right) \\ \vdots \\ \diag\left(\bb_n\right)\end{bmatrix}, \quad 
        \bb_i = \begin{bmatrix} b_1^{(i)}  \\ \vdots \\ b_T^{(i)} \end{bmatrix},\quad
        \bC = \begin{bmatrix} \diag\left(\bc_1\right) \\ \vdots \\ \diag\left(\bc_n\right)\end{bmatrix}, \quad 
        \bc_i = \begin{bmatrix} c_1^{(i)}  \\ \vdots \\ c_T^{(i)} \end{bmatrix}.
$$

The final multiplication will therefore look as

$$
    \underbrace{\bC^\top \bA^{-1} \bB}_{\bM}\bx
    = \left(\sum_{i=1}^n \diag(\bc_i) \ba^{-1} \diag(\bb_i)\right)\bx
    = \left(\sum_{i=1}^n\left(\bc_i\bb_i^\top\right)\odot \ba^{-1}\right)\bx.
$$

Finally, using the properties of the Hadamard product we get to the Structured Masked Attention form that we were looking for

$$
    \left(\bC^\top \bA^{-1} \bB \right)\bx
    = \left(\sum_{i=1}^n\left(\bc_i\bb_i^\top\right)\odot \ba^{-1}\right)\bx
    = \left(\left(\begin{bmatrix}
        \bc_1 & \dots & \bc_n
    \end{bmatrix}
    \begin{bmatrix}
        \bb_1^\top \\ \vdots \\ \bb_n^\top
    \end{bmatrix}\right)\odot \ba^{-1}\right)\bx.
$$

This mean that the SSM dynamics can be interpreted as a structured masked attention mechanism. Note that in the case of the dynamics being independent but different (i.e we would have $\ba_i$ rather than just $\ba$) the SSM dynamics would result in a sum of masked attentions, i.e

$$
    \left(\bC^\top \bA^{-1} \bB \right)\bx
    = \left(\sum_{i=1}^n\left(\bc_i\bb_i^\top\right)\odot \ba_i^{-1}\right)\bx
$$

{% details Code %}
```julia
using LinearAlgebra, Test, SymSemiseparableMatrices

T = 10          # Sequence length
n = 6           # State size

# Here we treat the matrices in terms of their states and not sequence lengths
a_blks = rand(T-1)
a = Bidiagonal(ones(T),-a_blks,:L)
A = kron(I(n),a)        # Kronecker product with identity

# The blocks are now size equal to the sequence and a block for each state!
B_blks = [rand(T) for _ in 1:n]
C_blks = [rand(T) for _ in 1:n]  
# Collecting the blocks into the B and C
B = vcat(Diagonal.(B_blks)...)
C = vcat(Diagonal.(C_blks)...) 

# We want to see if the full matrix M = C'*(A\B) can be written as
# structured masked attention ie. 
# M = (CB')\circ a^{-1}. For this we start by computing a^{-1}
ai = inv(a) # We ignore here that inv(a) is semiseparable
@test C'*(A\B) ≈ sum(i-> Diagonal(C_blks[i])*ai*Diagonal(B_blks[i]),1:n)
@test C'*(A\B) ≈ sum(i->(C_blks[i]*B_blks[i]') .* ai , 1:n)

# We can collect the terms and write is as Structured Masked Attention!
Cn = hcat(C_blks...)
Bn = hcat(B_blks...)
@test C'*(A\B) ≈ (Cn*Bn').*ai

# We can apply the semiseparable structure of the inverse when multiplying!
ai_blks = [prod(a_blks[1:i-1],init=1.0) for i in 1:T]
u = vcat(ai_blks...)
v = vcat(inv.(ai_blks)...)
# "SymSemiseparableCholesky" represents the matrix A^{-1} = tril(UV') 
ais = SymSemiseparableCholesky(u',v')
# Efficient products using the structure of "ais" and diagonal B and C
x = randn(T)
@test C'*(A\(B*x)) ≈ sum(i-> C_blks[i].*(ais*(B_blks[i].*x)),1:n)
```
{% enddetails %}

<!-- ### Delta-net -->
