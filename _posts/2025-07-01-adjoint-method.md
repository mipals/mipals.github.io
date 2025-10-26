---
layout: distill
title: Backpropagation is The Adjoint Method
description: Backpropagation is often introduced as something developed within the field of machine learning. However, the story is that backpropagation really is just a special case of the adjoint method. This note is in two parts. In the first part we review the adjoint method while in the second part we describe how backpropagation is a special case of the adjoint method with a structure that result in scalable computations algorithms.
tags: optimization constrained-optimization adjoint-method neural-networks backpropagation
giscus_comments: true
date: 2025-10-25 12:00:00
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
toc:
  - name: The Adjoint Method
  - name: Backpropagation and the adjoint method

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

<!-- The adjoint method allows us to efficiently compute the gradient of the objective function of an equality constrained optimization of the form

$$
\begin{equation}
    \begin{aligned}
        \min_{\theta}     \quad & L(u,\theta), \quad\ \quad\ \left(\text{some objective function}\right)\\
        \text{subject to }\quad & f(u,\theta) = 0,    \quad \left(\text{some equality constraint}\right) \\
    \end{aligned}
\end{equation}
$$

without having to explicitly compute the sensitivity of the implicitly defined variable $u$ of the optimization problem $\left(\text{i.e.}\ \frac{\mathrm{d}u}{\mathrm{d}\theta} \in \mathbb{R}^{n_u \times n_\theta}\right)$. The key reason why this is important is that forming the sensitivities explicitly would scale as $\mathcal{O}(n_un_\theta)$, which would make it impossible to solve large scale problems. The adjoint method can be used to resolve the issue by eliminating the need to compute the sensitivities at all. The first step in the derivation of the adjoint method is to introduce the Lagrangian of the objective function, i.e. -->

# The adjoint method

In many engineering applications we are interested in solving equality constrained optimization problems where the optimization variable ($\theta$) implicitly defines another variable ($u$) that then appears in the objective function<d-footnote>A famous example of this is [topology optimization](https://en.wikipedia.org/wiki/Topology_optimization).</d-footnote>. Written out this means solving optimization problems of the following form

$$
\begin{equation}
    \begin{aligned}
        \min_{\theta \in \mathbb{R}^{n_\theta}}     \quad & L(u,\theta), \quad\ \quad\ \left(\text{some objective function}\right)\\
        \text{subject to }\quad & f(u,\theta) = 0,    \quad \left(\text{some equality constraint}\right) \\
        & u \in \mathbb{R}^{n_u}.
    \end{aligned}
\end{equation}
$$

Most methods for solving equality constrained optimization problems requires the computation of gradient of the objective function with respect to the optimization variable $\theta$. This is where the problem comes in: Naively computing the gradient requires the computation of the sensitivities of the implicitly defined variable $u$ with respect to the optimization variable $\theta$ $\left(\text{i.e.}\ \frac{\mathrm{d}u}{\mathrm{d}\theta} \in \mathbb{R}^{n_u \times n_\theta}\right)$. This result in a computational bottleneck as forming the sensitivities scales as $\mathcal{O}(n_un_\theta)$, meaning that adding a new parameter adds $n_u$ additional sensitivities (and similarly adding one more variable would add $n_\theta$ additional sensitivities). 

To resolve this computational bottleneck the adjoint method was introduced. The first step in the derivation of the adjoint method is to introduce the Lagrangian of the objective function, i.e.

$$
\begin{equation}
    \mathcal{L}(u,\theta,\lambda) = L(u,\theta) + \lambda^\top f(u,\theta).
\end{equation}
$$

It is easy to see that whenever $u$ satisfy the equality constraint $f(u,\theta)=0$, then the Lagrangian is equal to the original objective function. This in turn mean that the two gradients $\nabla_\theta L(u,\theta,\lambda)$ and $\nabla_\theta \mathcal{L}(u,\theta,\lambda)$ are equal whenever $u$ satisfy the equality constraint $f(u,\theta) = 0$. The reason why the introduction of the additional term in the Lagrangian is useful is that $\lambda$ can be set to be anything. In particular, we aim to set in a way so that we can avoid computing the otherwise computational expensive sensitivities $\frac{\mathrm{d}u}{\mathrm{d}\theta}$. We start by computing the total derivative of the Lagrangian with respect to $\theta$<d-footnote>The total derivative is the transpose of the gradient i.e. $\left(\frac{\mathrm{d}L}{\mathrm{d}\theta}\right)^\top = \nabla_\theta L$. That is multiplying the change in $\theta$ with the total derivative (rather than the transpose of the gradient) gives the change in the function $L$.</d-footnote> 

$$
\begin{equation}
    \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta} = \frac{\partial L}{\partial\theta} + \frac{\partial L}{\partial u}\frac{\mathrm{d}u}{\mathrm{d}\theta} + \lambda^\top\left(\frac{\partial f}{\partial\theta} + \frac{\partial f}{\partial u}\frac{\mathrm{d}u}{\mathrm{d}\theta}\right).
\end{equation}
$$

Now we collect the terms that depend on the undesired $\frac{\mathrm{d}u}{\mathrm{d}\theta}$ result in

$$
\begin{equation}
    \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta} = \frac{\partial L}{\partial\theta} + \lambda^\top\frac{\partial f}{\partial\theta} + \left(\frac{\partial L}{\partial u} + \lambda^\top\frac{\partial f}{\partial u}\right)\frac{\mathrm{d}u}{\mathrm{d}\theta} .
\end{equation}
$$

As we can choose $\lambda$ freely a natural idea is to set it in a way such that the term in front of the undesired term vanishes. This means that we can choose $\lambda$ as the solution to the equation

$$
\begin{equation}
    \frac{\partial L}{\partial u} + \lambda^\top\frac{\partial f}{\partial u} = 0 \quad \Rightarrow \quad \lambda^\top = -\frac{\partial L}{\partial u}\left(\frac{\partial f}{\partial u}\right)^{-1}.
\end{equation}
$$  

Inserting $\lambda^\top$ back into the equation we find that the gradient of the Lagrangian with respect to $\theta$ is given by

$$
\begin{equation}
    \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta} = \frac{\partial L}{\partial\theta} \underbrace{- \frac{\partial L}{\partial u}\left(\frac{\partial f}{\partial u}\right)^{-1}}_{\lambda^\top}\frac{\partial f}{\partial\theta} \left(= \frac{\mathrm{d}L}{\mathrm{d}\theta}\right).
\end{equation}
$$

To conclude: The adjoint method is a simple way to avoid the computational bottleneck of computing the sensitivities $\frac{\mathrm{d}u}{\mathrm{d}\theta}$ by cleverly computing the so-called adjoint variable $\lambda$ in a way that eliminates the need to compute the problematic sensitivities. 


### Example: Linearly constrained problem with structure
In order to illustrate the adjoint method we consider a simple linearly constrained problem of the form (inspiration from problem 38 in <d-cite key="bright2025matrixcalculusformachine"></d-cite>)

$$
\begin{aligned}
    L(u,\theta) &= \left(c^\top u(\theta)\right)^2, \\
    f(u,\theta) &= A(\theta)u - b = 0,
\end{aligned}
$$

where $A(\theta) \in \mathbb{R}^{n\times n}$ is a symmetric tridiagonal matrix that depends on the parameters $\theta \in \mathbb{R}^{2n-1}$ as

$$
A = 
\begin{bmatrix}
    \theta_1        & \theta_{n+1}  &            &        & 0 \\
    \theta_{n+1}    & \theta_2      & \ddots     &        &   \\
                    & \ddots        & \ddots     & \ddots &   \\
                    &               & \ddots     & \theta_{n-1} & \theta_{2n-1} \\
    0               &               &            & \theta_{2n-1}     & \theta_n
\end{bmatrix}.
$$

Now in order to compute the gradient of interested we start by computing the adjoint variable $\lambda$ as

$$
\lambda^\top = -\frac{\partial L}{\partial u}\left(\frac{\partial f}{\partial u}\right)^{-1} = -2\left(c^\top u\right)c^\top A(\theta)^{-1}.
$$

Note that in practice we do not form $A(\theta)^{-1}$ explicitly but rather compute $\lambda$ by solving the linear system $ A(\theta)^\top\lambda = -2c\left(c^\top u\right)$. Using the adjoint variable we can compute the gradient of $L$ with respect to $\theta$ as

$$
    \frac{\mathrm{d}L}{\mathrm{d}\theta} 
    = \underbrace{\frac{\partial L}{\partial\theta}}_{=0} + \lambda^\top\frac{\partial f}{\partial\theta} 
    = \lambda^\top\frac{\partial A}{\partial\theta}u.
$$

In the concrete case of the derivatives of $A$ with respect to $\theta_i$ we have that

$$
\frac{\partial A}{\partial \theta_i} = 
\begin{bmatrix}
    \delta_{i,1}    & \delta_{i,n+1}    &            &                  & 0 \\
    \delta_{i,n+1}  & \delta_{i,2}      & \ddots     &                  &   \\
                    & \ddots            & \ddots     & \ddots           &   \\
                    &                   & \ddots     & \delta_{i,n-1}   & \delta_{i,2n-1} \\
    0               &                   &            & \delta_{i,2n-1}  & \delta_{i,n}
\end{bmatrix}.
$$

Using this it follows that the $i$th component of the gradient can be computed as

$$
\lambda^\top\frac{\partial A}{\partial \theta_i}u = 
\begin{cases}
    \lambda_i u_i,                                    \quad &i \leq n, \\
    \lambda_{i+1-n}u_{i-n} + \lambda_{i-n} u_{i+1-n}, \quad &i > n.
\end{cases}
$$

{% details Code %}
```julia
using LinearAlgebra 
n = 1000        # Number of linear constraints
c = rand(n)     # Objective vector
b = rand(n)     # Linear equality vector
θ = rand(2n-1)  # Parameters of the equality constraint

# Parametrizing linear equality constraint matrix
A(θ) = SymTridiagonal(θ[1:n], θ[n+1:end]) 

# Objective function
f(θ) = (c' * (A(θ) \ b))^2 

# Partial derivatives
partial(i,n,λ,u) = i <= n ?  λ[i]*u[i] : λ[i+1-n]*u[i-n] + λ[i-n]*u[i+1-n]

# Gradient computation
function ∇f(θ)
    n = length(b)
    M = A(θ)                # Defining constraint matrix
    u = M \ b               # Computing solution
    λ = M \ (-2*c*(c'*u))   # Adjoint variables
    return [partial(i,n,λ,u) for i = 1:2n-1]
end

# Testing against ForwardDiff
using Test, ForwardDiff
@test ForwardDiff.gradient(f,θ) ≈ ∇f(θ)
```
{% enddetails %}


# Backpropagation and the adjoint method

Section 5.5 of <d-cite key="edelman:backprop"></d-cite> includes an example of how the adjoint method and backpropagation of neural networks are similar. In neural networks we are often interested in minimizing a loss function of the form

$$
    L(\theta; u_0) = \| \Phi_N(\theta; u_0) - y \|_2^2 + \lambda \mathcal{R}(\theta), \quad \theta \in \mathbb{R}^k, \quad u_0 \in \mathbb{R}^m, \quad y \in \mathbb{R}^n,
$$

where the notation "$;u_0$" is used in order to highlight that $u_0$ is a constant input (i.e. most often the "data") and $\mathcal{R}(\theta)$ is some regularization function (e.g. $\mathcal{R}(\theta) = \Vert\theta\Vert_2^2$). Furthermore $\Phi_N(\theta;u_0): \mathbb{R}^k \to \mathbb{R}^n$ is a neural network with $N$ layers that given a set of constant inputs $u_0$ maps the parameters $\theta$ to an output. Now, the above objective functions does not include any equality constraints. However, one should realize that the a $N$-layer neural network is nothing more than a series of composition of $N$ functions

$$
    \Phi_N(\theta; u_0) = \Phi_N(\Phi_{N-1}(\cdots(\Phi_1(\theta_1; u_0)\cdots,\theta;u_0),\theta;u_0), 
$$

Now using the notation that $u_i$ is the output of the $i$th layer of the network we can write the propagation through the network as a large nonlinear system of equations that have to be satisfied

$$
    f(u,\theta) = u - \Phi(u,\theta) 
    = 
    \underbrace{\begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_N \end{bmatrix}}_{u}
    -
    \begin{bmatrix} \Phi_1(\theta; u_0) \\ \Phi_2(u_1, \theta; u_0) \\ \vdots \\ \Phi_N(u_1,\ldots,u_{N-1}, \theta; u_0) \end{bmatrix} = 
    \begin{bmatrix}0 \\ 0 \\ \vdots \\ 0\end{bmatrix}.
$$

Using this notation (and removing the explicit dependence on $x_0$) we see that optimizing a neural network is really nothing more solving the following equality constrained optimization problem

$$
\begin{aligned}
    \min_{\theta}     \quad & L(u,\theta) = \| u_N - y \|_2^2 + \lambda \mathcal{R}(\theta), \\
    \text{subject to }\quad & f(u,\theta) = 0. \\
\end{aligned}
$$

We can now use the adjoint method to compute the gradient of $L$ with respect to $\theta$. As a reminder this means that

$$
    \frac{\mathrm{d}L}{\mathrm{d}\theta} = \frac{\partial L}{\partial\theta} \underbrace{- \frac{\partial L}{\partial u}\left(\frac{\partial f}{\partial u}\right)^{-1}}_{\lambda^\top}\frac{\partial f}{\partial\theta}.
$$


For simplicity, we assume that $u_N$ is just a scalar, that is $u_N = e_N^\top u$ where $e_N$ is the $N$'th canonical basis vector. In this case we have that

$$
    \frac{\partial L}{\partial u} = 2(e_N^\top u - y)e_N^\top  = 2(u_N - y)e_N^\top = g_N^\top,
$$

Now what is left to compute is $\frac{\partial f}{\partial u}$ and $\frac{\partial f}{\partial\theta}$. Starting with $\frac{\partial f}{\partial u}$ we note because $f$ only propagates the $u_i$'s forwards the resulting partial derivative is a lower block triangular matrix of the form

$$
    \frac{\partial f}{\partial u} =
    \begin{bmatrix}
        I                                   & 0      & \cdots                                   & 0       \\
        \frac{\partial\Phi_2}{\partial u_1} & I      & \cdots                                   & 0       \\
        \vdots                              & \vdots & \ddots                                   & \vdots  \\
        \frac{\partial\Phi_N}{\partial u_1} & \cdots & \frac{\partial\Phi_N}{\partial u_{N-1}}  & I
    \end{bmatrix}
    = L,
$$

Now is a good place to stop and think of a key concept of backpropagation: The resulting computational graph should result in a Directed Acyclic Graph (DAG). This is indeed equivalent to stating that $ \frac{\partial f}{\partial u}$ should be a lower triangular matrix. This is an important property as the adjoint method requires us to invert $ \frac{\partial f}{\partial u} $, which can be done cheaply for a lower triangular matrix. Even more importantly is that the diagonal blocks are the identity, meaning that forward/backward substitutions can be done without any matrix inversions at all. 

What is left is to compute $\frac{\partial f}{\partial \theta}$, which standardly is done as

$$
    \frac{\partial f}{\partial \theta} =  
    -\underbrace{\begin{bmatrix}
        \frac{\partial\Phi_1}{\partial \theta_1} & \cdots  & \frac{\partial\Phi_1}{\partial \theta_k}       \\
        \vdots                                   & \ddots  & \vdots  \\
        \frac{\partial\Phi_N}{\partial \theta_1} & \cdots  & \frac{\partial\Phi_N}{\partial \theta_k}
    \end{bmatrix}}_{M^\top}
    = -M^\top.
$$

Now, in case of the layers not sharing any parameters the matrix $M^\top$ will have a block diagonal structure of the form

$$
    M^\top = \text{blkdiag}\left(\frac{\partial\Phi_1}{\partial\theta_1}, \ldots, \frac{\partial\Phi_N}{\partial\theta_N}\right), \quad 
    \theta = \begin{bmatrix}\theta_1 \\ \vdots \\ \theta_N \end{bmatrix},
$$

where $\theta_i$ are the parameters of layer $i$. 

Putting everything together we find that the gradient of $L$ with respect to $\theta$ is given by

$$
\begin{aligned}
    \frac{\mathrm{d}\mathcal{L}}{\mathrm{d}\theta} 
    &= \frac{\partial L}{\partial\theta} - \frac{\partial L}{\partial u}\left(\frac{\partial f}{\partial u}\right)^{-1}\frac{\partial f}{\partial\theta}\\
    &= \lambda\frac{\partial \mathcal{R}}{\partial\theta} - \left( g_N^\top L^{-1}(- M^\top)\right)\\
    &= \lambda\frac{\partial \mathcal{R}}{\partial\theta} + 2(u_N - y)e_N^\top L^{-1}M^\top.
\end{aligned}
$$

As noted previously both $L^{-1}$ and $M^\top$ are structured, meaning that the above can be computed efficiently.

{% details How to compute the elements of $L$ and $M^\top$ %}
For a standard linear layer an activation function $\sigma$ is usually applied element wise. In practice this means that we really should look at the rows of $\Phi_i$ separately. For row $j$th the gradients are easily seen to be

$$
\begin{alignat*}{2}
    \nabla_{u_{i-1}}\sigma(w_j^\top u_{i-1} + b_j) &= \sigma'(w_j^\top u_{i-1} + b_j)w_j,\quad\ \quad &&\text{Goes in to $L$}\\
    \nabla_{w_j}\sigma(w_j^\top u_{i-1} + b_j) &= \sigma'(w_j^\top u_{i-1} + b_j)u_{i-1},\quad  &&\text{Goes in to $M^\top$}\\
    \nabla_{b_j}\sigma(w_j^\top u_{i-1} + b_j) &= \sigma'(w_j^\top u_{i-1} + b_j), \quad\ \quad &&\text{Goes in to $M^\top$}
\end{alignat*}
$$

Using the above we see that 

$$
    \frac{\partial\Phi_i}{\partial u_{i-1}} = \text{diag}(\sigma'(W_i u_{i-1} + b))W_i,
$$

where $\sigma'(\cdot)$ is applied element wise. Now for $\frac{\partial\Phi_i}{\partial\theta_i}$ we have to pick a way of how to vectorize the parameters. Here we choose 

$$
    \theta_i = \begin{bmatrix} \text{vec}(W_i) \\ b_i\end{bmatrix},
$$

where $\text{vec}(W)$ vectorizes by stacking the columns of $W$. Using the vectorization we can write the sought after sensitivity as

$$
    \frac{\partial\Phi_i}{\partial\theta_i} = \text{diag}(\sigma'(W_i u_{i-1} + b))\left( \begin{bmatrix}u_{i-1} & 1\end{bmatrix} \otimes I_{n_i}\right).
$$

While the Kronecker above does result in a sparse matrix, it is even more structured. Using the standard property of Kronecker products, i.e. $(A\otimes B )v = B \text{mat}(v) A^\top $, where $\text{mat}(v)$ is the inverse operation of $\text{vec}(W)$.-


A final remark here is that the diagonal matrices that goes into the elements of $L$ and $M^\top$ are the same. That is if we define

$$
\begin{aligned}
    D   &= \text{diag}(\sigma'(W_1 u_{0} + b_1), \dots, \sigma'(W_N u_{N-1} + b_N)) \\
        &= \text{blkdiag}(\text{diag}(\sigma'(W_1 u_{0} + b_1)), \dots, \text{diag}(\sigma'(W_N u_{N-1} + b_N))),
\end{aligned}
$$

then it follows that

$$
\begin{aligned}
    L       &= I - D\text{blkdiag}(W_1, \dots, W_{N-1},-1), \quad \\
    M^\top  &= D\text{blkdiag}\left(\begin{bmatrix}u_0 & 1\end{bmatrix} \otimes I_{n_1}, \dots, \begin{bmatrix}u_{N-1} & 1\end{bmatrix} \otimes I_{n_N}\right).
\end{aligned}
$$

with the little unusual notation of $\text{blkdiag}(\cdot,-1)$ means that is the lower block diagonal matrix. 

{% enddetails %}

{% details Julia implementation %}
```julia
# add https://github.com/mipals/BlockDiagonalMatrices.jl.git # For BlockDiagonalMatrices
using Test, ForwardDiff, LinearAlgebra, BlockBandedMatrices, SparseArrays, BlockDiagonalMatrices, Kronecker
h(x)  =  exp(-x) # activation function
∇h(x) = -exp(-x) # derivative of activation function

# Forward pass including the derivatives
function forward_pass(u0,θ)
    x0 = u0
    diags = empty([first(θ)[2]])
    krons = empty([first(θ)[2]' ⊗ I(2) ])
    for (W,b) in θ
        push!(krons,[x0; 1]' ⊗ I(length(b))) # Lazy Kronecker producect using Kronecker.jl
        tmp = W*x0 + b  # Can be used for both forward pass and derivative pass
        x0 = h.(tmp)
        push!(diags, ∇h.(tmp))
    end
    return krons, diags, x0
end
# Block backsubstitution: Solving L^(-T)y = b
function backsub(dblks,wblks,b)
    y  = convert.(eltype(wblks[1]), copy(b))
    j0 = length(b)
    i0 = length(b) - size(wblks[end],1)
    @views for (D,blk) in (zip(reverse(dblks),reverse(Transpose.(wblks))))
        i1,j1 = size(blk)
        tmp = D .* y[j0-j1+1:j0]
        mul!(y[i0-i1+1:i0], blk, tmp, 1, 1)
        j0 -= j1
        i0 -= i1
    end
    return y
end
# Helper function thats packs a vector θ into Ws and bs
function pack_θ(θ, Ws_sizes, bs_sizes)
    We = empty([ones(eltype(θ),1,1)])
    be = empty([ones(eltype(θ),1)])
    i = 1
    for (W_size,b_size) in zip(Ws_sizes, bs_sizes)
        j = i+prod(W_size)
        push!(We, reshape(θ[i:j-1], W_size...))
        i = j + b_size
        push!(be, θ[j:i-1])
    end
    return We, be
end
# Evaluating f using the forward pass
function eval_f(θ, Ws_sizes, bs_sizes, u0)
    We,be = pack_θ(θ, Ws_sizes, bs_sizes)
    _,_,uN = forward_pass(u0, zip(We,be))
    return uN
end
# Gradient computation using the adjoint method
function ∇f(θ, Ws_sizes, bs_sizes, u0; y=0.0)
    # First pack vector parameters to matrices
    We,be = pack_θ(θ, Ws_sizes, bs_sizes)
    # Forward parse includes derivative information
    krons, ddiags, uN = forward_pass(u0, zip(We,be))
    # We here use that L' = I - blkdiag(W_1,..., W_N)D' and M^T = D*K 
    D = Diagonal(vcat(ddiags...))
    K = BlockDiagonal(krons)
    g = zeros(eltype(θ), sum(w_sizes -> w_sizes[1], Ws_sizes))
    g[end] = 2*(uN[1] - y) # uN is a 1x1 matrix so extract the scalar
    # The final step is to evaluate the gradient from the right
    grad_adjoint = (backsub(ddiags,We[2:end],g')*D)*K
    return grad_adjoint'
end
# Forward difference to test the adjoint gradient implementation
function fd(θ, Ws_sizes,bs_sizes,u0,i;e=1e-6,y=2.0)
    f0 = eval_f(θ, Ws_sizes, bs_sizes, u0)
    θ[i] += e
    f1 = eval_f(θ, Ws_sizes, bs_sizes, u0)
    θ[i] -= e
    return sum(((f1[1] - y)^2 - (f0[1] - y)^2)/e)
end

# Setting up parameters 
layer_sizes = [50,40,30,20,10,1]
N = length(layer_sizes) - 1
init(sizes...) = 0.01*randn(sizes...)
Ws = [init(layer_sizes[i+1],layer_sizes[i]) for i=1:N]
bs = [init(layer_sizes[i+1]) for i = 1:N]
u0 = init(layer_sizes[1],1)[:]
θ  = zip(Ws,bs)

Ws_sizes = size.(Ws)
bs_sizes = length.(bs)

# First we compute the forward pass
θvec = vcat([[W[:]; b] for (W,b) in θ]...)

## Testing the gradient
y = 3.0 # We aim to have the final output be 3.0
grad_adjoint = ∇f(θvec, Ws_sizes, bs_sizes, u0;y=y)
idx = 4000
@test fd(θvec, Ws_sizes,bs_sizes,u0,idx;e=1e-5,y=y) ≈ grad_adjoint[idx] atol=1e-6

# Optimizing to get the output y
for iter = 1:1000
    grad = ∇f(θvec, Ws_sizes, bs_sizes, u0; y=y) # Y is the output value we want
    θvec -= 0.001*grad
end
@test eval_f(θvec,Ws_sizes,bs_sizes,u0)[1] ≈ y
@test ∇f(θvec, Ws_sizes, bs_sizes, u0;y=y) ≈ zeros(length(θvec)) atol=1e-10

```
{% enddetails %}


<!-- $$
    \frac{\mathrm{d}L}{\mathrm{d}\theta} 
    = \lambda\frac{\partial \mathcal{R}}{\partial\theta} + 
    \begin{bmatrix}0 & \dots & 2(u_N - y)\end{bmatrix} 
    (\begin{bmatrix} I & 0 & \dots & 0\end{bmatrix})^{-1}
    \begin{bmatrix}\begin{bmatrix}u_0 & 1\end{bmatrix} \otimes I_{n_1} & & \\ & \ddots & \\ & & \begin{bmatrix}u_{N-1} & 1\end{bmatrix} \otimes I_{n_N}\end{bmatrix}.
$$ -->

<!-- ## Example: ODEs
Given a cost function $l$ that depends on the solution $u$ as well as the parameters $\theta \in \mathbb{R}^{n_\theta}$, we can efficiently compute its gradient using the so-called adjoint method. The main objective of the adjoint method is to eliminate the expensive computation of $\left(\frac{\mathrm{d}u}{\mathrm{d}\theta} \in \mathbb{R}^{n_u \times n_\theta}\right)$ when computing the gradient of a scalar loss function w.r.t. $\theta$. As this operation scales as $\mathcal{O}(nn_\theta)$ this is highly important. In general, we are trying to solve an ODE constrained optimization problem of the form

$$
\begin{equation}
    \begin{aligned}
        \min_{\theta}     \quad & L(u,\theta) = \int_{t_0}^T l(u,\theta)\ \mathrm{d}t\\
        \text{subject to }\quad &\frac{\mathrm{d}u}{\mathrm{d}t} - f(u,\theta,t) = 0, \quad \left(\text{ODE constraint}\right) \\
                          \quad &u(t_0,\theta) - v_{0}(\theta) = 0.\quad \left(\text{initial condition constraint}\right)
    \end{aligned}
\end{equation}
$$

The Lagrange form of the optimization problem is given by

$$
\begin{equation}
    \begin{aligned}
        \mathcal{L}(u,\theta,\lambda) = L(u,\theta) + \int_{t_0}^T\lambda^\top(t)\left(\frac{\mathrm{d}u}{\mathrm{d}t} - f(u,\theta,t)\right)\mathrm{d}t + \mu^\top\left(u(t_0,\theta) - v_0(\theta)\right),
    \end{aligned}
\end{equation}
$$

where it is important to note that the two additional terms add nothing to the cost function if $u$ is a solution to the ODE. We now do some manipulation with the aim of choosing $\lambda$ and $\mu$ in such a way that we can avoid computing the sensitivities $\frac{\mathrm{d}u}{\mathrm{d}\theta}$. We start by introducing the gradient of $\mathcal{L}$ with respect to $\theta$ as

$$
    \nabla_\theta\mathcal{L} 
    = \int_{t_0}^T \left(\frac{\partial l}{\partial \theta} + \frac{\partial l}{\partial u}\frac{\mathrm{d}u}{\mathrm{d}\theta}\right)\ \mathrm{d}t 
    + \int_{t_0}^T\lambda^\top\left(\frac{\mathrm{d}}{\mathrm{d}\theta}\frac{\mathrm{d}u}{\mathrm{d}t} - \left(\frac{\partial f}{\partial \theta} + \frac{\partial f}{\partial u}\frac{\mathrm{d}u}{\mathrm{d}\theta}\right)\right)\mathrm{d}t 
    + \mu^\top\left.\left(\frac{\mathrm{d}u}{\mathrm{d}\theta} - \frac{\mathrm{d}v}{\mathrm{d}\theta}\right)\right|_{t_0}
$$

We now apply integration by parts on the first term in the second integral

$$
\begin{equation}
    \int_{t_0}^T\lambda^\top\frac{\mathrm{d}}{\mathrm{d}\theta}\frac{\mathrm{d}u}{\mathrm{d}t}\mathrm{d}t = \left[\lambda^\top\frac{\mathrm{d}u}{\mathrm{d}\theta}\right]_{t_0}^T - \int_{t_0}^T\frac{\mathrm{d}\lambda^\top}{\mathrm{d}t}\frac{\mathrm{d}u}{\mathrm{d}\theta}\mathrm{d}t.
\end{equation}
$$

The idea is to collect the terms that depend on $\frac{\mathrm{d}u}{\mathrm{d}\theta}$ and then try to rewrite the equation in a way that we can avoid computing it

$$
    \nabla_\theta\mathcal{L} 
    = \int_{t_0}^T \left(\frac{\partial l}{\partial \theta} - \lambda^\top\frac{\partial f}{\partial \theta}\right) + \left(\frac{\partial l}{\partial u} - \frac{\mathrm{d}\lambda^\top}{\mathrm{d}t} - \lambda^\top\frac{\partial f}{\partial u}\right)\frac{\mathrm{d}u}{\mathrm{d}\theta}\mathrm{d}t 
    + \left[\lambda^\top\frac{\mathrm{d}u}{\mathrm{d}\theta}\right]_{t_0}^T
    + \mu^\top\left.\left(\frac{\mathrm{d}u}{\mathrm{d}\theta} - \frac{\mathrm{d}v}{\mathrm{d}\theta}\right)\right|_{t_0}.
$$

We see that we can get rid of the sensitivities $\frac{\mathrm{d}u}{\mathrm{d}\theta}$ inside the last integral by choosing $\lambda$ as the solution to

$$
\begin{equation}
    \frac{\mathrm{d}\lambda}{\mathrm{d}t} = -\left(\frac{\partial f}{\partial u}\right)^\top\lambda + \left(\frac{\partial l}{\partial u}\right)^\top.
\end{equation}
$$

Furthermore, the sensitivities at $T$ of $\left[\lambda^\top\frac{\mathrm{d}u}{\mathrm{d}\theta}\right]_{t_0}^T$ can be disregarded if we choose $\lambda(T)=0$. As such the gradient of the Lagrangian with respect to the parameters $\theta$ can be written as

$$
\begin{equation}
    \nabla_\theta\mathcal{L}(u,\theta,\lambda) = \int_{t_0}^T \left(\frac{\partial l}{\partial \theta} - \lambda^\top\frac{\partial f}{\partial \theta}\right)\ \mathrm{d}t - \lambda(t_0)^\top\frac{\mathrm{d}u}{\mathrm{d}\theta}(t_0) + \mu^\top\left.\left(\frac{\mathrm{d}u}{\mathrm{d}\theta} - \frac{\mathrm{d}v}{\mathrm{d}\theta}\right)\right|_{t_0}.
\end{equation}
$$

Finally, setting $\mu=\lambda(t_0)$ we have that the gradient of the Lagrangian with respect to the parameters $\theta$ is given by

$$
\begin{equation}
    \nabla_\theta\mathcal{L}(u,\theta,\lambda) = \int_{t_0}^T \left(\frac{\partial l}{\partial \theta} - \lambda^\top\frac{\partial f}{\partial \theta}\right)\ \mathrm{d}t - \frac{\mathrm{d}v}{\mathrm{d}\theta}(t_0).
\end{equation}
$$

In the case of the initial condition not depending on the parameters. -->

<!-- To summarize the results of the adjoint method for neural networks we have the following table

| Adjoint Method                        |  Matrix                               |     Size     |
| ------------------------------------- | ------------------------------------- | ------------ |
| $\frac{\partial f}{\partial u}$       | $I - \tilde{L}$                       | $N \times N$ |
| $\frac{\partial f}{\partial \theta}$  | $-M^{\mathsf{T}}$                     | $N \times k$ |
| $\frac{\partial u}{\partial \theta}$  | $(I - \tilde{L})^{-1} M^{\mathsf{T}}$ | $N \times k$ |
 -->


