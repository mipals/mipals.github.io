---
layout: distill
title: The Adjoint Method
description: and why its easier than people make it out to be
tags: optimization constrained-optimization adjoint-method
giscus_comments: true
date: 2025-08-01 12:06:00
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
# toc:
#   - name: The Adjoint Method


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

Most methods for solving equality constrained optimization problems requires the computation of gradient of the objective function with respect to the optimization variable $\theta$. This is where the problem comes in: Naively computing the gradient requires the computation of the sensitivities of the implicitly defined variable $u$ with respect to the optimization variable $\theta$ $\left(\text{i.e.}\ \frac{\mathrm{d}u}{\mathrm{d}\theta} \in \mathbb{R}^{n_u \times n_\theta}\right)$. This result in a computational bottleneck as forming the sensitivities scales as $\mathcal{O}(n_un_\theta)$, meaning that adding a new parameter adds $n_u$ additional sensitivities (and one adding one more variable would add $n_\theta$ additional sensitivities). 

To resolve this computational bottleneck we can make use the adjoint method. The first step in the derivation of the adjoint method is to introduce the Lagrangian of the objective function, i.e.

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


### Example
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
