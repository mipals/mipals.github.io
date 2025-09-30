---
layout: distill
title: Conic Optimization
description: introduction and implementation details
tags: optimization conic-optimization sparsity quasi-definiteness interior-point-methods
giscus_comments: true
date: 2025-07-28 12:06:00
featured: true
# citation: true

authors:
  - name: Mikkel Paltorp
    url: "https://mipals.github.io"
    affiliations:
      name: Technical University of Denmark

bibliography: 2025-07-29-conic-optimization.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
  - name: Theory
  - name: Computing Step Directions
  - name: Solving the KKT system
  - name: Sparsity exploitation
  - name: Semi-definite cones

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

## Introduction
This note introduces some central ideas behind the implementation of efficient interior-point methods for solving conic programs. A larger portion of the theory section is based on the papers for the conic solvers Clarabel <d-cite key="goulart2024clarabelinteriorpointsolverconic"></d-cite> and CVXOPT <d-cite key="vandenberghe2010cvxopt"></d-cite>. What this note does different is that includes some essential implementation details that were left out of the original two papers (e.g. iterative refinement, matrix equilibration, and handling of the structure of the scaling matrices). 

## Theory
We will in this text consider optimization problems where the objective function is at most quadratic and the constraints are conic (including the *zero cone* in order to model equality constraints). The general form of such problems is the following

$$
\begin{alignat*}{3}
    &\text{minimize}\quad  && \frac{1}{2}x^\top Px + q^\top x \\
    &\text{subject to} && Ax + s = b\\
    &\ && s \in \mathcal{K},
\end{alignat*}
$$

where $x \in \mathbb{R}^n$ is the decision variable, $s \in \mathbb{R}^m$ is the slack variable, and $P = P^\top\succeq 0 \in \mathbb{R}^{n\times n}$, $q \in \mathbb{R}^n$, $A\in\mathbb{R}^{m\times n}$, $b\in \mathbb{R}^n$. Furthermore, $\mathcal{K}$ is a composite cone made from the Cartesian product the following cone types:

* The **zero cone**: $\mathcal{Z}_n := \lbrace 0\rbrace^n$ (used to model equality constraints). 

* The **nonnegative cone**: $\mathbb{R}_+^n :=  \lbrace x \in \mathbb{R}^n\ \|\ x \geq 0 \rbrace$.

* The **second-order cone**: $\mathcal{Q}_n := \lbrace (u,v) \in \mathbb{R}\times\mathbb{R}^{n-1}\ \|\ \|v\|_2 \leq u \rbrace$.

* The **positive semidefinite cone**: $\mathcal{S}_n := \lbrace x \in \mathbb{R}^{n(n+1)/2}\ \|\ \mathbf{smat}(x) \succeq 0 \rbrace$.

* The **exponential cone**: $\mathcal{K}_\text{exp} := \lbrace (z_1,z_2,z_3)\in\mathbb{R}^3\ \|\ z_2 \exp\left(\frac{z_1}{z_2}\right) \leq z_3,\ z_2 > 0 \rbrace$.

* The **power cone**: $\mathcal{K}_\text{pow} := \lbrace (z_1,z_2,z_3) \in \mathbb{R}^3\ \|\ z_1^\alpha z_2^{1-\alpha} \geq \|z_3\|, z_1,z_2> 0 \rbrace$.


where $\mathbf{smat}(x)$ creates a symmetric matrix from $x$. This can be done in a plethora of ways, but and often used definition is

$$
\begin{equation}
    \mathbf{smat}(x)  = 
    \begin{bmatrix}
        x_1 & x_2/\sqrt{2} & \cdots & x_n/\sqrt{2}\\
        x_2/\sqrt{2} & x_{n+1} & \cdots & x_{2n-1}/\sqrt{2}\\
        \vdots & \vdots & \ddots & \vdots\\
        x_n/\sqrt{2} & x_{2n-1}/\sqrt{2} & \cdots & x_{n(n+1)/2}
    \end{bmatrix}
\end{equation}
$$

where the scaling with $\sqrt{2}$ have been introduced in order to preserve the inner product. I.e. that $x^\top y = \text{tr}\left(\mathbf{smat}(x)\mathbf{smat}(y)\right)$. Furthermore, we define the inverse operation $\mathbf{svec}(X)$ that creates a vector from a symmetric matrix $X$.

### The dual problem
The optimization problem of interest equivalently be written as

$$
\begin{alignat*}{3}
    &\text{minimize}\quad  &&\frac{1}{2}x^\top Px + q^\top x\\
    &\text{subject to} && b - Ax \in \mathcal{K}.
\end{alignat*}
$$

Note that the conic constraint $b - Ax \in \mathcal{K}$ is equivalent to the *generalized inequality*

$$
    0 \preceq_{\mathcal{K}}	s = b - Ax.
$$

However, the standard form (similar to LPs) requires that the inequality is turned the other way. As such we negate the expression so that, resulting in 

$$
    Ax - b \preceq_\mathcal{K} 0
$$

We now write up the Lagrangian as

$$
    L(x,z) =  \frac{1}{2}x^\top Px + q^\top x + z^\top\left(Ax - b\right)
$$

Since $-(Ax -b) \in \mathcal{K} $  we must have that $z \in \mathcal{K}^\ast$ in order to avoid the Lagrangian being unbounded from below. 
Now since we want to find a stationary point we compute the gradient and set it to zero

$$
    \nabla_x L(x,z) = Px + q + A^\top z = 0.
$$

Now notice that the Lagrangian can be written as

$$
    x^\top\left(\frac{1}{2} Px + q + A^\top z\right) - b^\top z
$$

The thing inside the parentheses, is *almost* equal to our constraint on the gradient. However, we can do a standard mathematical trick of adding zero followed by rearranging the expressions

$$
    x^\top\left(\frac{1}{2} Px + q + A^\top z\right) + \left(\underbrace{\frac{1}{2}x^\top Px - \frac{1}{2}x^\top Px}_{=0}\right) - b^\top z  = x^\top\underbrace{\left(Px + q + A^\top z\right)}_{\nabla_xL(x,z) = 0} - b^\top z - \frac{1}{2}x^\top Px
$$

As such the dual formulation of the problem is

$$
\begin{aligned}
	\text{maximize}&\quad 	 -\frac{1}{2}x^\top P x - b^\top z\\
	\text{subject to}&\quad Px + q + A^\top z = 0\\	
	&\quad z \in \mathcal{K}^\ast.\\
\end{aligned}
$$

### Homogenous self-embedding
First we define the duality gap $\gamma$ as

$$
\begin{aligned}
    \gamma :&= \left(\frac{1}{2}x^\top Px + q^\top x\right) - \left(-\frac{1}{2}x^\top Px - b^\top z\right)  \\
    &= x^\top P x + q^\top x + b^\top z = s^\top z
\end{aligned}
$$

where we used that $q = -Px - A^\top z$ and $s = b - Ax$. By combining the constraints of the primal and dual problems and setting the duality gap equal to zero we get the standard KKT conditions

$$
\begin{aligned}
    Ax + s          &= b\\
    Px + A^\top z   &= -q\\
    s^\top z        &= 0\\
    (s,z)           &\in \mathcal{K}\times\mathcal{K}^\ast.
\end{aligned}
$$

Solving non-linear KKT system directly result in

$$
\begin{aligned}
	\underset{(x,s,z)}{\text{minimize}}&\quad 	 s^\top z\\
	\text{subject to}&\quad x^\top Px + q^\top x + b^\top z = 0\\
        &\quad Px  + A^\top z + q\tau = 0\\
        &\quad Ax + s - b\tau = 0\\
	&\quad (s,z,\tau,\kappa) \in \mathcal{K}\times\mathcal{K}^\ast.\\
\end{aligned}
$$

An issue with the above is that the duality gap is explicitly set to zero. This condition can be relaxed after change of variables $x \rightarrow x/\tau,  z\rightarrow z/\tau,$ and $s \rightarrow s/\tau$ and instead solve

$$
\begin{aligned}
	\underset{(x,s,z,\tau,\kappa)}{\text{minimize}}&\quad 	 s^\top z + \tau\kappa\\
	\text{subject to}&\quad \frac{1}{\tau}x^\top Px + q^\top x + b^\top z = -\kappa\\
        &\quad Px  + A^\top z + q\tau = 0\\
        &\quad Ax + s - b\tau = 0\\
	&\quad (s,z,\tau,\kappa) \in \mathcal{K}\times\mathcal{K}^\ast\times\mathbb{R}_+\times\mathbb{R}_+.\\
\end{aligned}
$$

### (Conjugate) Barrier Functions
 For each nonzero cone we define a strictly convex *barrier function* $f: \mathcal{K}\rightarrow \mathbb{R}$ and its associated *conjugate barrier* $f_\ast: \mathcal{K}^\ast\rightarrow\mathbb{R}$. In particular, we use barrier functions that meet the following definition


> A function $f: \mathcal{K}\rightarrow\mathbb{R}$ is called a logarithmically homogenous self-concordant barrier with degree $\nu\geq 1$ ($\nu$-LHSCB) for the convex cone $\mathcal{K}$ if it satisfies the following properties
$$
    \begin{alignat*}{2}
        &|\nabla^3f(x)[r,r,r]| \leq 2(\nabla^2f(x)[r,r])^{3/2}\quad && \forall x \in \textup{int}\left(\mathcal{K}\right), r\in\mathbb{R}^d,\\
        &f(\lambda x) = f(x) - \nu\log(\lambda), \quad && \forall x \in \mathcal{K}, \lambda > 0.
    \end{alignat*}
$$

The *conjugate barrier function* $f_\ast: \mathcal{K}^\ast\rightarrow\mathbb{R}$ is defined as <d-footnote>This is not to be confused with the *convex conjugate* of a function, which is denoted by $f^\ast(y) = \text{sup}\{y^\top x - f(x),\ x\in\text{int}\left(\mathcal{K}\right)\}$. The relation between the two conjugates are $f_\ast(y) = f^\ast(-y)$ and $\nabla f_\ast(y) = -\nabla f^\ast(-y)$.</d-footnote>

$$
    f_\ast(y) := \underset{x\in\text{int}\left(\mathcal{K}\right)}{\text{sup}}\left\{-y^\top x - f(x)\right\}.
$$

The function $f_\ast$ is $\nu$-LHSCB for $\mathcal{K}^\ast$. The primal gradient $\nabla f$ satisfies

$$
    x^\top \nabla f(x) = -\nu, \quad \forall x\in\text{int}\left(\mathcal{K}\right).
$$

In cases where the conjugate barrier function is known only through the definition (i.e. rather than a closed-form representation) we can compute its gradient as the solution to

$$
    \nabla f_\ast(y) := -\text{arg} \underset{x\in\text{int}(\mathcal{K})}{\text{sup}}\lbrace-y^\top x - f(x)\rbrace.
$$

The relations collectively ensure that

$$
\begin{align*}
    f_\ast(y) 
    &= -y^\top\left(-\nabla f_\ast(y)\right) - f(-\nabla f_\ast(y))\\
    &= -\nu - f(-\nabla f_\ast(y)).
\end{align*}
$$

And also

$$
    -\nabla f_\ast(-\nabla f(x)) = x, \forall x \in \text{int}\left(\mathcal{K}\right),\quad -\nabla f(-\nabla f_\ast(y)) = y,\ \forall y \in \text{int}\left(\mathcal{K}\right)^\ast.
$$


### Central Path
We start by defining the function $G$ as 

$$
    G(x,z,s,\tau,\kappa) := 
    \begin{bmatrix} 0\\ s\\ \kappa \end{bmatrix} -
    \begin{bmatrix} P & A^\top & q \\ -A & 0 & b \\ -q^\top & -b^\top & 0\end{bmatrix}
    \begin{bmatrix} x\\z\\\tau\end{bmatrix} + 
    \begin{bmatrix} 0 \\ 0 \\ \tau^{-1}x^\top P x\end{bmatrix}.
$$

In <d-cite key="goulart2024clarabelinteriorpointsolverconic"></d-cite> $G$ is used to show the existence of solutions and infeasibility detection.


Now assume that $f:\mathcal{K} \rightarrow \mathbb{R}$ is a $\nu$-LHSCB function on $\mathcal{K}$ with conjugate barrier $f_\ast$ for some $\nu \geq 1$. Given any initial $v^0 \in\mathcal{C}$, we define the *central path* of $v^\ast(\mu)$ as the unique solution to

$$
\begin{align}\label{eq:central:path}
    G(v) =&\ \mu G(v^0)\\
    s = -\mu\nabla f_\ast(z),\ & z = -\mu\nabla f(s), \label{eq:central:path:2}
\end{align}
$$

which implies that

$$
    s^\top z/\nu = \mu.
$$

If the cone $\mathcal{K}$ is symmetric, then we can replace the condition with a simpler condition defined in terms of the Jordan product on $\mathcal{K}$:

$$
\begin{equation}\label{eq:jordan}
    s\circ z = \mu e,
\end{equation}
$$

where e is the standard idempotent of $\mathcal{K}$. The core of the implemented interior-point algorithm amounts to a Newton-like method for computing the solution to the system of equation representing the central path. 

### Scaling Matrices
We now introduce the idea of a scaling matrix $H$. The choice of scaling matrix $H$ depends on which way we linearize the central path. For symmetric cones the most common choice is the Nesterov-Todd scaling. For nonsymmetric cones the central path is defined by the set of point satisfying \eqref{eq:central:path}. For the zero cone we set $H=0$.

#### Symmetric cones
For symmetric cones the central path described with \eqref{eq:jordan} is linearized. The Nesterov-Todd scaling exploits the self-scaled property of symmetric cones to compute a unique scaling point $w\in\mathcal{K}$ satisfying

$$
    H(w)s = z.
$$

We can factorize $H(w)$ as $H^{-1}(\omega) = W^\top W$ and we set $H=H^{-1}(w)$. The factors $w, W$ are computing following <d-cite key="vandenberghe2010cvxopt"></d-cite> except for second-order cones where we instead apply a sparse factorization strategy of <d-cite key="domahidi2013a"></d-cite>. 
<!-- Two sparse factorization strategies are described in \autoref{sec:soc}. -->

#### Nonsymmetric cones
As nonsymmetric cones are not self-scaled, and we can not just \eqref{eq:jordan} when describing the central path. Instead, we must linearize \eqref{eq:central:path:2} instead. A general primal-dual scaling strategy suitable for nonsymmetric cones was consequently proposed in <d-cite key="Tunel2001"></d-cite>, and used later in <d-cite key="dahl2022a"></d-cite>, which relies on the satisfaction of two secant equations

$$
    Hz = s, \quad H\nabla f(s) = \nabla f_\ast(z).
$$

We now define *shadow iterates* as

$$
    \tilde{z} := - \nabla f(s), \quad \tilde{s} := -\nabla f_\ast(z),
$$

with

$$
    \tilde{\mu} = \langle \tilde{s}, \tilde{z}\rangle\nu^{-1}
$$

A scaling matrix $H$ can be obtained from the rank-4 Broyden-Fletcher-Goldfarb-Shanno (BFGS) update, which is commonly used in quasi-Newton methods,

$$
    H := H_{\text{BFGS}} := Z(Z^\top S)^{-1}Z^\top + H_a - H_a S(S^\top H_a S)^{-1}S^\top H_a,
$$

where $Z := [z,\tilde{z}]$, $S:=[s,\tilde{s}]$, and $H_a \succ 0$ is an approximation of the Hessian. In our implementation we choose $H_a = \mu\nabla^2f_\ast(z)$ following <d-cite key="dahl2022a"></d-cite>. Putting things together means that the computation of $H_{\text{BFGS}}$ reduces to a rank-3-update as

$$
\begin{align*}
    H_{\text{BFGS}} = \mu\nabla^2f_\ast(z) + &(2\mu\nu)^{-1}\delta_s\left(s + \mu\tilde{s} + (\mu\tilde{\mu} -1)^{-1}\delta_s\right)^\top\\
     + &(2\mu\nu)^{-1}\left(s + \mu\tilde{s} + (\mu\tilde{\mu} -1)^{-1}\delta_s\right)\delta_s^\top\\
     -&\mu\frac{\left(\nabla^2 f_\ast(z)\tilde{z} - \tilde{\mu}\tilde{s}\right)\left(\nabla^2 f_\ast(z)\tilde{z} - \tilde{\mu}\tilde{s}\right)^\top}{\langle \tilde{z},\nabla^2f_\ast(z)\tilde{z}\rangle - \nu\tilde{\mu}^2}.
\end{align*}
$$

A more detailed description can be found in <d-cite key="dahl2022a"></d-cite>.

{% details Click here to know more %}
The above is defined in terms of the conjugate barrier function $f_\ast(z)$. This in contrast to <d-cite key="dahl2022a"></d-cite> that defines it terms of the barrier function. 

The approach of using the dual follows that of ECOS-Exp described in <d-cite key="serrano"></d-cite> where symmetric cones are scaled by the barrier while the exponential cones are scaled by the conjugate barrier. Clarabel might be going for the conjugate barrier above to be able to be easily compatible with the formulation in <d-cite key="serrano"></d-cite> - Or simply because it implemented the dual-scaling approach before the primal-dual-scaling approach. In addition, as described in <d-cite key="chen2023a"></d-cite>, the dual scaling result in augmented sparse conjugate Hessians for *some* nonsymmetric cones.
{% enddetails %}

### Solver initialization
The initialization strategies depend on whether the objective is linear or quadratic or not and whether the cones are symmetric or not. However, in all cases the initial scalar values are set as $\tau^0= \kappa^0 = 1$.

#### Symmetric Cones
The initialization strategies below can be thought of as projections of the equality constraints onto cones in the direction of $e$.

#### Quadratic Objective
If $P\neq 0$ the initialization happens by solving the following linear system

$$
    \begin{bmatrix}P & A^\top \\ A & -I_E\end{bmatrix}
    \begin{bmatrix}x \\ z\end{bmatrix}
    = \begin{bmatrix}-q \\ b\end{bmatrix}
$$

with $I_E = I - E$ with $E$ being a matrix with ones on the diagonal corresponding to equality constraints but otherwise zero. The above linear system corresponding to solving the following optimization problem

$$
\begin{align}
    \text{minimize}\quad &\frac{1}{2}x^\top P x + q^\top x + \frac{1}{2}\|Ax - b\|_2^2\\
    \text{subject to}\quad & EAx = Eb,\\
\end{align}
$$

We set $x^0 = x$ and

$$
    s^0 = 
    \begin{cases}
        -z, \quad &    \alpha_\text{p} < -\epsilon\\
        -z + (\epsilon + \alpha_\text{p})e, \quad &\text{otherwise}
    \end{cases}, \quad z^0 = 
    \begin{cases}
        z, \quad &    \alpha_\text{d} < -\epsilon\\
        z + (\epsilon + \alpha_\text{d})e, \quad &\text{otherwise}
    \end{cases}.
$$

where $\alpha_\text{p} = \inf\lbrace\alpha\ \|\ -z + \alpha e\in\mathcal{K}\rbrace$ and $\alpha_\text{d} = \inf\lbrace\alpha\ \|\ -s + \alpha e\in\mathcal{K}\rbrace$.

#### Linear Objective
In the case of $P=0$ instead solve two linear systems

$$
    \begin{bmatrix}0 & A^\top \\ A & -I_E\end{bmatrix}
    \begin{bmatrix} x & t\\ s & z\end{bmatrix}
    = \begin{bmatrix} 0 & -q\\ b & 0\end{bmatrix}.
$$

From this we set

$$
    x^0 = x, \quad s^0 = 
    \begin{cases}
        -s, \ &    \alpha_\text{p} < -\epsilon\\
        -s + (\epsilon + \alpha_\text{p})e, \ &\text{otherwise}
    \end{cases},
    \quad 
    z^0 = 
    \begin{cases}
        z, \ &    \alpha_\text{d} < -\epsilon\\
        z + (\epsilon + \alpha_\text{d})e, \ &\text{otherwise}
    \end{cases},
$$

where $\alpha_\text{p} = \text{inf}\lbrace \alpha\ \|\ -s + \alpha e\in\mathcal{K}\rbrace$ and $\alpha_\text{d} = \text{inf}\lbrace \alpha\ \|\ z + \alpha e\in\mathcal{K}\rbrace$.


#### Nonsymmetric cones
When $\mathcal{K}$ contains any nonsymmetric cone, one should instead apply a unit initialization strategy. In this case, we initialize both primal and dual variables at a point on the central path satisfying $z=s=-\nabla f_\ast(z)$ (corresponding to $\mu^0= 1$). This is equivalent to solving the unconstrained optimization problem

$$
    \underset{z}{\text{minimize}}\quad \frac{1}{2}\|z\|_2^2 + f_\ast(z),
$$

which is strictly convex and has a unique solution. It yields $s^0_\text{sym} = z^0_\text{sym} = e$ for symmetric cones, and

$$
    s^0_\text{exp} = z^0_\text{exp} \approx (-1.051383, 0.556409, 1.258967),
$$

for exponential cones, and

$$
    s^0_\text{pow} = z^0_\text{pow} \approx (\sqrt{1 + \alpha}, \sqrt{2 - \alpha},0),
$$

for power cones with parameter $\alpha$.

### Termination criteria
All termination criteria are based on unscaled problem data and iterates, i.e. after the Ruiz scaling has been reverted.

#### Feasibility Checks
For checks of primal and dual feasibility we introduce the normalized variables
$\overline{x} = x/\tau$, $\overline{s} = s/\tau$, $\overline{z}=z/\tau$ and define the primal and dual residual as follows

$$
\begin{alignat*}{2}
        r_p &:= A\overline{x} + \overline{s} - b\quad && \text{(primal residual)}\\
        r_d &:= P\overline{x} + A^\top\overline{z} + q\quad &&\text{(dual residual)}.
\end{alignat*}
$$

Likewise, we define the primal and dual objectives as

$$
\begin{alignat*}{2}
        g_p &:= \frac{1}{2}\overline{x}^\top P\overline{x} - q^\top\overline{x}\quad && \text{(primal objective)}\\
        g_d &:= -\frac{1}{2}\overline{x}^\top P\overline{x} -b^\top\overline{z}\quad &&\text{(dual objective)}.
\end{alignat*}
$$

Using the above definitions we can declare convergence if **all** the following three holds:

$$
\begin{alignat*}{2}
        \|r_p\| &< \epsilon_f\cdot\text{max}\{1, \|b\|_\infty + \|\overline{x}\| + \|\overline{s}\|\} \quad&& \text{(primal convergence)}\\
        \|r_d\| &< \epsilon_f\cdot\text{max}\{1, \|q\|_\infty + \|\overline{x}\| + \|\overline{z}\|\} \quad&&\text{(dual convergence)}\\
        |g_p - g_d| &< \epsilon_f\cdot\text{max}\{1, \text{min}\{|g_p|, |g_d|\}\} \quad&& \text{(duality gap convergence)}
\end{alignat*}
$$

We specify a default value of $\epsilon_f = 10^{-8}$ as well as a weaker threshold of $\epsilon_f = 10^{-5}$ when testing for "near optimality" in cases of early termination (e.g. lack of progress, timeout, iterations limit, etc.).

#### Infeasibility checks
When testing or infeasibility we do *not* normalize iterates, but rather work directly with the unscaled variables since infeasibility corresponds to the case where $\tau\rightarrow 0$. We declare primal infeasibility if the following holds

$$
\begin{alignat*}{2}
        \|A^\top z\| &< -\epsilon_{i,r}\cdot\text{max}\{1, \|x\| + \|z\|\} \quad&& \text{(primal infeasibility)}\\
        b^\top z &< -\epsilon_{i,a} \quad&& \text{(primal infeasibility)}.
\end{alignat*}
$$

Similarly, we define dual infeasibility if the following hold

$$
\begin{alignat*}{2}
        \|Px\| &< -\epsilon_{i,r}\cdot\text{max}\{1, \|x\|\}\cdot\left(b^\top z\right) \quad&& \text{(dual infeasibility)}\\
        \|Ax + s\| &< -\epsilon_{i,r}\cdot\text{max}\{1, \|x\| + \|s\|\}\cdot\left(q^\top x\right)\quad&& \text{(dual infeasibility)}\\
        b^\top z &< -\epsilon_{i,a} \quad&& \text{(dual infeasibility)}
\end{alignat*}
$$

We set the relative and absolute tolerances as $\epsilon_{i,r}=\epsilon_{i,a}=10^{-8}$ and allow for weaker thresholds to declare "near infeasibility" certificates in cases of early termination.


## Computing Step Directions
The linearization of the homogenous embedding gives the following system of equations, where $d=(d_x,d_z,d_\tau,d_s,d_\kappa)$ is a vector of residuals

$$
\begin{aligned}
    \begin{bmatrix}0\\\Delta s\\\Delta\kappa\end{bmatrix}
    -
    &\begin{bmatrix}
        P & A^\top & q\\
        -A & 0 & b\\
        -(q + 2P\xi)^\top & -b^\top & \xi^\top P\xi
    \end{bmatrix}
    \begin{bmatrix}\Delta x\\\Delta z\\\Delta\tau\end{bmatrix}
    =
    -\begin{bmatrix}d_x\\d_z\\d_\tau\end{bmatrix}\\
    &H\Delta z + \Delta s = -d_s,\quad \kappa\Delta\tau + \tau\Delta\kappa = -d_\kappa,
\end{aligned}
$$

where $\xi = x/\tau$ and $H\in\mathbb{R}^{m\times m}$ is a positive definite matrix that is normally referred to as the *scaling matrix*. Note that in order to allow for nonsymmetric cones we use $H$, which is in contrast to <d-cite key="vandenberghe2010cvxopt"></d-cite> that uses $\lambda \circ (W\Delta z + W^{-\top}\Delta s) = d_s$ with $H=W^\top W$.\\


We now eliminate $(\Delta s, \Delta\kappa)$ from the system resulting in

$$
\begin{equation}\label{eq:3x3:system}
\begin{aligned}
    \begin{bmatrix}0\\\Delta s\\\Delta\kappa\end{bmatrix}
    -
    &\begin{bmatrix}
        P & A^\top & q\\
        -A & H & b\\
        -(q + 2P\xi)^\top & -b^\top & \xi^\top P\xi + \kappa\tau^{-1}
    \end{bmatrix}
    \begin{bmatrix}\Delta x\\\Delta z\\\Delta\tau\end{bmatrix}
    =
    -\begin{bmatrix}d_x\\d_z - d_s\\d_\tau - d_\kappa\tau^{-1}\end{bmatrix}\\
    &\Delta s = -d_s - H\Delta z, \quad \Delta\kappa = -\left(d_\kappa + \kappa\Delta\tau\right)\tau^{-1}.
\end{aligned}
\end{equation}
$$

The above system can be solved by a pair of linear system with a common coefficient matrix, meaning that it only has to be factorized once

$$
\begin{equation}\label{eq:kkt}
    \begin{bmatrix}P & A^\top \\ A & -H\end{bmatrix}
    \begin{bmatrix}[c|c] \Delta x_1 & \Delta x_2\\ \Delta z_1 & \Delta z_2 \end{bmatrix}
    = 
    \begin{bmatrix}[c|c] d_x & -q\\ -(d_z - d_s) & b\end{bmatrix}
\end{equation}
$$

From the above solution we can recover the search direction $(\Delta x,\Delta z,\Delta s, \Delta\tau, \Delta\kappa)$ as follows

$$
\begin{align*}
    \Delta x &= \Delta x_1 + \Delta\tau\cdot\Delta x_2,\\
    \Delta z &= \Delta z_1 + \Delta\tau\cdot\Delta z_2\\
    \Delta s &= -d_s - H\Delta z,\\ 
    \Delta\tau &= \frac{d_\tau - d_\kappa\tau^{-1} + \left(2P\xi + q\right)^\top\Delta x_1 + b^\top\Delta z_1}{\kappa\tau^{-1} + \xi^\top P\xi - \left(2P\xi + q\right)^\top\Delta x_2 - b^\top \Delta z_2}\\
    &= \frac{d_\tau - d_\kappa\tau^{-1} +q^\top\Delta x_1 + b^\top\Delta z_1 + 2\xi^\top P\Delta x_1}{\|\Delta x_2 - \xi\|_P^2 - \|\Delta x_2\|_P^2  -q^\top\Delta x_2 - b^\top\Delta z_2}\\
    \Delta\kappa &= -\left(d_\kappa + \kappa\Delta\tau\right)\tau^{-1}.
\end{align*}
$$

The actual step is then obtained by computing the maximal step size $\alpha$ that ensures that the new update is still in the interior of the conic constraints. 

### The affine and centering directions
At every interior-point iteration we solve KKT system for two sets of right-hand sides each, corresponding to the so-called affine and centering directions. In short this means solving \eqref{eq:kkt} twice, however since the system matrix stays the same and the right-hand side $(-q,b)$ stays the same the computation requires only a single numerical factorization\footnote{As the sparsity pattern does not change the symbolic factorization is reused} and three solves.

The two sets of right-hand sides corresponds to a *predictor* step and a corrector step *corrector*. The steps are also known as the *affine* step and the *centering* step. In our case the affine step has the right-hand side of

$$
\begin{equation}
    d_x = r_x,\ d_z=r_z,\ d_\tau=\kappa\tau,\ d_s=s,
\end{equation}
$$

while the corrector step have the right-hand side of

$$
\begin{align*}
    d_x = (1-\sigma)r_x, 
    d_z &= (1-\sigma)r_z, d_\tau=(1-\sigma)r_\tau, d_\kappa = \kappa\tau + \Delta\kappa\Delta\tau - \sigma\mu\\
    d_s &= 
    \begin{cases}
        W^\top\left(\lambda \diamond (\lambda\circ\lambda  - \sigma\mu e + \eta)\right)\quad &(\text{symmetric})    \\
        s + \sigma\mu\nabla f_\ast(z) + \eta \quad &(\text{nonsymmetric}),
    \end{cases}
\end{align*}
$$

where $\diamond$ denotes the inverse operator of $\circ$. Computation of a higher-order correction term $\eta$ is a heuristic technique that is known to accelerate the convergence of IPMs significantly. The choice of this term varies depending on the choice of the scaling matrix $H$ and whether a given constraint is symmetric or not. For symmetric cones we use the Mehrotra correction $\eta= (W^{-\top}\Delta s)\circ (W\Delta z)$, while for nonsymmetric cones we compute $\eta$ using the 3rd-order correction from <d-cite key="dahl2022a"></d-cite>, i.e.

$$
\begin{equation}
    \eta = -\frac{1}{2}\nabla^3 f_\ast(z)[\Delta z, \nabla^2 f_\ast (z)^{-1}\Delta s].
\end{equation}
$$
We set $\sigma = (1 - \alpha_{\text{aff}})^3$ where $\alpha_{\text{aff}}$ is the step size of the affine step.


## Solving the KKT system
The main computational efforts in the interior point method is to solve the Karush-Kuhn-Tucker (KKT) system of the form

$$
\begin{equation}\label{eq:kkt:initial}
    \underbrace{\begin{bmatrix}
        P & A^\top\\
        A & -H\\
    \end{bmatrix}}_{K}
    \underbrace{\begin{bmatrix} x \\ z \end{bmatrix}}_{y}
    =
    \underbrace{\begin{bmatrix} b_x \\ b_z \end{bmatrix}}_{b},
\end{equation}
$$

The scaling matrix, however, has the unfortunate property that it becomes ill-conditioned when the iterates is close to the boundary. As such one instead solves the regularized problem

$$
\begin{equation}\label{eq:kkt:diag}
    \underbrace{\begin{bmatrix}
        P + \epsilon I & A^\top\\
        A & -\left(H + \epsilon I\right)\\
    \end{bmatrix}}_{K + D}
    \underbrace{\begin{bmatrix} x \\ z \end{bmatrix}}_{y}
    =
    \underbrace{\begin{bmatrix} b_x \\ b_z \end{bmatrix}}_b,
\end{equation}
$$

where $D$ denotes the added diagonal part. Note that while solving \eqref{eq:kkt:diag} is more numerically stable, it is not the system that we actually aim to solve i.e. \eqref{eq:kkt:initial}. The regularization $\epsilon$ is computed as

$$
\begin{equation}
    \epsilon = \epsilon_\text{constant} + \epsilon_\text{proportion}\|\text{diag}(K)\|_\infty.
\end{equation}
$$

(Default settings: $\epsilon_\text{constant} = 1e-8$, $\epsilon_\text{proportion}=\text{eps}(T)^2$).
It turns out that the effects of $D$ can be removed using a process *inspired by (standard) iterative refinement*. In Clarabel and other software the process is still referred to as iterative refinement which is a slight abuse of nomenclature as iterative refinement was originally introduced to correct floating point errors in the solution and not correcting for a regularization term. More importantly the underlying equations are also slightly different. As such it is important to disable any iterative refinement applied supplied by the direct sparse solver software used and instead let Clarabel handle it internally.

### Regularized Iterative Refinement
As described in <d-cite key="osqp"></d-cite>, the regularization term can be removed by solving a sequence of systems of the form

$$
\begin{equation}\label{eq:ir:0}
    (K + D)\Delta x_m = b - Kx_m,\quad x_{m+1} = x_m + \Delta x_m.
\end{equation}
$$

The sequence $\{x_m\}$ converges towards to the true solution $Kx = b$ provided that it exist. To show this first notice that we can rewrite \eqref{eq:ir:0} as 

$$
\begin{equation}\label{eq:ir:2}
    (K + D)x_{m+1} - Dx_m = b.
\end{equation}
$$

Now assume that $x^0$ is the solution to the equation $Kx^0 = b$ (which is the equation we aim to solve). Now we perform the mathematicians favorite trick of adding 0 (in this case $Dx^0 - Dx^0$), it follows that

$$
\begin{equation}\label{eq:ir:1}
    (K+D)x^0 - Dx^0 = b.
\end{equation}
$$

We now subtract \eqref{eq:ir:1} from \eqref{eq:ir:2}

$$
\begin{equation}
    (K+D)(x_{m+1} - x^0) - D(x_m - x^0) = 0.
\end{equation}
$$

Rewriting the above gives us

$$
\begin{equation}
    x_{m+1} - x^0 = \left(K+D\right)^{-1}D(x_m - x^0).
\end{equation}
$$

The final step is to take norms on both sides resulting in

$$
\begin{equation}
    \|x_{m+1} - x^0\| \leq \|\left(K+D\right)^{-1}D\|\|x_m - x^0\|.
\end{equation}
$$

Thus, if $\|\left(K+D\right)^{-1}D\| \leq 1$ we have that $x_m \rightarrow x^0$ for $m\rightarrow \infty$.

### Matrix Equilibration
The equilibration in Clarabel is a modified version of Ruiz scaling <d-cite key="ruizscaling"></d-cite>, similar as to what is done in the OSQP package <d-cite key="osqp"></d-cite>. The thing is based on the scaling of the *constant* part of the KKT system (i.e. the part within the primal-dual scaling matrices) as shown below

$$
\begin{equation}
    M = 
    \begin{bmatrix}
        P & A^\top \\
        A & 0
    \end{bmatrix}.
\end{equation}
$$

The aim is to find a diagonal scaling matrix $S$ such that 

$$
\begin{equation}
    SMS = 
    \begin{bmatrix}
        D & 0 \\ 0 & E
    \end{bmatrix}
    \begin{bmatrix}
        P & A^\top \\
        A & 0 
    \end{bmatrix}
    \begin{bmatrix}
        D & 0 \\ 0 & E
    \end{bmatrix}
    = 
    \begin{bmatrix}
        DPD & DA^\top E \\ EAD & 0
    \end{bmatrix},
\end{equation}
$$

is better conditioned. The above can be seen as a transformation of input data to a conic optimization problem

$$
\begin{alignat*}{3}
    &\text{minimize}\quad  && \frac{1}{2}\bar{x}^\top \bar{P}\bar{x} + \bar{q}^\top \bar{x} \\
    &\text{subject to} && \bar{A}\bar{x} + \bar{s} = \bar{b}\\
    &\ && \bar{s} \in \mathcal{K},
\end{alignat*}
$$

where $\bar{x} = D^{-1}x$, $\bar{P} = cDPD$, $\bar{q} = cDq$, $\bar{A} = EAD$.

The main idea of the equilibration procedure is to scale the rows (and columns) for the matrix $M$ so that they have equal $\ell_p$-norm. An efficient approximate solution to this problem is the so-called Modified Ruiz scaling described in <d-cite key="osqp"></d-cite> based on the Ruiz scaling described in <d-cite key="ruizscaling"></d-cite>.

<!-- \begin{algorithm}[H]
    \caption{Modified Ruiz Scaling}\label{alg:ruiz}
    \KwData{$c=1, S=I, \delta = 0, \bar{P}=P, \bar{q} = q, \bar{A}=A, \bar{\mathcal{K}} = \mathbf{K}$}
    \While{$\| 1- \delta\|_\infty > \varepsilon_\text{equilibration}$}{
        \While{$i < m+n$}{
        $\delta_i \leftarrow 1/\sqrt{\|M_i\|_\infty}$
        }
        $\bar{P}, \bar{q}, \bar{A}, \bar{\mathcal{K}}\leftarrow \text{scale using } \text{diag}(\delta)$\\
        $\gamma \leftarrow 1 / \max\{\text{mean}(\|\bar{P}_i\|_\infty), \|\bar{q}\|_\infty\}$\\
        $\left(\bar{P}, \bar{q}\right) \leftarrow \left(\gamma\bar{P},\gamma\bar{q} \right)$\\
        $S \leftarrow \text{diag}\left(\delta\right)S, c \leftarrow \gamma c$
    }
    \Return{$S,c$}
\end{algorithm} -->

### Sparse Solvers
More often than not the main computational time in an interior-point method is solving \eqref{eq:kkt:diag}. Luckily, the linear system is often sparse leading to fast factorization algorithms. As such it is important it utilizes an efficient implementation of a direct sparse solver if one wants the interior-point solver to be computationally efficient. In Clarabel it is possible to utilize the following sparse solvers

* (Rust) fear-rs. Multithreaded. (Actively maintained - Gets continuously improved).
* (Rust and Julia) Built-in Quasi-Definite LDL (QDLDL) based on  <d-cite key="davis2005a"></d-cite>. Single threaded.
* (Rust and Julia) PanuaPardiso (Requires license). Multithreaded.
* (Julia) oneMKL-Pardiso (Only Intel CPUs). Multithreaded.
* (Julia) cuDSS (CUDA Direct Sparse Solver). GPU-based.
* (Julia) libHSL (Requires license). Multithreaded.
* (Julia) CHOLMOD. Multithreaded.


## Sparsity exploitation
### Augmented sparsity
In numerical computations one often meet a matrix of the form *sparse-plus-low-rank*. The general form of this is the following

$$
\begin{equation}
    H 
    := S - \sum_{j=1}^{n_V}v_iv_i^\top  + \sum_{i=1}^{n_U}u_iu_i^\top  
    = S - VV^\top + UU^\top ,
\end{equation}
$$

where $S$ is sparse and $V = \begin{bmatrix}v_1 & \dots & v_{n_V}\end{bmatrix}$ and $U = \begin{bmatrix}u_1 & \dots & u_{n_U}\end{bmatrix}$ with $n_U, n_V \ll n$. The *bad* thing of the above matrix is that forming the outer products (i.e. assembling the *plus-low-rank* part) will ruin the sparsity of $S$ and $H$ in turn will be a dense matrix. Given that our aim is to solve a system of the form $Hx = b$ lets look at the structure of $Hx$ by performing the product

$$
\begin{equation}
    Hx 
    = Sx - V(\underbrace{V^\top x}_{-t_V}) + U(\underbrace{U^\top x}_{t_U})
    = Sx + Vt_V + Ut_U.
\end{equation}
$$

The above together with the constraints $V^\top x +  t_V = 0$ and $U^\top x - t_U = 0$ we can solve $Hx=b$ by solving the following equivalent system

$$
\begin{equation}
    \underbrace{\begin{bmatrix}
        S & V & U\\
        V^\top & I_{n_V}\\
        U^\top & 0 & - I_{n_U}
    \end{bmatrix}}_{H_\text{aug}}\begin{bmatrix}x\\ t_V \\ t_U\end{bmatrix}
    =
    \begin{bmatrix} b \\ 0 \\ 0 \end{bmatrix}.
\end{equation}
$$

Note that while $H_\text{aug}$ is *sparse* it is larger. An important property of the augmented system is that if $S\in\mathcal{S}$ and $D - VV^\top \succ 0$ then $H_\text{aug}$ is *Quasi-Definite* and therefore *strongly factorizable*, meaning that for every permutation $P$ one can find $L$ and $D$ so that $PH_\text{aug}P^\top = LDL^\top$ <d-cite key="vanderbei"></d-cite>.

### Second-order cones
#### The Clarabel approach
Clarabel uses the same approach as in ECOS <d-cite key="domahidi2013a"></d-cite>. In short, they utilize that the structure of $W_k^\top W_k$ for a second-order cone is augmented-sparse, meaning that it can be split as follows

$$
\begin{equation}\label{eq:clarabel:uv}
    W_k^\top W_k  = S_k - v_kv_k^\top + u_ku_k^\top,
\end{equation}
$$

where $S_k - v_kv_k^\top \succ 0$. We can generalize the above for all second-order cones by defining

$$
    S = \textbf{blkdiag}\left(S_1,S_2,\dots, S_n\right), \quad 
    V = \textbf{blkdiag}\left(v_1,v_2,\dots, v_n\right), \quad 
    U = \textbf{blkdiag}\left(u_1,u_2,\dots, u_n\right),
$$ 

meaning that 

$$
    W^\top W = S - VV^\top + UU^\top.
$$

As such the KKT of the form

$$
\begin{bmatrix}
	P & A^\top \\
	A & -W^\top W
\end{bmatrix}
\begin{bmatrix} x \\ z \end{bmatrix}
= 
\begin{bmatrix} b_x \\ b_z \end{bmatrix}
$$

can equivalently be solved by solving

$$
\begin{bmatrix}
	P & A^\top & 0 & 0 \\
	A & -S & -V^\top & -V^\top\\
 	0 & -V &  -I & 0\\
	0 & -U & 0 & I\\
\end{bmatrix}
\begin{bmatrix} x \\ z \\ t_V \\ t_U \end{bmatrix}
= 
\begin{bmatrix} b_x \\ b_z \\ 0 \\0 \end{bmatrix},
$$

The approach is simple as it leaves $A$ intact and does not scale $z$. However, one has to deal with computing $V$ and $U$, which is not unique. In fact Clarabel and ECOS applies different approaches when calculating $V$ and $U$. A detail here is that one wants to compute $V$ and $U$ in a way for which $S - VV^\top \succ 0$ so that the system remains quasi-definite (and therefore *strongly factorizable*).


#### The CVXOPT approach
Another approach is that of CVXOPT. Here the *diagonal-plus-low-rank* structure of the scaling matrix of a second-order cone (and its inverse) is used, i.e. that

$$
\begin{equation}\label{eq:cvxopt:uv}
    W_k = D_k + u_ku_k^\top, \quad     W_k^{-1} = E_k + v_kv_k^\top,
\end{equation}
$$

where the $D_k$ and $D_k$ are diagonal. We can again bunch all the scaling matrices together by defining

$$
    D = \textbf{blkdiag}\left(D_1,D_2,\dots, D_n\right), \quad 
    E = \textbf{blkdiag}\left(E_1,E_2,\dots, E_n\right),
$$

so that

$$
    W = D + UU^\top, \quad     W^{-1} = E + VV^\top.
$$

Note that the $U$ and $V$ is defined as previous, but are different since the $u_k$ and $v_k$ comes from \eqref{eq:cvxopt:uv} rather than \eqref{eq:clarabel:uv}. We now show how the rank-update and the known inverse can be used explicitly in order to preserve sparsity. The KKT system of interest using Nesterov-Todd scaling is the following

$$
\begin{bmatrix}
	P & A^\top \\
	A & -W^\top W
\end{bmatrix}
\begin{bmatrix} x \\ z \end{bmatrix}
= 
\begin{bmatrix} b_x \\ b_z \end{bmatrix}
$$

By scaling the $z$-variable and multiplying the last row with $W^{-1}$ one find that

$$
\begin{bmatrix}
	P & A^\top 	\left(E + VV^\top\right) \\
	\left(E + VV^\top\right)A & -I
\end{bmatrix}
\begin{bmatrix} x \\ Wz \end{bmatrix}
= 
\begin{bmatrix} b_x \\ W^{-1}b_z \end{bmatrix}.
$$

For simplicity, we now define $S = A^\top V$

$$
\begin{bmatrix}
	P & A^\top E + SV^\top \\
	EA + VS^\top & -I
\end{bmatrix}
\begin{bmatrix} x \\ Wz \end{bmatrix}
= 
\begin{bmatrix} b_x \\ W^{-1}b_z \end{bmatrix}.
$$

We can now introduce new variables $\alpha,\beta \in \mathbb{R}^{n}$ resulting in

$$
\begin{bmatrix}
	P & A^\top E &  S & 0\\
	E^\top A & -I & 0 & V\\
	S^\top & 0& 0 & -I \\
	0 & V^\top & -I & 0
\end{bmatrix}
\begin{bmatrix} x \\ Wz \\ \alpha \\ \beta \end{bmatrix}
= 
\begin{bmatrix} b_x \\ W^{-1}b_z \\ 0 \\ 0 \end{bmatrix}.
$$

Where we have used that $E = E^\top$ to show why the system is also symmetric. Now the above preserves sparsity. For SOCs we have that $EE = I $, so we can reduce the above as

$$
\begin{bmatrix}
	P & A^\top  &  S & 0\\
	A & -I & 0 & EV\\
	S^\top & 0& 0 & -I \\
	0 & V^\top E^\top & -I & 0
\end{bmatrix}
\begin{bmatrix} x \\ EWz \\ \alpha \\ \beta \end{bmatrix}
= 
\begin{bmatrix} b_x \\ EW^{-1}b_z \\ 0 \\ 0 \end{bmatrix}.
$$

### Roated Second-order cones
By rotating the rotated second-order cone back to the second-order cone form we can exploit the sparsity similar to as the regular second-order cone. This rotation can be done by applying the following transformation

$$
\begin{equation}
    R = 
    \begin{bmatrix}
        \frac{1}{\sqrt{2}} &  \frac{1}{\sqrt{2}} & 0 & \cdots &  0\\
        \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} & 0 & \cdots &  0\\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 &\cdots & 1
    \end{bmatrix}.
\end{equation}
$$

### Power Cones
The conjugate barrier for certain power cones have a Hessian that, similar to that of second-order cones, can be represented using augmented sparsity <d-cite key="chen2023a"></d-cite>. The intuition here is that e.g. the generalized power cone reduce to a second-order cone when $d_1=1$. Given that the augmented sparsity is only available for the conjugate barrier the following is only relevant in the case of *dual scaling*.

> The generalized power cone is parametrized by $\alpha \in \mathbb{R}_{++}^{d_1}$ such that 
    $$
    \sum_{i\in \lbrack d_1 \rbrack} \alpha_i = 1
    $$ 
> and is defined as
    $$
    \mathcal{K}_\textup{gpow}(\alpha,d_1,d_2) = \left\{(u,w) \in \mathbb{R}_+^{d_1}\times \mathbb{R}^{d_2}\ \Big|\ \prod_{i=1,\dots,d_1} u_i^{\alpha_i} \geq \|w\|\right\}
    $$
> with dual cone
    $$
    \mathcal{K}_\textup{gpow}^\ast(\alpha,d_1,d_2) = \left\{(u,w) \in \mathbb{R}_+^{d_1}\times \mathbb{R}^{d_2}\ \Big|\ \prod_{i=1,\dots,d_1} \left(\frac{u_i}{\alpha_i}\right)^{\alpha_i} \geq \|w\|\right\}
    $$


>    The generalized power cone is parametrized by $\alpha \in \mathbb{R}_{++}^{d_1}$ such that
    $$
    \sum_{i\in [d_1]} \alpha_i = 1
    $$
>    and is defined as
    $$
    \mathcal{K}_\textup{powm}(\alpha,d_1) = \left\{(u,w) \in \mathbb{R}_+^{d_1}\times \mathbb{R}\ \Big|\ \prod_{i=1,\dots,d_1} u_i^{\alpha_i} \geq w\right\}
    $$
>    with dual cone
    $$
    \mathcal{K}_\textup{powm}^\ast(\alpha,d_1) = \left\{(u,w) \in \mathbb{R}_+^{d_1}\times \mathbb{R}\ \Big|\ \prod_{i=1,\dots,d_1} \left(\frac{u_i}{\alpha_i}\right)^{\alpha_i} \geq -w\right\}
    $$

>    The relative entropy cone is defined as
    $$
    \mathcal{K}_\textup{rel}(\alpha,d_1) = \textup{cl}\left\{(u,v,w) \in \mathbb{R}\times\mathbb{R}_{++}^{d}\times \mathbb{R}_{++}^d\ \Big|\  u \geq \sum_{i=1,\dots, d}w_i\ln\left(\frac{w_i}{v_i}\right)\right\}
    $$
>   with dual cone
    $$
    \mathcal{K}_\textup{rel}^\ast(\alpha,d_1) = \textup{cl}\left\{(u,v,w) \in \mathbb{R}\times\mathbb{R}_{++}^{d}\times \mathbb{R}_{++}^d\ \Big|\  w_i \geq u\left(\ln\left(\frac{u}{v_i}\right) - 1\right) \forall i = 1,\dots, d\right\}
    $$

### Generalized Power Cones
As shown in <d-cite key="chen2023a"></d-cite> the Hessian of the conjugate barrier for the generalized power cones are augmented sparse with $n_u=1$ and $n_v = 2$, i.e. that

$$
\begin{equation}
    H^\ast(z) = D + pp^\top  - qq^\top - rr^\top.
\end{equation}
$$

where $z:=(u,w)$ and $D - qq^\top - rr^\top \succ 0$ with

$$
\begin{equation}
    D = 
    \begin{bmatrix}
        \ddots  & & & \\
        &       & \frac{\tau_i\phi}{u_i\zeta} + \frac{1 - \alpha_i}{u_i^2}& &\\
        &       & & \ddots  & \\
        &       & & & \frac{2}{\zeta}I_{d_2}
    \end{bmatrix},\ 
     p = \begin{bmatrix} p_0 \frac{\tau}{\zeta} \\ p_1 \frac{w}{\zeta}\end{bmatrix},\  
     q = \begin{bmatrix} q_0 \frac{\tau}{\zeta} \\ 0\end{bmatrix},\ 
     r = \begin{bmatrix} 0 \\ r_1 \frac{w}{\zeta}\end{bmatrix},
\end{equation}
$$

where

$$
\begin{alignat*}{2}
    p_0 &= \sqrt{\frac{\phi(\phi + \|w\|^2)}{2}}, \quad && p_1  = -2\sqrt{\frac{2\phi}{\phi + \|w\|^2}}\\
    q_0 &= \sqrt{\frac{\phi\zeta}{2}}, \quad && r_1 = 2\sqrt{\frac{\zeta}{\phi + \|w\|^2}}\\
\end{alignat*}
$$

and $\phi = \prod_{i\in [d_1]}\left(\frac{u_i}{\alpha_i}\right)^{2\alpha_i}$, $\tau_i = \frac{2\alpha_i}{u_i}$ and $\zeta = \phi - \|w\|^2$.

### Power mean cone
For the power mean cone it is shown in <d-cite key="chen2023a"></d-cite> that the Hessian of the conjugate barrier is augmented sparse with $n_u=1$ and $n_v = 2$, i.e. that

$$
\begin{equation}
    H^\ast(z) = D + pp^\top  - qq^\top - rr^\top.
\end{equation}
$$

where $z:=(u,w)$ and $D - qq^\top - rr^\top \succ 0$ with

$$
\begin{equation}
    D = 
    \begin{bmatrix}
        \ddots  & & & \\
        &       & \frac{\tau_i\phi}{u_i} + \frac{1 - \alpha_i}{u_i^2}& &\\
        &       & & \ddots &\\
        &       & & & \theta
    \end{bmatrix},\ 
     p = \begin{bmatrix} p_0 \tau \\ p_1 \frac{1}{\zeta}\end{bmatrix},\  
     q = \begin{bmatrix} q_0 \tau \\ 0\end{bmatrix},\ 
     r = \begin{bmatrix} 0 \\ r_1 \frac{1}{\zeta}\end{bmatrix},
\end{equation}
$$

where

$$
\theta = 1 + w^{-2},\quad p_0 = \phi,\quad p_1  = 1,\quad q_0 = \sqrt{\zeta \phi },\quad r_1 = \zeta
$$

and

$\phi = \prod_{i\in [d_1]}\left(\frac{u_i}{\alpha_i}\right)^{\alpha_i}$, $\zeta = \phi + w$ and $\tau_i = \frac{\alpha_i}{u_i\zeta}, i = 1,\dots,d$.

### Relative Entropy Cone
For the relative entropy cone it is shown in <d-cite key="chen2023a"></d-cite> that the Hessian of the conjugate barrier is sparse with the following structure 

$$
\begin{equation}
    H^\ast(z) = 
    \begin{bmatrix}
        \nabla^2_{u,u} f_\ast & \nabla^2_{u,v} f_\ast & \nabla^2_{u,w} f_\ast\\
        \nabla^2_{v,u} f_\ast & D_{v,v} & D_{v,w}\\
        \nabla^2_{w,u} f_\ast & D_{w,v} & D_{w,w}\\
    \end{bmatrix}
\end{equation}
$$

where $z:=(u,v,w)$ and $H^\ast(z) \succ 0$

$$
\begin{align}
    D_{v,v} &= \textbf{diag}\left(\begin{bmatrix} \frac{u(\gamma_1 + u)}{\gamma_1^2v_1^2} + \frac{1}{v_1^2} & \dots &  \frac{u(\gamma_d + u)}{\gamma_n^2v_d^2} + \frac{1}{v_d^2} \end{bmatrix}\right)\\
    D_{w,w} &= \textbf{diag}\left(\begin{bmatrix} \gamma_1^{-2} & \dots & \gamma_d^{-2} \end{bmatrix}\right)\\
    D_{v,w} = D_{w,v} &= \textbf{diag}\left(\begin{bmatrix}u(\gamma_1^2v_1)^{-1} & \dots & u(\gamma_d^2v_d)^{-1} \end{bmatrix}\right),
\end{align}
$$

with $\gamma_i = w_i - u\left(\ln\left(\frac{u}{v_i}\right) - 1 \right) > 0, i = 1,\dots,d$.


## Semi-definite cones
We start by introducing the *vectorization* of a symmetric matrix $M$ defined below

$$
\begin{equation}
    \mathbf{svec}(M) := \left[M_{1,1}, \sqrt{2}M_{2,1}, M_{2,2}, \sqrt{2}M_{3,1}, \dots, \sqrt{2}M_{n,n-2},\sqrt{2}M_{n,n-1}, M_{n,n}\right]^\top.
\end{equation}
$$

Note that the scaling of $\sqrt{2}$ is in order to preserve inner-products i.e. that $\mathbf{tr}(L^\top M) = \mathbf{svec}(L)^\top\mathbf{svec}(M)$. We equivalently introduce the *matrization* of a vector (and denote it by $\mathbf{smat}$) which unsurprisingly have the property $M = \mathbf{smat}(\mathbf{svec}(M))$ as

$$
\begin{equation}
    \mathbf{smat}(m) := 
    \begin{bmatrix}
        m_1 & m_2/\sqrt{2} & \dots & m_{n(n-1)/2}/\sqrt{2}\\
        m_2/\sqrt{2} & m_{3} & \dots & m_{n(n-1)/2 - 1}/\sqrt{2}\\
        \vdots & \vdots & \ddots & \vdots\\
        m_{n(n-1)/2}/\sqrt{2} & m_{n(n-1)/2+1}/\sqrt{2} & \dots & m_{n(n+1)/2}
    \end{bmatrix}.
\end{equation}
$$

Note that the scaling ensures that we preserve inner-products i.e. that $u^\top v = \mathbf{tr}(\mathbf{smat}(u)\mathbf{smat}(v))$.

### Computing dense $W_k^\top W_k$
In the CVXOPT the Nesterov-Todd scaling of a semidefinite cone is described by the product

$$
\begin{equation}\label{eq:fast:sdp:scaling}
    W_k\mathbf{svec}(V) = \mathbf{svec}\left(R_k^\top V R_k\right), 
\end{equation}
$$

However, stating it like this gives no insights into the structure of $W_k$, which is needed when assembling the KKT system (to be precise $W_k^\top W_k$ is needed for the KKT assembly). It turns out that the structure comes from the *symmetric* Kronecker product given by

$$
\begin{equation}
\left(G\otimes_s H\right)\mathbf{svec}(V) = \frac{1}{2}\mathbf{svec}\left(HSG^\top + GSH^\top\right).
\end{equation}
$$

Using the above we can see that this means that the Nesterov-Todd scaling of the semidefinite cone is

$$
\begin{equation}
\left(R^\top\otimes_s R^\top\right)\mathbf{svec}(V) = \mathbf{svec}\left(R^\top V R\right),
\end{equation}
$$

from which it becomes clear that $W_k = \left(R_k^\top\otimes_s R_k^\top\right)$. We're not almost ready to describe $W_k^\top W_k$, but first we need to introduce the following property of the symmetric Kronecker product <d-cite key="onethekronecker"></d-cite>

$$
\begin{equation}
\left(A\otimes_s B\right)\left(C\otimes_s D\right) = \frac{1}{2}\left(AC\otimes_s BD + AD\otimes_s BC\right).
\end{equation}
$$

Using the above it follows that

$$
\begin{equation}
\begin{aligned}
    W_k^\top W_k 
    &= \left(R_k\otimes_s R_k\right) \left(R_k^\top\otimes_s R_k^\top\right) \\
    &= R_kR_k^\top \otimes_s R_kR_k^\top
\end{aligned}
\end{equation}
$$

Now how do we define the symmetric Hadamard product? First we need to introduce a mapping $Q$ that works as follows

$$
\begin{equation}
    Q\mathbf{vec}(S) = \mathbf{svec}(S), \quad Q^\top\mathbf{svec}(S) = \mathbf{vec}(S)
\end{equation}
$$

Using $Q$ it is now possible to describe the symmetric Kronecker product as 

$$
\begin{equation}
    G\otimes_s H = \frac{1}{2}Q\left(G\otimes H + H\otimes G\right)Q^\top
\end{equation}
$$

This definite intuitively makes sense. The $Q^\top$ in front can be seen to lift a $\mathbf{svec}$  to $\mathbf{vec}$ i.e. from a symmetric vector to a regular vector. We then apply this to a regular Kronecker product (which is symmetric due to the sum) and then pull the product down back to $\mathbf{svec}$ using $Q$. Note that in general $G\otimes_s H$ *is not symmetric*. Instead, it represents a single half a otherwise symmetric product. In the case where $H=G$, as is the case of NT-scaling, the product is symmetric and in fact

$$
\begin{equation}
W_k^\top W_k = R_kR_k^\top\otimes_s R_kR_k^\top = Q\left(R_kR_k^\top\otimes R_kR_k^\top\right)Q^\top
\end{equation}
$$


## Appendix

### Jordan product for symmetric cones
The Jordan product between elements inside the various symmetric cones is given by

$$
\begin{equation}
    u\circ v = 
    \begin{cases}
        \begin{bmatrix}u_1v_1 & u_2v_2 & \dots & u_nv_n\end{bmatrix}^\top\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        \begin{bmatrix}u^\top v & u_1v_2^\top + u_2^\top v_1\end{bmatrix}^\top\ & \mathcal{K}_k = \mathcal{Q}_n\\
        \frac{1}{2}\mathbf{svec}\left(\mathbf{smat}(u)\mathbf{smat}(v) + \mathbf{smat}(v)\mathbf{smat}(u)\right)\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases}
\end{equation}
$$

#### Idempotent for symmetric cones
The idempotent for the three symmetric cones is as follows

$$
\begin{equation}
    e_k = 
    \begin{cases}
        \begin{bmatrix}1 & 1 & \dots & 1\end{bmatrix}^\top\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        \begin{bmatrix}1 & 0 & \dots & 0\end{bmatrix}^\top\ & \mathcal{K}_k = \mathcal{Q}_n\\
        \mathbf{svec}\left(I_n\right)\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases}
\end{equation}
$$

Note that $e^\top (z\circ s) = z^\top s$ and $e^\top e = m$.

In addition, the inverse, square, and square root are defined by the relations $u^{-1}\circ u = e$, $u^2 = u\circ u$, and $u^{1/2}\circ u^{1/2} = u$.

### Barrier Functions
#### Logarithmic barrier
For a composite cone $\mathcal{K} = \mathcal{K}_1\times\mathcal{K}_2\times\dots\times\mathcal{K}_K$ we have that

$$
\begin{equation}
    f(u) = \sum_{k=1}^K f_k(u_k), \quad
    f_k(u) = 
    \begin{cases}
        -\sum_{j=1}^n \log(u_j)\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        -\frac{1}{2}\log(u^\top J u )\ & \mathcal{K}_k = \mathcal{Q}_n\\
        \log\det\mathbf{smat}(u)\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases},
\end{equation}
$$

where $u = \begin{bmatrix}u_1 & u_2 & \dots & u_k \end{bmatrix}^\top$ with $u_k \in \mathcal{K}_k$ and 

$$
\begin{equation}
    J = \begin{bmatrix}1 & 0 \\ 0 & -I_{n-1}\end{bmatrix},
\end{equation}
$$

meaning that $u^\top J u = u_1^2 - v_2^\top v_1 \geq 0$ for all $u\in\mathcal{K}_\text{SOC}$. Note that $f(tu) = f(u) - m\log(t)$ for $t > 0$ where

$$
\begin{equation}
    m = m_1 + \dots + m_K, \quad
    m_k = 
    \begin{cases}
        n\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        1\ & \mathcal{K}_k = \mathcal{Q}_n\\
        n\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases}.
\end{equation}
$$

We refer to $m$ as the degree of the cone $\mathcal{K}$. 

#### Gradient of Logarithmic barrier
We denote the gradients of $f$ and $f_k$ at $u$ as $g(u) = \nabla f(u)$ and $g_k(u_k) = \nabla f_k(u_k)$:

$$
\begin{equation}
    g_k(u_k) = 
    \begin{cases}
        -\mathbf{diag}(u_k)^{-1}1_n\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        -\left(u_k^\top Ju_k\right)^{-1}Ju_k\ & \mathcal{K}_k = \mathcal{Q}_n\\
        -\mathbf{svec}\left(\mathbf{smat}(u_k)^{-1}\right)\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases},
\end{equation}
$$

where $1_n$ is a $n$-vector of ones. It can be shown that $-g(u) \in \mathcal{K}$ and that $u^\top g(u) = -m$ for $u\in\mathcal{K}$.

#### Hessian of Logarithmic barrier
We denote the Hessian of $f$ and $f_k$ as $H(u) = \nabla^2 f(u)$ and $H_k(u_k) = \nabla^2f_k(u_k)$:

$$
\begin{equation}
    H_k(u_k) = 
    \begin{cases}
        \mathbf{diag}(u_k)^{-2}\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        \left(u_k^\top Ju_k\right)^{-2}\left(2 Ju_ku_k^\top J - (u_k^\top J u_k)J\right)\ & \mathcal{K}_k = \mathcal{Q}_n\\
        R_kR_k^\top \otimes_s R_kR_k^\top\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases},
\end{equation}
$$

where $R_k = L_1V\Lambda_k^{-1/2} = L_2^{-\top}U\Lambda_k^{1/2}$ with

$$
\begin{equation}
    \mathbf{smat}(s_k) = L_1L_1^\top, \quad \mathbf{smat}(z_k) = L_2L_2^\top, \quad L_2^\top L_1 = U\Lambda_k V^\top.
\end{equation}
$$

In addition, we have that

$$
\begin{equation}
    H_k(u_k)^{-1} = 
    \begin{cases}
        \mathbf{diag}(u_k)^{2}\ & \mathcal{K}_k = \mathbb{R}_+^n\\
        2 u_ku_k - (u_k^\top J u_k)J\ & \mathcal{K}_k = \mathcal{Q}_n\\
        \left(R_kR_k^\top\right)^{-1} \otimes_s \left(R_kR_k^\top\right)^{-1}\ & \mathcal{K}_k = \mathcal{S}_n
    \end{cases}.
\end{equation}
$$

Further properties of the Hessian of the logarithmic barrier function is 

$$
\begin{align*}
    H(u)^{-1} = &H(u^{-1}), \quad H(u)^{1/2} = H(u^{1/2})\\
    H(u)u = u^{-1}, \quad &H(u^{1/2})u = e, \quad \left(H(u)v\right)^{-1} = H(u)^{-1}v^{-1}.
\end{align*}
$$

### Self-scaled property
The following property is known as the *self-scaled* property of the barrier function. Suppose that $w \in \text{int}\left(\mathcal{K}\right)_k$. Then $\nabla^2 f(w) u \in \text{int}\left(\mathcal{K}\right)_k$ for all $u\in \text{int}\left(\mathcal{K}\right)_k$ and

$$
\begin{equation}
    f(\nabla^2 f(w) u) = f(u) - 2f(w).
\end{equation}
$$

A consequence of self-scaled cones property is that for every $x$ in the interior of the primal cone $\mathcal{K}$ and every $s$ in the interior of the corresponding dual cone $\mathcal{K}^\ast$ there is a unique scaling point $w$ in the interior of $\mathcal{K}$ such theta <d-cite key="todd1998a"></d-cite>

$$
\begin{equation}\label{eq:scaling:point}
\nabla^2 f(w)z = s.
\end{equation}
$$

The theory also implies that given $t=-\nabla f(w)$ then we have that $\nabla^2 f_\ast(t) = \left(\nabla^2 f(w)\right)^{-1}$, meaning that \eqref{eq:scaling:point} can be rewritten as

$$
z = \left(\nabla^2 f(w)\right)^{-1}s \quad\Rightarrow\quad z= \nabla^2 f_\ast(t)s.
$$

Furthermore, the scaling point can also be used to map the gradients of the barrier and the conjugate barrier as

$$
\nabla^2 f(w) \nabla f_\ast(s) = \nabla f(z).
$$

Using the above we can rewrite the linearization of the central path 

$$
\nabla^2 f(w)\Delta z + \Delta s = -s - \mu\nabla f(z),
$$

as 

$$
\Delta z + \nabla^2 f_\ast (t)\Delta s = - z - \mu\nabla f_\ast(s).
$$