---
layout: distill
title: Polar Express
description: Approximating matrix functions using only matrix-matrix products. In this note we explain how the Polar Express algorithm can be used to approximate the polar transform used in the Muon optimizer. Furthermore we show how similar ideas can be used to project matrices onto the cone of semidefinite matrices using only matrix-matrix products. The reason why such methods are of interest is that matrix-matrix products are very efficient to compute on modern hardware such as GPUs.
tags: linear-algebra matrix-functions GPUs
giscus_comments: true
date: 2025-10-20 12:00:00
featured: true
citation: true

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
    - name: The Polar Express algorithm for Polar Normalization
    - name: Projection onto the semidefinite cone

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
### The Singular Value Transformation

The singular value transformation is defined for any odd function $f$, as

$$
    f(M) := Uf(\Sigma) V^\top, \quad M = U\Sigma V^\top.
$$

The requirement of $f$ being odd means that the singular value transformation and a matrix function overlap when the matrix $M$ is square symmetric. To see this we start by using the eigendecomposition of $M$ as $M=Q\Lambda Q^\top$ for which the following hold when $f$ is odd

$$
\begin{aligned}
    f(M) = Qf(M)Q^\top 
    &= Q\text{diag}(f(\text{sign}(\lambda_1)|\lambda_1|),\dots,f(\text{sign}(\lambda_n)|\lambda_n|))Q^\top\\
    &= Q\text{diag}(\text{sign}(\lambda_1)f(|\lambda_1|),\dots,\text{sign}(\lambda_n)f(|\lambda_n|))Q^\top\\
    &= Q\text{diag}(\text{sign}(\lambda_1),\dots, \text{sign}(\lambda_n)) f(|\Lambda|) Q^\top\\
    &= Uf(\Sigma)V^\top,
\end{aligned} 
$$

where the sign function is defined as the odd function

$$
    \text{sign}(x) := 
    \begin{cases}
        1, & x > 0, \\
        0, & x = 0, \\
        -1, & x < 0,
    \end{cases}
$$

# Approximating the Polar Normalization
Polar normalization corresponds to the singular value transformation when $f$ itself is the sign function, i.e.

$$
    \text{polar}(M) = U\text{diag}(\text{sign}(\sigma_1), \dots, \text{sign}(\sigma_k))V^\top = UV^\top.
$$

Recently there have been an increased interest in polar normalization as it is a key ingredient in the recently proposed Muon algorithm for optimizing large language models <d-cite key="jordan2024muon"></d-cite>. In short the steps of the Muon algorithm is <d-cite key="bernstein2025manifolds"></d-cite>

$$
\begin{aligned}
    M_t     &= \beta M_{t-1} + (1 - \beta) G_t, \\
    W_{t+1} &= W_t - \lambda \text{polar}(M_t).
\end{aligned}
$$

A distinct disadvantage is that the polar step in the naive implementation requires the computation of an SVD which is computationally expensive and not very GPU friendly (it is primarily memory bound rather than compute bound). Recently the polar express algorithm have been proposed as a method for approximating the matrix sign function using just matrix-matrix products <d-cite key="polar-express"></d-cite>. The key idea is rather simple: Approximate the function $f$ using a composition of polynomials e.g.

$$
    f(x) \approx p_k(p_{k-1}(\dots p_2(p_1(x))\dots)).
$$

Given that the above is just polynomials it can be evaluated using just additions and multiplications. 

### Example: Newton-Schulz
The idea of approximating functions using repeated polynomial application is not. For example the Newton-Schulz algorithm take the path of applying the same polynomial over and over, i.e. $p_1=p_2=\dots=p_k$, in order to approximate the sign function. For order 3 and 5 the polynomials look as follows
$$
    p(x) = \frac{3}{2}x - \frac{1}{2}x^3, \quad p(x) = (15x - 10x^3 + 3x^5)/8.
$$

While the Newton-Schulz method converges for all $x\in [-1,1]$ the convergence is only fast for values close to $\pm 1$. Reversely for values that are close to zero the converges is slow. The slow convergence is exactly what the Polar Express method explained shortly aims to fix. 

A note here is that for non-square matrices we can not simply do $M^n$. Instead what is done is that $M^{2n+1} = M(M^\top M)^{n} = (MM^\top)^n M$

{% details Newton-Schulz %}
```julia
using LinearAlgebra
```
{% enddetails %}

### The Polar Express Algorithm

The basic question that the polar express algorithm aim to solve is as follows: Given a set of $T$ polynomials $\{p_i(x) = a_1^ix + a_3^ix^3 + a_5x^5\}_{i=1}^T$, can we find optimal set of coefficients that minimizes the maximum error over a given interval $[\ell,u]$? That is we want to solve

$$
    p^\star = \underset{\substack{p = p_T \circ p_{T-1} \circ \cdots \circ p_1 \\ p_t \in \mathbb{P}_d^{\text{odd}}}}{\text{argmin}} \; \max_{\substack{M \in \mathbb{R}^{m \times n} \\ \sigma(M) \subset [\ell, u]}} \; \| \operatorname{polar}(M) - p(M) \|_2.
$$

In <d-cite key="polar-express"></d-cite> they show that the above is solved by a greedy approach, with the caveat is the convergence is only guaranteed if the smallest singular value is larger than $\ell$, which we do not know in advance. A comment here is that the main focus in the paper is on $\texttt{Float16}$. Here the machine precision is $\epsilon_\text{mach} = 2^{-7} = 0.0078125$. Therefore, they suggest to set $\ell = 10^{-3}$ and $u=1$. In addition, they provide a few modifications in order to make the method more robust. These additions are skipped here, but can be found in the original paper.

{% details Polar Express for sign computation %}
```julia
using LinearAlgebra
```
{% enddetails %}

The first step is then to find a polynomial approximation of the sign function on the interval $[-1,-\epsilon]\cup[\epsilon,1]$ for some small $\epsilon > 0$. That is we solve

$$
\begin{aligned}
&\min_{\substack{f_1,\ldots,f_T}} \; \max_{x \in [-1,-\epsilon] \cup [\epsilon,1]} \; \big| f_T \circ f_{T-1} \circ \cdots \circ f_1(x) - f_{\mathrm{sign}}(x) \big|\\
\quad \text{subject to} \quad & f_t \in \mathbb{R}_{d_t}^{\text{odd}}[x], \; t = 1,\ldots,T.
\end{aligned}
$$



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



# Projection onto the semidefinite cone
The **positive semidefinite cone** is the set of all real symmetric $n\times n$ positive semidefinite matrices. That is

$$
    \mathbb{S}_+^n := \lbrace X \in \mathbb{R}^{n\times n}\ |\ X\succeq 0 \rbrace.
$$

Algorithms that aim to solve semidefinite programming (SDP) off course need to ensure that the solution actually satisfy the positive semidefinite constraint. In interior point methods this is done by only taking steps for which the next iterate remain inside the cone, while first order methods most often rely on projecting back onto the set after taking a step, i.e. 

$$
    \Pi_{\mathbb{S}_+^n}(X) 
    = \arg\min_{Y \in \mathbb{S}_+^n} \|Y - X\|_F 
    = U \text{diag}(\text{ReLu}(\lambda_1), \dots, \text{ReLu}(\lambda_n)) U^T.
$$

While the idea of projecting back onto the semidefinite cone using just matrix-matrix products is not new (a similar fixed-point approach was suggested in <d-cite key="fixedpoint"></d-cite>) the main takeaway is that the ideas of Polar Express can be used to find a faster converging method <d-cite key="polar-express-sdp"></d-cite>. An initial idea would be to directly find polynomials that approximate the ReLU function, i.e. solving

$$
\begin{aligned}
    &\inf_{f_1,\ldots,f_T} \; \max_{x \in [-1,1]} \; \big| f_T \circ f_{T-1} \circ \cdots \circ f_1(x) - f_{\mathrm{ReLU}}(x) \big| \\
    \quad \text{subject to} \quad &f_t \in \mathbb{R}_{d_t}[x], \; t = 1,\ldots,T.
\end{aligned}
$$

where $T$ is the total number of polynomials in the composition, $d_t$ is the degree of the $t$'th polynomial which are fixed in advance as part of a "matrix-matrix" multiplication budget. Unfortunately directly solving the above is challenging, and instead a two-stage approach is used. The idea is to realize that we can write the ReLu function 

$$
    \text{ReLu}(x) = \frac{1}{2}x\left(1 + \text{sign}(x)\right)
$$

The first step is to compute optimal coefficients for approximating the sign function using the Polar Express approach. The second step is then to refine the coefficients of the polynomials found in the first step by minimizing the error to the ReLu function on the entire interval $[-1,1]$. 

$$
\ell(f_T,\ldots,f_1) := \max_{x \in [-1,1]} \Big| \frac{1}{2} x \big( 1 + f_T \circ f_{T-1} \circ \cdots \circ f_1(x) \big) - f_{\mathrm{ReLU}}(x) \Big|.
$$


# Code
```julia
using LinearAlgebra

poly_coeffs = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]
safe_poly_coeffs = [coeffs ./ (1.01, 1.01^3, 1.01^5) for coeffs in poly_coeffs[1:end-1]]

psd_coeffs = [
        ( 8.3119043343,  -23.0739115930,  16.4664144722 ),
        ( 4.1439360087,   -2.9176674704,   0.5246212487 ),
        ( 4.0257813209,   -2.9025002398,   0.5334261214 ),
        ( 3.5118574347,   -2.5740236523,   0.5050097282 ),
        ( 2.4398158400,   -1.7586675341,   0.4191290613 ),
        ( 1.9779835097,   -1.3337358510,   0.3772169049 ),
        ( 1.9559726949,   -1.3091355170,   0.3746734515 ),
        ( 1.9282822454,   -1.2823649693,   0.3704626545 ),
        ( 1.9220135179,   -1.2812524618,   0.3707011753 ),
        ( 1.8942192942,   -1.2613293407,   0.3676616051 )
]

function polynomial_approxmation(G, method=:msign; steps=10)
    # Tranpose G if required
    if_tranpose = >(size(G)...) 
    G = if_tranpose ? G : G'
    # Normalize (Shuld also push to float16)
    X = G/norm(G)
    # Getting element types of X
    elt = eltype(X)
    # Polynomial composition (not at the end the same polynomial is repeatedly applied)
    for step in 1:steps
        # This step forces to F64
        if method == :msign
            a1,a3,a5 = elt.(step <= length(safe_poly_coeffs) ? safe_poly_coeffs[step] : poly_coeffs[end])
        elseif method == :psd_projection
            a1,a3,a5 = elt.(step <= length(psd_coeffs) ? psd_coeffs[step] : psd_coeffs[end])
        end
        # Evaluating the polynomial using Horners method x*(a1 + x^2*(a3 + a5*x^2))
        X2 = X * X'      # Defining the square variable
        Y = a5*X2 + a3*I # Inner parentheses
        Y = Y*X2  + a1*I # Middle parentheses
        X = Y*X          # Outer parentheses
    end
    # If PSD projection is sought remember last step
    G = method == :msign ?  X : G*(I + X)/2
    # Transpose back if needed
    G = if_tranpose ? G' : G
    return G
end

function polar(G, steps=10)
    return polynomial_approxmation(G, :msign, steps=steps)
end

function relu(G, steps=10)
    return polynomial_approxmation(G, :psd_projection, steps=steps)
end


####
using LinearAlgebra
N = 5
A = rand(N,N)
A = A + A'

relu(x::Number) = x < 0 ? zero(x) : x


A_eigvals = eigvals(A)
Aeig = eigen(A)
Asvd = svd(A)


Aeig_sign = Aeig.vectors*Diagonal(sign.(Aeig.values))*Aeig.vectors'
Asvd_sign = Asvd.U*Diagonal(sign.(Asvd.S))*Asvd.Vt
Asvd_iter = msign(A,20)
eigvals(Aeig_sign) ./ sign.(A_eigvals)
eigvals(Asvd_sign) ./ sign.(A_eigvals)
eigvals(Asvd_iter) ./ sign.(A_eigvals)


Aeig_relu = Aeig.vectors*Diagonal(relu.(Aeig.values))*Aeig.vectors'
Asvd_relu = Asvd.U*Diagonal(relu.(Asvd.S))*Asvd.Vt # No signs so cant be used!
Asvd_iter = relu(A)

eigvals(Aeig_relu) ./ A_eigvals
eigvals(Asvd_relu) ./ A_eigvals
eigvals(Asvd_iter) ./ A_eigvals





function ppolar(x,steps=10; method=:psign)
    y = x
    elt = eltype(x)
    for step in 1:steps
        if method == :psign
            a1,a3,a5 = elt.(step <= length(safe_poly_coeffs) ? safe_poly_coeffs[step] : poly_coeffs[end])
        else
            a1,a3,a5 = elt.(step <= length(psd_coeffs) ? psd_coeffs[step] : psd_coeffs[end])
        end
        y = a1*y + a3*y^3 + a5*y^5
    end
    return method == :psign ? y : x * (1 + y)/2
end

function psign(x,steps=10)
    return ppolar(x,steps; method=:psign)
end
function ppsd(x,steps=10)
    return ppolar(x,steps; method=:psd)
end


## Sign
using Plots, LinearAlgebra
function NewtonSchulz(x, steps=10; order=5)
    p(x) = order == 5 ? (15*x - 10*x^3 + 3*x^5)/8 :  3/2*x - x^3/2
    for step in 1:steps
        x = p(x)
    end
    return x
end

n = 1000
x = sort(2*rand(n)) .- 1
p1 = plot(x, sign.(x), linewidth=3, legend=:topleft, label="sign(x)")
for i in 1:3:10
    plot!(p1,x, NewtonSchulz.(x,i), linewidth=2, label="NewtonShulz$(i)", color=:green)
    plot!(p1,x, psign.(x,i), linewidth=2, label="Polar$(i)", linestyle=:dash,color=:red)
end
display(p1)


p2 = plot()
for i in 1:2:10
    plot!(p2,x, ppsd.(x,i))
end
display(p2)


### WHY?
p(x) = I + x^2
p(A)

Aeig.vectors*Diagonal(p.(Aeig.values))*Aeig.vectors'



i = 5
plot(x,abs.(sign.(x) - psign.(x,i)) .+ eps(Float64),yscale=:log10)
plot!(x,abs.(sign.(x) - NewtonSchulz.(x,i)) .+ eps(Float64),yscale=:log10)


plot(x, sign.(x), label="sign",legend=:topleft)
plot!(x, psign.(x,i), label="psign")
plot!(x, NewtonSchulz.(x,i), label="NewtonSchulz")



using WGLMakie

x = LinRange(0, 2π, 200)
fig = Figure(resolution=(800,400))
ax = Axis(fig[1,1])
lines!(ax, x, sin.(x), label="sin")
lines!(ax, x, cos.(x), label="cos")
axislegend(ax)  # or Legend(fig[...] , ... ) and place where desired

# show in browser (starts a local server)
display(fig)
```


