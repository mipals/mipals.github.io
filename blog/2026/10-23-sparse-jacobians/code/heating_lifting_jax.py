# Newton steps for the heating problem with asdex, automatic sparse
# differentiation in JAX, on the residual lifted with every partial sum
# of the integral as an unknown. JAX has no sparse direct solver on the
# CPU, so each step factorizes the sparse Jacobian with SciPy.
import time

import asdex
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

jax.config.update("jax_enable_x64", True)


# The heating residual with the partial sums sₖ = sₖ₋₁ + exp(-xₖ) as
# unknowns, z = [x; s], so that every equation reads only a few unknowns
def lifted(z, lam=2.0):
    n = z.size // 2
    x, s = z[:n], z[n:]
    h = 1 / (n + 1)
    xp = jnp.pad(x, 1)  # u(0) = u(1) = 0
    F = (xp[:-2] - 2 * x + xp[2:]) / h**2 + lam * jnp.exp(-x) / (h * s[-1]) ** 2
    S = s - jnp.concatenate([jnp.zeros(1), s[:-1]]) - jnp.exp(-x)
    return jnp.concatenate([F, S])


def lift(x):  # the partial sums that belong to x
    return jnp.concatenate([x, jnp.cumsum(jnp.exp(-x))])


# The pattern and the coloring, and a compiled sparse Jacobian
def prepare(z):
    coloring = asdex.jacobian_coloring(lifted, z)
    return coloring, jax.jit(asdex.jacobian_from_coloring(lifted, coloring))


def step(jac, z):
    J = jac(z)  # a BCOO matrix
    rows, cols = np.asarray(J.indices).T
    J = scipy.sparse.csc_matrix((np.asarray(J.data), (rows, cols)), shape=J.shape)
    return scipy.sparse.linalg.splu(J).solve(np.asarray(lifted(z)))


# Newton from x = 0
z = lift(jnp.zeros(1000))
_, jac = prepare(z)
for k in range(21):
    r = np.linalg.norm(lifted(z))
    print(f"  iteration {k}: |G(z)| = {r}")
    if r < 1e-9 * (1 + np.linalg.norm(z)):
        break
    z = z - step(jac, z)


# Seconds for the preparation, which includes compiling the Jacobian,
# and the best of three later steps
def best(f, runs=3):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


for n in (500, 1000, 2000, 4000, 8000):
    key0, key1 = jax.random.split(jax.random.key(n))
    z, z_new = lift(0.1 * jax.random.uniform(key0, (n,))), lift(0.1 * jax.random.uniform(key1, (n,)))
    t = time.perf_counter()
    coloring, jac = prepare(z)
    jac(z).data.block_until_ready()
    t_prepare = time.perf_counter() - t
    print(f"n = {n}: {coloring.num_colors} colors, nnz {jac(z).nse}, "
          f"prepare {t_prepare:.2e} s, step {best(lambda: step(jac, z_new)):.2e} s")
