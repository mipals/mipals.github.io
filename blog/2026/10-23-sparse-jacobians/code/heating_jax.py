# Newton steps for the nonlocal heating problem with JAX,
# u'' + λ exp(-u) / (∫₀¹ exp(-u) dx)² = 0. JAX only offers products
# with J, so a step forms the dense J with jax.jacfwd (n Jacobian-vector
# products) or jax.jacrev (n vector-Jacobian products) and solves densely.
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def heating(x, lam=2.0):
    h = 1 / (x.size + 1)
    s = jnp.sum(jnp.exp(-x))    # h * s is the integral
    xp = jnp.pad(x, 1)          # u(0) = u(1) = 0
    return ((xp[:-2] - 2 * x + xp[2:]) / h**2
            + lam * jnp.exp(-x) / (h * s) ** 2)


def newton_step(jac):
    return jax.jit(lambda x: jnp.linalg.solve(jac(heating)(x), heating(x)))


# Seconds for the first call, which compiles, and the best of three
# later calls at a new point
def prepare_and_step(step, x, x_new, runs=3):
    t = time.perf_counter()
    jax.block_until_ready(step(x))
    prepare = time.perf_counter() - t
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        jax.block_until_ready(step(x_new))
        times.append(time.perf_counter() - t)
    return prepare, min(times)


for n in (500, 1000, 2000, 4000, 8000):
    x = 0.1 * jax.random.uniform(jax.random.key(0), (n,))
    x_new = 0.1 * jax.random.uniform(jax.random.key(1), (n,))
    fwd, rev = newton_step(jax.jacfwd), newton_step(jax.jacrev)
    p_fwd, s_fwd = prepare_and_step(fwd, x, x_new)
    p_rev, s_rev = prepare_and_step(rev, x, x_new)
    print(f"n = {n}: jacfwd prepare {p_fwd:.2e} s, step {s_fwd:.2e} s; "
          f"jacrev prepare {p_rev:.2e} s, step {s_rev:.2e} s")
