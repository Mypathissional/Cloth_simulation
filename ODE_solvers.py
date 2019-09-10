import numpy as np
from tqdm import tqdm


def ExplicitEuler(x0, f, t0, T, n_steps, savestep=5):
    h = (T - t0) / n_steps
    x = x0
    solutions = [x0]
    for t in tqdm(range(n_steps)):
        x = x + h * f(x)

        if t % savestep == 0:
            solutions.append(x)
    return solutions


def FixedPoint(f, x_n, n=100):
    x_new = f(x_n)
    if np.linalg.norm(x_n - x_new) >= 0.000001 or n == 0:
        return FixedPoint(f, x_new, n - 1)
    else:
        return x_new


def step(x_old, x_new, h, f):
    return x_old + h * f(x_new)


def ImplicitEuler(x0, f, t0, T, n_steps, savestep=2):
    h = (T - t0) / n_steps
    x = x0
    solutions = [x0]
    for t in tqdm(range(n_steps)):
        def update_f(x_new): return step(x, x_new, h, f)
        x = FixedPoint(update_f, x)
        if t % savestep == 0:
            solutions.append(x)

    return solutions


def RK4(x0, f, t0, T, n_steps, savestep=2):
    r = (T - t0) / n_steps
    x = x0
    solutions = [x0]
    for t in tqdm(range(n_steps)):
        h1 = f(x)
        h2 = f(x + 0.5 * r * h1)
        h3 = f(x + 0.5 * r * h2)
        h4 = f(x + r * h3)
        x = x + r * (h1 * 1. / 6 + 0.5 * h2 + 0.5 * h3 + h4)
        if t % savestep == 0:
            solutions.append(x)

    return solutions
