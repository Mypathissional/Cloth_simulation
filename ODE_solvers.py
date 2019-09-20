import numpy as np
from tqdm import tqdm


def FixedPoint(f, x_n, n=50):
    x_new = f(x_n)
    if np.linalg.norm(x_n - x_new) >= 0.00001 and n > 0:
        return FixedPoint(f, x_new, n - 1)
    else:
        return x_new


class ExplicitEulerStep:
    def __init__(self):
        self.multistep = False
        self.method = "Explicit_Euler"

    def __call__(self, x, f, r):
        return x + r * f(x)


def ImplicitStep(x_old, x_new, r, f):
    return x_old + r * f(x_new)


class ImplicitEulerStep:
    def __init__(self):
        self.multistep = False
        self.method = "Implicit_Euler"
    def __call__(self, x, f, r):
        def increment(x_new): return ImplicitStep(
            x_old=x, x_new=x_new, r=r, f=f)
        return FixedPoint(increment, x)


class RK4Step:
    def __init__(self):
        self.multistep = False
        self.method = "Runge_Kutta_4_order"
    def __call__(self, x, f, r):

        h1 = f(x)
        h2 = f(x + 0.5 * r * h1)
        h3 = f(x + 0.5 * r * h2)
        h4 = f(x + r * h3)
        x = x + r * (h1 * 1. / 6 + 1./3 * h2 + 1./3 * h3 + 1./6 * h4)

        return x


class AdamStep:
    def __init__(self, f_prev, order):
        self.multistep = True
        self.order = order
        self.f_prev = f_prev
        self.method = "Adam_"+str(self.order)+"_order"
        if self.order == 3 and not isinstance(f_prev, list):
            raise TypeError("order should be list not " +
                            str(type(self.order)))

    def __call__(self, x, f, r, f_prev):
        f_next = f(x)
        if self.order == 2:
            return x + r * (1.5 * f_prev + 0.5 * f_next), f_next
        if self.order == 3:
            return x + r / 12. * (23 * f_next - 16 * f_prev[0] + 5 * f_prev[1])


class AdaptiveRungeKuttaFehlbergStep:
    def __init__(self):
        self.multistep = False
        self.order = 4 # low order term
        self.method = "AdaptiveRungeKutta_"+str(self.order)+"_order"
        self.c = np.array([0, 1. / 4., 3. / 8., 12. / 13., 1., 0.5])
        self.a2 = np.array([0.25])
        self.a3 = np.array([3. / 32., 9. / 32.])
        self.a4 = np.array([1932. / 2197., -7200. / 2197., 7296. / 2197.])
        self.a5 = np.array([439. / 216., -8, 3680. / 513., -845. / 4104.])
        self.a6 = np.array(
            [-8. / 27., 2., -3544. / 2565., 1859. / 4104., -11. / 40.])
        self.b1 = np.array([16. / 135, 0, 6656. / 12825.,
                            28561. / 56430, -9 / 50., 2. / 55.])
        self.b2 = np.array(
            [25. / 216., 0, 1408. / 2565., 2197. / 4104., -0.2, 0])

    def __call__(self, x, f, r):
        h1 = f(x)
        h2 = f(x + r * self.a2[0] * h1)
        h3 = f(x + r * (self.a3[0] * h1 + self.a3[1] * h2))
        h4 = f(x + r * (self.a4[0] * h1 + self.a4[1] * h2 * self.a4[2] * h3))
        h5 = f(x + r * (self.a5[0] * h1 + self.a5[1]
                        * h2 * self.a5[2] * h3 + self.a5[3] * h4))
        h6 = f(x + r * (self.a6[0] * h1 + self.a6[1] * h2 *
                        self.a6[2] * h3 + self.a6[3] * h4 + self.a6[4] * h5))

        x1_prime = x + r * (self.b1[0] * h1 + self.b1[1] * h2 + self.b1[2]
                            * h3 + self.b1[3] * h4 + self.b1[4] * h5 + self.b1[5] * h6)
        x1_tilda = x + r * (self.b2[0] * h1 + self.b2[1] * h2 + self.b2[2]
                            * h3 + self.b2[3] * h4 + self.b2[4] * h5 + self.b2[5] * h6)

        return x1_prime, x1_tilda


def solve_ODE(x0, f, t0, T, n_steps, n_frames, solver):
    h = (T - t0) / n_steps
    savestep = int(n_steps / n_frames)
    solutions = [x0]
    x = x0
    if solver.multistep:
        f_prev = solver.f_prev
    for t in tqdm(range(n_steps)):
        if not solver.multistep:
            x = solver(x, f, h)
        else:
            x, f_prev = solver(x, f, h, f_prev)
        if t % savestep == 0:
            solutions.append(x)
    return solutions

def solve_adaptive_ODE(x0, f, t0, T, n_steps, n_frames, solver, tolerance = 0.000001):
    h = (T - t0) / n_steps
    savestep = int(n_steps / n_frames)
    solutions = [x0]
    x = x0
    counter = 1
    t = t0
    step = 1

    for counter in tqdm(range(n_steps*3)):
        if(t < T):
            counter += 1
            h = min(h, T-t)
            # higher order calculation first
            x1_prime, x1_tilda = solver(x, f, h)
            ratio = tolerance/np.max(np.abs(x1_prime-x1_tilda))
            if ratio >= 1:
                x = x1_prime
                step += 1
                t = t + h
                if step % savestep == 0:
                    print(step)
                    solutions.append(x)
            h = h * pow(ratio, 1./(solver.order+1))

    return solutions
