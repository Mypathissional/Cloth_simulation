from time import time
import ipdb
import numpy as np
from Cloth_visualization import plot_cloth
from ODE_solvers import ExplicitEuler, ImplicitEuler, RK4
from Cloth import Cloth
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from tqdm import tqdm
import os


M, N = 10, 10
fixed = np.array([[0,  0], [0,  N - 1]])
Forces = {}
Forces["K"] = 1
Forces["stiffness"] = 0
Forces["D"] = 0
interval_length = 0.001
cloth = Cloth(N, N, interval_length, fixed, Forces)

init_velocity_ind = [[M - 1], [N - 1]]

init_velocity = 0.1
# os.makedirs(save_dir)
Xv = cloth.init_Xv(N, N, init_velocity_ind, init_velocity)
t0 = 0.
t1 = 0.1
n_steps = 100
n_save = 1

solutions = RK4(Xv, cloth, t0, t1, n_steps, n_save)

fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d', aspect='equal', autoscale_on = False)
ax = p3.Axes3D(fig)
ax.set_xlim3d([-M, M])
ax.set_xlabel('X')

ax.set_ylim3d([-N, N])
ax.set_ylabel('Y')

ax.set_zlim3d([-10.0, 10.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')


def flatten(l): return [item for sublist in l for item in sublist]


def update_f(i):

    return plot_cloth(solutions[i], M, N, fig, ax)


X, Y, Z = solutions[0][0], solutions[0][1], solutions[0][2]
X, Y, Z = flatten(X), flatten(Y), flatten(Z)


n_frames = int(n_steps / n_save)
dt = (t1 - t0) / n_frames
interval = 500 * dt

anim = animation.FuncAnimation(fig, update_f, frames=n_frames,
                               interval=interval, blit=False, repeat=True)

plt.show()
