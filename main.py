import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LightSource
import numpy as np
from ODE_solvers import *
from Cloth import Cloth
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3


# Init params

save_to_file = True
M, N = 20, 20
fixed = np.array([[0,  0, M - 1, M - 1, ], [0,  N - 1, 0, N - 1]])
Forces = {}
Forces["K"] = 50.
Forces["stiffness"] = 50.
Forces["D"] = 20,
interval_length = 0.1
cloth = Cloth(N, N, interval_length, fixed, Forces)

init_velocity_ind = [[int(M / 2)], [int(N * 0.5)]]

init_velocity = 15
# os.makedirs(save_dir)
Xv = cloth.init_Xv(N, N, init_velocity_ind, init_velocity)
t0 = 0.
t1 = 1.
n_steps = 100
n_frames = 20

# Define method
print("Solving ODE")
solver = ImplicitEulerStep()
solutions = solve_ODE(Xv, cloth, t0, t1, n_steps, n_frames, solver)


fig = plt.figure()
ax = p3.Axes3D(fig)


def update_f(i, render=True):

    ax.clear()

    ax.set_xlim3d([0 - 1, M * interval_length + 1])
    ax.set_xlabel('X')

    ax.set_ylim3d([0 - 1, N * interval_length + 1])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-10.0, 10.0])
    ax.set_zlabel('Z')

    ax.set_title('3D Test')
    X, Y, Z = solutions[i][0], solutions[i][1], solutions[i][2]

    if render:
        grey = np.array([1, 1, 1])

        light = LightSource(0, 0)
        rgb = np.ones((Z.shape[0], Z.shape[1], 3))
        grey_surface = light.shade_rgb(rgb * grey, Z)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False,
                        facecolors=grey_surface)
    else:
        ax.plot_wireframe(X, Y, Z, c='grey')
    ax.scatter(X, Y, Z, marker='o', c='grey', linewidths=0.02)
    return ax


if save_to_file:
    savestring = solver.method + ''.join(["_" + key +
                                      "=" + str(Forces[key]) for key in Forces])
    savestring += ",interval_length" + str(interval_length)
    print("Final Timestep Simulation")
    update_f(len(solutions)-1)
    plt.show()
    fig.savefig(savestring+".png")
    anim = animation.FuncAnimation(fig, update_f, frames=len(solutions),
                                   interval=5, blit=False, repeat=True)
    print("Saving to a file")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=18000)
    anim.save( savestring + '.mp4', writer=writer, dpi=fig.dpi)

    print("Animation is saved")
else:
    anim = animation.FuncAnimation(fig, update_f, frames=len(solutions),
                                   interval=5, blit=False, repeat=True)
    print(" 3D Animation of cloth simulation")
    plt.show()
    print("Animation is saved")
