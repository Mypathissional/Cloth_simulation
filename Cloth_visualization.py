import ipdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
N = 5

x = np.linspace(0, N - 1, N)
y = np.linspace(0, N - 1, N)
X, Y = np.meshgrid(x, y)
X, Y = X.transpose(), Y.transpose()
Z = np.zeros((N, N))


def get_neighbours(i, j, M, N):
    neighbours = []
    if i < M - 1:
        neighbours.append([i + 1, j])
        if j > 0:
            neighbours.append([i + 1, j - 1])
        if j < N - 1:
            neighbours.append([i + 1, j + 1])

    if i > 0:
        neighbours.append([i - 1, j])
        if j > 0:
            neighbours.append([i - 1, j - 1])
        if j < N - 1:
            neighbours.append([i - 1, j + 1])

    if j > 0:
        neighbours.append([i, j - 1])

    if j < N - 1:
        neighbours.append([i, j + 1])

    return neighbours



def plot_cloth(data, M, N, fig, ax):
    ax.clear()
    X, Y, Z = [data[0], data[1], data[2]]
    #x, y, z = [], [], []
    plot_list = []
    for i in range(N):
        for j in range(N):
            neighbours = get_neighbours(i, j, N, N)
            for n in neighbours:
                # x.append([X[i, j], X[n[0], n[1]]])
                # y.append([Y[i, j], Y[n[0], n[1]]])
                # z.append([Z[i, j], Z[n[0], n[1]]])
                x = [X[i, j], X[n[0], n[1]]]
                y = [Y[i, j], Y[n[0], n[1]]]
                z = [Z[i, j], Z[n[0], n[1]]]
                plot_list.append(ax.plot(x, y, z, c='g'))
    plot_list.append(ax.scatter(X, Y, Z, c='b', marker='o', linewidths=5))
    return plot_list


    # for i in range(len(x)):
    #     plt.plot(x[i], y[i], z[i], c='g')

    # fig.savefig(save_dir)
    #plt.show()
    #plt.savefig(save_dir, format='png')




# animation function


def plot_grid(X, Y, Z, N):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, c='b', marker='o', linewidths=5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    for i in range(N):
        plt.plot([X[i][0], X[i][N - 1]], [Y[i][0], Y[i][N - 1]],
                 [Z[i][0], Z[i][N - 1]], c='g')

    for i in range(N):
        plt.plot([X[0][i], X[N - 1][i]], [Y[0][i], Y[N - 1][i]],
                 [Z[0][i], Z[N - 1][i]], c='g')

    # plotting diagonals

    for j in range(N):
        plt.plot([X[0][j], X[N - 1 - j][N - 1]],
                 [Y[0][j], Y[N - 1 - j][N - 1]], c='g')

    for j in range(N):
        plt.plot([X[j][0], X[N - 1][N - 1 - j]],
                 [Y[j][0], Y[N - 1][N - 1 - j]], c='g')

    for j in range(N):
        plt.plot([X[0][j], X[j][0]],
                 [Y[0][j], Y[j][0]], c='g')

    for j in range(N):
        plt.plot([X[j][N - 1], X[N - 1][j]],
                 [Y[j][N - 1], Y[N - 1][j]], c='g')

    plt.show()
    return fig
