import numpy as np


class SpringForce:
    def __init__(self, K, L0, D):
        """
        K - stiffness
        L0 - rest length
        D - spring damping
        """
        self.K = K
        self.L0 = L0
        self.D = D

    def __call__(self, p, q, v_p, v_q):
        """
        The spring force,
        given spring connecting two particles located at p and q,
        with velocities v_p and v_q
        and stiffness K, rest length L0 and damping D.
        """
        distance = np.linalg.norm(p - q)

        spring_force_on_p = self.K * (self.L0 - distance) * (p - q) / distance
        damping_force_on_p = self.D * (v_q - v_p) * distance
        force_on_p = spring_force_on_p + damping_force_on_p

        return force_on_p


class Cloth:
    def __init__(self, N, M, fixed, Forces):
        self.Forces = Forces
        self.M = M
        self.N = N
        self.fixed = fixed

        self.__init_forces()

    def init_Xv(self, M, N, init_velocity_ind, velocity):
        x = np.linspace(0, M - 1, M)*0.001
        y = np.linspace(0, N - 1, N)*0.001
        X, Y = np.meshgrid(x, y)
        X, Y = X.transpose(), Y.transpose()
        Z = np.zeros((M, N))
        Vx = np.zeros((M, N))
        Vy = np.zeros((M, N))
        Vz = np.zeros((M, N))
        Vz[init_velocity_ind[0], init_velocity_ind[1]] = velocity
        Xv = np.transpose(
            np.dstack((X, Y, Z, Vx, Vy, Vz)), axes=(2, 0, 1))

        return Xv

    def __init_forces(self):
        # Horizontal and vertical spring force
        self.SF_hv = SpringForce(
            self.Forces["K"], 0.001, self.Forces["D"])
        # Diagonal spring force
        self.SF_diagonal = SpringForce(
            self.Forces["K"], 0.001*np.sqrt(2), self.Forces["D"])
        self.SF_stiffness = SpringForce(
            self.Forces["stiffness"], 0.001*2, self.Forces["D"])

    # Xv = (x(t),y(t).z(t),vx(t),vy(t),vz(t))
    def __call__(self, Xv):

        Xv_prime = np.zeros((6, self.M, self.N))
        # x(t)' = vx(t), y(t)' = vy(t), z(t)' = vz(t)
        #Xv_prime[:3, :, :] = Xv[3:]
        # gravitational force on z
        Xv_prime[:3, :, :] = Xv[3:]
        Xv_prime[5, :, :] -= 0

        for i in range(self.M):
            for j in range(self.N):

                # Adding SpringForces
                if j < self.N - 1:

                    force_on_ij = self.SF_hv(
                        Xv[:3, i, j], Xv[:3, i, j + 1], Xv[3:, i, j], Xv[3:, i, j + 1])

                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i, j + 1] -= force_on_ij

                if i < self.M - 1:
                    force_on_ij = self.SF_hv(
                        Xv[:3, i, j], Xv[:3, i + 1, j], Xv[3:, i, j], Xv[3:, i + 1, j])
                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i + 1, j] -= force_on_ij

                if i < self.M - 1 and j < self.N - 1:
                    force_on_ij = self.SF_diagonal(
                        Xv[:3, i, j], Xv[:3, i + 1, j + 1], Xv[3:, i, j], Xv[3:, i + 1, j + 1])
                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i + 1, j + 1] -= force_on_ij

                if i > 0 and j < self.N - 1:
                    force_on_ij += self.SF_diagonal(
                        Xv[:3, i, j], Xv[:3, i - 1, j + 1], Xv[3:, i, j], Xv[3:, i - 1, j + 1])
                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i - 1, j + 1] -= force_on_ij

                if j < self.N - 2:
                    force_on_ij = self.SF_stiffness(
                        Xv[:3, i, j], Xv[:3, i, j + 2], Xv[3:, i, j], Xv[3:, i, j + 2])
                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i, j + 2] -= force_on_ij

                if i < self.M - 2:
                    force_on_ij += self.SF_stiffness(
                        Xv[:3, i, j], Xv[:3, i + 2, j], Xv[3:, i, j], Xv[3:, i + 2, j])

                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i + 2, j] -= force_on_ij

                if j > 1:
                    force_on_ij = self.SF_stiffness(
                        Xv[:3, i, j], Xv[:3, i, j - 2], Xv[3:, i, j], Xv[3:, i, j - 2])
                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i, j - 2] -= force_on_ij

                if i > 1:
                    force_on_ij += self.SF_stiffness(
                        Xv[:3, i, j], Xv[:3, i - 2, j], Xv[3:, i, j], Xv[3:, i - 2, j])

                    Xv_prime[3:, i, j] += force_on_ij
                    Xv_prime[3:, i - 2, j] -= force_on_ij
        Xv_prime[:, self.fixed[0], self.fixed[1]] = 0

        return Xv_prime
