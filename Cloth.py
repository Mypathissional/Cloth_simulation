import numpy as np
np.seterr(all='warn')

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
    def __init__(self, N, M, interval_length, fixed, Forces):
        self.Forces = Forces
        self.M = M
        self.N = N
        self.fixed = fixed
        self.interval_length = interval_length

        self.__init_forces()

    def init_Xv(self, M, N, init_velocity_ind, velocity):
        x = np.linspace(0, M - 1, M) * self.interval_length
        y = np.linspace(0, N - 1, N) * self.interval_length
        X, Y = np.meshgrid(x, y)
        X, Y = X.transpose(), Y.transpose()
        Z = np.zeros((M, N),dtype=np.float128)
        Vx = np.zeros((M, N),dtype=np.float128)
        Vy = np.zeros((M, N), dtype=np.float128)
        Vz = np.zeros((M, N),dtype=np.float128)
        Vz[init_velocity_ind[0], init_velocity_ind[1]] = velocity
        Xv = np.transpose(
            np.dstack((X, Y, Z, Vx, Vy, Vz)), axes=(2, 0, 1))
        Xv = np.array(Xv,dtype=np.float128)
        return Xv

    def __init_forces(self):
        # Horizontal and vertical spring force
        self.SF_structural = SpringForce(
            self.Forces["K"], self.interval_length, self.Forces["D"])
        # Diagonal spring force
        self.SF_flexion = SpringForce(
            self.Forces["K"], self.interval_length * np.sqrt(2), self.Forces["D"])
        self.SF_sheer = SpringForce(
            self.Forces["stiffness"], self.interval_length * 2, self.Forces["D"])

    # Xv = (x(t),y(t).z(t),vx(t),vy(t),vz(t))
    def __call__(self, Xv):

        Xv_prime = np.zeros((6, self.M, self.N))
        # x(t)' = vx(t), y(t)' = vy(t), z(t)' = vz(t)
        # gravitational force on z
        Xv_prime[0] = Xv[3]
        Xv_prime[1] = Xv[4]
        Xv_prime[2] = Xv[5]
        Xv_prime[5] -= 9.8

        for i in range(self.M):
            for j in range(self.N):
                try:
                # Adding SpringForces
                    if j < self.N - 1:

                        f_ij = self.SF_structural(
                            Xv[:3, i, j], Xv[:3, i, j + 1], Xv[3:, i, j], Xv[3:, i, j + 1])

                        Xv_prime[3:, i, j] += f_ij
                        Xv_prime[3:, i, j + 1] -= f_ij

                    if i < self.M - 1:
                        f_ij = self.SF_structural(
                            Xv[:3, i, j], Xv[:3, i + 1, j], Xv[3:, i, j], Xv[3:, i + 1, j])
                        Xv_prime[3:, i, j] += f_ij
                        Xv_prime[3:, i + 1, j] -= f_ij

                    if i < self.M - 1 and j < self.N - 1:
                        f_ij = self.SF_flexion(
                            Xv[:3, i, j], Xv[:3, i + 1, j + 1], Xv[3:, i, j], Xv[3:, i + 1, j + 1])
                        Xv_prime[3:, i, j] += f_ij
                        Xv_prime[3:, i + 1, j + 1] -= f_ij

                        f_ij1 = self.SF_flexion(
                            Xv[:3, i + 1, j], Xv[:3, i, j + 1], Xv[3:, i + 1, j], Xv[3:, i, j + 1])

                        Xv_prime[3:, i + 1, j] += f_ij1
                        Xv_prime[3:, i, j + 1] -= f_ij1

                    if j < self.N - 2:
                        f_ij = self.SF_sheer(
                            Xv[:3, i, j], Xv[:3, i, j + 2], Xv[3:, i, j], Xv[3:, i, j + 2])
                        Xv_prime[3:, i, j] += f_ij
                        Xv_prime[3:, i, j + 2] -= f_ij

                    if i < self.M - 2:
                        f_ij += self.SF_sheer(
                            Xv[:3, i, j], Xv[:3, i + 2, j], Xv[3:, i, j], Xv[3:, i + 2, j])

                        Xv_prime[3:, i, j] += f_ij
                        Xv_prime[3:, i + 2, j] -= f_ij
                except:
                    import ipdb
                    ipdb.set_trace()
        Xv_prime[:, self.fixed[0], self.fixed[1]] = 0.

        return Xv_prime
