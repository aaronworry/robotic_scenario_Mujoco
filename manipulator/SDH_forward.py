import casadi as ca
import numpy as np

# here is version using casadi to describe the fk model

class Arm_FK():
    def __init__(self, nq, dh_list = None):
        self.nq = nq
        self.DH_list = dh_list
        self.q = ca.SX.sym("theta", self.nq)

    def set_DH(self, dh_list):
        assert len(dh_list) == self.nq
        self.DH_list = dh_list

    def set_theta_list(self, theta_list):
        self.theta_list = theta_list

    def forward_compute(self, q):
        """
        :param q: position of all joints, a list of float
        :return: the state of the ee, including position and orientation

        Note: all representation are defined in the arm-base coordinate system
        """
        T = np.eye(4)
        for i in range(self.nq):
            Ti = self.computeTij(self.DH_list[i], q[i])
            T = T @ Ti
        return T

    def ca_forward(self, idx):
        """

        :return: the casadi.Fuction of forward
        """
        T = ca.DM.eye(4)
        assert 1 <= idx <= self.nq
        for i in range(idx):
            Ti = self.buildTij(self.DH_list[i], self.q[i])
            T = T @ Ti
        return ca.Function("arm_fk", [self.q], [T])

    def forward(self, q, idx):
        """
        :return: the ca of forward
        """
        T = ca.DM.eye(4)
        assert 1 <= idx <= self.nq
        for i in range(idx):
            Ti = self.buildTij(self.DH_list[i], q[i])
            T = T @ Ti
        return T[3, :3], T[:3, :3]

    def buildTij(self, dh, q_ca):
        # 后续考虑将dh替换为ca符号
        alpha, a, d = dh[0], dh[1], dh[2]
        T_a = ca.DM.eye(4)
        T_a[0, 3] = a

        T_d = ca.DM.eye(4)
        T_d[2, 3] = d

        Rzt = ca.vertcat(
            ca.horzcat(ca.cos(q_ca), -ca.sin(q_ca), 0, 0),
            ca.horzcat(ca.sin(q_ca), ca.cos(q_ca), 0, 0),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1))

        Rxa = ca.vertcat(
            ca.horzcat(1, 0, 0, 0),
            ca.horzcat(0, ca.cos(alpha), -ca.sin(alpha), 0),
            ca.horzcat(0, ca.sin(alpha), ca.cos(alpha), 0),
            ca.horzcat(0, 0, 0, 1))

        return Rzt @ T_d @ Rxa @ T_a

    def computeTij(self, dh, q):
        # q = 机器人的current_q + dh.theta

        alpha, a, d = dh[0], dh[1], dh[2]
        T_a = np.eye(4)
        T_a[0, 3] = a

        T_d = np.eye(4)
        T_d[2, 3] = d

        Rzt = np.array([[np.cos(q), -np.sin(q), 0., 0.],
            [np.sin(q), np.cos(q), 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])

        Rxa = np.array([[1., 0., 0., 0.],
            [0., np.cos(alpha), -np.sin(alpha), 0.],
            [0., np.sin(alpha), np.cos(alpha), 0.],
            [0., 0., 0., 1.]])
        """
        np.array([
                [np.cos(q), -np.sin(q)*np.cos(alpha),  np.sin(q)*np.sin(alpha), a*np.cos(q)],
                [np.sin(q),  np.cos(q)*np.cos(alpha), -np.cos(q)*np.sin(alpha), a*np.sin(q)],
                [0,              np.sin(alpha),                np.cos(alpha),               d],
                [0,              0,                            0,                           1]
                ])
        SDH 坐标系i建立在关节i+1处, DH[0].alpha = pi/2 or -pi/2
        """
        return Rzt @ T_d @ Rxa @ T_a

    def ca_jacobian(self):
        Ji = []
        Zi = []
        Oi = []
        Zi.append(ca.DM([0, 0, 1]))
        Oi.append(ca.DM([0, 0, 0]))
        for i in range(self.nq):
            T0i = self.ca_forward(i + 1)
            Zi.append(T0i[0:3, 2])
            Oi.append(T0i[0:3, 3])

        for i in range(self.nq):
            Jv = ca.cross(Zi[i], Oi[-1] - Oi[i])
            Ji.append(ca.vertcat(Jv, Zi[i]))
        Ja = ca.horzcat([Ji[i] for i in range(self.nq)])

        return ca.Function('jacobian', [self.q], [Ja])

    def rotation2quaternion(self, R):
        """

        :param R: a casadi pramater
        :return: a quaternion with casadi symbol
        """
        dim = 3
        trace_R = ca.trace(R)
        # for i in range(dim):
        #     trace_R += R[i, i]

        theta = ca.arccos((trace_R - 1) / 2)
        omega = ca.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]) / (2 * ca.sin(theta))
        q = ca.vertcat(omega * ca.sin(theta / 2), ca.cos(theta / 2))
        return q

    def quaternion_error(self, q1, q2):
        """
        :param q1:
        :param q2:
        :return:
        """
        # so3mat_list = [[0, -q2[2], q2[1]],
        #            [q2[2], 0, -q2[0]],
        #            [-q2[1], q2[0], 0]]
        sx_matrix = ca.MX.zeros(3, 3)

        sx_matrix[0, 1] = -q2[2]
        sx_matrix[0, 2] = q2[1]
        sx_matrix[1, 0] = q2[2]
        sx_matrix[1, 2] = -q2[0]
        sx_matrix[2, 0] = -q2[1]
        sx_matrix[2, 1] = q2[0]

        return q1[3] * q2[:3] - q2[3] * q1[:3] + sx_matrix @ q1[:3]

    def compute_jacobian(self, q):
        z0 = np.array([0, 0, 1])  # Z-axis in base frame， 所有关节的转轴在关节坐标系下的转轴为z轴

        # Transformation matrices and Jacobians
        T = [np.eye(4)]  # Transformation matrices
        Jv = []  # Linear velocity Jacobians
        Jw = []  # Angular velocity Jacobians

        J_ee = np.zeros((6, self.nq))

        # Forward kinematics to compute T and Jacobians
        for i in range(self.nq):
            # Extract DH parameters
            theta = q[i] + self.theta_list[i]  # Add joint position

            # Compute transformation matrix
            T_i = self.computeTij(self.DH_list[i], theta)
            T.append(T[-1] @ T_i)              # 关节在机械臂基坐标系的位姿

            # Extract rotation matrix and position
            R = T[-1][:3, :3]
            p = T[-1][:3, 3]
            T_ee_base = T[-1] @ np.eye(4)
            R_ee = T_ee_base[:3, :3]
            p_ee = T_ee_base[:3, 3]

            # Compute Jacobians
            Jv_i = np.zeros((self.nq, 3))
            Jw_i = np.zeros((self.nq, 3))
            for j in range(i+1):
                R_prev = T[j][:3, :3]
                p_prev = T[j][:3, 3]
                z_prev = R_prev @ z0

                if j < i:
                    Jv_i[j, :] = np.cross(z_prev, p - p_prev)
                    Jw_i[j, :] = z_prev
                else:
                    Jw_i[j, :] = z_prev

                if i == self.nq - 1:
                    J_ee[:3, j] = np.cross(z_prev, p_ee - p_prev)
                    J_ee[3:, j] = z_prev

            Jv_i = np.array(Jv_i).T
            Jw_i = np.array(Jw_i).T
            Jv.append(Jv_i)
            Jw.append(Jw_i)
        return J_ee

if __name__ == "__main__":
    SDH_list = [[0., 1., 0.], [0., 1., 0.], [0., 1., 1.]]
    fk = Arm_FK(3, SDH_list)
    T = np.eye(4)
    lj = fk.forward_compute([np.pi/6, np.pi/6, np.pi/6])
    print(lj @ T)
    fk.set_theta_list([0., 0., 0.])
    print(fk.compute_jacobian([np.pi/6, np.pi/6, np.pi/6]))

