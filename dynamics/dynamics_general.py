import numpy as np

def newton_euler_dynamics(DH_params, CoM, masses, inertias, g, q, qd, qdd, T_ee, f_ext, n_ext):
    """
    Implements the Newton-Euler method to compute joint torques.
    MDH description

    Args:
        DH_params (list of dict): DH parameters for each joint (a, alpha, d, theta).
        CoM (list of np.array): centre of mass of eack links, in link frame.
                It's noted that CoM in SDH and MDH are different.
        masses (list): List of masses for each link.
        inertias (list): List of inertia tensors for each link (3x3 numpy arrays).
        g (np.array): Gravitational acceleration vector (e.g., [0, 0, -9.81]).
        q (np.array): Joint positions (n,).
        qd (np.array): Joint velocities (n,).
        qdd (np.array): Joint accelerations (n,).
        T_ee (4x4 matrix): describe the ee in last joint frame.
        f_ext (np.array): external force applied by ee, in base frame, (n, ).
        n_ext (np.array): external torque applied by ee, in base frame, (n, ).

    Returns:
        tau (np.array): Joint torques (n,).
    """
    n = len(DH_params)  # Number of joints
    z0 = np.array([0, 0, 1])  # Z-axis in base frame

    # Initialize variables
    T = []  # Transformation matrices                                        n
    omega = [np.zeros(3)]  # Angular velocity         base frame
    omegad = [np.zeros(3)]  # Angular acceleration                           n + 1
    v = [np.zeros(3)]  # Linear velocity                                     n + 1
    vd = [g]  # Linear acceleration (base link)                              n + 1
    vc = []   # Linear velocity of com of link                               n
    vd_c = [] # Linear acceleration of com of link                           n
    T_last_base = np.eye(4)

    # Forward pass: Compute velocities and accelerations
    for i in range(n):
        # Extract DH parameters
        a, alpha, d, theta = DH_params[i]['a'], DH_params[i]['alpha'], DH_params[i]['d'], DH_params[i]['theta']
        theta += q[i]  # Add joint position

        # Compute transformation matrix MDH
        T_i = np.array([
            [np.cos(theta), -np.sin(theta), 0., a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
            [0, 0, 0, 1]
        ])
        #T_i 表示第i个joint坐标系下的值 在 i-1下的描述           T[0] 表示第一个joint下的值在base frame下的描述
        T.append(T_i)
        T_last_base = T_last_base @ T_i      # 最后一个joint在base frame的描述
        # Extract rotation matrix
        R = T[-1][:3, :3]

        # Compute angular velocity and acceleration
        # 在link坐标系下的描述
        omega_i = R.T @ omega[-1] + qd[i] * z0
        omegad_i = R.T @ omegad[-1] + np.cross(R.T @ omega[-1], qd[i] * z0) + qdd[i] * z0

        # Compute linear velocity and acceleration
        v_i = R.T @ (v[-1] + np.cross(omega[-1], T[-1][:3, 3]))
        vd_i = R.T @ (vd[-1] + np.cross(omegad[-1], T[-1][:3, 3]) + np.cross(omega[-1], np.cross(omega[-1], T[-1][:3, 3])))
        # 将CoM转移到机械臂基坐标系上， CoM在SDH和MDH下的描述不同，传入参数时需注意
        # MDH: CoM =   SDH: -p + CoM
        # SDH: CoM =   MDH: p + CoM
        v_c_i = v_i + np.cross(omega_i, CoM[i])
        vd_c_i = vd_i + np.cross(omegad_i, CoM[i]) + np.cross(omega_i, np.cross(omega_i, CoM[i]))

        # Append results
        omega.append(omega_i)       # 角速度
        omegad.append(omegad_i)     # 角加速度
        v.append(v_i)               # 关节线速度
        vd.append(vd_i)             # 关节加速度
        vc.append(v_c_i)            # link质心的速度
        vd_c.append(vd_c_i)         # link质心的加速度

    # Backward pass: Compute forces and torques
    # 0 base, joint_i, -1 f_ext
    # f0, n0： base受到地面或机器人底盘的力，力矩
    f = [np.zeros(3) for _ in range(n + 2)]
    n_torque = [np.zeros(3) for _ in range(n + 2)]

    f[-1] = T_last_base[:3, :3].T @ f_ext   # 向量在不同坐标系下的描述只和旋转有关
    n_torque[-1] = T_last_base[:3, :3].T @ n_ext
    tau = np.zeros(n)
    T.append(T_ee)

    for i in reversed(range(1, n + 1)):
        # Extract rotation matrix and link properties
        R = T[i][:3, :3]
        p = T[i][:3, 3]
        m = masses[i - 1]
        I = inertias[i - 1]
        # Compute force and torque
        Fi = m * vd_c[i-1]
        Ni = I @ omegad[i] + np.cross(omega[i], I @ omega[i])

        f[i] = R @ f[i+1] + Fi
        n_torque[i] = R @ n_torque[i + 1] + np.cross(p, R @ f[i + 1]) + Ni + np.cross(CoM[i-1], Fi)
        # Compute joint torque
        tau[i-1] = np.dot(n_torque[i], z0)
    R = T[0][:3, :3]
    p = T[0][:3, 3]
    f[0] = R @ f[1]
    n_torque[0] = R @ n_torque[1] + np.cross(p, R @ f[1])
    return tau

def compute_dynamics(MDH_params, masses, inertias, g, q, qd):
    """
    Computes the Inertia Matrix (M), Gravity Vector (G), and Coriolis Matrix (C) for a manipulator.

    Args:
        DH_params (list of dict): DH parameters for each joint (a, alpha, d, theta).
        masses (list): List of masses for each link.
        inertias (list): List of inertia tensors for each link (3x3 numpy arrays).
        g (np.array): Gravitational acceleration vector (e.g., [0, 0, -9.81]).
        q (np.array): Joint positions (n,).
        qd (np.array): Joint velocities (n,).

    Returns:
        M (np.array): Inertia matrix (n x n).
        G (np.array): Gravity vector (n,).
        C (np.array): Coriolis matrix (n x n).
    """
    n = len(MDH_params)  # Number of joints
    z0 = np.array([0, 0, 1])  # Z-axis in base frame， 所有关节的转轴在关节坐标系下的转轴为z轴

    # Transformation matrices and Jacobians
    T = [np.eye(4)]  # Transformation matrices
    Jv = []  # Linear velocity Jacobians
    Jw = []  # Angular velocity Jacobians

    # Forward kinematics to compute T and Jacobians
    for i in range(n):
        # Extract DH parameters
        a, alpha, d, theta = MDH_params[i]['a'], MDH_params[i]['alpha'], MDH_params[i]['d'], MDH_params[i]['theta']
        theta += q[i]  # Add joint position

        # Compute transformation matrix
        T_i = np.array([
            [np.cos(theta), -np.sin(theta), 0., a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
            [0, 0, 0, 1]
        ])
        T.append(T[-1] @ T_i)              # 关节在机械臂基坐标系的位姿

        # Extract rotation matrix and position
        R = T[-1][:3, :3]
        p = T[-1][:3, 3]

        # Compute Jacobians
        Jv_i = np.zeros((n, 3))
        Jw_i = np.zeros((n, 3))
        for j in range(i + 1):
            R_prev = T[j+1][:3, :3]
            p_prev = T[j+1][:3, 3] 
            z_prev = R_prev @ z0

            if j < i:
                Jv_i[j, :] = np.cross(z_prev, p - p_prev)
                Jw_i[j, :] = z_prev
            else:
                Jw_i[j, :] = z_prev

        Jv_i = np.array(Jv_i).T
        Jw_i = np.array(Jw_i).T
        Jv.append(Jv_i)
        Jw.append(Jw_i)

    # Compute Inertia Matrix M
    M = np.zeros((n, n))
    for i in range(n):
        M += masses[i] * (Jv[i].T @ Jv[i]) + Jw[i].T @ inertias[i] @ Jw[i]

    # Compute Gravity Vector G
    G = np.zeros(n)
    for i in range(n):
        G += masses[i] * (Jv[i].T @ g)

    # Compute Coriolis Matrix C
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += 0.5 * (M[i, j] + M[i, k] - M[j, k]) * qd[k]

    return M, G, C

# Example usage: 6-DOF manipulator
MDH_params = [
    {'a': 0.0, 'alpha': 0., 'd': 0.123, 'theta': 0.0},
    {'a': 0.0, 'alpha': -np.pi/2, 'd': 0.0, 'theta': -np.pi * 174.22 / 180},
    {'a': 0.285, 'alpha': 0.0, 'd': 0.0, 'theta': -100.78 / 180 * np.pi},
    {'a': -0.022, 'alpha': np.pi / 2, 'd': 0.25, 'theta': 0.0},
    {'a': 0.0, 'alpha': -np.pi / 2, 'd': 0.0, 'theta': 0.0},
    {'a': 0.0, 'alpha': np.pi / 2, 'd': 0.091, 'theta': 0.0}
]

masses = [5, 5, 5, 5, 5, 5]  # Link masses
inertias = [np.eye(3) for _ in range(6)]  # Inertia tensors
g = np.array([0, 0, -9.81])  # Gravity vector
q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # Joint positions
qd = np.zeros(6)  # Joint velocities
qdd = np.zeros(6)  # Joint accelerations
T_ee = np.eye(4)
T_ee[2, 3] = 0.145
CoM = [np.array([0.02, 0., 0.]), np.array([0.02, 0., 0.]), np.array([0.02, 0., 0.]), np.array([0.02, 0., 0.]), np.array([0.02, 0., 0.]), np.array([0.02, 0., 0.])]

tau = newton_euler_dynamics(MDH_params, CoM, masses, inertias, g, q, qd, qdd, T_ee, np.array([0., 0., 0.]), np.array([0., 0., 0.]))
print("Joint torques:", tau)

M, G, C = compute_dynamics(MDH_params, masses, inertias, g, q, qd)
print("Inertia Matrix (M):")
print(M)
print("\nGravity Vector (G):")
print(G)
print("\nCoriolis Matrix (C):")
print(C)
