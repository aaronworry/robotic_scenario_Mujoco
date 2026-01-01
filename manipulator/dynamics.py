import numpy as np

# 3 vector -> so3 -> SO3         rotation
def VecToso3(v_so3):
    """
    Converts a 3-vector to a so(3) representation
    :param v_so3:  A 3-vector
    :return: The skew symmetric representation of v_so3
    """
    return np.array([[0,      -v_so3[2],  v_so3[1]],
                     [v_so3[2],       0, -v_so3[0]],
                     [-v_so3[1], v_so3[0],       0]])

def so3ToVec(so3):
    """

    :param so3: A 3x3 skew-symmetric matrix
    :return:
    """
    return np.array([so3[2][1], so3[0][2], so3[1][0]])


def AxisAng3(exp_c3):
    """
    Converts a 3-vector of exponential coordinates for rotation into axis-angle form
    :param exp_c3: a 3-vector of exponential coordinates for rotation
    :return: A unit rotation axis, and the rotation angle
    """
    norm = np.linalg.norm(exp_c3)
    return (exp_c3 / norm, norm)

def MatrixExp3(so3):
    """
    Computes the matrix exponential of a matrix in so(3)
    :param so3: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3: SO3
    """
    # Rodrigues' formula
    omega = so3ToVec(so3)
    if abs(np.linalg.norm(omega)) <= 1e-6:
        return np.eye(3)
    else:
        unit_omega, theta = AxisAng3(omega)
        omgmat = so3 / theta
        unit_omega = np.expand_dims(unit_omega, axis=-1)
        # return np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * unit_omega @ unit_omega.T + np.sin(theta) * omgmat
        return np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatrixLog3(SO3):
    """
    Computes the matrix logarithm of a rotation matrix, SO3 -> so3
    :param SO3: A 3x3 rotation matrix
    :return: The matrix logarithm of SO3
    """
    # https://zhuanlan.zhihu.com/p/369659467
    acosinput = (np.trace(SO3) - 1) / 2.0
    if acosinput >= 1:
        # theta = 0 且 旋转轴未被定义
        return np.zeros((3, 3))
    elif acosinput <= -1:
        # theta = \pi
        if abs(1 + SO3[2][2]) > 1e-6:
            # (I + SO3)[:, 3]
            omega = (1.0 / np.sqrt(2 * (1 + SO3[2][2]))) * np.array([SO3[0][2], SO3[1][2], 1 + SO3[2][2]])
        elif abs(1 + SO3[1][1]) > 1e-6:
            omega = (1.0 / np.sqrt(2 * (1 + SO3[1][1]))) * np.array([SO3[0][1], 1 + SO3[1][1], SO3[2][1]])
        else:
            omega = (1.0 / np.sqrt(2 * (1 + SO3[0][0]))) * np.array([1 + SO3[0][0], SO3[1][0], SO3[2][0]])
        # omega 是一个单位向量
        return VecToso3(np.pi * omega)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (SO3 - np.array(SO3).T)


# rotation + translation <-> SE3
def RpToTrans(R, p):
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def TransToRp(T):
    return T[0: 3, 0: 3], T[0: 3, 3]



# 6 vector -> se3 -> SE3         rotation + translation
def VecTose3(V):
    """
    Converts a spatial velocity vector into a 4x4 matrix in se3
    :param V: A 6-vector representing a spatial velocity
    :return: The 4x4 se3 representation of V
    """
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.array([[0., 0., 0., 0.]])]

def se3ToVec(se3):
    """
    Converts a se3 matrix into a spatial velocity vector
    :param se3: a 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3
    """
    return np.r_[[se3[2][1], se3[0][2], se3[1][0]],
                 [se3[0][3], se3[1][3], se3[2][3]]]

def AxisAng6(exp_c6):
    """
    Converts a 6-vector of exponential coordinates into screw axis-angle form
    :param exp_c6: A 6-vector of exponential coordinates for rigid-body motion S * theta
    :return: The corresponding normalized screw axis, The distance traveled along/about S
    """
    theta = np.linalg.norm([exp_c6[0], exp_c6[1], exp_c6[2]])
    if abs(theta) <= 1e-6:
        theta = np.linalg.norm([exp_c6[3], exp_c6[4], exp_c6[5]])
    return (np.array(exp_c6 / theta), theta)

def MatrixExp6(se3):
    """
    Computes the matrix exponential of a se3 representation of exponential coordinates, se3 -> SE3
    :param se3: a matrix in se3
    :return: The matrix exponential of se3 : SE3
    """
    # https://zhuanlan.zhihu.com/p/557532018
    omega = so3ToVec(se3[:3, :3])
    if abs(np.linalg.norm(omega)) <= 1e-6:
        return np.r_[np.c_[np.eye(3), se3[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omega)[1]
        omgmat = se3[:3, :3] / theta
        # J = I + (1 - np.cos(theta)) / theta * omgmat + (theta - np.sin(theta)) /theta * np.dot(omgmat,omgmat)
        # \rho = se3[:3, 3]
        # J\rho = np.dot(J, se3[:3, 3])
        return np.r_[np.c_[MatrixExp3(se3[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * omgmat + (theta - np.sin(theta)) * np.dot(omgmat,omgmat), se3[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def MatrixLog6(T):
    """
    Computes the matrix logarithm of a homogeneous transformation matrix, SE3 -> so3
    :param T: A matrix in SE3
    :return: The matrix logarithm of T
    """
    # https://zhuanlan.zhihu.com/p/557532018
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)), [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        # J_inv = np.eye(3) - theta / 2. * u + (1.  -  theta / 2. / np.tan(theta / 2.)) * np.dot(u, u)
        # u = omgmat / theta
        # J_inv = np.eye(3) - 1 / 2. * omgmat + (1.  / theta / theta - 1. / 2. / np.tan(theta / 2.) / theta) * np.dot(omgmat, omgmat)
        return np.r_[np.c_[omgmat, np.dot(np.eye(3) - omgmat / 2.0 + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) * np.dot(omgmat,omgmat) / theta, [T[0][3], T[1][3], T[2][3]])],
                     [[0, 0, 0, 0]]]

def TransInv(T):
    """
    Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    """

    R, p = T[:3, :3], T[:3, 3]
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)],
                 [[0, 0, 0, 1]]]


# adjoint matrix of vec_se3, SE3
def Adjoint(T):
    """
    Computes the adjoint representation of a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T
    """
    R, p = T[:3, :3], T[:3, 3]
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]


def ad(V):
    """
    Calculate the 6x6 matrix [adV] of the given 6-vector, Used to calculate the Lie bracket [V1, V2] = [adV1]V2

    :param V: A 6-vector spatial velocity
    :return: The corresponding 6x6 matrix [adV]

    Example Input:
        V = np.array([1, 2, 3, 4, 5, 6])
    Output:
        np.array([[ 0, -3,  2,  0,  0,  0],
                  [ 3,  0, -1,  0,  0,  0],
                  [-2,  1,  0,  0,  0,  0],
                  [ 0, -6,  5,  0, -3,  2],
                  [ 6,  0, -4,  3,  0, -1],
                  [-5,  4,  0, -2,  1,  0]])
    """
    omgmat = VecToso3([V[0], V[1], V[2]])
    return np.r_[np.c_[omgmat, np.zeros((3, 3))],
                 np.c_[VecToso3([V[3], V[4], V[5]]), omgmat]]




# dynamics
def InverseDynamics(q, dq, ddq, g, F, M, G, S):
    """
    This function uses forward-backward Newton-Euler iterations to solve the equation:
    tau = M(q)ddq + c(q,dq) + g(q) + Jtr(q)F

    :param q: n-vector of joint variables
    :param dq: n-vector of joint rates
    :param ddq: n-vector of joint accelerations
    :param g: Gravity vector g
    :param F: Spatial force applied by the end-effector expressed in frame {n+1}
    :param M: List of link frames {i} relative to {i-1} at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :return: The n-vector of required joint forces/torques

    Example Input (3 Link Robot):
        q = np.array([0.1, 0.1, 0.1])
        dq = np.array([0.1, 0.2, 0.3])
        ddq = np.array([2, 1.5, 1])
        g = np.array([0, 0, -9.8])
        F = np.array([1, 1, 1, 1, 1, 1])

        # 正运动学 D-H矩阵
        M01 = np.array([[1, 0, 0,        0],
                        [0, 1, 0,        0],
                        [0, 0, 1, 0.089159],
                        [0, 0, 0,        1]])
        M12 = np.array([[ 0, 0, 1,    0.28],
                        [ 0, 1, 0, 0.13585],
                        [-1, 0, 0,       0],
                        [ 0, 0, 0,       1]])
        M23 = np.array([[1, 0, 0,       0],
                        [0, 1, 0, -0.1197],
                        [0, 0, 1,   0.395],
                        [0, 0, 0,       1]])
        M34 = np.array([[1, 0, 0,       0],
                        [0, 1, 0,       0],
                        [0, 0, 1, 0.14225],
                        [0, 0, 0,       1]])

        G1 = np.diag([0.010267, 0.010267, 0.00666, 3.7, 3.7, 3.7])
        G2 = np.diag([0.22689, 0.22689, 0.0151074, 8.393, 8.393, 8.393])
        G3 = np.diag([0.0494433, 0.0494433, 0.004095, 2.275, 2.275, 2.275])
        G = np.array([G1, G2, G3])
        M = np.array([M01, M12, M23, M34])
        S = np.array([[1, 0, 1,      0, 1,     0],
                          [0, 1, 0, -0.089, 0,     0],
                          [0, 1, 0, -0.089, 0, 0.425]]).T
    Output:
        np.array([74.69616155, -33.06766016, -3.23057314])
    """
    n = len(q)

    Mi = np.eye(4)

    Ai = np.zeros((6, n))
    AdTi = [[None]] * (n + 1)

    AdTi[n] = Adjoint(TransInv(M[n]))

    # v
    Vi = np.zeros((6, n + 1))

    # v_dot
    Vdi = np.zeros((6, n + 1))
    Vdi[:, 0] = np.r_[[0, 0, 0], -np.array(g)]

    Fi = np.array(F).copy()
    taulist = np.zeros(n)

    # forward: calculate velocity of CoM of Links
    for i in range(n):
        Mi = np.dot(Mi, M[i])
        Ai[:, i] = np.dot(Adjoint(TransInv(Mi)), np.array(S)[:, i])
        AdTi[i] = Adjoint(np.dot(MatrixExp6(VecTose3(Ai[:, i] * -q[i])), TransInv(M[i])))
        Vi[:, i + 1] = np.dot(AdTi[i], Vi[:,i]) + Ai[:, i] * dq[i]
        Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i])  + Ai[:, i] * ddq[i] + np.dot(ad(Vi[:, i + 1]), Ai[:, i]) * dq[i]
    # backward: calculate force
    for i in range (n - 1, -1, -1):
        Fi = np.dot(np.array(AdTi[i + 1]).T, Fi) + np.dot(np.array(G[i]), Vdi[:, i + 1]) - np.dot(np.array(ad(Vi[:, i + 1])).T, np.dot(np.array(G[i]), Vi[:, i + 1]))
        taulist[i] = np.dot(np.array(Fi).T, Ai[:, i])
    return taulist

def MassMatrix(q, M, G, S):
    """
    Computes the mass matrix of an open chain robot based on the given configuration
    :param q: A list of joint variables
    :param M: List of link frames i relative to i-1 at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :return: The numerical inertia matrix M(thetalist) of an n-joint serial chain at the given configuration thetalist

    This function calls InverseDynamics n times, each time passing a
    ddq vector with a single element equal to 1 and all other inputs set to 0.
    Each call of InverseDynamics generates a single column, and these columns
    are assembled to create the inertia matrix.
    """
    n = len(q)
    Mass = np.zeros((n, n))
    for i in range (n):
        ddq = [0] * n
        ddq[i] = 1
        Mass[:, i] = InverseDynamics(q, [0] * n, ddq, [0, 0, 0], [0, 0, 0, 0, 0, 0], M, G, S)
    return Mass

def VelQuadraticForces(q, dq, M, G, S):
    """
    Computes the Coriolis and centripetal terms in the inverse dynamics of an open chain robot
    :param q: A list of joint variables
    :param dq: A list of joint rates
    :param M: List of link frames i relative to i-1 at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :return: The vector c(thetalist,dthetalist) of Coriolis and centripetal terms for a given thetalist and dthetalist

    This function calls InverseDynamics with g = 0, F = 0, and ddq = 0.
    """
    return InverseDynamics(q, dq, [0] * len(q), [0, 0, 0], [0, 0, 0, 0, 0, 0], M, G, S)

def GravityForces(q, g, M, G, S):
    """
    Computes the joint forces/torques an open chain robot requires to overcome gravity at its configuration
    :param q: A list of joint variables
    :param g: 3-vector for gravitational acceleration
    :param M: List of link frames i relative to i-1 at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :return: The joint forces/torques required to overcome gravity at q

    This function calls InverseDynamics with F = 0, dq = 0, and ddq = 0
    """
    n = len(q)
    return InverseDynamics(q, [0.] * n, [0] * n, g, [0, 0, 0, 0, 0, 0], M, G, S)

def EndEffectorForces(q, F, M, G, S):
    """
    Computes the joint forces/torques an open chain robot requires only to create the end-effector force F
    :param q: A list of joint variables
    :param F: Spatial force applied by the end-effector expressed in frame {n+1}
    :param M: List of link frames i relative to i-1 at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :return: The joint forces and torques required only to create the end-effector force F

    This function calls InverseDynamics with g = 0, dq = 0, and ddq = 0
    """
    n = len(q)
    return InverseDynamics(q, [0.] * n, [0] * n, [0., 0., 0.], F, M, G, S)

def ForwardDynamics(q, dq, tau, g, F, M, G, S):
    """
    Computes forward dynamics in the space frame for an open chain robot
    :param q: A list of joint variables
    :param dq: A list of joint rates
    :param tau: An n-vector of joint forces/torques
    :param g: Gravity vector g
    :param F: Spatial force applied by the end-effector expressed in frame {n+1}
    :param M: List of link frames i relative to i-1 at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :return: The resulting joint accelerations

    This function computes ddq by solving:
    M(q) * ddq = tau - c(q, dq) - g(q) - Jtr(q) * F
    """
    Mq = MassMatrix(q, M, G, S)
    C = VelQuadraticForces(q, dq, M, G, S)
    Gra = GravityForces(q, g, M, G, S)
    E = EndEffectorForces(q, F, M, G, S)

    return np.dot(np.linalg.inv(Mq), (tau - C - Gra - E))


# control
def ComputeTorque(q, dq, e_int, g, M, G, S, q_des, dq_des, ddq_des, Kp, Ki, Kd):
    """
    Computes the joint control torques at a particular time instant
    :param q: n-vector of joint variables
    :param dq: n-vector of joint rates
    :param e_int: n-vector of the time-integral of joint errors
    :param g: Gravity vector g
    :param M: List of link frames i relative to i-1 at the home position
    :param G: Spatial inertia matrices Gi of the links
    :param S: Screw axes Si of the joints in a space frame, in the format of a matrix with axes as the columns
    :param q_des: n-vector of reference joint variables
    :param dq_des: n-vector of reference joint velocities
    :param ddq_des: n-vector of reference joint accelerations
    :param Kp: The feedback proportional gain (identical for each joint)
    :param Ki: The feedback integral gain (identical for each joint)
    :param Kd: The feedback derivative gain (identical for each joint)
    :return: The vector of joint forces/torques computed by the feedback linearizing controller at the current instant
    """

    e = np.subtract(q_des, q)
    return np.dot(MassMatrix(q, M, G, S), Kp * e + Ki * (np.array(e_int) + e) + Kd * np.subtract(dq_des, dq)) \
           + InverseDynamics(q, dq, ddq_des, g, [0, 0, 0, 0, 0, 0], M, G, S)



def exp3(v):
    """
    angular velocity to SO3
    :param v: angular velocity 3-vector
    :return:
    """
    theta = np.linalg.norm(v)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    alpha_vxvx = (1. - ctheta) / theta**2 if theta > 0.01 else 0.5 - theta**2 / 24
    alpha_vx = stheta / theta if theta > 0.01 else 1. - theta**2 / 6
    v_m = np.expand_dims(v, axis=-1)
    result = alpha_vxvx * (v_m @ v_m.T)
    result[0, 1] -= alpha_vx * v[2]
    result[1, 0] += alpha_vx * v[2]
    result[0, 2] += alpha_vx * v[1]
    result[2, 0] -= alpha_vx * v[1]
    result[1, 2] -= alpha_vx * v[0]
    result[2, 1] += alpha_vx * v[0]

    ctheta = ctheta if theta > 0.01 else 1. - theta**2 / 2.
    result[0, 0] += ctheta
    result[1, 1] += ctheta
    result[2, 2] += ctheta
    return result

def Jexp3(v):
    """
    dexp3_r_dr
    :param v: angular velocity 3-vector
    :return: devirative of exp3(r)
    """
    theta = np.linalg.norm(v)
    n2 = theta ** 2
    n_inv = 1. / theta
    n2_inv = n_inv * n_inv

    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    a = 1. - n2 / 6. if theta < 0.01 else stheta * n_inv
    b = -0.5 - n2 / 24. if theta < 0.01 else (ctheta - 1.) * n2_inv
    c = 1. / 6. - n2 / 120. if theta < 0.01 else n2_inv * (1. - a)
    result = a * np.ones((3, 3))
    result[0, 1] = -b * v[2]
    result[1, 0] = -result[0, 1]
    result[0, 2] = b * v[1]
    result[2, 0] = -result[0, 2]
    result[1, 2] = -b * v[0]
    result[2, 1] = -result[1, 2]
    v_m = np.expand_dims(v, axis=-1)
    result += c * (v_m @ v_m.T)
    return result

def norm_R(R):
    res = np.zeros((3, 3))
    res[:3, 0] = R[:3, 0] / np.linalg.norm(R[:3, 0])
    res[:3, 1] = R[:3, 1] / np.linalg.norm(R[:3, 1])
    res[:3, 2] = np.cross(res[:3, 0], res[:3, 1])
    res[:3, 0] = np.cross(res[:3, 1], res[:3, 2])
    return res

def compute_theta_axis(index, value, R):
    i0 = index
    i1 = (index + 1) % 3
    i2 = (index + 2) % 3

    temp = 1. if R[i2, i1] >= R[i1, i2] else -1.
    s = np.sqrt(value + 1e-8) * temp

    axis = np.array([0., 0., 0.])
    axis[i0] = s / 2.
    axis[i1] = 1. / 2. / s * (R[i1, i0] + R[i0, i1])
    axis[i2] = 1. / 2. / s * (R[i2, i0] + R[i0, i2])

    w = 1. / 2. / s * (R[i2, i1] - R[i1, i2])
    axis_norm = np.linalg.norm(axis)

    theta = 2. * np.arctan2(axis_norm, w)
    axis /= axis_norm
    return theta, axis

def log3(SO3):
    """
    这个函数可能有问题，不建议使用
    :param SO3: a 3x3 matrix
    :return: 3-vector for angular velocity
    """

    R_normed = norm_R(SO3)
    tr = np.trace(R_normed)
    ctheta = (tr - 1.) / 2.

    val_singular = 2. * np.diag(R_normed) - tr + 1.
    theta_0, axis_0 = compute_theta_axis(0, val_singular[0], R_normed)
    theta_1, axis_1 = compute_theta_axis(1, val_singular[1], R_normed)
    theta_2, axis_2 = compute_theta_axis(2, val_singular[2], R_normed)

    if val_singular[0] >= val_singular[1]:
        if val_singular[0] >= val_singular[2]:
            theta_singular = theta_0
            angle_axis_singular = axis_0
        else:
            theta_singular = theta_2
            angle_axis_singular = axis_2
    else:
        if val_singular[1] >= val_singular[2]:
            theta_singular = theta_1
            angle_axis_singular = axis_1
        else:
            theta_singular = theta_2
            angle_axis_singular = axis_2

    acos_expansion = np.sqrt(2. * (1. - ctheta) + 1e-8)
    if tr <= 3. - 1e-3:
        if tr >= -1. + 1e-3:
            theta_nominal = np.arccos(ctheta)
        else:
            theta_nominal = np.pi - acos_expansion
    else:
        theta_nominal = acos_expansion

    antisymmetric_R = unSkew(R_normed)
    norm_antisymmetric_R_squared = np.linalg.norm(antisymmetric_R) ** 2
    t = theta_nominal / np.sin(theta_nominal) if theta_nominal >= 1e-3 else 1. + norm_antisymmetric_R_squared / 6. + 3. / 40. * norm_antisymmetric_R_squared**2
    theta = theta_nominal if ctheta >= -1. + 1e-3 else theta_singular
    axis = t * antisymmetric_R if ctheta >= -1. + 1e-3 else theta_singular * angle_axis_singular
    return axis, theta


def Jlog3(theta, axis):
    # theta = np.linalg.norm(axis)
    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    st_1mct = stheta / (1. - ctheta)
    alpha = 1. / 12. + theta * theta / 720 if theta < 0.01 else 1. / theta / theta - st_1mct / 2. / theta
    diag_value = 0.5 * (2. - theta * theta / 6.) if theta < 0.01 else 0.5 * theta * st_1mct

    axis_m = np.expand_dims(axis, axis=-1)
    Jlog_ = alpha * (axis_m @ axis_m.T)
    Jlog_ += np.eye(3) * diag_value

    Jlog_ += VecToso3(0.5 * axis)
    return Jlog_

def unSkew(M):
    """
    一种安全操作， 对于任意一个矩阵A    0.5(A - A.T) 是一个skew symmetric matrix
    :param M:
    :return:
    """
    res = np.array([0.] * 3)
    res[0] = 0.5 * (M[2, 1] - M[1, 2])
    res[1] = 0.5 * (M[0, 2] - M[2, 0])
    res[2] = 0.5 * (M[1, 0] - M[0, 1])
    return res

# 以下代码和pinocchio的描述不一致
# 在pinocchio中 前面3个是位置 后面是旋转
# 此处 前面是旋转 后面是位置
def exp6(vec):
    """

    :param vec: 6-vector, first 3 is angular, last 3 is trans
    :return: SE3
    """
    w = vec[:3]
    v = vec[3:]
    theta = np.linalg.norm(w)
    t2 = theta**2
    inv_t2 = 1. / t2
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    alpha_wxv = 0.5 - t2 / 24. if theta < 0.01 else (1. - ctheta) * inv_t2
    alpha_v = 1. - t2 / 6. if theta < 0.01 else stheta / theta
    alpha_w = 1. / 6. - t2 / 120. if theta < 0.01 else (1. - alpha_v) * inv_t2
    diagonal_term = 1. - t2 / 2. if theta < 0.01 else ctheta

    trans = alpha_v * v + (alpha_w * w.dot(v)) * w + alpha_wxv * np.cross(w, v)

    w_m = np.expand_dims(w, axis=-1)
    rot = alpha_wxv * (w_m @ w_m.T)
    rot[0, 1] -= alpha_v * w[2]
    rot[1, 0] += alpha_v * w[2]
    rot[0, 2] += alpha_v * w[1]
    rot[2, 0] -= alpha_v * w[1]
    rot[1, 2] -= alpha_v * w[0]
    rot[2, 1] += alpha_v * w[0]
    rot[0, 0] += diagonal_term
    rot[1, 1] += diagonal_term
    rot[2, 2] += diagonal_term

    result = np.eye(4)
    result[:3,:3] = rot
    result[:3, 3] = trans
    return result



def Jexp6(vec):
    """

    :param vec: 6-vector, first 3 is angular, last 3 is trans
    :return: Derivative of SE3
    """
    w = vec[:3]
    v = vec[3:]
    theta = np.linalg.norm(w)
    t2 = theta**2
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    t_inv = 1. / theta
    t2_inv = t_inv * t_inv
    inv_2_2ct = 1. / 2. / (1. - ctheta)

    beta = 1. / 12. + t2 / 720. if theta < 0.01 else t2_inv - stheta * t_inv * inv_2_2ct
    # 为了简化计算，beta' 多除了一个 t
    beta_dot_over_theta = 1. / 360. if theta < 0.01 else -2. * t2_inv * t2_inv + (1. + stheta * t_inv) * t2_inv * inv_2_2ct

    result = np.zeros((6, 6))
    result[3:, 3:] = Jexp3(w)
    result[:3, :3] = result[3:, 3:]

    p = result[:3, :3].T @ v
    wTp = np.dot(w, p)

    w_m = np.expand_dims(w, axis = -1)
    p_m = np.expand_dims(p, axis = -1)

    J = VecToso3(0.5 * p) + beta_dot_over_theta * wTp * (w_m @ w_m.T) - (t2 * beta_dot_over_theta + 2. * beta) * (p_m @ w_m.T) + wTp * beta * np.eye(3) + beta * (w_m @ p_m.T)
    result[3:, :3] = -1 * result[:3, :3] @ J
    return result


def log6(SE3):
    """

    :param SE3:
    :return:
    """
    R = SE3[:3, :3]
    p = SE3[:3, 3]
    antisymmetric_R = unSkew(R)
    t2 = np.linalg.norm(antisymmetric_R) ** 2
    tr = np.trace(R)

    w, theta = log3(R)

    stheta = np.sin(theta)
    ctheta = np.cos(theta)

    alpha = 1. - t2 / 12. - t2 * t2 / 720. if tr >= 3. - 1e-3 else theta * stheta / (2. * (1. - ctheta))
    beta = 1. / 12. + t2 / 720. if tr >= 3. - 1e-3 else 1. / (theta * theta) - stheta / (2. * theta * (1. - ctheta))

    result = np.array([0.] * 6)
    result[:3] = w
    result[3:] = alpha * p - 0.5 * np.cross(w, p) + (beta * np.dot(w, p)) * w
    return result


def Jlog6(SE3):
    """

    :param SE3:
    :return:
    """
    R = SE3[:3, :3]
    p = SE3[:3, 3]

    w, t = log3(R)
    A = Jlog3(t, w)
    D = A

    t2 = t * t
    tinv = 1. / t
    t2inv = tinv * tinv
    stheta = np.sin(t)
    ctheta = np.cos(t)

    inv_2_2ct = 1. / (2. * (1. - ctheta))

    beta = 1. / 12. + t2 / 720. if t < 0.01 else t2inv - stheta * tinv * inv_2_2ct

    # 为了简化计算，beta' 多除了一个 t
    beta_dot_over_theta = 1. / 360. if t < 0.01 else -2 * t2inv * t2inv + (1. + stheta * tinv) * t2inv * inv_2_2ct

    wTp = np.dot(w, p)

    # 这里乘上了一个t
    v3_temp = beta_dot_over_theta * wTp * w - (t2 * beta_dot_over_theta + 2. * beta) * p

    w_m = np.expand_dims(w, axis = -1)
    p_m = np.expand_dims(p, axis = -1)
    C = v3_temp * w_m.T + beta * w_m @ p_m.T + wTp * beta * np.eye(3) + VecToso3(0.5 * p)
    B = C @ A
    result = np.zeros((6, 6))
    result[:3, :3] = A
    result[3:, :3] = B
    result[3:, 3:] = D
    return result



if __name__ == "__main__":
    so3 = np.array([[0., 1., 2.], [-1., 0., 3.], [-2., -3., 0.]])
    v3 = so3ToVec(so3)

    SO3 = MatrixExp3(so3)
    so3_re = MatrixLog3(SO3)
    v3_re = so3ToVec(so3_re)
    axis_re, theta_re = AxisAng3(v3_re)

    SO3_re = MatrixExp3(so3_re)
    so3_temp = MatrixLog3(SO3_re)
    v3_temp = so3ToVec(so3_temp)
    axis_temp, theta_temp = AxisAng3(v3_temp)


    se3 = np.zeros((4, 4))
    se3[:3, :3] = so3
    se3[:3, 3] = np.array([0.5, 1., 0.7])
    SE3 = MatrixExp6(se3)
    se3 = MatrixLog6(SE3)
    v6 = se3ToVec(se3)
    SE3_temp = exp6(v6)
    print(SE3_temp, log6(SE3_temp))
    print(Jlog6(SE3_temp))
    print(Jexp6(v6))




