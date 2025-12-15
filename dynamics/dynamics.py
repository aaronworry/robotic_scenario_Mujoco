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
    omega = so3ToVec(so3)
    if abs(np.linalg.norm(omega)) <= 1e-6:
        return np.eye(3)
    else:
        theta = AxisAng3(omega)[1]
        omgmat = so3 / theta
        return np.eye(3) + np.sin(theta) * omgmat + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatrixLog3(SO3):
    """
    Computes the matrix logarithm of a rotation matrix, SO3 -> so3
    :param SO3: A 3x3 rotation matrix
    :return: The matrix logarithm of SO3
    """
    acosinput = (np.trace(SO3) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if abs(1 + SO3[2][2]) > 1e-6:
            omega = (1.0 / np.sqrt(2 * (1 + SO3[2][2]))) * np.array([SO3[0][2], SO3[1][2], 1 + SO3[2][2]])
        elif abs(1 + SO3[1][1]) > 1e-6:
            omega = (1.0 / np.sqrt(2 * (1 + SO3[1][1]))) * np.array([SO3[0][1], 1 + SO3[1][1], SO3[2][1]])
        else:
            omega = (1.0 / np.sqrt(2 * (1 + SO3[0][0]))) * np.array([1 + SO3[0][0], SO3[1][0], SO3[2][0]])
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
                 np.array([[0., 0., 0., 1.]])]

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
    omega = so3ToVec(se3[:3, :3])
    if abs(np.linalg.norm(omega)) <= 1e-6:
        return np.r_[np.c_[np.eye(3), se3[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omega)[1]
        omgmat = se3[:3, :3] / theta
        return np.r_[np.c_[MatrixExp3(se3[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * omgmat + (theta - np.sin(theta)) * np.dot(omgmat,omgmat), se3[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def MatrixLog6(T):
    """
    Computes the matrix logarithm of a homogeneous transformation matrix, SE3 -> so3
    :param T: A matrix in SE3
    :return: The matrix logarithm of T
    """
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)), [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 1]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat, np.dot(np.eye(3) - omgmat / 2.0 + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) * np.dot(omgmat,omgmat) / theta, [T[0][3], T[1][3], T[2][3]])],
                     [[0, 0, 0, 1]]]

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



