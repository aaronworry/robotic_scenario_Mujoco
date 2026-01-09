import numpy as np

class RobotArmMDH:
    def __init__(self, mdh_params, masses, coms, inertias_com):
        """
        Initialize the robot arm with MDH parameters and dynamic properties.

        Args:
            mdh_params: List of 4-tuples per joint [(alpha, a, d, theta_offset), ...]
                        Note: theta_offset is added to the active joint variable q.
            masses: List of masses [m1, m2, ...]
            coms: List of CoM vectors (x, y, z) relative to the Link Frame.
            inertias_com: List of 3x3 Inertia matrices at the CoM.
        """
        self.n = len(mdh_params)
        self.mdh = mdh_params
        self.masses = masses
        self.coms = coms

        # Precompute Spatial Inertia Matrices at the Link Origin (Joint Frame)
        self.I_spatial = []
        for i in range(self.n):
            self.I_spatial.append(
                self._compute_link_inertia(masses[i], coms[i], inertias_com[i])
            )

        # Gravity vector (gravity acting DOWN implies base accelerating UP)
        # Format: [angular_x, angular_y, angular_z, linear_x, linear_y, linear_z]
        self.gravity = np.array([0., 0., 0., 0., 0., 9.81])

        # Active Joint Axis (Z-axis in MDH)
        self.S = np.array([0., 0., 1., 0., 0., 0.])

    def _rot_x(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _rot_z(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _skew(self, v):
        """Returns 3x3 skew symmetric matrix of a 3D vector."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def _compute_link_inertia(self, mass, com, I_3x3):
        """
        Converts Mass, CoM, and Inertia(at CoM) into a 6x6 Spatial Inertia Matrix
        expressed at the Link Frame Origin.

        Logic: I_link = X_com_to_link.T * I_com_spatial * X_com_to_link
        """
        # 1. Construct 6x6 Inertia at CoM
        # I_spatial = [ I_3x3    0   ]
        #             [   0     m*I3 ]
        I_spatial_com = np.zeros((6, 6))
        I_spatial_com[0:3, 0:3] = I_3x3
        I_spatial_com[3:6, 3:6] = np.eye(3) * mass

        # 2. Construct Transform from CoM to Link Origin
        # Position of CoM relative to Link Origin is 'com'
        # We need the transform X that moves motion vectors from Link to CoM.
        # But for Inertia transformation I_link = X^T * I_com * X,
        # X must be the Motion Transform from Link Origin to CoM.

        # Translation vector r = com
        # Rotation R = Identity (CoM frame is parallel to Link frame)
        c = np.array(com)

        # X_link_to_com = [ I    0 ]
        #                 [ -S(c)  I ]
        X = np.eye(6)
        X[3:6, 0:3] = -self._skew(c)

        # Transform Inertia
        I_link = X.T @ I_spatial_com @ X
        return I_link

    def _get_transform_mdh(self, i, q_val):
        """
        Calculates the Spatial Motion Transform X from Frame {i-1} to Frame {i}
        using Modified DH parameters: alpha_{i-1}, a_{i-1}, d_i, theta_i

        Returns:
            X_inv: Transform to map Motion from Parent(i-1) to Child(i).
                   v_i = X_inv @ v_{i-1}
        """
        alpha, a, d, theta_offset = self.mdh[i]
        theta = q_val + theta_offset


        T = np.eye(4)
        # RotX(alpha)
        T = T @ np.array([
            [1, 0, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha), 0],
            [0, np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 0, 1]
        ])
        # TransX(a)
        T = T @ np.array([
            [1, 0, 0, a],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # RotZ(theta)
        T = T @ np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # TransZ(d)
        T = T @ np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, d],
            [0, 0, 0, 1]
        ])

        R = T[0:3, 0:3]
        p = T[0:3, 3]

        # 3. Build Plucker Transform X (Parent -> Child)
        # We need the inverse: Motion from Parent to Child.
        # X_{i, i-1} = [ R^T    0 ]
        #              [ -R^T*S(p)  R^T ]

        R_t = R.T
        X_inv = np.zeros((6, 6))
        X_inv[0:3, 0:3] = R_t
        X_inv[3:6, 3:6] = R_t
        X_inv[3:6, 0:3] = -R_t @ self._skew(p)

        return X_inv

    def _crm(self, v):
        """Spatial Cross Product for Motion (v x)."""
        w = v[0:3]
        vel = v[3:6]
        res = np.zeros((6, 6))
        res[0:3, 0:3] = self._skew(w)
        res[3:6, 3:6] = self._skew(w)
        res[3:6, 0:3] = self._skew(vel)
        return res

    def _crf(self, v):
        """Spatial Cross Product for Force (v x*)."""
        return -self._crm(v).T

    # ==========================================================
    # RNEA: Recursive Newton-Euler Algorithm (Inverse Dynamics)
    # ==========================================================
    def inverse_dynamics(self, q, dq, ddq):
        """
        Recursive Newton Euler Algorithm (Inverse Dynamics).

        Args:
            q: Joint positions (rad)
            dq: Joint velocities (rad/s)
            ddq: Joint accelerations (rad/s^2)
        Returns:
            tau: Joint torques (Nm)
        """
        n = self.n

        # Initialize Buffers
        v = [np.zeros(6) for _ in range(n)]
        a = [np.zeros(6) for _ in range(n)]
        f = [np.zeros(6) for _ in range(n)]
        X_up = [np.eye(6) for _ in range(n)] # Transforms parent->i

        # --- 1. Forward Pass (Kinematics) ---
        v_parent = np.zeros(6)
        a_parent = self.gravity.copy() # Base acceleration mimics gravity

        for i in range(n):
            # Get transform from parent(i-1) to current(i)
            X_i = self._get_transform_mdh(i, q[i])
            X_up[i] = X_i

            # Joint velocity vector (Z-axis rotation)
            v_J = self.S * dq[i]

            # v_i = X * v_{i-1} + S * q_dot
            v[i] = X_i @ v_parent + v_J

            # a_i = X * a_{i-1} + S * q_ddot + v_i x (S * q_dot)
            # (Note: v_i x v_J is the Coriolis term)
            a[i] = (X_i @ a_parent) + (self.S * ddq[i]) + (self._crm(v[i]) @ v_J)

            v_parent = v[i]
            a_parent = a[i]

        # --- 2. Backward Pass (Forces/Torques) ---
        tau = np.zeros(n)
        f_next = np.zeros(6) # Force from child link (initially 0 for EE)

        for i in reversed(range(n)):
            # Net Force required for rigid body dynamics
            # F = I*a + v x* (I*v)
            I = self.I_spatial[i]

            # Newton-Euler Equation in Spatial form
            force_net = (I @ a[i]) + (self._crf(v[i]) @ (I @ v[i]))

            # Add forces transmitted from child link
            # We must transform f_next (in frame i+1) to frame i
            if i < n - 1:
                # X_up[i+1] maps motion i->i+1.
                # Its Transpose maps Force i+1->i
                f_child = X_up[i+1].T @ f_next
            else:
                f_child = np.zeros(6)

            f[i] = force_net + f_child

            # Extract Torque (Project onto Z-axis)
            tau[i] = np.dot(f[i], self.S)

            f_next = f[i]

        return tau

    def forward_dynamics(self, q, dq, tau):
        """
        Compute Joint Accelerations given Torque.

        Args:
            q: Joint positions (rad)
            dq: Joint velocities (rad/s)
            tau: Joint torques (Nm)
        Returns:
            ddq: Joint accelerations (rad/s^2)
        """
        n = self.n

        # --- Buffers ---
        v = [np.zeros(6) for _ in range(n)]
        c = [np.zeros(6) for _ in range(n)] # Coriolis acceleration (v x S*qd)
        pA = [np.zeros(6) for _ in range(n)] # Articulated Bias Force
        IA = [np.zeros((6,6)) for _ in range(n)] # Articulated Inertia
        X_up = [np.eye(6) for _ in range(n)] # Transform Parent->Child

        # Intermediate cache for Pass 3
        U = [np.zeros(6) for _ in range(n)]
        D = [0.0 for _ in range(n)]
        u = [0.0 for _ in range(n)] # Intermediate scalar force

        # ----------------------------------------------
        # Pass 1: Forward (Kinematics & Bias Forces)
        # ----------------------------------------------
        v_parent = np.zeros(6)

        for i in range(n):
            # Transform
            X_i = self._get_transform_mdh(i, q[i])
            X_up[i] = X_i

            # 1. Velocity
            v_J = self.S * dq[i]
            v[i] = X_i @ v_parent + v_J

            # 2. Coriolis Acceleration term (c)
            # c[i] = v[i] x S * dq[i]
            c[i] = self._crm(v[i]) @ v_J

            # 3. Spatial Bias Force (p) due to centrifugal/gyroscopic effects
            # p = v x* (I * v)
            I = self.I_spatial[i]
            p_bias = self._crf(v[i]) @ (I @ v[i])

            # Initialize Articulated variables with local rigid body values
            IA[i] = I.copy()
            pA[i] = p_bias.copy()

            v_parent = v[i]

        # ----------------------------------------------
        # Pass 2: Backward (Articulated Inertia)
        # ----------------------------------------------
        # We propagate Inertia and Forces from Tip to Base

        for i in reversed(range(n)):
            # 1. Project Articulated Inertia onto Joint Axis
            # U = IA * S
            U[i] = IA[i] @ self.S

            # D = S.T * IA * S (Scalar effective inertia)
            D[i] = np.dot(self.S, U[i])

            # 2. Compute Net Force scalar 'u'
            # u = tau - S.T * pA
            u[i] = tau[i] - np.dot(self.S, pA[i])

            # 3. Propagate to Parent (if not base)
            if i > 0:
                # Calculate Articulated Inertia of current link locked at joint
                # Ia = IA - U * U.T / D
                Ia = IA[i] - (np.outer(U[i], U[i]) / D[i])

                # Calculate Bias Force of current link locked at joint
                # pa = pA + Ia * c + U * (u / D)
                pa = pA[i] + (Ia @ c[i]) + (U[i] * (u[i] / D[i]))

                # Transform to Parent Frame and Add
                # X_up[i] transforms Parent->Child
                # X_up[i].T transforms Force Child->Parent
                # X_up[i].T * Matrix * X_up[i] transforms Inertia Child->Parent

                X = X_up[i]
                IA[i-1] += X.T @ Ia @ X
                pA[i-1] += X.T @ pa

        # ----------------------------------------------
        # Pass 3: Forward (Accelerations)
        # ----------------------------------------------
        ddq = np.zeros(n)
        a_parent = self.gravity.copy() # Base acceleration

        for i in range(n):
            X = X_up[i]

            # 1. Spatial Acceleration of frame i (before joint acceleration)
            # a' = X * a_parent + c
            a_prime = (X @ a_parent) + c[i]

            # 2. Joint Acceleration
            # ddq = (u - U.T * a') / D
            ddq[i] = (u[i] - np.dot(U[i], a_prime)) / D[i]

            # 3. Spatial Acceleration of link i
            # a = a' + S * ddq
            a_spatial = a_prime + (self.S * ddq[i])

            a_parent = a_spatial

        return ddq

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Define MDH Parameters: [alpha(i-1), a(i-1), d(i), theta_offset]
    # (Simplified values for demonstration)
    mdh_params = [
        (0,       0,     0,   0),    # Joint 1
        (-np.pi/2, 0,     0,   0),    # Joint 2
        (0,       0.43,  0.15, 0),    # Joint 3
        (-np.pi/2, 0.02,  0.43, 0),    # Joint 4
        (np.pi/2,  0,     0,   0),    # Joint 5
        (-np.pi/2, 0,     0,   0)     # Joint 6
    ]

    # 2. Define Mass (kg)
    masses = [5.0, 10.0, 8.0, 2.0, 1.5, 0.5]

    # 3. Define CoM [x, y, z] relative to Link Frame (local)
    # Often in MDH, the link frame is at the joint axis.
    coms = [
        np.array([0, 0, 0.1]),
        np.array([0.2, 0, 0]),
        np.array([0.1, 0, 0]),
        np.array([0, 0.05, 0]),
        np.array([0, 0, 0.02]),
        np.array([0, 0, 0.01])
    ]

    # 4. Define Inertia Matrices at CoM (kg*m^2)
    # Simplified diagonal matrices
    inertias = []
    for i in range(6):
        # Approximating as spheres/rods
        val = 0.01 * masses[i]
        I = np.diag([val, val, val])
        inertias.append(I)

    # Instantiate Robot
    robot = RobotArmMDH(mdh_params, masses, coms, inertias)

    # Inputs
    q =   np.array([0.0, -0.7, 1.5, 0.0, 0.5, 0.0])
    dq =  np.array([0.1, 0.0, 0.2, 0.0, 0.1, 0.0])
    ddq = np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1])

    # Compute
    tau = robot.inverse_dynamics(q, dq, ddq)

    print("Calculated Torques (Nm):")
    with np.printoptions(precision=4, suppress=True):
        print(tau)

    tau_input = np.array([10.0, 50.0, 20.0, 5.0, 1.0, 0.1])

    # Run ABA
    ddq_result = robot.forward_dynamics(q, dq, tau_input)

    print("Forward Dynamics Results (ABA):")
    print("Joint Accelerations (rad/s^2):")
    with np.printoptions(precision=4, suppress=True):
        print(ddq_result)
