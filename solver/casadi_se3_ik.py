import numpy as np
import casadi as ca
import os

class Arm_ik():
    def __init__(self, zero_SE3_list, axis_list, q_min, q_max):
        self.zero_SE3_list = zero_SE3_list
        self.axis_list = axis_list
        self.q_min = q_min
        self.q_max = q_max
        self.nq = len(self.zero_SE3_list)

    def forward(self, q):
        T = ca.DM.eye(4)
        i = 0
        for SE3 in self.zero_SE3_list:
            T_dh = ca.mtimes(ca.DM(SE3), self.matrixexp3(q[i], ca.SX(self.axis_list[i])))
            T = ca.mtimes(T, T_dh)
            i += 1
        return T

    def compute_T_ee(self, q_value):
        q_ca = ca.SX.sym("q_ca", self.nq)
        func = ca.Function("compute_ee", [q_ca], [self.forward(q_ca)])
        return func(q_value)

    def matrixlog3(self, SO3):
        """
        CasADi符号版本 旋转矩阵转李代数 (SO3 -> so3)
        :param SO3: CasADi的SX/MX/DM对象，3x3的旋转矩阵（必须是合法的正交矩阵、行列式=1）
        :return: CasADi的3x1向量，so3李代数，对应旋转轴+旋转角的紧凑表示
        """
        # 替换np.trace → CasADi矩阵迹运算
        acosinput = (ca.trace(SO3) - 1.0) / 2.0

        # ========== 分支1: acosinput >= 1 ，θ=0，无旋转，返回零向量 ==========
        zero_vec = ca.SX.zeros(3, 1)

        # ========== 分支2: acosinput <= -1 ，θ=π(180°) 奇异情况 ==========
        omega1 = (1.0 / ca.sqrt(2 * (1 + SO3[2,2]))) * ca.vertcat(SO3[0,2], SO3[1,2], 1 + SO3[2,2])
        omega2 = (1.0 / ca.sqrt(2 * (1 + SO3[1,1]))) * ca.vertcat(SO3[0,1], 1 + SO3[1,1], SO3[2,1])
        omega3 = (1.0 / ca.sqrt(2 * (1 + SO3[0,0]))) * ca.vertcat(1 + SO3[0,0], SO3[1,0], SO3[2,0])

        omega_pi = ca.if_else(ca.fabs(1 + SO3[2,2]) > 1e-6, omega1,
                    ca.if_else(ca.fabs(1 + SO3[1,1]) > 1e-6, omega2, omega3))
        pi_vec = ca.pi * omega_pi

        # ========== 分支3: -1 < acosinput < 1 ，常规非奇异情况 ==========
        theta = ca.arccos(acosinput)
        so3_mat = theta / (2.0 * ca.sin(theta)) * (SO3 - SO3.T)
        # 提取反对称矩阵的李代数向量：[so3[2,1], so3[0,2], so3[1,0]] 索引和原代码一致
        normal_vec = ca.vertcat(so3_mat[2,1], so3_mat[0,2], so3_mat[1,0])

        res = ca.if_else(acosinput >= 1.0, zero_vec, ca.if_else(acosinput <= -1.0, pi_vec, normal_vec))

        return res

    def matrixexp3(self, theta, axis):
        # 构造旋转轴的反对称矩阵
        skew_symmetric = ca.vertcat(
            ca.horzcat(0, -axis[2], axis[1]),
            ca.horzcat(axis[2], 0, -axis[0]),
            ca.horzcat(-axis[1], axis[0], 0)
        )

        # 罗德里格斯公式核心计算
        I = ca.DM.eye(3)
        R = I + ca.sin(theta) * skew_symmetric + (1 - ca.cos(theta)) * skew_symmetric @ skew_symmetric

        T = ca.blockcat([
                [R[0, 0], R[0, 1], R[0, 2], 0],
                [R[1, 0], R[1, 1], R[1, 2], 0],
                [R[2, 0], R[2, 1], R[2, 2], 0],
                [0, 0, 0, 1]
            ])

        # 数值稳定性：θ趋近于0时直接返回单位矩阵，避免分母为0
        res = ca.if_else(ca.fabs(theta) <= 1e-8, ca.DM.eye(4), T)
        return res


    def pose_error(self, q, T_tar):
        T_cur = self.forward(q)
        p_c = T_cur[:3, 3]
        p_d = T_tar[:3, 3]
        R_c = T_cur[:3, :3]
        R_d = T_tar[:3, :3]

        # 位置误差：期望减实际
        e_pos = p_d - p_c

        # 姿态误差：R_d * R_c^T 的李代数（旋转矩阵→角速度向量，误差最小）
        R_ref = R_c.T @ R_d


        e_att = self.matrixlog3(R_ref)
        return ca.vertcat(e_pos, e_att)

    def createSolver(self):
        # Creating symbolic variables
        self.cq = ca.SX.sym("q", self.nq, 1)
        self.cTf = ca.SX.sym("tf", 4, 4)


        self.error = ca.Function(
            "error",
            [self.cq, self.cTf],
            [
                self.pose_error(self.cq, self.cTf)
            ],
        )


        # Defining the optimization problem
        self.opti = ca.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.var_q_last = self.opti.parameter(self.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.regularization_cost = ca.sumsqr(self.var_q)
        self.smooth_cost = ca.sumsqr(self.var_q - self.var_q_last)

        # 误差的符号化计算
        error_vector = self.error(self.var_q, self.param_tf)
        pos_error = error_vector[:3]  # 位置误差
        ori_error = error_vector[3:]  # 姿态误差
        # 设置位置和姿态的权重
        weight_position = 1.0  # 位置权重
        weight_orientation = 0.5  # 姿态权重

        # 误差的cost
        self.error_cost = weight_position * ca.sumsqr(pos_error) + weight_orientation * ca.sumsqr(ori_error)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.q_min,
            self.var_q,
            self.q_max)
        )

        self.opti.minimize(200.0 * self.error_cost + 0.01 * self.regularization_cost + 1. * self.smooth_cost)

        ##### IPOPT #####
        opts = {
            'ipopt':{
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4,
                'sb': 'yes'
                # 'hessian_approximation':"limited-memory"
            },
            'print_time':False  # print or not
            #'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.nq)

    def ik(self, T , current_arm_motor_q = None, current_arm_motor_dq = None):
        if current_arm_motor_q is not None:
            self.init_data = current_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, T)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)

            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q


            dof = np.zeros(self.nq)
            dof[:len(sol_q)] = sol_q
            return dof

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)


            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q


            dof = np.zeros(self.nq)
            dof[:len(sol_q)] = self.init_data

            raise e

