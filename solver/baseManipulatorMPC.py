import numpy as np
import casadi as ca
import scipy
from piper_forward import Arm_FK             # 目前只支持piper
from scipy.spatial.transform import Rotation



def euler2quaternion(euler):
    """
    将欧拉角转化为四元数
    :param euler: a list with three angles in radius: [rx, ry, rz]
    :return: a quaternion:   qx, qy, qz, qw
    """
    r = Rotation.from_euler('xyz', euler, degrees=False)
    quat_xyzw = r.as_quat()
    return quat_xyzw


class BaseArmMPC():
    def __init__(self, dt, N, joint_num, p_range, v_range, a_range, Q, R, M, Q_pos, Q_ori, DH_list, arm_fk):
        self.dt = dt            # 0.005 s  对应piper控制频率
        self.N = N
        self.p_range = p_range
        self.v_range = v_range
        self.a_range = a_range

        self.nq = joint_num

        self.Q = Q
        self.R = R
        self.M = M

        self.Q_pos = Q_pos
        self.Q_ori = Q_ori

        self.nv = self.nq

        self.arm_fk = arm_fk(self.nq, DH_list)

        self.q = ca.SX.sym('q', self.nq)
        self.u = ca.SX.sym('v', self.nv)

        self.f_kinematics = ca.Function('f_dot', [self.q, self.u], [self.q + self.u * self.dt])


    def construct_opti_ee(self):
        self.opti = ca.Opti()

        self.X = self.opti.variable(self.N + 1, self.nq)
        self.U = self.opti.variable(self.N, self.nv)

        self.U_last = self.opti.parameter(self.N, self.nv)
        self.X_init = self.opti.parameter(1, self.nq)

        self.X_ref = self.opti.parameter(self.N+1, self.nq)
        self.U_ref = self.opti.parameter(self.N, self.nv)

        self.ee_ref = self.opti.parameter(self.N+1, 7)

        self.x_guess = None
        self.u_latest = None

        # 定义约束
        for k in range(self.N):
            # 状态转移约束
            self.opti.subject_to(self.X[k+1, :].T == self.f_kinematics(self.X[k, :], self.U[k, :]))

            # 关节限制约束
            self.opti.subject_to(self.opti.bounded(self.v_range[0], self.U[k, :], self.v_range[1])) # dq constraint
            self.opti.subject_to(self.opti.bounded(self.p_range[0], self.X[k, :], self.p_range[1])) # q constraint
            self.opti.subject_to(self.opti.bounded(self.a_range[0], self.U[k, :] - self.U_last[k, :], self.a_range[1]))

        # 最终状态约束
        self.opti.subject_to(self.opti.bounded(self.p_range[0], self.X[self.N, :], self.p_range[1])) # q constraint

        # 初值约束
        self.opti.subject_to(self.X[0, :] == self.X_init)


        # 损失函数
        obj = 0.
        for k in range(self.N):
            x_endpoint, R_endpoint = self.arm_fk.forward(self.X[k, :], self.nq)
            ori_x_endpoint = self.arm_fk.rotation2quaternion(R_endpoint)
            ee_state_error = x_endpoint - self.ee_ref[k, :3]            # 末端执行器位置误差
            ori_ee_state_error = ori_x_endpoint.T - self.ee_ref[k, 3:]    # 末端执行器姿态误差
            # state_error = self.X[k, :] - self.X_ref[k, :]               # 关节误差
            # control_error = self.U[k, :] - self.U_ref[k, :]             # 控制误差
            control_change = self.U[k, :] - self.U_last[k, :]           # 控制量变化 , 加速度

            # obj += ca.mtimes([state_error, self.Q, state_error.T])
            # obj += ca.mtimes([control_error, self.R, control_error.T])
            obj += ca.mtimes([control_change, self.M, control_change.T])

            obj += ca.mtimes([ee_state_error, self.Q_pos, ee_state_error.T])
            obj += ca.mtimes([ori_ee_state_error, self.Q_ori, ori_ee_state_error.T])


        x_endpoint_last, R_endpoint_last = self.arm_fk.forward(self.X[self.N, :], self.nq)
        ori_endpoint_last = self.arm_fk.rotation2quaternion(R_endpoint_last)
        last_ee_state_error = x_endpoint_last - self.ee_ref[self.N, :3]          # 最后的末端执行器误差
        last_ori_ee_state_error = ori_endpoint_last.T - self.ee_ref[self.N, 3:]    # 末端执行器姿态误差
        # last_state_error = self.X[self.N, :] - self.X_ref[self.N, :]             # 最后的关节误差

        # obj += ca.mtimes([last_state_error, self.Q, last_state_error.T])
        obj += ca.mtimes([last_ee_state_error, self.Q_pos, last_ee_state_error.T])
        obj += ca.mtimes([last_ori_ee_state_error, self.Q_ori, last_ori_ee_state_error.T])

        # 构建优化问题
        self.opti.minimize(obj)
        # 设置求解算法的参数
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 2000,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6,
            },
            'print_time': 0
        }

        # 定义优化问题, 使用ipopt求解器
        self.opti.solver("ipopt", opts)

    def set_target_pos(self, pos, quat):
        # all trajectory points are set as target position and orientation
        result = np.zeros((self.N+1, 7))
        for k in range(self.N + 1):
            for j in range(3):
                result[k, j] = pos[j]
            for j in range(4):
                result[k, j+3] = quat[j]
        return result


    def solve_ee(self, x_init, traj_ref):
        x_init = np.maximum(np.minimum(x_init, self.p_range[1]), self.p_range[0]).squeeze()
        # assert x_init[1] <= 0 and x_init[2] >= 0

        # Set initial guess for the optimization problem
        if self.x_guess is None:
            self.x_guess = np.ones((self.N+1, self.nq)) * x_init

        if self.u_latest is None:
            self.u_latest = np.zeros((self.N, self.nq))

        self.opti.set_initial(self.X, self.x_guess)
        self.opti.set_initial(self.U, self.u_latest)

        self.opti.set_value(self.ee_ref, traj_ref)
        self.opti.set_value(self.U_last, self.u_latest)
        self.opti.set_value(self.X_init, x_init)

        sol = self.opti.solve()

        self.x_guess = sol.value(self.X)
        self.u_latest = sol.value(self.U)

        return self.u_latest[0, :]

        # joint_position_control = current_state + u_res * dt, 输入到机器人上

    # 注意：MPC中的q为模型中的q
    # MPC的q0 = 机器人的current_q + dh.theta
    # MPC求解得到的q 需要减去 dh.theta, 才能输入给关节控制指令

if __name__ == "__main__":

    theta_list = [0., -ca.pi * 174.22 / 180, -100.78 / 180 * ca.pi, 0., 0., 0.]
    p_range = (ca.horzcat(-2.618, -ca.pi * 174.22 / 180, -2.967 - 100.78 / 180 * ca.pi, -1.745, -1.32, -2.094), ca.horzcat(2.618, 3.14 -ca.pi * 174.22 / 180, -100.78 / 180 * ca.pi, 1.745, 1.32, 2.094))
    v_range = (ca.horzcat(-1, -1, -1, -1, -1, -1), ca.horzcat(1, 1, 1, 1, 1, 1))
    a_range = (ca.horzcat(-0.5, -0.5, -0.5, -0.5, -0.5, -0.5), ca.horzcat(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
    Q = np.diag([200., 200., 200., 200., 200., 200.])
    R = np.diag([20., 20., 20., 20., 20., 20.])
    Q_pos = np.diag([200., 200., 200.])
    Q_ori = np.diag([200., 200., 200., 200.])

    dhs = [[0., 0., 0.123], [-ca.pi/2, 0., 0.], [0., 0.285, 0.], [ca.pi/2, -0.022, 0.25], [-ca.pi/2, 0., 0.], [ca.pi/2, 0., 0.091]]



    mpc = BaseArmMPC(0.02, 20, 6, p_range=p_range, v_range=v_range, a_range = a_range, Q=Q, R=R, M = R, Q_pos=Q_pos, Q_ori = Q_ori, DH_list = dhs, arm_fk=Arm_FK)
    mpc.construct_opti_ee()

    quat = euler2quaternion([0., 0., 0.])
    # start = mpc.set_target_pos([0.2, 0., 0.2], [quat[0], quat[1], quat[2], quat[3]])
    goal = mpc.set_target_pos([0.5, 0., 0.2], [quat[0], quat[1], quat[2], quat[3]])
    initial_state = [0. + theta_list[i] for i in range(len(theta_list))]
    u = mpc.solve_ee(np.array(initial_state), goal)

    print(np.array(initial_state) + u * 0.02 - np.array(theta_list))
