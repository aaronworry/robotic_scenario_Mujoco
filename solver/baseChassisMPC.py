import numpy as np
import casadi as ca
import time


class BaseChassisMPC():
    def __init__(self, dt, N, v_range, omega_range, Q, R):
        self.dt = dt         # timestep
        self.N = N           # Horizon width

        self.v_range = v_range    # range of velocity
        self.omega_range = omega_range # range of angular velocity
        self.v_dim = len(self.v_range)

        # two diag matrix
        self.Q = Q
        self.R = R

        self.state_dim = 3

        # x, u 是 opti 下的对象
        # 此函数必须重载
        x = ca.SX.sym("x", self.state_dim)
        u = ca.SX.sym("u", self.v_dim + 1)

        self.dynamic_func = ca.Function(
            "dynamic_func",
            [x, u],
            [
                np.array([u[0] * ca.cos(x[2]),
                u[0] * ca.sin(x[2]),
                u[1]])
            ],
        )

    def construct_opti(self, state_weight, control_weight, goal_state_weight):
        self.opti = ca.Opti()
        self.opt_x0 = self.opti.parameter(self.state_dim)
        self.opt_controls = self.opti.variable(self.N, self.v_dim + 1)
        self.opt_goal = self.opti.parameter(self.N + 1, self.state_dim)
        self.opt_states = self.opti.variable(self.N + 1, self.state_dim)

        # 初始条件约束
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x0.T )

        # 控制指令约束
        for i in range(len(self.v_range)):
            self.opti.subject_to(self.opti.bounded(self.v_range[i, 0], self.opt_controls[:, i], self.v_range[i, 1]))
        self.opti.subject_to(self.opti.bounded(self.omega_range[0], self.opt_controls[:, -1], self.omega_range[1]))

        # 运动学方程
        for j in range(self.N):
            x_next = self.opt_states[j, :] + self.dt * self.dynamic_func(self.opt_states[j, :], self.opt_controls[j, :]).T
            self.opti.subject_to(self.opt_states[j+1, :] == x_next)

        # 目标函数
        obj = 0.
        self.state_weight = state_weight
        self.control_weight = control_weight
        self.goal_state_weight = goal_state_weight
        for i in range(self.N):
            obj += self.state_weight * ca.mtimes([(self.opt_states[i, :] - self.opt_goal[i, :]), self.Q, (self.opt_states[i, :] - self.opt_goal[i, :]).T])
            obj += self.control_weight * ca.mtimes([self.opt_controls[i, :], self.R, self.opt_controls[i, :].T])
        obj += self.goal_state_weight * ca.mtimes([(self.opt_states[self.N, :] - self.opt_goal[self.N, :]), self.Q, (self.opt_states[self.N, :] - self.opt_goal[self.N, :]).T])

        # 构建优化问题
        self.opti.minimize(obj)
        # 设置求解算法的参数
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-4,
            },
            'print_time': False
        }
        # 定义优化问题, 使用ipopt求解器
        self.opti.solver("ipopt", opts)

    def solve(self, current_state, goal_state):
        self.opti.set_value(self.opt_x0, current_state)
        self.opti.set_value(self.opt_goal, goal_state)

        try:
            sol = self.opti.solve_limited()
            u_res = self.opti.value(self.opt_controls)
            state_res = self.opti.value(self.opt_states)
        except:
            state_res = np.repeat(current_state, self.N + 1, axis = 0)
            u_res = np.zeros((self.N, self.v_dim + 1))

        return state_res, u_res[0, :]

class NonholonomicMPC():
    def __init__(self, dt, N, v_range, omega_range, Q, R):
        self.dt = dt
        self.N = N

        self.v_range = v_range
        self.omega_range = omega_range

        self.Q = Q
        self.R = R


    def forward(self, x, u):
        # 速度控制
        # x y theta v omega; v omega
        x_next = ca.horzcat(
            x[0] + self.dt * x[3] * ca.cos(x[2]),
            x[1] + self.dt * x[3] * ca.sin(x[2]),
            x[2] + self.dt * x[4],
            u[0],
            u[1]
        )
        # 加速度控制
        # x y theta v omega;  v_dot omega_dot
        """
        x_next = ca.horzcat(
            x[0] + self.dt * x[3] * ca.cos(x[2]),
            x[1] + self.dt * x[3] * ca.sin(x[2]),
            x[2] + self.dt * x[4],
            x[3] + self.dt * u[0],
            x[4] + self.dt * u[1]
        )
        """
        # 加速度控制
        # x y theta x_dot y_dot omega;  v_dot omega_dot
        """
        x_next = ca.horzcat(
            x[0] + self.dt * x[3],
            x[1] + self.dt * x[4],
            x[2] + self.dt * x[5],
            x[3] + self.dt * (u[0] * ca.cos(x[2]) - x[4] * x[5]),
            x[4] + self.dt * (u[0] * ca.sin(x[2]) + x[3] * x[5]),
            x[5] + self.dt * u[1]
        )
        """
        # x_next[2] = ca.fmod((x_next[2] + ca.pi), (2 * ca.pi)) - ca.pi
        return x_next

    def angleDiff(self, a, b):
        a = ca.fmod((a + ca.pi), (2*ca.pi)) - ca.pi # convert to [-pi.pi)
        b = ca.fmod((b + ca.pi), (2*ca.pi)) - ca.pi

        angle_diff = ca.if_else(
            a * b >= 0,
            a - b,
            ca.if_else(
                a > b,
                ca.if_else(
                    a - b <= ca.pi,
                    a - b,
                    a - b - 2 * ca.pi
                ),
                ca.if_else(
                    a - b > -ca.pi,
                    a - b,
                    a - b + 2 * ca.pi
                )
            )
        )

        return angle_diff

    def construct_opti(self):
        self.opti = ca.Opti()

        self.X = self.opti.variable(self.N + 1, 5)
        self.U = self.opti.variable(self.N, 2)

        self.X_init = self.opti.parameter(1, 5)

        self.X_ref = self.opti.parameter(self.N, 3)
        self.U_ref = self.opti.parameter(self.N, 2)

        self.X_guess = None
        self.U_guess = None

        # 添加约束
        # 初始位置约束
        self.opti.subject_to(self.X[0, :] == self.X_init)

        # 运动学约束
        for i in range(self.N):
            self.opti.subject_to(self.X[i + 1, :] == self.forward(self.X[i, :], self.U[i, :]))

        # 控制指令约束
        for i in range(self.N):
            self.opti.subject_to(self.opti.bounded(self.v_range[0], self.U[i, 0], self.v_range[1])) # acc constraint
            self.opti.subject_to(self.opti.bounded(self.omega_range[0], self.U[i, 1], self.omega_range[1])) # vel constraint

        # 设置目标函数
        obj = 0.
        for i in range(self.N):
            state_error = ca.horzcat(self.X[i+1, :2] - self.X_ref[i, :2],
                                     self.angleDiff(self.X[i+1, 2], self.X_ref[i, 2])
                                     )
            control = self.U[i, :]
            obj += ca.mtimes([state_error, self.Q, state_error.T]) + ca.mtimes([control, self.R, control.T])

        # 构建优化问题
        self.opti.minimize(obj)
        # 设置求解算法的参数
        opts = {'ipopt.max_iter':2000,
                'ipopt.print_level':0,
                'print_time':0,
                'ipopt.acceptable_tol':1e-8,
                'ipopt.acceptable_obj_change_tol':1e-6}
        # 定义优化问题, 使用ipopt求解器
        self.opti.solver("ipopt", opts)

    def solve(self, x_init, x_ref):
        if self.X_guess is None:
            self.X_guess = np.ones((self.N+1, 5)) * x_init

        if self.U_guess is None:
            self.U_guess = np.zeros((self.N, 2))

        self.opti.set_initial(self.X, self.X_guess)
        self.opti.set_initial(self.U, self.U_guess)

        self.opti.set_value(self.X_ref, x_ref)
        self.opti.set_value(self.X_init, x_init)

        sol = self.opti.solve()

        ## obtain the initial guess of solutions of the next optimization problem
        self.X_guess = sol.value(self.X)
        self.U_guess = sol.value(self.U)
        return self.U_guess[0, :]


if __name__ == "__main__":
    mpc = NonholonomicMPC(0.02, 20, np.array([-2., 2.]), np.array([-1., 1.]), np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]]), np.array([[0.5, 0.], [0., 0.4]]))
    mpc.construct_opti()

    goal = np.zeros((20, 3))
    dtheta = np.pi / 2. / 20.
    for i in range(20):
        vec = np.array([np.cos(np.pi * 1.5 + i * dtheta), np.sin(np.pi * 1.5 + i * dtheta)])
        goal[i, :] = np.array([vec[0] * 0.5, vec[1] * 0.5 + 0.5, i * dtheta])

    u = mpc.solve(np.array([0., 0., 0., 0., 0.]), goal)

    print(u)
