import numpy as np
from utils.util import hadamard_sum
from optimizer.continuum_manipulator_velocities_solver import VelocityOptimizer


class LSRobotGenerator():
    def __init__(self, d0, h0, a, n, theta_init, le0 = 0., b = 0.22, dtheta = np.pi/6, tol = 1e-4):
        """
        a, b:    paramether of logarithmic spiral function             rho(theta) = a * e^(b * theta)
        theta_init:   the theta of first module that correspond to logarithmic spiral function           [rad]
        n:    number of modules
        dtheta：   delta theta  along the logarithmic spiral function
        d0, le0, h0, l0:  the parameter of the trapezoidal section of first module
            four point :   [0, 0], [d0, h0], [d0, h0 + le0], [0, l0]
        """
        self.a = a
        self.b = b
        self.dtheta = dtheta
        self.n = n
        self.h0 = h0
        self.le0 = le0
        self.d0 = d0
        self.theta_init = theta_init
        self.tol = tol
        
        assert(self.a > 0.)
        assert(self.d0 > 0.)
        assert(self.h0 > 0.)
        assert(self.dtheta > 0. and self.dtheta < np.pi)
        assert(self.theta_init > self.n * self.dtheta + self.tol)
        
        self.l0 = self.calculate_length(self.theta_init, self.dtheta)
        assert(self.l0 > self.h0 + self.le0)
        
        if self.le0 < self.tol:
            self.le0 = self.l0 - self.h0 - np.tan(self.dtheta - np.arctan2(self.h0, self.d0)) * self.d0
        else:
            assert(np.arctan2(self.h0, self.d0) + np.arctan2(self.l0 - self.le0 - self.h0, self.d0) == self.dtheta)
            pass

        self.result = np.zero((1, 4))
        self.result[0,:] = np.array([self.l0, self.le0, self.h0, self.d0])
        
        
    def center_ls_function(self, theta):
        rho = np.exp(sel.b * theta) * (np.exp(2 * np.pi * self.b) + 1.) * self.a / 2.
        point = np.array([rho * np.cos(theta), rho * np.sin(theta)])
        return point
        
    def calculate_length(self, theta, delta_theta):
        point1= self.center_ls_function(theta)
        point2 = self.center_ls_function(theta - delta_theta)
        return np.linalg.norm(point1 - point2)
        
    def construct_robot_module_parameters(self):
        # theta from +infity to 0
        last_l, last_le, last_h, last_d = self.l0, self.le0, self.h0, self.d0
        last_theta = self.theta_init
        for i in range(1, self.n):
            l = self.calculate_length(last_theta, self.dtheta)
            ratio = l / last_l
            le = ratio * last_le
            d = ratio * last_d
            h = ratio * last_h
            self.result = np.vstack((self.result, np.array([l, le, h, d])))
            
            last_l, last_le, last_h, last_d = l, le, h, d
        return self.result
        
    
class RobotModule2D():
    def __init__(self, params):
        self.l = params[0]
        self.le = params[1]
        self.h = params[2]
        self.d = params[3]
        
        self.position = None
        self.orientation = None
        self.init_orientation = None
        self.init_position = None
        
        self.point_matrix = np.zeros((6, 2))
        self.init_point_matrix = np.zeros((6, 2))

        self.init_cable_point = np.zeros((4, 2))
        self.cable_point = np.zeros((4, 2))
        
        self.cable_inner_length_left = 0.
        self.cable_inner_length_right = 0.
        
        
    def initial_module(self, position, orientation):
        self.init_point_matrix[0, :] = np.array([0., 0.])
        self.init_point_matrix[1, :] = np.array([self.h, self.d])
        self.init_point_matrix[2, :] = np.array([self.h + self.le, self.d])
        self.init_point_matrix[3, :] = np.array([self.l, 0.])
        self.init_point_matrix[4, :] = np.array([self.h + self.le, - self.d])
        self.init_point_matrix[5, :] = np.array([self.h, - self.d])
        
        self.init_orientation = orientation / np.linalg.norm(orientation)
        self.init_position = position
        
    def initial_cable_point(self, bottom_left, bottom_right, top_left, top_right):
        # input: global position in init state
        # can only be used in initialization() of the robot
        
        theta = -1 * np.arctan2(self.init_orientation[1], self.init_orientation[0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        self.init_cable_point[0,:] = bottom_left - self.init_position
        self.init_cable_point[1,:] = top_left - self.init_position
        self.init_cable_point[2,:] = top_right - self.init_position
        self.init_cable_point[3,:] = bottom_right - self.init_position
        
        self.cable_inner_length_left = np.linalg.norm(top_left - bottom_left)
        self.cable_inner_length_right = np.linalg.norm(top_right - bottom_right)
        
        inv_tran = np.transpose(R @ self.init_cable_point.T)
        self.init_cable_point = inv_tran
        
    
    def get_bottom_left(self):
        return self.cable_point[0]
        
    def get_bottom_left_iitial(self):
        return self.init_cable_point[0]
        
    def get_top_left(self):
        return self.cable_point[1]
        
    def get_top_left_initial(self):
        return self.init_cable_point[1]
        
    def get_top_right(self):
        return self.cable_point[2]
        
    def get_top_right_initial(self):
        return self.init_cable_point[2]
        
    def get_bottom_right(self):
        return self.cable_point[3]
        
    def get_bottom_right_initial(self):
        return self.init_cable_point[3]
        
        
        
    def reset(self):
        self.update(self.init_position, self.init_orientation)
        
        
    def update(self, position, orientation):
        self.orientation = orientation / np.linalg.norm(orientation)
        self.position = position
        theta = np.arctan2(orientation[1], orientation[0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        tran = np.transpose(R @ self.init_point_matrix.T)
        for i in range(len(self.point_matrix)):
            self.point_matrix[i, :] = self.position + tran[i, :]
        
        trans2 = np.transpose(R @ self.init_cable_point.T)
        for i in range(len(init_cable_point)):
            self.cable_point[i, :] = self.position + trans2[i, :]
        
            
        
        
            
        


class TwoStringContinuumRobot():
    """
    module are along the x-axis
    y > 0:   right
    y < 0:   left
    """
    def __init__(self, trapezoidal_param, y, taper_angle = np.pi/12):
        
        # y <= trapezoidal_param[0][3]
        self.trapezoidal_param = trapezoidal_param
        self.taper_angle = taper_angle
        self.n = len(self.trapezoidal_param)
        #                                    h,            d                       l - le - h                  d
        delta_theta = np.arctan2(trapezoidal_param[0][2], trapezoidal_param[0][3]) + np.arctan2(trapezoidal_param[0][0] - trapezoidal_param[0][1] - trapezoidal_param[0][2], trapezoidal_param[0][3])
        self.theta_ub = [delta_theta] * self.n
        self.theta_lb = [-delta_theta] * self.n
        delta_theta_0 = np.arctan2(trapezoidal_param[0][2], trapezoidal_param[0][3])
        self.theta_ub[0] = delta_theta_0
        self.theta_lb[0] = -delta_theta_0
        
        self.modules = []
        self.initial_position_matrix = np.zero((self.n, 2))
        self.initial_theta_list = [0.] * self.n
        self.position_matrix = np.zero((self.n, 2))
        self.theta_list = [0.] * self.n
        self.intial_orientations = np.zeros((self.n, 2))
        self.orientations = np.zeros((self.n, 2))
        
        # string state
        self.cable_length_left = 0.
        self.cable_length_right = 0.
        self.cable_length_left_outer = 0.
        self.cable_length_right_outer = 0.
        self.cable_length_left_inner = 0.
        self.cable_length_right_inner = 0.
        
        self.cable_length_max_left = 0.
        self.cable_length_min_left = 0.
        self.cable_length_max_right = 0.
        self.cable_length_min_right = 0.
        self.end_point_left = None
        self.end_point_right = None
        self.start_point_left = np.array([0., -y])
        self.start_point_right = np.array([0., y])
        
        self.end_effector_point_left = np.array([y / np.tan(self.taper_angle / 2), 0.])
        self.end_effector_point_right = np.array([y / np.tan(self.taper_angle / 2), 0.])
        
        self.v_opt = VelocityOptimizer(2, self.n, 2)
        
        self.initialization()
    
    def intersection(point1, point2, point3, point4):        
        """
            line1 : point1, point2
            line2: point3, point4
        """
        x1,y1 = point1[0], point1[1]
        x2,y2 = point2[0], point2[1]
        x3,y3 = point3[0], point3[1]
        x4,y4 = point4[0], point4[1]
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1
        
        A2 = y4 - y3
        B2 = x3 - x4
        C2 = A2 * x3 + B2 * y3
        
        denominator = A1 * B2 - A2 * B1
        if denominator == 0:
            return None  # The lines are parallel.
        
        x = (B2 * C1 - B1 * C2) / denominator
        y = (A1 * C2 - A2 * C1) / denominator
        
        return np.array([x, y])


    
    def initialization(self):
        position = np.array([0., 0.])
        orientation = np.array([1., 0.])
        for i in range(self.n):
            module = RobotModule2D(trapezoidal_param[i, :])
            module.initial_module(position, orientation)
            
            # intersection must exist
            bottom_right = self.intersection(self.start_point_right, self.end_effector_point_right, module.init_point_matrix[0], module.init_point_matrix[1])
            bottom_left = self.intersection(self.start_point_left, self.end_effector_point_left, module.init_point_matrix[0], module.init_point_matrix[5])
            top_right = self.intersection(self.start_point_right, self.end_effector_point_right, module.init_point_matrix[2], module.init_point_matrix[3])
            top_left = self.intersection(self.start_point_left, self.end_effector_point_left, module.init_point_matrix[4], module.init_point_matrix[3])
            module.initial_cable_point(bottom_left, bottom_right, top_left, top_right)
            
            self.cable_length_left_inner += module.cable_inner_length_left
            self.cable_length_right_inner += module.cable_inner_length_right
            
            self.initial_position_matrix[i, :] = position
            self.initial_orientations[i, :] = orientation
            self.initial_theta_list[i] = 0.
            position[0] += module.l
            self.modules.append(module)
            
            if i == self.n - 1:
                self.end_point_left = top_left
                self.end_point_right = top_right
                
        self.forward(self.theta_lb)
        self.cable_length_min_left = self.cable_length_left
        self.cable_length_max_right = self.cable_length_right
        self.forward(self.theta_ub)
        self.cable_length_max_left = self.cable_length_left
        self.cable_length_min_right = self.cable_length_right
        
        
            
    def reset(self):
        for i in range(self.n):
            self.modules[i].reset()
        self.theta_list = self.initial_theta_list
        self.position_matrix = self.initial_position_matrix
        self.orientations = self.initial_orientations
        
        
    def update(self):
        self.cable_length_left_outer = 0.
        self.cable_length_right_outer = 0.
        for i in range(self.n):
            self.modules[i].update(self.position_matrix[i, :], self.orientations[i, :]))
            if i == 0:
                self.cable_length_left_outer += np.linalg.norm(start_point_left - self.modules[i].get_bottom_left())
                self.cable_length_right_outer += np.linalg.norm(start_point_right - self.modules[i].get_bottom_right())
            else:
                self.cable_length_left_outer += np.linalg.norm(self.modules[i-1].get_top_left() - self.modules[i].get_bottom_left())
                self.cable_length_right_outer += np.linalg.norm(self.modules[i-1].get_top_right() - self.modules[i].get_bottom_right())
            
            
        self.cable_length_left = self.cable_length_left_inner + self.cable_length_left_outer
        self.cable_length_right = self.cable_length_right_inner + self.cable_length_right_outer
        
    def cable_length_outer(self, theta_list):
        cable_length_left_outer = 0.
        cable_length_right_outer = 0.
        for i in range(self.n):
            R = np.array([[np.cos(theta_list[i]), -np.sin(theta_list[i])], [np.sin(theta_list[i]), np.cos(theta_list[i])]])
            if i == 0:
                A_l_s = self.start_point_left
                A_r_s = self.start_point_right
                A_l_1_out_initial = self.modules[i].get_bottom_left_initial()
                A_r_1_out_initial = self.modules[i].get_bottom_right_initial()
                cable_length_left_outer += np.linalg.norm(R @ A_l_1_out_initial - A_l_s)
                cable_length_right_outer += np.linalg.norm(R @ A_r_1_out_initial - A_r_s)
            else:
                A_l_i_in = self.modules[i-1].get_top_left_initial()
                A_r_i_in = self.modules[i-1].get_top_right_initial()
                A_l_i_out_initial = self.modules[i].get_bottom_left_initial()
                A_r_i_out_initial = self.modules[i].get_bottom_right_initial()
                P_i = self.modules[i-1].init_point_matrix[3, :]
                cable_length_left_outer += np.linalg.norm(R @ A_l_i_out_initial + P_i - A_l_i_in)
                cable_length_right_outer += np.linalg.norm(R @ A_r_i_out_initial + P_i - A_r_i_in)
        return np.array([cable_length_left_outer, cable_length_right_outer])
        
    
    def random_control(self):
        # only for test
        theta_list = [0.] * self.n
        for i in range(self.n):
            theta = np.random.rand(1)[0] * (self.theta_ub[i] - self.theta_lb[i]) + self.theta_lb[i]
            theta_list[i] = theta
        return theta_list
            
    
    def forward(self, theta_list):
        self.theta_list = theta_list
        total_theta = theta_list[0]
        self.position_matrix[0, :] = np.array([0., 0.])
        self.orientations[0, :] = np.array([np.cos(total_theta), np.sin(total_theta)])
        for i in range(1, self.n):
            length = self.modules[i-1].l
            self.position_matrix[i, :] = self.position_matrix[i-1, :] + length * self.orientations[i-1, :]
            dtheta = theta_list[i]
            total_theta += dtheta
            self.orientations[i, :] = np.array([np.cos(total_theta), np.sin(total_theta)])
        
        self.update()
        
    def compute_Jacobian(self):
        """
        l' = J theta'
        J_0 = left_cable
        J_1 = right_cable
        """
        J = np.zeros((2, self.n))
        for i in range(self.n):
            dR_dq = np.array([[-np.sin(self.theta_list[i]), -np.cos(self.theta_list[i])], [np.cos(self.theta_list[i]), -np.sin(self.theta_list[i])]])
            if i == 0:
                A_l_s = self.start_point_left
                A_r_s = self.start_point_right
                A_l_1_out = self.modules[i].get_bottom_left()
                A_r_1_out = self.modules[i].get_bottom_right()
                A_l_1_out_initial = np.reshape(self.modules[i].get_bottom_left_initial(), (2, 1))
                A_r_1_out_initial = np.reshape(self.modules[i].get_bottom_right_initial(), (2, 1))
                r_l = np.reshape(A_l_1_out - A_l_s, (2, 1))
                r_r = np.reshape(A_r_1_out - A_r_s, (2, 1))
                temp_l =  A_l_1_out_initial @ r_l.T / np.linalg.norm(r_l)
                temp_r = A_r_1_out_initial @ r_r.T / np.linalg.norm(r_r)
                J[0][i] = hadamard_sum(temp_l, dR_dq)
                J[1][i] = hadamard_sum(temp_r, dR_dq)
            else:
                R = np.array([[np.cos(self.theta_list[i]), -np.sin(self.theta_list[i])], [np.sin(self.theta_list[i]), np.cos(self.theta_list[i])]])
                A_l_i_in = self.modules[i-1].get_top_left_initial()
                A_r_i_in = self.modules[i-1].get_top_right_initial()
                A_l_i_out = np.reshape(self.modules[i].get_bottom_left_initial(), (2, 1))
                A_r_i_out = np.reshape(self.modules[i].get_bottom_right_initial(), (2, 1))
                P_i = self.modules[i-1].init_point_matrix[3, :]
                r_l = np.reshape((R @ A_l_i_out + P_i) - A_l_i_in, (2, 1))
                r_r = np.reshape((R @ A_r_i_out + P_i) - A_r_i_in, (2, 1))
                temp_l =  A_l_i_out @ r_l.T / np.linalg.norm(r_l)
                temp_r = A_r_i_out @ r_r.T / np.linalg.norm(r_r)
                J[0][i] = hadamard_sum(temp_l, dR_dq)
                J[1][i] = hadamard_sum(temp_r, dR_dq)
        return J
    
    
    def step_v(self, cable_velocity, dt):
        """
        控制时，只考虑一条绳提供速度
        建模时，同时考虑
        
        
        最小作用量原理计算 更新之后的 theta_list
        
        min \sum |d \theta|         st  cable_velocity = \sum f(d \theta)            l + cable_velocity * dt = \sum g (d \theta + \theta)
        ======> 锥规划           acados不支持锥规划
        min \sum z_i
        s. t.
            cable_velocity = \sum f(d \theta)
            l + cable_velocity * dt = \sum g (d \theta + \theta)
            z_i >= |d \theta_i|
        
        """
        J = self.compute_Jacobian()
        # cable_velocity = J @ x
        # 和下面等效:  但是下面构建优化问题时，需要调用函数，耗时间
        # 对于 [cos -sin         可以考虑构建一个对称矩阵且 R^T R = I 的元素，以略去三角函数     比如 x^T x = 1     [x[0], -x[1]; x[1], x[0]]
        #     [sin cos ]
        # self.cable_length_left_outer - cable_velocity * dt = self.cable_length_outer(self.theta_list + x * dt)[0]
        # self.cable_length_right_outer - cable_velocity * dt = self.cable_length_outer(self.theta_list + x * dt)[1]
        
        # 一个step中可能会调用多次求解器
        #    当角度变换之后被边界条件限制，记录此次的时间，设置边界条件处的mask=1，继续调用求解器
        #    求解的过程中，会出现很多的角速度为0的情况
        pass
        
    
    
    def step_f(self, cable_force):
        pass
        
        
        
        
    
        
        
        