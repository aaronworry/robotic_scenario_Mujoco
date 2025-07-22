import numpy as np


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
    # for visualization
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
        
    def initial_module(self, position, orientation):
        self.init_point_matrix[0, :] = np.array([0., 0.])
        self.init_point_matrix[1, :] = np.array([self.h, self.d])
        self.init_point_matrix[2, :] = np.array([self.h + self.le, self.d])
        self.init_point_matrix[3, :] = np.array([self.l, 0.])
        self.init_point_matrix[4, :] = np.array([self.h + self.le, - self.d])
        self.init_point_matrix[5, :] = np.array([self.h, - self.d])
        
        self.init_orientation = orientation / np.linalg.norm(orientation)
        self.init_position = position
        
    def reset(self):
        self.update(self.init_position, self.init_orientation)
        
        
    def update(self, position, orientation):
        self.orientation = orientation / np.linalg.norm(orientation)
        self.position = position
        theta = np.arctan2(orientation[1], orientation[0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        
        tran = np.transpose(R @ self.init_point_matrix.T)
        for i in range(6):
            self.point_matrix[i, :] = self.position + self.init_point_matrix[i, :]
        
            
        


class TwoStringContinuumRobot():
    """
    module are along the x-axis
    """
    def __init__(self, trapezoidal_param, y, taper_angle = np.pi/12):
        self.trapezoidal_param = trapezoidal_param
        self.taper_angle = taper_angle
        self.n = len(self.trapezoidal_param)
        #                                    h,            d                       l - le - h                  d
        delta_theta = np.arctan2(trapezoidal_param[0][2], trapezoidal_param[0][3]) + np.arctan2(trapezoidal_param[0][0] - trapezoidal_param[0][1] - trapezoidal_param[0][2], trapezoidal_param[0][3])
        
        self.theta_ub = [delta_theta] * self.n
        self.theta_lb = [-delta_theta] * self.n
        self.modules = []
        self.initial_position_matrix = np.zero((self.n, 2))
        self.initial_theta_list = [0.] * self.n
        self.position_matrix = np.zero((self.n, 2))
        self.theta_list = [0.] * self.n
        
        # string state
        self.end_point1 = None
        self.end_point2 = None
        self.start_point1 = np.array([0., y])
        self.start_point2 = np.array([0., -y])
        
        self.initialization()
        
        
    def initialization(self):
        position = np.array([0., 0.])
        for i in range(self.n):
            module = RobotModule2D(trapezoidal_param[i, :])
            module.initial_module(position, np.array([1., 0.]))
            self.initial_position_matrix[i, :] = position
            self.initial_theta_list[i] = 0.
            position[0] += module.l
            self.modules.append(module)
            
    def reset(self):
        for i in range(self.n):
            self.modules[i].reset()
        self.theta_list = self.initial_theta_list
        self.position_matrix = self.initial_position_matrix
        
        
    def update(self):
        # for visualization
        for i in range(self.n):
            self.modules[i].update(self.position_matrix[i, :], np.array([np.cos(self.theta_list[i]), np.sin(self.theta_list[i])]))
        
    
    def random_control(self):
        # only for test
        theta_list = [0.] * self.n
        for i in range(self.n):
            theta = np.random.rand(1)[0] * (self.theta_ub[i] - self.theta_lb[i]) + self.theta_lb[i]
            theta_list[i] = theta
        return theta_list
            
    
    def forward(self, theta_list):
        total_theta = 0.
        for i in range(1, self.n):
            dtheta = self.theta_list[i-1]
            total_theta += dtheta
            length = self.modules[i-1].l
            self.position_matrix[i, :] = self.position_matrix[i-1, :] + length * np.array([np.cos(total_theta), np.sin(total_theta)])
        
        
    
        
        
        