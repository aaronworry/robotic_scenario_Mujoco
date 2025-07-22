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
        
    
        

class TwoStringContinuumRobot():
    def __init__(self, trapezoidal_param, taper_angle = np.pi/12):
        self.trapezoidal_param = trapezoidal_param
        self.taper_angle = taper_angle