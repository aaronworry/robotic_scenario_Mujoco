import time
import numpy as np

from gym.spaces import Box

from environment.baseEnv import BaseEnv
from visualization.matplot_2d_visual import MatplotViewer
from robot.logarithmic_spiral_2d_robot import TwoStringContinuumRobot, LSRobotGenerator


class Target():
    def __init__(self):
        pass

class LS2D_tracking_Env(BaseEnv):
    def __init__(self, numerical_simulation = True):
        super().__init__(numerical_simulation = True)
        """
        observe_space
        action_space
        state_ub
        state_lb
        control_ub
        control_lb
        """
        self.dt = 0.1
        

        
    def initialization(self):
        """
        initial the environment
        add robot with initial_state
        add object with initial_state
        """
        generator_param = np.array([[1., 0.571, 0.214, 0.8]]*5)
        self.robot = TwoStringContinuumRobot(generator_param, 0.8)
        self.viewer = MatplotViewer(X_limit=[-1., 15.], Y_limit=[-10., 10.])
        self.viewer.show()
        
        
    def render(self):
        """
        read point and line of robots
        draw
        """
        robot_draw = []
        cable_draw = []
        point_draw = []
        last_tr = self.robot.start_point_right
        last_tl = self.robot.start_point_left
        for i in range(self.robot.n):
            module_vertice = self.robot.modules[i].get_vertice_matrix()
            for j in range(6):
                robot_draw.append([module_vertice[j-1], module_vertice[j]])
            
            bl = self.robot.modules[i].get_bottom_left()
            tl = self.robot.modules[i].get_top_left()
            
            br = self.robot.modules[i].get_bottom_right()
            tr = self.robot.modules[i].get_top_right()
            
            cable_draw.append([br, tr])
            cable_draw.append([bl, tl])
            cable_draw.append([last_tr, br])
            cable_draw.append([last_tl, bl])
            
            last_tr = tr
            last_tl = tl
            
            point_draw.append([bl])
            point_draw.append([tl])
            point_draw.append([br])
            point_draw.append([tr])
            
            
        self.viewer.render([robot_draw], [cable_draw], self.dt)
            
        
        
        
    def reset(self):
        """
        set the environment to initial state
        set robot to initial_state
        set object to initial_state
        """
        self.robot.reset()
        self.render()
        
    def step(self, action, dt):
        """
        update state of environment, including objects and robots
        """
        theta_list = self.robot.random_control()
        self.robot.forward(theta_list)
        self.render()
        return 0
        
    def reward(self):
        """
        calculate the reward based on states and actions
        """
        return 0
        
    def get_obs(self):
        """
        get observation for other algorithms
        """
        return 0