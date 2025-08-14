import numpy as np
from robot.baseManipulator import BaseManipulator
import pinocchio


# this can also deploy end-effector

"""
需要实现速度到力的方法，可调用 优化， 等Controller
"""

class Kinova(BaseManipulator):
    def __init__(self, model_file, base_position, joint_pos, joint_velocity):
        super().__init__(base_position, joint_pos, joint_velocity)
        self.load_parameters(model_file)
        self.control_mode = None
        
    def load_parameters(self, model_file):
        """
        get data from URDF files.
        link length, link size, link mass, link CoM, link Inertia
        using pinocchio
        """
        urdf_package = os.path.dirname(model_file)
        urdf = urdf_package + "/GEN3-LITE.urdf"
        self.robot = pinocchio.RobotWrapper.BuildFromURDF(urdf, urdf_package)
        self.robot.data = self.robot.model.createData()
        
        self.robot.model.inertias[link].lever[axis]
        pinocchio.centerOfMass(self.robot.model, self.robot.data, pose, True)
        
        pass
        
    
    def set_control_mode(self, mode):
        """
        control mode of the joints: angle_position, angle_velocity, 
        """
        self.control_mode = mode
        
    def get_control_mode(self):
        return self.control_mode
        
    def step(self, target_position, target_orientation):
        if self.control_mode == "position":
            return self.ik(target_position, target_orientation)
        joint_velocity = self.ik_velocity(target_position, target_orientation)
        if self.control_mode == "velocity":
            return joint_velocity
        elif self.control_mode == "torque":
            return self.ik_force(joint_velocity)
        return None
    
    def set_controller(self):
        pass
        
        # 在控制器中实现PID
        
    def move_to(self):
        pass
        # PID control 
        
        
        
        # Cartesian impedance:
        self.thr_cart_error = 0.001  # m
        self.Kd = np.eye(3) * 40
        self.Dd = np.eye(3) * 3
        self.error_cart_MAX = 0.1  # m
        self.thr_dynamic = 0.3  # rad/s

        # Null space:
        self.K_n = np.eye(6) * 0.125
        self.D_n = np.eye(6) * 0.025

        # Base
        self.thr_pos_error = 0.01  # m
        self.thr_rot_error = np.deg2rad(10)
        self.K_pos = 4
        self.gain_pos_MAX = 1
        self.K_rot = 0.5
        self.gain_rot_MAX = 0.5
        


        