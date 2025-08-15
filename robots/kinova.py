import numpy as np
from robot.baseManipulator import BaseManipulator
from controller.manipulator_controller import Manipulator_PD_Controller
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
        self.set_controller()
        
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
        
    def step_position(self, target_position):
        force = self.controller.force_control_end_effector(self.end_effector.position, self.end_effector.velocity, target_position, np.zeros(3))
        torque = self.force_to_joint_torque(force)
        return torque
        
    def set_controller(self):
        self.controller = Manipulator_PD_Controller()
        
    def force_to_joint_torque(self, force):
        pass
        
    def ik(self, target_position, target_orientation):
        pass
        
    def move_to(self, position, orientation = None):
        if orientation == None:
            self.step_position(position)
        else:
            pass
        # PID control 
        
        


        