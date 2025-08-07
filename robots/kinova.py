import numpy as np
from robot.baseManipulator import BaseManipulator
import pinocchio

class Kinova(BaseManipulator):
    def __init__(self, model_file, base_position, joint_pos, joint_velocity):
        super().__init__(base_position, joint_pos, joint_velocity)
        self.load_parameters(model_file)
        
    def load_parameters(self, model_file):
        """
        get data from URDF files.
        link length, link size, link mass, link CoM, link Inertia
        using pinocchio
        """
        
        pass
        
    
    
    def set_controller(self):
        pass
        
    def check_collision(self, objects):
        
        pass
        