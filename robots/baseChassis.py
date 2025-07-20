import numpy as np

class Chassis():
    def __init__(self, position, orientation, velocity):
        self.position = position
        self.orientation = orientation / np.linalg.norm(orientation)
        self.velocity = velocity
        self.target_position = None
        self.target_orientation = None
        self.target_velocity = None
        self.action = None
        self.controller = None
        self.set_initial_state(position, orientation, velocity)

    def _set_controller(self):
        """
        self.controller = DifferentialDriver()
        
        """
        raise NotImplementedError

    def set_target_state(self, position, orientation, velocity):
        self.target_position = position
        self.target_orientation = orientation / np.linalg.norm(orientation)
        self.target_velocity = velocity
        
    def get_target_state(self):
        return self.target_position, self.target_orientation, self.target_velocity

    
    def ik(self, velocity):
        action = self.controller.run_ik(velocity)
        return action
    
    def set_state(self, position, orientation, velocity):
        if self.controller is not None:
            self.position = position
            self.orientation = orientation / np.linalg.norm(orientation)
            self.velocity = velocity

    def get_state(self):
        return self.position, self.orientation, self.velocity

    def set_initial_state(self, position, orientation, velocity):
        self.init_position = position
        self.init_orientation = orientation / np.linalg.norm(orientation)
        self.init_velocity = velocity
        
    def get_initial_state(self):
        return self.init_position, self.init_orientation, self.init_velocity


    def reset(self, position, orientation, velocity):
        self.position = position
        self.orientation = orientation / np.linalg.norm(orientation)
        self.velocity = velocity
        self.action = None
        self.target_position = None

    def check_collision(self):
        """
        based on shape of chassis, judge the collision
        
        should be implemented in simulation
        """
        raise NotImplementedError
