import numpy as np
from robots.baseChassis import Chassis
from controller.chassis_controller import QuadrupedDriven

# To be continue...

class Go1(Chassis):
    def __init__(self, position, orientation, velocity, parameters, whether_in_simulator = True):
        super().__init__(position, orientation, velocity)
        
        
        self.mass = parameters[mass]
        
        self.controller = self._set_controller(whether_in_simulator)
        
    def _set_controller(self, whether_in_simulator):
        if whether_in_simulator:
            return QuadrupedDriven()
        else:
            return None
        
    def check_collision_point(self, point):
        
        raise NotImplementedError
        
    def check_collision(self, points):
        for point in points:
            if self.check_collision_point(point):
                return True, point
        return False, None