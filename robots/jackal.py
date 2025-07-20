import numpy as np
from robots.baseChassis import Chassis
from controller.chassis_controller import DifferentialController

class Jackal(Chassis):
    def __init__(self, position, orientation, velocity, parameters):
        super().__init__(position, orientation, velocity)
        
        self.height = parameters[height]
        self.width = parameters[width]
        self.length = parameters[length]
        self.wheel_distance_width = parameters[wheel_distance_width]
        self.wheel_distance_length = parameters[wheel_distance_length]
        self.mass = parameters[mass]
        
        self.controller = self._set_controller(whether_in_simulator)
        
    def _set_controller(self, whether_in_simulator):
        if whether_in_simulator:
            return DifferentialController(self.wheel_distance_width, self.wheel_distance_length)
        else:
            return None
        
    def check_collision_point(self, point):
        point_ref = point - self.position
        distance = np.linalg.norm(point_ref)
        distance_in_length = abs(np.dot(point_ref, self.orientation))
        distance_in_width = np.sqrt(distance**2 - distance_in_length**2)
        if distance_in_length <= self.length / 2. or distance_in_width <= self.width / 2.:
            return True
        return False
        
    def check_collision(self, points):
        for point in points:
            if self.check_collision_point(point):
                return True, point
        return False, None