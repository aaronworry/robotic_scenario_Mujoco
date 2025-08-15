import numpy as np

# wheel velocity

class MecanumController():
    def __init__(self):
        self.velocity = None

    def run(self, wheels_lf, wheels_lb, wheels_rf, wheels_rb):
        return velocity 

    def run_ik(self, velocity):
        
        wheels_lf = 0.
        wheels_lb = 0. # left back
        wheels_rf = 0. # right front
        wheels_rb = 0. 
        
        return wheels_lf, wheels_lb, wheels_rf, wheels_rb
        
    def step(self, target_position, target_orientation):
        """
        output torque of wheels
        """
        pass
        
        
    def direction_to_wheel_torques(self, speed, direction: np.ndarray) -> np.ndarray:
        # 机器人底盘相对坐标系
        """Calculate the required wheel torques to move in the given direction."""
        angle = np.arctan2(direction[1], direction[0])

        torque_A = self.calculate_torque(angle)
        torque_B = self.calculate_torque(-angle)
        return np.array([torque_A, torque_B, torque_B, torque_A]) * speed


    def rotation_to_wheel_torques(self, angular_speed, rotation: float) -> np.ndarray:
        """Calculate the required wheel torques to move in the given rotation."""
        return np.array([1, -1, 1, -1]) * np.sign(rotation) * angular_speed


    def calculate_torque(self, angle: float) -> float:
        """Calculate the required wheel torque to match the given moving angle."""
        if angle < -np.pi / 2:
            return -1
        if angle < 0:
            return 1 + angle * (4 / np.pi)
        if angle < np.pi / 2:
            return 1
        return 3 - angle * (4 / np.pi)




class DifferentialController():
    def __init__(self):
        pass