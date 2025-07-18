import numpy as np
import pybullet as p

def get_euler_from_quaternion(quaternion):
    return list(p.getEulerFromQuaternion(quaternion))

def restrain(theta):
    if theta > np.pi:
        theta -= 2*np.pi
    elif theta < -np.pi:
        theta += 2*np.pi
    return theta

class MecanumController():
    def __init__(self):
        self.velocity = None

    def control(self, linear_velocity, lateral_velocity):
        return linear_velocity, lateral_velocity

    def control_position(self, cur_position, cur_orientation, position, orientation):
        delta_position = position - cur_position
        delta_orientation = restrain(orientation - cur_orientation)
        distance = np.sqrt(delta_position[0]**2 + delta_position[1]**2)
        v = max(min(0.5 * distance / 3., 0.5), 0.15)
        vx = v * delta_position[0] / distance
        vy = v * delta_position[1] / distance
        w = delta_orientation
        if distance < 0.7:
            if abs(w) < 0.2:
                return 0., 0., 0., True
            else:
                return 0., 0., 0., True
        return vx, vy, 0., False

class DifferentialController():
    def __init__(self):
        pass