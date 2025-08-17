import numpy as np

# output : wheel velocity

class MecanumController():
    def __init__(self, wheel_radius, length_between_wheel, width_between_wheel, develop_type = "X"):
    
        """
        develop_type: O
        /     \
                    -> x
        \     /
        develop_type: X       The axial direction of the roller towards the robot's centre.
        \     /
                    -> x
        /     \
        """
        self.L = length_between_wheel
        self.W = width_between_wheel
        self.r = wheel_radius
        self.develop_type = develop_type
        

    def global_to_frame(self, global_velocity, global_angular_velocity, global_orientation):
        relative_angular_speed =  global_angular_velocity
        relative_velocity = np.array([0., 0.])
        relative_velocity[0] = np.dot(global_velocity, global_orientation)
        relative_velocity[1] = np.sqrt(global_velocity[0]**2 + global_velocity[1]**2 - relative_velocity[0]**2)
        return relative_velocity, relative_angular_speed

    def fk(self, wheel_lf_speed, wheel_rf_speed, wheel_lb_speed, wheel_rb_speed):
        relative_velocity = np.array([0., 0.])
        relative_angular_speed = 0.
        if self.develop_type == "X":
            relative_velocity[0] = (wheel_lf_speed + wheel_rf_speed + wheel_lb_speed + wheel_rb_speed) / 4.
            relative_velocity[1] = (-wheel_lf_speed + wheel_rf_speed - wheel_lb_speed + wheel_rb_speed) / 4.
            relative_angular_speed = (-wheel_lf_speed - wheel_rf_speed + wheel_lb_speed + wheel_rb_speed) / (2 * self.L + 2 * self.W)
        elif self.develop_type == "O":
            relative_velocity[0] = (wheel_lf_speed + wheel_rf_speed + wheel_lb_speed + wheel_rb_speed) / 4.
            relative_velocity[1] = (wheel_lf_speed - wheel_rf_speed + wheel_lb_speed - wheel_rb_speed) / 4.
            relative_angular_speed = (-wheel_lf_speed - wheel_rf_speed + wheel_lb_speed + wheel_rb_speed) / (2 * self.L + 2 * self.W)
            
        return relative_velocity, relative_angular_speed
            

    def ik(self, relative_velocity, relative_angular_speed):
        if self.develop_type == "X":
            wheel_lf_speed = relative_velocity[0] - relative_velocity[1] - relative_angular_speed * (self.W + self.L) / 2.
            wheel_rf_speed = relative_velocity[0] + relative_velocity[1] - relative_angular_speed * (self.W + self.L) / 2.
            wheel_lb_speed = relative_velocity[0] - relative_velocity[1] + relative_angular_speed * (self.W + self.L) / 2.
            wheel_rb_speed = relative_velocity[0] + relative_velocity[1] + relative_angular_speed * (self.W + self.L) / 2.
        elif self.develop_type == "O":
            wheel_lf_speed = relative_velocity[0] + relative_velocity[1] - relative_angular_speed * (self.W + self.L) / 2.
            wheel_rf_speed = relative_velocity[0] - relative_velocity[1] - relative_angular_speed * (self.W + self.L) / 2.
            wheel_lb_speed = relative_velocity[0] + relative_velocity[1] + relative_angular_speed * (self.W + self.L) / 2.
            wheel_rb_speed = relative_velocity[0] - relative_velocity[1] + relative_angular_speed * (self.W + self.L) / 2.
        
        return np.array([wheel_lf_speed, wheel_rf_speed, wheel_lb_speed, wheel_rb_speed])
        
        
    def motor_output_speed(self, wheel_speed):
        """
        m/s  -> rpm
        """
        return (wheel_speed / self.r) * 30 / np.pi
        
        
    def step(self, target_position, target_orientation, current_position, current_orientation, speed, angular_speed):
        """
        output velocity of wheels
        """
        direction = (target_position - current_position) / np.linalg.norm(target_position - current_position)
        global_v = speed * direction
        delta_orientation = np.arctan2(target_orientation[1], target_orientation[0]) - np.arctan2(current_orientation[1], current_orientation[0])
        if delta_orientation > np.pi:
            delta_orientation -= 2 * np.pi
        elif delta_orientation < -np.pi:
            delta_orientation += 2 * np.pi
        global_w = angular_speed * np.sign(delta_orientation)
        v, w = self.global_to_frame(global_v, global_w, current_orientation)
        four_wheel_speed = self.ik(v, w)
        motor_rpm = np.array([0.]*4)
        for i in range(4):
            motor_rpm[i] = self.motor_output_speed(four_wheel_speed[i])
        
        return motor_rpm
        
    

class DifferentialController():
    def __init__(self, wheel_radius, length_between_wheel, width_between_wheel, develop_type = "2"):
        """
        develop_type = "4" / "2"
        """
        
        self.L = length_between_wheel
        self.W = width_between_wheel
        self.r = wheel_radius
        self.develop_type = develop_type

    def fk(self, wheel_lf_speed, wheel_rf_speed, wheel_lb_speed, wheel_rb_speed):
        relative_velocity = np.array([0., 0.])
        relative_angular_speed = 0.
        if self.develop_type == "4":
            relative_velocity[0] = (wheel_lf_speed + wheel_rf_speed + wheel_lb_speed + wheel_rb_speed) / 4.
            relative_angular_speed = (-wheel_lf_speed - wheel_rf_speed + wheel_lb_speed + wheel_rb_speed) / (2 * self.L + 2 * self.W)
        elif self.develop_type == "2":
            relative_velocity[0] = (wheel_lf_speed + wheel_rf_speed) / 2.
            relative_angular_speed = (-wheel_lf_speed -wheel_lb_speed + wheel_rf_speed + wheel_rb_speed) / (2 * self.W)
            
        return relative_velocity, relative_angular_speed
            

    def ik(self, relative_velocity, relative_angular_speed):
        if self.develop_type == "4":
            wheel_lf_speed = relative_velocity[0] - relative_angular_speed * self.W / 2.
            wheel_rf_speed = relative_velocity[0] + relative_angular_speed * self.W / 2.
            wheel_lb_speed = wheel_lf_speed
            wheel_rb_speed = wheel_rf_speed
        elif self.develop_type == "2":
            wheel_lf_speed = relative_velocity[0] - relative_angular_speed * self.W / 2.
            wheel_rf_speed = relative_velocity[0] + relative_angular_speed * self.W / 2.
            wheel_lb_speed = 0.
            wheel_rb_speed = 0.
        
        return np.array([wheel_lf_speed, wheel_rf_speed, wheel_lb_speed, wheel_rb_speed])
        
        
    def motor_output_speed(self, wheel_speed):
        """
        m/s  -> rpm
        """
        return (wheel_speed / self.r) * 30 / np.pi
        
        
    def step(self, target_position, target_orientation, current_position, current_orientation):
        k_rho = 1.
        k_alpha = 4.
        k_beta = -1.
        
        dx = target_position[0] - current_position[0]
        dy = target_position[1] - current_position[1]
        
        rho = np.sqrt(dx**2 + dy**2)
        alpha = np.arctan2(dy, dx) - np.arctan2(curent_orientation[1], current_orientation[0])
        beta = np.arctan2(target_orientation[1], target_orientation[0]) - np.arctan2(dy, dx)
        
        # Normalize angles
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        beta = (beta + np.pi) % (2 * np.pi) - np.pi

        v = k_rho * rho
        w = k_alpha * alpha + k_beta * beta
        four_wheel_speed = self.ik(np.array([v, 0.]), w)
        motor_rpm = np.array([0.]*4)
        for i in range(4):
            motor_rpm[i] = self.motor_output_speed(four_wheel_speed[i])
        
        return motor_rpm
        
        
        
    
class OmniController():
    def __init__(self):
        pass
        