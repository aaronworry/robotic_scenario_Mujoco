import numpy as np


# generate torque of joint motor

class Manipulator_PD_Controller():
    def __init__(self, Kp = np.eye(3) * 40, Kd = np.eye(3) * 3, error_cart_MAX = 0.1, thr_cart_error = 0.001):
        self.Kp = Kp
        self.Kd = Kd
        self.error_cart_MAX = error_cart_MAX
        self.thr_cart_error = thr_cart_error

        
    def set_parameters(self, Params):
        """
        Params:  a dict
            {"P": , "D": }
        """
        if "P" in Params:
            self.Kp = Params["P"]
        if "D" in Params:
            self.Kd = Params["D"]
            
    def get_parameters(self):
        params = {"P": np.zeros((3, 3)), "D": np.zeros((3, 3))}
        params["P"] = self.Kp
        params["D"] = self.Kd
        return params
        
    def force_control_end_effector(self, x, dx, x_target, dx_target):
        """
        target and current state of the end-effector
        """
        x_e = np.zeros(3)
        error = x_target - x
        magnitude = np.linalg.norm(error)
        if magnitude > self.thr_cart_error:
            vector = error / magnitude           # 单位向量
            self.x_e = vector * min(self.error_cart_MAX, magnitude)     # 最大不超过 error_cart_MAX
        dx_e = dx_target - dx
        force = self.Kp @ x_e + self.Kd @ dx_e    # PD control
        return force
        # manipulator对象中需要 实现 将 force 转化为 关节 torque
        
       
        
        
    def run_ik_velocity(self, target_position, target_orientation):
        # 使用优化器算
        pass
        
    def run_ik_torque(self, joint_velocity):
        pass
        