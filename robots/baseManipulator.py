import numpy as np

class BaseManipulator():
    def __init__(self, base_position, joint_pos, joint_velocity):
        self.base_position = base_position
        self.joint_pos = joint_pos
        self.joint_velocity = joint_velocity
        
        self.set_initial_state(base_position, joint_pos, joint_velocity)
        self.controller = None
        
        """
        target of end-effector
        """
        self.target_position = None
        self.target_orientation = None
        
        self.joint_torque = None
        

    def set_initial_state(self, position, joint_pos, joint_velocity)
        self.init_base_position = position
        self.init_joint_pos = joint_pos
        self.init_joint_velocity = joint_velocity
        
    def get_initial_state(self):
        return self.init_base_position, self.init_joint_pos, self.init_joint_velocity
        
    def set_state(self, position, joint_pos, joint_velocity):
        if self.controller is not None:
            self.base_position = position
            self.joint_pos = joint_pos
            self.joint_velocity = joint_velocity

    def get_state(self):
        return self.base_position, self.joint_pos, self.joint_velocity
        
    def set_target_pos(self, position, orientation):
        self.target_position = position
        self.target_orientation = orientation / np.linalg.norm(orientation)
        
    def get_target_pos(self):
        return self.target_position, self.target_orientation
        
    def set_controller(self):
        
        raise NotImplementedError
        
    def ik(self, target_position, target_orientation):
        # solve an optimization problem,  f(q+dq) = target_position, target_orientation
        # min \sum |dq|
        pass
    
    
    def ik_velocity(self, target_position, target_orientation):
        # output the despired joint velocity, w.r.t. current state
        # ik , Jacobian Matrix
        # should be realized
        joint_velocity = self.controller.run_ik_velocity(target_position, target_orientation)
        return joint_velocity
        
    def ik_force(self, joint_velocity):
        # output the torque to motor, w.r.t. current state
        # PID or other method
        # should provide api for researchers
        joint_torque = self.controller.run_ik_torque(joint_velocity)
        return joint_torque
        
        
    def reset(self):
        self.base_position = self.init_base_position
        self.joint_pos = self.init_joint_pos
        self.joint_velocity = self.init_joint_velocity
        self.target_orientation = None
        self.target_position = None
        

    def check_collision(self):
        """
        based on shape of link, judge the collision
        
        should be implemented in numerical simulation
        in Mujoco or other simulator, this function can be ignored
        """
        pass
        
        
    