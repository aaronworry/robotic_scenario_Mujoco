import numpy as np
import math
import gym

class BaseChassis():
    def __init__(self, env, pose, orientation):
        self.env = env
        self.pose = pose
        self.init_pose = self.pose

        self._last_step_simulation_count = -1
        self.position = self.pose[0]
        self.orientation = orientation

        self.state = None

        self.target_position = None
        self.target_orientation = None
        self.action = None
        self.awaiting_action = False

        self.waypoint_pos = None
        self.waypoint_ori = None

        self.controller = None

        self.collision = False   #set([self.id])
        self.collision_obstacle = False
        self.collision_vehicle = False

        self.id = self._createBody()

    def update(self):
        self.pose = 
        self.position = 
        self.orientation = 

    def _createBody(self):
        pass

    def _set_controller(self, controller):
        self.controller = controller

    def set_target(self, position, orientation):
        self.target_position = position
        self.target_orientation = orientation

    def set_init_position(self, pose):
        self.init_pose = pose

    def get_position(self):
        position, _ = 
        self.position = position
        return self.position

    def step(self):
        pass

    def get_state(self):
        pass

    def set_init_pose(self, pose):
        self.init_pose = pose

    def reset(self):
        self.action = None
        self.target_position = None
        self.waypoint_pos = None
        self.waypoint_ori = None
        self.reset_pose(self.init_pose)

    def reset_pose(self, pose):
        
        self._last_step_simulation_count = -1

    def check_for_collisions(self):
        for contact_point in 
            body_b_id = contact_point[2]
            if body_b_id in self.collision:
                continue
            if body_b_id in self.env.obstacle_collision_set:
                self.collision_obstacle = True
            if body_b_id in self.env.robot_collision_set:
                self.collision_vehicle = True
            if self.collision_obstacle or self.collision_vehicle:
                break
