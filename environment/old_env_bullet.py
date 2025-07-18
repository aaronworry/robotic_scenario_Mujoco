import math
import pkgutil
import sys
from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from pathlib import Path
from pprint import pprint
from baseVehicle import Vehicle
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
from baseVehicle import heading_to_orientation

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

class Env():
    def __init__(self, robot_config=None, thing_config=None, num_thing=10, env_name='small_empty', show_gui=False):
        self.robot_config = robot_config
        self.thing_config = thing_config
        self.num_thing = num_thing
        self.env_name = env_name
        self.show_gui = show_gui

        self.room_length = 4
        self.room_width = 4
        self.receptacle_position = (self.room_length / 2 - 0.05, self.room_width / 2 - 0.05, 0)

        if self.show_gui:
            self.p = bc.BulletClient(connection_mode=p.GUI)
            self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.p = bc.BulletClient(connection_mode=p.DIRECT)

        self.p.resetDebugVisualizerCamera() #

        #whether robot poses are out of date
        self.step_simulation_count = 0

        self.robot_spawn_bounds = None
        self.thing_spawn_bounds = None

        if self.robot_config is None:
            self.robot_config = [{'grab_vehicle':1}]
        self.num_robots = sum(sum(g.values()) for g in self.robot_config)  #vehicle and UAV
        self.robot_group_types = [next(iter(g.keys())) for g in self.robot_config]

        if self.thing_config is None:
            self.thing_config = [{'small_cube': 1}]
        self.num_things = sum(sum(g.values()) for g in self.thing_config) #small big
        self.thing_group_types = [next(iter(g.keys())) for g in self.thing_config]

        self.robot_ids = None
        self.robots = None
        self.robot_groups = None

        self.thing_ids = None
        self.things = None
        self.thing_groups = None

        self.obstacle_ids = None

        # track of env state
        self.obstacle_collision_body_b_ids_set = None
        self.robot_collision_body_b_ids_set = None
        self.available_thing_ids_set = None
        self.removed_thing_ids_set = None

        self.steps = None
        self.simulation_steps = None
        self.inactivity_steps = None

    def reset(self):
        self.p.resetSimulation()
        self.p.setRealTimeSimulation(0)
        self.p.setGravity(0, 0, -9.8)

        self._create_env()
        self._reset_poses()
        self._step_simulation_until_still()

        # Set awaiting new action for first robot
        self._set_awaiting_new_action()

        self.steps = 0
        self.simulation_steps = 0
        self.inactivity_steps = 0

        return self.get_state()

    def store_new_action(self):
        # save the action, so it can be used to train some NN
        pass

    def step(self, action):

        sim_steps = self._execute_actions()
        self._set_awaiting_new_action()

        for thing_id in self.available_thing_ids_set.copy():
            thing_position = self.get_thing_position(thing_id)

            closest_robot = self.robots[np.argmin([distance(robot.get_position(), thing_position) for robot in self.robots])]

            # thing is pushed by robot
            if isinstance(closest_robot, PushVehicle):
                closest_robot.process_thing_position()   #??

            # thing when be moved to goal
            if self.thing_position_in_receptacle(thing_position):
                # closest_robot need unload the thing
                closest_robot.porcess_cube_success() #?
                self.remove_thing(thing_id)
                self.available_thing_ids_set.remove(thing_id)

        self.steps += 1
        self.simulation_steps += sim_steps
        if sum(robot.things for robot in self.robots) > 0:
            self.inactivity_steps = 0
        else:
            self.inactivity_steps += 1

        done = len(self.removed_thing_ids_set) == self.num_things or self.inactivity_steps >= self.inactivity_cutoff #100

        for robot in self.robots:
            if robot.awaiting_new_action or done:
                robot.compute_reward_and_states(done=done)

        state = [[None for _ in g] for g in self.robot_groups] if done else self.get_state()
        reward = [[robot.reward if (robot.awaiting_new_action or done) else None for robot in robot_group] for
                  robot_group in self.robot_groups]
        info = {
            'steps': self.steps,
            'simulation_steps': self.simulation_steps,
            'distance': [[robot.distance if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_things': [[robot.cumulative_things if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_distance': [[robot.cumulative_distance if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_reward': [[robot.cumulative_reward if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_obstacle_collisions': [[robot.cumulative_obstacle_collisions if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_robot_collisions': [[robot.cumulative_robot_collisions if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'total_things': sum(robot.cumulative_things for robot in self.robots),
            'total_obstacle_collisions': sum(robot.cumulative_obstacle_collisions for robot in self.robots),
            'total_robot_collisions': sum(robot.cumulative_robot_collisions for robot in self.robots),
        }

        return state, reward, done, info

    

    def _create_env(self):
        floor_thickness = 10
        wall_thickness = 1.4
        room_length_with_walls = self.room_length + 2 * wall_thickness
        room_width_with_walls = self.room_width + 2 * wall_thickness
        floor_half_extents = (room_length_with_walls / 2, room_width_with_walls / 2, floor_thickness / 2)
        floor_collision_shape_id = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half_extents)
        floor_visual_shape_id = self.p.createVisualShape(p.GEOM_BOX, halfExtents=floor_half_extents)
        self.p.createMultiBody(0, floor_collision_shape_id, floor_visual_shape_id, (0, 0, -floor_thickness / 2))

        obstacle_color = (0.9, 0.9, 0.9, 1)
        self.obstacle_ids = []
        for obstacle in self._get_obstacles(wall_thickness):
            half_height = 0.005 if 'low' in obstacle else 0.002
            obstacle_half_extents = (obstacle['x_len'] / 2, obstacle['y_len'] / 2, half_height)
            obstacle_collision_shape_id = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
            obstacle_visual_shape_id = self.p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=obstacle_color)
            obstacle_id = self.p.createMultiBody(0, obstacle_collision_shape_id, obstacle_visual_shape_id, (obstacle['position'][0], obstacle['position'][1], 0.1), heading_to_orientation(obstacle['orientation']))
            self.obstacle_ids.append(obstacle_id)

        receptacle_color = (1, 87.0 / 255, 89.0 / 255, 1)  # Red
        receptacle_collision_shape_id = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=(0, 0, 0))
        receptacle_visual_shape_id = self.p.createVisualShape(p.GEOM_BOX, halfExtents=(0.05, 0.05, 0.0001), rgbaColor=receptacle_color, visualFramePosition=(0, 0, 0.0001))
        self.receptacle_id = self.p.createMultiBody(0, receptacle_collision_shape_id, receptacle_visual_shape_id, self.receptacle_position)

        self.robot_collision_body_b_ids_set = set()
        self.robot_ids = []
        self.robots = []
        self.robot_groups = [[] for _ in range(len(self.robot_config))]
        for robot_group_index, g in enumerate(self.robot_config):
            robot_type, count = next(iter(g.items()))
            for _ in range(count):
                robot = Vehicle.get_robot(robot_type, self, robot_group_index)
                self.robots.append(robot)
                self.robot_groups[robot_group_index].append(robot)
                self.robot_ids.append(robot.id)

        cube_half_extents = (0.005, 0.005, 0.005)
        cube_collision_shape_id = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half_extents)
        cube_visual_shape_id = self.p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half_extents, rgbaColor=(237.0 / 255, 201.0 / 255, 72.0 / 255, 1))
        small_cube_mass = 0.024  # 24 g
        self.thing_ids = []
        for _ in range(self.num_things):
            thing_id = self.p.createMultiBody(small_cube_mass, cube_collision_shape_id, cube_visual_shape_id)
            self.thing_ids.append(thing_id)

        self.obstacle_collision_body_b_ids_set = set(self.obstacle_ids)
        self.robot_collision_body_b_ids_set.update(self.robot_ids)
        self.available_thing_ids_set = set(self.thing_ids)
        self.removed_thing_ids_set = set()





    def _reset_poses(self):
        for robot in self.robots:
            pos_x, pos_y, orientation = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
            robot.reset_pose(pos_x, pos_y, orientation)

        for thing_id in self.thing_ids:
            pos_x, pos_y, orientation = self._get_random_thing_pose()
            self.reset_thing_pose(thing_id, pos_x, pos_y, orientation)

        done = False
        while not done:
            done = True
            self.step_simulation()
            for robot in self.robots:
                reset_robot_pose = False

                # Check if robot is stacked on top of a cube
                if robot.get_position(set_z_to_zero=False)[2] > 0.001:  # 1 mm
                    reset_robot_pose = True

                # Check if robot is inside an obstacle or another robot
                for contact_point in self.p.getContactPoints(robot.id):
                    if contact_point[2] in self.obstacle_collision_body_b_ids_set or contact_point[2] in self.robot_collision_body_b_ids_set:
                        reset_robot_pose = True
                        break

                if reset_robot_pose:
                    done = False
                    pos_x, pos_y, orientation = self._get_random_robot_pose(padding=robot.RADIUS,
                                                                        bounds=self.robot_spawn_bounds)
                    robot.reset_pose(pos_x, pos_y, orientation)

    def _get_random_thing_pose(self):
        done = False
        while not done:
            pos_x, pos_y = self._get_random_position(padding=0.05, bounds=self.0.1)

            # Only spawn cubes outside of the receptacle
            if self.receptacle_id is None or not self.cube_position_in_receptacle((pos_x, pos_y)):
                done = True

        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return pos_x, pos_y, heading

    def _get_random_robot_pose(self, padding=0, bounds=None):
        position_x, position_y = self._get_random_position(padding=padding, bounds=bounds)
        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return position_x, position_y, heading

    def _get_random_position(self, padding=0, bounds=None):
        low_x = -self.room_length / 2 + padding
        high_x = self.room_length / 2 - padding
        low_y = -self.room_width / 2 + padding
        high_y = self.room_width / 2 - padding
        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            if x_min is not None:
                low_x = x_min + padding
            if x_max is not None:
                high_x = x_max - padding
            if y_min is not None:
                low_y = y_min + padding
            if y_max is not None:
                high_y = y_max - padding
        position_x, position_y = self.room_random_state.uniform((low_x, low_y), (high_x, high_y))
        return position_x, position_y

    def _step_simulation_until_still(self):
        # Kick-start gravity
        for _ in range(2):
            self.step_simulation()

        movable_body_ids = self.robot_ids + self.thing_ids
        prev_positions = []
        sim_steps = 0
        done = False
        while not done:
            # Check whether any bodies moved since last step
            positions = [self.p.getBasePositionAndOrientation(body_id)[0] for body_id in movable_body_ids]
            if len(prev_positions) > 0:
                done = True
                for prev_position, position in zip(prev_positions, positions):
                    change = distance(prev_position, position)
                    # Ignore removed cubes (negative z)
                    if position[2] > -0.0001 and change > 0.0005:  # 0.5 mm
                        done = False
                        break
            prev_positions = positions

            self.step_simulation()
            sim_steps += 1

            if sim_steps > 800:
                break

    def _set_awaiting_new_action(self):
        if sum(robot.awaiting_new_action for robot in self.robots) == 0:
            for robot in self.robots:
                if robot.is_idle():
                    robot.awaiting_new_action = True
                    break

    def _execute_actions(self):    #帧同步？
        sim_steps = 0
        while True:
            if any(robot.is_idle() for robot in self.robots):
                break

            self.step_simulation()
            sim_steps += 1
            for robot in self.robots:
                robot.step()

        return sim_steps



