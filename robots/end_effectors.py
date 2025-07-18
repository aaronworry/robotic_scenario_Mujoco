import os
import numpy as np
import math
import pybullet as p
from attrdict import AttrDict
from collections import namedtuple
from threading import Thread
import time
import quaternion


SUCTION_BASE_URDF = os.path.join(os.path.dirname(__file__),'../../assets/ur5_defi/suction/suction-base.urdf')
SUCTION_HEAD_URDF = os.path.join(os.path.dirname(__file__),'../../assets/ur5_defi/suction/suction-head.urdf')
ROBOTIQ_BASE_URDF = os.path.join(os.path.dirname(__file__), '../../assets/ur5_defi/gripper/robotiq_2f_85.urdf')
ROBOTIQ_BASE_URDF_new = os.path.join(os.path.dirname(__file__), '../../assets/gripper/robotiq_2f_85.urdf')


class Robotiq85():
    def __init__(self, env, robot, ee, obj_ids):
        self.env = env
        self.robot = robot
        orientation = (0., np.pi/2, 0.)
        self.open_close_flag = False
        self.orientation = np.asarray(p.getQuaternionFromEuler(orientation))
        # pose = ((0.5, 0.5, 3), p.getQuaternionFromEuler((np.pi, 0, 0)))
        pose = self.robot.get_end_effector_pose()  # need code
        self.id = p.loadURDF(ROBOTIQ_BASE_URDF_new, pose[0], pose[1], physicsClientId=self.env.client)
        self.constraint_id = p.createConstraint(parentBodyUniqueId=robot.id,
                           parentLinkIndex=ee,
                           childBodyUniqueId=self.id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=(0, 0, 0),
                           parentFramePosition=(0, 0, 0),
                           childFramePosition=(0, 0, 0.01), physicsClientId=self.env.client)
        self.obj_ids = obj_ids
        self.activated = False
        self.thing = None

        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = p.getNumJoints(self.id, physicsClientId=self.env.client)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()

        # get jointInfo and index of dummy_center_indicator_link
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                #self.controllable_joints.append(jointID)
                p.setJointMotorControl2(self.id, jointID, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity, controllable)
            self.joints[singleInfo.name] = singleInfo
            # register index of dummy center link

        self.gripper_main_control_joint_name = "robotiq_2f_85_right_driver_joint"
        self.mimic_joint_name = ["robotiq_2f_85_right_follower_joint",
                            "robotiq_2f_85_right_spring_link_joint",
                            "robotiq_2f_85_left_driver_joint",
                            "robotiq_2f_85_left_follower_joint",
                            "robotiq_2f_85_left_spring_link_joint"]
        self.mimic_multiplier = [-1, 1, 1, -1, 1]
        self.open_theta = 0.835
        self.close_theta = 0.835
        self.open_gripper()

        self.hold_on_thread = Thread(target=self.thread_hold_on)
        self.hold_on_thread.daemon = True
        self.hold_on_thread.start()

    def thread_hold_on(self):
        while True:
            self.still()
            time.sleep(0.01)

    def remove(self):
        p.removeConstraint(self.constraint_id, physicsClientId=self.env.client)
        p.removeBody(self.id, physicsClientId=self.env.client)

    def still(self):
        # current_joint = p.getJointState(self.id, 1, physicsClientId=self.env.client)[0]
        current_joint = self.target_theta
        p.setJointMotorControlArray(
            self.id,
            [6, 3, 8, 5, 10],
            p.POSITION_CONTROL,
            [current_joint,
             -current_joint,
             -current_joint,
             current_joint,
             current_joint],
            positionGains=np.ones(5))

    def get_position(self):
        # get the position of EE
        position, _ = p.getBasePositionAndOrientation(self.id, physicsClientId=self.env.client)
        self.position = position
        return self.position

    def control_angle(self, angle, threshold):
        # gripper_opening_angle = 0.715 - math.asin((angle - 0.010) / 0.1143)  # angle calculation
        gripper_opening_angle = angle
        current_joint = p.getJointState(self.id, self.joints[self.gripper_main_control_joint_name].id, physicsClientId=self.env.client)[0]
        different = abs(gripper_opening_angle - current_joint)
        if different < threshold:
            self.open_close_flag = False
            self.still()
            self.target_theta = angle

        # Move with constant velocity
        if gripper_opening_angle > current_joint:
            step_joint = current_joint + 0.02
        elif gripper_opening_angle < current_joint:
            step_joint = current_joint - 0.02
        else:
            step_joint = current_joint
        self.target_theta = step_joint

        p.setJointMotorControl2(self.id,
                                self.joints[self.gripper_main_control_joint_name].id,
                                p.POSITION_CONTROL,
                                targetPosition=step_joint,
                                force=self.joints[self.gripper_main_control_joint_name].maxForce,
                                maxVelocity=self.joints[self.gripper_main_control_joint_name].maxVelocity)
        for i in range(len(self.mimic_joint_name)):
            joint = self.joints[self.mimic_joint_name[i]]
            p.setJointMotorControl2(self.id, joint.id, p.POSITION_CONTROL,
                                    targetPosition=step_joint * self.mimic_multiplier[i],
                                    force=joint.maxForce,
                                    maxVelocity=joint.maxVelocity)

    def close_gripper(self):
        if self.open_close_flag == False:
            self.open_close_flag = True
        self.control_angle(self.close_theta, 0.01)

    def open_gripper(self):
        if self.open_close_flag == False:
            self.open_close_flag = True
        self.control_angle(self.open_theta, 0.01)

    def grab_thing(self, thing):
        thing.finished = True
        self.thing = thing
        p.resetBaseVelocity(thing.id, linearVelocity=[0., 0., 0.], angularVelocity=[0., 0., 0.],
                            physicsClientId=self.env.client)
        body_pose = p.getLinkState(self.id, 0, physicsClientId=self.env.client)
        obj_pose = p.getBasePositionAndOrientation(thing.id, physicsClientId=self.env.client)
        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
        self.contact_constraint = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1,
                                                     childBodyUniqueId=thing.id, childLinkIndex=-1,
                                                     jointType=p.JOINT_FIXED, jointAxis=None,
                                                     parentFramePosition=obj_to_body[0],
                                                     childFramePosition=[0., 0., 0.], physicsClientId=self.env.client)
        self.activated = True
        # change the set
        # self.env.available_thing_ids_set.remove(self.thing)
        # self.env.removed_thing_ids_set.add(self.thing)

    def throw_thing(self):

        # when UR place things, this function will be excuted, release the constraint
        reward = 0
        if self.activated:
            self.activated = False

            # rigid object
            if self.contact_constraint is not None:
                p.removeConstraint(self.contact_constraint, physicsClientId=self.env.client)
                self.contact_constraint = None
                reward = 1
                self.thing = None

        return reward

    def release(self):
        if self.activated:
            self.activated = False

class Robotiq2F85():
    def __init__(self, env, robot):
        self.robot = robot
        self.activated = False
        self.object = None
        self.env = env
        pose = self.robot.get_end_effector_pose()  # need code
        self.id = p.loadURDF(ROBOTIQ_BASE_URDF_new, pose[0], pose[1])
        orientation = (0., 0., -np.pi/2) # (0., -np.pi/2, 0.)
        self.contact_constraint = None
        self.constraint = p.createConstraint(parentBodyUniqueId=self.robot.id,
                           parentLinkIndex=7,
                           childBodyUniqueId=self.id,
                           childLinkIndex=0,
                           jointType=p.JOINT_FIXED,
                           jointAxis=(0, 0, 0),
                           parentFramePosition=(0, 0, 0),
                           childFramePosition=(0, 0, 0.01),
                           childFrameOrientation=p.getQuaternionFromEuler(orientation), physicsClientId = self.env.client)
        self.setup()

    def setup(self):
        for i in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, lateralFriction=1.0, spinningFriction=1.0,
                             rollingFriction=0.0001,
                             frictionAnchor=True, physicsClientId = self.env.client)
        self.joints = [p.getJointInfo(self.id, i, physicsClientId = self.env.client) for i in range(p.getNumJoints(self.id))]
        self.joints = [joint_info[0] for joint_info in self.joints if joint_info[2] == p.JOINT_REVOLUTE]
        p.setJointMotorControlArray(self.id, self.joints, p.POSITION_CONTROL, targetPositions=[0.0]*len(self.joints), positionGains=[1.0]*len(self.joints))
        self.open()
        self.hold_on_thread = Thread(target=self.thread_hold_on)
        self.hold_on_thread.daemon = True
        self.hold_on_thread.start()

    def open(self):
        p.setJointMotorControl2(
            self.id,
            1,
            p.VELOCITY_CONTROL,
            targetVelocity=-5,
            force=10000)

    def close(self):
        p.setJointMotorControl2(
            self.id,
            1,
            p.VELOCITY_CONTROL,
            targetVelocity=5,
            force=10000)

    def still(self):
        # hold on the state of ee
        current_joint = p.getJointState(self.id, 1, physicsClientId=self.env.client)[0]
        p.setJointMotorControlArray(
            self.id,
            [6, 3, 8, 5, 10],
            p.POSITION_CONTROL,
            [current_joint,
             -current_joint,
             -current_joint,
             current_joint,
             current_joint],
            positionGains=np.ones(5))

    def thread_hold_on(self):
        while True:
            self.still()
            time.sleep(0.01)

    def get_position(self):
        position, _ = p.getBasePositionAndOrientation(self.id, physicsClientId=self.env.client)
        self.position = position
        return self.position

    def grab_thing(self, thing):
        thing.finished = True
        self.object = thing
        body_pose = p.getLinkState(self.id, 0, physicsClientId=self.env.client)
        obj_pose = p.getBasePositionAndOrientation(self.object.id, physicsClientId=self.env.client)
        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
        self.contact_constraint = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1,
                                                     childBodyUniqueId=self.object.id, childLinkIndex=-1,
                                                     jointType=p.JOINT_FIXED, jointAxis=None,
                                                     parentFramePosition=obj_to_body[0],
                                                     childFramePosition=[0., 0., 0.], physicsClientId=self.env.client)
        self.activated = True

    def throw_thing(self):
        # when UR place things, this function will be excuted, release the constraint
        if self.activated:
            self.activated = False
        if self.contact_constraint is not None:
            p.removeConstraint(self.contact_constraint, physicsClientId=self.env.client)
            self.contact_constraint = None
            self.object = None


    def release(self):
        if self.activated:
            self.activated = False

    # no using ...
    def transform_orientation(self, orientation):
        A = np.quaternion(*orientation)
        B = np.quaternion(*p.getQuaternionFromEuler([0, np.pi / 2, 0]))
        C = B * A
        return quaternion.as_float_array(C)

    def set_pose(self, pose):
        p.resetBasePositionAndOrientation(
            self.id,
            pose[0],
            self.transform_orientation(pose[1]))

    def update_ee_pose(self):
        self.set_pose(self.robot.get_end_effector_pose())


class Suction():
    def __init__(self, env, robot, obj_ids):
        self.env = env   # environment
        self.position = None
        self.robot = robot

        # bind the end_effector with UR
        pose = self.robot.get_end_effector_pose()
        self.id_base = p.loadURDF(SUCTION_BASE_URDF, pose[0], pose[1], physicsClientId=self.env.client)
        orientation = (0., 0., 0.)
        self.robot_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot.id,
            parentLinkIndex=7,
            childBodyUniqueId=self.id_base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01), childFrameOrientation=p.getQuaternionFromEuler(orientation),
            physicsClientId=self.env.client)
        self.id_body = p.loadURDF(SUCTION_HEAD_URDF, pose[0], pose[1], physicsClientId=self.env.client)
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot.id,
            parentLinkIndex=7,
            childBodyUniqueId=self.id_body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08), childFrameOrientation=p.getQuaternionFromEuler(orientation),
            physicsClientId=self.env.client)
        p.changeConstraint(self.constraint_id, maxForce=50, physicsClientId=self.env.client)

        # things that can be grabed in the task, maybe not use?
        self.obj_ids = obj_ids
        self.activated = False

        # get the init position of ee, maybe not use?
        self.init_position = None

        # the constraint between ee and thing
        self.contact_constraint = None

        # the thing EE will grab or has grabed
        self.thing = None

    def remove(self):
        # delete constraint
        p.removeConstraint(self.constraint_id, physicsClientId=self.env.client)
        p.removeConstraint(self.robot_constraint_id, physicsClientId=self.env.client)
        # delete model
        p.removeBody(self.id_body, physicsClientId=self.env.client)
        p.removeBody(self.id_base, physicsClientId=self.env.client)

    def still(self):
        pass



    def get_position(self):
        # get the position of EE
        position, _ = p.getBasePositionAndOrientation(self.id_body, physicsClientId=self.env.client)
        if self.init_position is None:
            self.init_position = position
        self.position = position
        return self.position

    def detect_collision_with_thing(self, thing):
        points = p.getContactPoints(bodyA=self.id_body, linkIndexA=0)
        if points:
            for point in points:
                obj_id, contact_link = point[2], point[4]
            if obj_id == thing.id:
                return True
        return False

    def grab_thing(self, thing):
        # create constraint when UR grab things
        thing.finished = True
        self.thing = thing
        p.resetBaseVelocity(thing.id, linearVelocity=[0., 0., 0.], angularVelocity=[0., 0., 0.],
                            physicsClientId=self.env.client)
        body_pose = p.getLinkState(self.id_body, 0, physicsClientId=self.env.client)
        obj_pose = p.getBasePositionAndOrientation(thing.id, physicsClientId=self.env.client)
        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
        self.contact_constraint = p.createConstraint(parentBodyUniqueId=self.id_body, parentLinkIndex=-1,
                                                     childBodyUniqueId=thing.id, childLinkIndex=-1,
                                                     jointType=p.JOINT_FIXED, jointAxis=None,
                                                     parentFramePosition=obj_to_body[0],
                                                     childFramePosition=[0., 0., 0.], physicsClientId=self.env.client)
        self.activated = True
        # change the set
        # self.env.available_thing_ids_set.remove(self.thing)
        # self.env.removed_thing_ids_set.add(self.thing)

    def throw_thing(self):

        # when UR place things, this function will be excuted, release the constraint
        reward = 0
        if self.activated:
            self.activated = False

            # rigid object
            if self.contact_constraint is not None:
                p.removeConstraint(self.contact_constraint, physicsClientId=self.env.client)
                self.contact_constraint = None
                reward = 1
                self.thing = None

        return reward


    def activate(self):
        if not self.activated:
            points = p.getContactPoints(bodyA=self.id_body, linkIndexA=0, physicsClientId=self.env.client)
            if points:
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                    if obj_id in self.obj_ids:
                        body_pose = p.getLinkState(self.id_body, 0, physicsClientId=self.env.client)
                        obj_pose = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.env.client)
                        world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                        obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
                        self.contact_constraint = p.createConstraint(parentBodyUniqueId=self.id_body,
                            parentLinkIndex=0,
                            childBodyUniqueId=obj_id,
                            childLinkIndex=contact_link,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0, 1), physicsClientId=self.env.client)
                self.activated = True

    def release(self):
        if self.activated:
            self.activated = False

            # rigid object
            if self.contact_constraint is not None:
                p.removeConstraint(self.contact_constraint, physicsClientId=self.env.client)
                self.contact_constraint = None


    def detect_contact(self):
        body, link = self.id_body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint, physicsClientId=self.env.client)
                body, link = info[2], info[3]
            except:
                self.contact_constraint = None
                pass

        points = p.getContactPoints(bodyA=body, linkIndexA=link, physicsClientId=self.env.client)
        if self.activated:
            points = [point for point in points if point[2] != self.id_body]
        if points:
            return True
        return False

    def check_grasp(self):
        # judge whether the EE grab something
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint, physicsClientId=self.env.client)[2]
        return suctioned_object is not None


