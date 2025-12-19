import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import os

class Arm_ik():
    def __init__(self, ee_frame) -> None:
        self.frame_name = ee_frame

    def buildFromMJCF(self, mcjf_file):
        self.arm = pin.RobotWrapper.BuildFromMJCF(mcjf_file)
        self.createSolver()

    def buildFromURDF(self, urdf_file):
        self.arm = pin.RobotWrapper.BuildFromURDF(urdf_file, package_dirs=[os.path.dirname(urdf_file)])
        self.createSolver()

    def createSolver(self):
        self.model = self.arm.model
        self.data = self.arm.data

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.ee_id = self.model.getFrameId(self.frame_name)

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.ee_id].translation - self.cTf[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.ee_id].rotation @ self.cTf[:3,:3].T)
                )
            ],
        )

        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ee_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # 误差的符号化计算
        error_vector = self.error(self.var_q, self.param_tf)
        pos_error = error_vector[:3]  # 位置误差
        ori_error = error_vector[3:]  # 姿态误差
        # 设置位置和姿态的权重
        weight_position = 1.0  # 位置权重
        weight_orientation = 0.1  # 姿态权重

        # 误差的cost
        self.error_cost = weight_position * casadi.sumsqr(pos_error) + weight_orientation * casadi.sumsqr(ori_error)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
        # self.opti.minimize(200.0 * self.translational_cost + 200.*self.rotation_cost + 0.01 * self.regularization_cost + 0. * self.smooth_cost)
        self.opti.minimize(200.0 * self.error_cost + 0.01 * self.regularization_cost + 1. * self.smooth_cost)

        ##### IPOPT #####
        opts = {
            'ipopt':{
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4,
                'sb': 'yes'
                # 'hessian_approximation':"limited-memory"
            },
            'print_time':False  # print or not
            #'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq)

    def ik(self, T , current_arm_motor_q = None, current_arm_motor_dq = None):
        if current_arm_motor_q is not None:
            self.init_data = current_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf, T)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            # self.smooth_filter.add_data(sol_q)
            # sol_q = self.smooth_filter.filtered_data

            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))
            sol_tauff = np.concatenate([sol_tauff, np.zeros(self.model.nq - sol_tauff.shape[0])], axis=0)

            info = {"sol_tauff": sol_tauff, "success": True}

            dof = np.zeros(self.model.nq)
            dof[:len(sol_q)] = sol_q
            return dof

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            # self.smooth_filter.add_data(sol_q)
            # sol_q = self.smooth_filter.filtered_data

            if current_arm_motor_dq is not None:
                v = current_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))
            import ipdb; ipdb.set_trace()
            sol_tauff = np.concatenate([sol_tauff, np.zeros(self.model.nq - sol_tauff.shape[0])], axis=0)

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_arm_motor_q} \nright_pose: \n{T}")

            info = {"sol_tauff": sol_tauff * 0.0, "success": False}

            dof = np.zeros(self.model.nq)
            # dof[:len(sol_q)] = current_arm_motor_q
            dof[:len(sol_q)] = self.init_data

            raise e


if __name__ == "__main__":

    arm = Arm_ik("joint6")
    arm.buildFromURDF("../../assets/piper_description/piper_description.urdf")
    theta = np.pi
    tf = np.array([
            [1, 0, 0, 0.2],
            [0, np.cos(theta), -np.sin(theta), 0.0],
            [0, np.sin(theta), np.cos(theta), 0.3],
        ])
    tf = np.vstack((tf, [0, 0, 0, 1]))
    dof = arm.ik(tf)
    print(f"DoF: {dof}")

