import datetime
import numpy as np
import timeit
import gurobipy as gp
from gurobipy import GRB
import math


class StateOptimizer():
    def __init__(self, dim, n_modules, n_cables, q_current, q_s_range, q_range, dt, current_kinetic_energy = 0., current_potential_energy = 0.):
        # q_current  should be a list
        self.dim = dim
        self.n_modules = n_modules
        self.n_cables = n_cables
        self.dt = dt
        self.q_current = q_current
        self.current_kinetic_energy = current_kinetic_energy
        self.current_potential_energy = current_potential_energy
        self.q_s_range = q_s_range
        self.q_range = q_range
        
        self.delta_theta = np.zeros((self.dim - 1, self.n_modules))
        
        self.model = gp.Model("CR_velocity")
        self.c = self.model.addVars(self.dim - 1, self.n_modules, lb = 0., ub = 5., vtype=GRB.CONTINUOUS, name="c")
        self.q_dot = self.model.addVars(self.dim - 1, self.n_modules, lb = -5., ub = 5., vtype=GRB.CONTINUOUS, name = "qdot")
        self.q = self.model.addVars(self.dim - 1, self.n_modules, lb = q_range[0], ub = q_range[1], vtype=GRB.CONTINUOUS, name = "q")
        self.dl = self.model.addVars(self.n_cables, lb = -5., ub = 5. vtype=GRB.CONTINUOUS, name = "delta_cable_length")
        self.Ke = self.model.addVar(lb = 0., ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Ke")
        self.Kp = self.model.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Kp")
        self.friction_loss = self.model.addVar(lb = 0., ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="friction_loss")
        self.cable_input_work = self.model.addVar(lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype=GRB.CONTINUOUS, name="work")
        self.create_opt()
        
        
    def create_opt(self):
        # c >= |q_dot|
        for i in range(self.dim - 1):
            for j in range(self.n_modules):
                self.model.addConstr(self.c[i, j] >= self.q_dot[i, j], name = "cone_pos_"+str(i)+str(j))
                self.model.addConstr(self.c[i, j] >= -1 * self.q_dot[i, j], name = "cone_neg_"+str(i)+str(j))
        
        # q_dot dt + q_current = q
        for i in range(self.dim - 1):
            for j in range(self.n_modules):
                self.model.addConstr(self.q[i, j] == self.q_current[i, j] + self.q_dot[i, j] * self.dt, name = "joint_velocity_"+str(i)+str(j))
                
        # first q_s 
        for i in range(self.dim - 1):
            self.model.addConstr(self.q[i, 0] <= self.q_s_range[1], name = "first_q_max")
            self.model.addConstr(self.q[i, 0] >= self.q_s_range[0], name = "first_q_min")
            
        # kinetic_energy
        self.model.addConstr(self.Ke == 0., name = "kinetic_energy")
        # potential_energy
        self.model.addConstr(self.Kp == 0., name = "potential_energy")
        
        """
        以下约束需要更新后的参数，update_opt
        """
        # energy
        self.model.addConstr(self.cable_input_work - self.friction_loss == self.Ke + self.Kp - self.current_kinetic_energy - self.current_potential_energy, name = "energy_balance")
        # work
        self.model.addConstr(gp.quicksum(0.*self.dl[j] for j in range(self.cables)) == self.cable_input_work, name = "cable_work")
        # friction_loss      # q_current q  q_dot
        self.model.addConstr(self.friction_loss == 0., name = "friction_loss")
        

        J = np.zeros((self.n_cables, self.n_modules))
        # delta l = J q_dot dt
        for i in range(self.n_cables):
            self.model.addConstr(gp.quicksum(J[i, j] * self.q_dot[i, j] * self.dt for j in range(self.n_modules)) == self.dl[i], name = "cable_delta_length_"+str(i))

        # 最小作用量
        obj = gp.quicksum(self.c[i, j] for i, j in np.ndindex(self.delta_theta.shape))
        self.model.setObjective(obj, GRB.MINIMIZE)
    
    def update_opt(self, J, cable_force, friction_matrix, current_kinetic_energy, current_potential_energy):
        for i in range(self.n_cables):
            self.model.remove(self.model.getConstrByName("cable_delta_length_"+str(i)))
        self.model.remove(self.model.getConstrByName("energy_balance"))
        self.model.remove(self.model.getConstrByName("cable_work"))
        self.model.remove(self.model.getConstrByName("friction_loss"))
        self.model.update()
        for i in range(self.n_cables):
            self.model.addConstr(gp.quicksum(J[i, j] * self.q_dot[i, j] * self.dt for j in range(self.n_modules)) == self.dl[i], name = "cable_delta_length_"+str(i))
        self.model.update()
        
        
    def solve(self):
        starttime = timeit.default_timer()
        self.model.optimize()
        t_diff = timeit.default_timer() - starttime

        print("*********************************************************")
        print("Time to solve (ms)=",t_diff*1000)
        print("*********************************************************")
        if self.model.status == GRB.Status.OPTIMAL:
            print('Optimal Solution found')
            for key, value in self.q_dot.items():
                self.delta_theta[key] = self.q_dot[key].x
            return True
        elif self.model.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            return False
        elif self.model.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
            return False
        elif self.model.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
            return False
        else:
            print('Optimization ended with status %d' % self.model.status)
            return False





if __name__ == "__main__":
    opt = VelocityOptimizer(2, 5, 1)
    if(opt.solve()):
        print(opt.delta_theta)
    print("========")
    J = np.random.rand(1, 5) - 0.5
    print(J)
    opt.update_opt(J, mask = np.ones((1, 5)), cable_velocity = np.array([0.03]))
    if(opt.solve()):
        print(opt.delta_theta)