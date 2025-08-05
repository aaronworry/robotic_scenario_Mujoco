import matplotlib.pyplot as plt
import datetime
import numpy as np
import timeit
import gurobipy as gp
from gurobipy import GRB
import math


class VelocityOptimizer():
    def __init__(self, dim, n_modules, n_cables):
        self.dim = dim
        self.n_modules = n_modules
        self.n_cables = n_cables
        
        self.delta_theta = np.zeros((self.dim - 1, self.n_modules))
        
        self.model = gp.Model("CR_velocity")
        self.c = self.model.addVars(self.dim - 1, self.n_modules, lb = 0., ub = 0.52, vtype=GRB.CONTINUOUS, name="c")
        self.q_dot = self.model.addVars(self.dim - 1, self.n_modules, lb = -0.52, ub = 0.52, vtype=GRB.CONTINUOUS, name = "qdot")
        self.create_opt()
        
        
    def create_opt(self):
        # cone:  c >= |q_dot|
        """
        for i in range(self.dim - 1):
            for j in range(self.n_modules):
                self.model.addConstr(self.c[i, j] >= self.q_dot[i, j], name = "cone_pos_"+str(i)+str(j))
                self.model.addConstr(self.c[i, j] >= -1 * self.q_dot[i, j], name = "cone_neg_"+str(i)+str(j))
        """
        for i in range(self.dim - 1):
            for j in range(self.n_modules):
                self.model.addGenConstrAbs(self.c[i, j], self.q_dot[i, j], name = "cone_pos_"+str(i)+str(j))
        
        mask = np.zeros((self.n_cables, self.n_modules))
        J = np.zeros((self.n_cables, self.n_modules))
        cable_velocity = np.array([0.] * self.n_cables)
        # l_dot = J q_dot
        for i in range(self.n_cables):
            self.model.addConstr(gp.quicksum(mask[i, j] * J[i, j] * self.q_dot[i, j] for j in range(self.n_modules)) == cable_velocity[i], name = "velocity_constr_"+str(i))

        obj = gp.quicksum(self.c[i, j] for i, j in np.ndindex(self.delta_theta.shape))
        self.model.setObjective(obj, GRB.MINIMIZE)
    
    def update_opt(self, J, mask, cable_velocity):
        for i in range(self.n_cables):
            self.model.remove(self.model.getConstrByName("velocity_constr_"+str(i)))
        self.model.update()
        for i in range(self.n_cables):
            self.model.addConstr(gp.quicksum(mask[i, j] * J[i, j] * self.q_dot[i, j] for j in range(self.n_modules)) == cable_velocity[i], name = "velocity_constr_"+str(i))
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