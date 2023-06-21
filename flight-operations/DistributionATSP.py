from pyomo.environ import *
import numpy as np
#import cplex
import sys

class Distribution_ATSP(object):
    def __init__(self, UB, LB, N, tt):
        self.UB = UB   #arrive time upper bound
        self.LB = LB   #arrive time lower bound
        self.N = N     #distribution 0,...,N
        self.tt = tt   #travel time
        # self.mm = mm   #mean time

    def set_model(self):
        model = ConcreteModel()
        #arrive time, [1,...,N+1]
        model.t = Var(RangeSet(1, self.N+2), initialize=0, within=NonNegativeReals, bounds=(0, 100000))
        #1 = yes from i to j, 0 = no; [1,...,N+1][1,...,N+1]
        model.y = Var(RangeSet(1, self.N+1), RangeSet(1, self.N+1), initialize=0, within=Binary)

        ########################### obj min t_N ###############################################
        def obj_rule(m):
            #return m.t[self.N+2] + sum(0.2*(abs(m.t[i+1] - self.mm[i])) for i in range(self.N+1))
            return m.t[self.N + 2] #+ sum(0.2 * ((m.t[i + 1] - self.mm[i])**2) for i in range(self.N + 1))

        model.obj = Objective(rule=obj_rule, sense=minimize)

        ############################ const 2 ################################################
        def time_traveltime(m, ii):  # ii: 2,...,N+1;
            return m.t[ii] >= self.tt[0][ii-1] * m.y[1, ii]

        model.timeTraveltime = Constraint(RangeSet(2, self.N+1), rule=time_traveltime)

        ############################ const 4 ################################################
        def y_j(m, jj):  # jj: 2,...,N+1;
            return sum(m.y[i, jj] for i in range(1, self.N + 2)) == 1

        model.yj = Constraint(RangeSet(1, self.N + 1), rule=y_j)

        ############################ const 5 ################################################
        def y_i(m, ii):  # ii: 2,...,N+1;
            return sum(m.y[ii, j] for j in range(1, self.N + 2)) == 1

        model.yi = Constraint(RangeSet(1, self.N + 1), rule=y_i)

        ############################ const 6 ################################################
        def t_i1(m, ii):  # ii: 2,...,N+1;
            return m.t[ii] + self.tt[ii-1][0] <= m.t[self.N+2]

        model.ti1 = Constraint(RangeSet(2, self.N + 1), rule=t_i1)

        ############################ const 7 ################################################
        def t_i2(m, ii):  # ii: 2,...,N+1;
            return m.t[ii] <= self.UB[ii-1]

        model.ti2 = Constraint(RangeSet(2, self.N + 1), rule=t_i2)

        ############################ const 8 ################################################
        def t_i3(m, ii):  # jj: 2,...,N+1;
            return m.t[ii] >= self.LB[ii - 1]

        model.ti3 = Constraint(RangeSet(2, self.N + 1), rule=t_i3)


        ########################### const 9 ################################################
        def y_i2(m, ii, jj):  # ii: 2,...,N+1;
            if ii == jj:
                return m.y[ii, jj] == 0
            if ii != jj:
                return m.y[ii, jj] >= -1
        model.yi2 = Constraint(RangeSet(1, self.N + 1), RangeSet(1, self.N + 1), rule=y_i2)


        ############################ const 3 ################################################
        def time_y(m, ii, jj):  # ii: 2,...,N+1; jj: 2,...,N+1;
            if ii != jj:
                # if m.y[ii, jj] == 1:
                #     return m.t[ii] - m.t[jj] + self.tt[ii - 1][jj - 1] <= 0
                # if m.y[ii, jj] == 0:
                #     return m.t[ii] - m.t[jj] <= self.UB[ii-1] - self.LB[jj-1]

                #return m.t[ii] - m.t[jj] + (self.UB[ii-1] - self.LB[jj-1] + self.tt[ii -1][jj - 1]) * m.y[ii, jj] <= (self.UB[ii-1] - self.LB[jj-1])
                return m.t[ii] - m.t[jj] + self.tt[ii - 1][jj - 1] <= (self.UB[ii - 1] - self.LB[jj - 1] + self.tt[ii - 1][jj - 1]) * (1-m.y[
                    ii, jj])
            if ii == jj:
                return m.t[ii] >= 0

        model.timey = Constraint(RangeSet(2, self.N + 1), RangeSet(2, self.N + 1), rule=time_y)



        return model




