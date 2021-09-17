import numpy as np
from ContextualLearner import *
from DataParameters import prices

class Contextual_UCB1_learner(ContextualLearner):
    def __init__(self, n_arms,context, rev):
        super().__init__(n_arms,context, rev)
        self.arms_parameters = [np.zeros([n_arms, 2]) for _ in range(self.context.structure)]

    def updateContext(self, context):
        ret = False
        if self.context.getRuleName() != context.getRuleName():
            ret = True
            self.context = context
        return ret

    def offlineTraining(self, database):
        self.arms_parameters = [np.zeros([self.n_arms, 2]) for _ in range(self.context.structure)]
        for day in range(0, len(database.Database)):
            for customer in database.Database[day]:
                self.arms_parameters[self.context.evaluate(customer.feature)][prices.index(database.DailySpecifics[day][0][database.contextHistory[day//7].evaluate(customer.feature)]), 0] += customer.accepted
                self.arms_parameters[self.context.evaluate(customer.feature)][prices.index(database.DailySpecifics[day][0][database.contextHistory[day//7].evaluate(customer.feature)]), 1] += 1

    def pull_arm(self):
        if self.t <self.n_arms:
            return [prices[self.t] for _ in range(self.context.structure)]
        else:
            ret = []
            for c in range(self.context.structure):
                idx = np.zeros(10)
                for i, x in enumerate(self.arms_parameters[c]):
                    idx[i] = self.rev(x[0]/x[1] +  np.sqrt(2*np.log(self.t)/(x[1])), i,self.mean_returns_estimates[c])
                ret.append(prices[np.argmax(idx)])
            return ret

    def update(self, pulled_arms, rewards,n_cust):
        self.t+=1
        self.update_observations(pulled_arms, rewards,n_cust)
        for c in range(self.context.structure):
            self.arms_parameters[c][pulled_arms[c], 0] += rewards[c]
            self.arms_parameters[c][pulled_arms[c], 1] += n_cust[c]