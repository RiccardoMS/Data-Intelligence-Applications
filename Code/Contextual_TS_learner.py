import numpy as np
from ContextualLearner import *
from DataParameters import prices


class Contextual_TS_learner(ContextualLearner):
    def __init__(self, n_arms,context, rev):
        super().__init__(n_arms,context, rev)
        self.beta_parameters = [np.ones([n_arms, 2]) for _ in range(self.context.structure)]

    def updateContext(self, context):
        ret = False
        if self.context.getRuleName() != context.getRuleName():
           ret = True
           self.context = context
        return ret

    def offlineTraining(self, database):
        self.beta_parameters = [np.ones([self.n_arms, 2]) for _ in range(self.context.structure)]
        for day in range(0, len(database.Database)):
            for customer in database.Database[day]:
                self.beta_parameters[self.context.evaluate(customer.feature)][prices.index(database.DailySpecifics[day][0][database.contextHistory[day // 7].evaluate(customer.feature)]), 0] += customer.accepted
                self.beta_parameters[self.context.evaluate(customer.feature)][prices.index(database.DailySpecifics[day][0][database.contextHistory[day // 7].evaluate(customer.feature)]), 1] += 1.0 - customer.accepted

    def pull_arm(self):
        if self.t < self.n_arms:
            return [prices[self.t] for _ in range(self.context.structure)]
        else:
            ret = []
            for c in range(self.context.structure):
                idx = np.zeros(10)
                for i, x in enumerate(self.beta_parameters[c]):
                    idx[i] = self.rev(np.random.beta(x[0], x[1]),i,self.mean_returns_estimates[c])
                ret.append(prices[np.argmax(idx)])
        return ret

    def update(self, pulled_arms, rewards, n_cust):
        self.t += 1
        self.update_observations(pulled_arms, rewards, n_cust)
        for c in range(0,self.context.structure):
            self.beta_parameters[c][pulled_arms[c], 0] += rewards[c]
            self.beta_parameters[c][pulled_arms[c], 1] += n_cust[c] - rewards[c]
