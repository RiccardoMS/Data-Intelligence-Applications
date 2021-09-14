import numpy as np
from Learner import *


class TS_learner(Learner):
    def __init__(self, n_arms, rev):
        super().__init__(n_arms, rev)
        self.beta_parameters = np.ones([n_arms, 2])

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        else:
            #print(self.rewards_per_arm)
            idx = np.zeros(10)
            for i, x in enumerate(self.beta_parameters):
               idx[i] = self.rev(np.random.beta(x[0], x[1]),i)
        return np.argmax(idx)

    def update(self, pulled_arm, reward, n_cust):
        self.t += 1
        self.update_observations(pulled_arm, reward, n_cust)
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += n_cust - reward