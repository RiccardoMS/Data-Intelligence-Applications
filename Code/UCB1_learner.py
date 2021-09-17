import numpy as np
from Learner import *

class UCB1_learner(Learner):
    def __init__(self, n_arms, rev):
        super().__init__(n_arms, rev)
        self.arms_parameters = np.zeros([n_arms, 2])

    def pull_arm(self):
        if self.t <self.n_arms:
            return self.t
        else:
            idx = np.zeros(10)
            for i, x in enumerate(self.arms_parameters):
                idx[i] = self.rev(x[0]/x[1] + np.sqrt(2*np.log(self.t)/(x[1])), i)
        return np.argmax(idx)

    def update(self, pulled_arm, reward, n_cust):
        self.t+=1
        self.update_observations(pulled_arm, reward, n_cust)
        self.arms_parameters[pulled_arm, 0] += reward
        self.arms_parameters[pulled_arm, 1] += n_cust



