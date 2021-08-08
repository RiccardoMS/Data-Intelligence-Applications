import numpy as np
from Learner import *

class UCB1_learner(Learner):
    def __init__(self, n_arms, samples_per_round, rev):
        super().__init__(n_arms, samples_per_round, rev)

    def pull_arm(self):
        if self.t <self.n_arms:
            return self.t
        else:
            #print(self.rewards_per_arm)
            avg = ([np.average(np.array(it)) for it in self.rewards_per_arm])
            upper = np.sqrt(2*np.log(self.t+1)/np.array([len(x)+1 for x in self.rewards_per_arm]))
            idx = np.zeros(10)
            for i in range(10):
                idx[i] = self.rev((avg[i] + upper[i]),i)
        return np.argmax(idx)

    def update(self, pulled_arm, reward):
        self.t+=1
        self.update_observations(pulled_arm, reward)



