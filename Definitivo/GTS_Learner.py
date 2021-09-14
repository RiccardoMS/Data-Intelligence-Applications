import numpy as np
from BiddingLearner import *
from scipy.stats import norm


class GTS_Learner(BiddingLearner):
    def __init__(self, n_arms, rev):
        super().__init__(n_arms, rev)
        self.means_rev = np.zeros(n_arms)
        self.sigmas_rev = np.ones(n_arms)*50
        self.revenues_per_arm = [[] for i in range(n_arms)]
        self.treshold = 0.2

    def eligible_arms(self):
        ret = []
        for i in range(0,self.n_arms):
            if norm.cdf(0,loc=self.means_rev[i],scale=self.sigmas_rev[i]) < self.treshold:
                    ret.append(i)
        return ret

    def pull_arm(self):
        if self.t<10:
            return self.t
        else:
            candidates = self.eligible_arms()
            vals = []
            for i in candidates:
                vals.append(np.random.normal(self.means_rev[i], self.sigmas_rev[i]))
            return candidates[np.argmax(vals)]

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.revenues_per_arm[pulled_arm].append(self.rev(reward[0], reward[1]))
        self.means_rev[pulled_arm] = np.mean(self.revenues_per_arm[pulled_arm])
        n_samples = len(self.revenues_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas_rev[pulled_arm] = np.std(self.revenues_per_arm[pulled_arm])/ n_samples
