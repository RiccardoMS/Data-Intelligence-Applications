import numpy as np

class Learner:

    def __init__(self, n_arms, rev):
        self.n_arms = n_arms
        self.t = 0
        self.rev = rev
        self.collected_revenues = np.array([])
        self.collected_arms = x = [[] for i in range(n_arms)]

    def update_observations(self, pulled_arm, reward, n_cust):
        self.collected_revenues = np.append(self.collected_revenues, self.rev((reward/n_cust),pulled_arm))
        self.collected_arms[pulled_arm].append(1)
        for x in range(self.n_arms):
            if x!= pulled_arm:
               self.collected_arms[x].append(0)



