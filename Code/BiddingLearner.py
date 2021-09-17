import numpy as np


class BiddingLearner:
    def __init__(self, n_arms, rev):
        self.n_arms = n_arms
        self.t = 0
        self.rev = rev

        self.collected_rewards_NDC = np.array([])
        self.collected_rewards_CPC = np.array([])
        self.collected_revenues = np.array([])
        self.collected_arms =[[] for i in range(n_arms)]

    def update_observations(self, pulled_arm, reward):
        self.collected_rewards_NDC = np.append(self.collected_rewards_NDC,reward[0])
        self.collected_rewards_CPC = np.append(self.collected_rewards_CPC,reward[1])
        self.collected_revenues    = np.append(self.collected_revenues, self.rev(reward[0], reward[1]))
        self.collected_arms[pulled_arm].append(1)
        for x in range(self.n_arms):
            if x != pulled_arm:
                self.collected_arms[x].append(0)
