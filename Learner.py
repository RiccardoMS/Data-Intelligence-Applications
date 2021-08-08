import numpy as np



class Learner:

    def __init__(self, n_arms,samples_per_round, rev):
        self.n_arms = n_arms
        self.t = 0
        self.samples_per_round = samples_per_round
        self.rev = rev
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])
        self.collected_revenues = np.array([])
        self.collected_arms = x = [[] for i in range(n_arms)]

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward/self.samples_per_round)
        self.collected_rewards = np.append(self.collected_rewards, reward/self.samples_per_round)
        self.collected_revenues = np.append(self.collected_revenues, self.rev((reward/self.samples_per_round),pulled_arm))
        self.collected_arms[pulled_arm].append(1)
        for x in range(self.n_arms):
            if x!= pulled_arm:
               self.collected_arms[x].append(0)



