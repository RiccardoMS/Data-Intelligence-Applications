import numpy as np


class Environment():
    def __init__(self,n_arms, probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm,n_samples):
        reward = np.random.binomial(n_samples, self.probabilities[pulled_arm])
        return reward

