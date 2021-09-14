import numpy as np
from DataParameters import C1,C2,C3


class ContextualEnvironment:
    def __init__(self,n_arms):
        self.n_arms = n_arms
        self.mapping = dict ({
            '[0, 0]': C1['conversion rate'],
            '[0, 1]': C2['conversion rate'],
            '[1, 0]': C3['conversion rate'],
            '[1, 1]': C3['conversion rate']
        })

    def round(self, feature, pulled_arm):
        reward = np.random.binomial(1,self.mapping[str(feature)][pulled_arm])
        return reward




