import numpy as np
from DataParameters import C1,C2,C3,weights, bids
from DataGenerativeFunctions import Ndc_a,Cpc_a

class JointEnvironment:
    def __init__(self,n_prices,n_bids):
        self.n_prices = n_prices
        self.n_bids = n_bids
        self.bids = bids
        self.mapping = dict ({
            '[0, 0]': C1['conversion rate'],
            '[0, 1]': C2['conversion rate'],
            '[1, 0]': C3['conversion rate'],
            '[1, 1]': C3['conversion rate']
        })

    def sampledAcceptance(self, feature, pulled_arm):
        reward = np.random.binomial(1,self.mapping[str(feature)][pulled_arm])
        return reward

    def BidRewards(self, pulled_arm):
        rew_ndc = Ndc_a([bids[pulled_arm]]).item()
        rew_cpc = Cpc_a([bids[pulled_arm]]).item()
        return rew_ndc, rew_cpc