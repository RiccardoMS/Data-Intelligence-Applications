import numpy as np
from DataGenerativeFunctions import Ndc_a, Cpc_a
from DataParameters import bids


class BiddingEnvironment():
    def __init__(self, bids):
        self.bids = bids

    def round(self, pulled_arm):
        rew_ndc = Ndc_a([bids[pulled_arm]]).item()
        rew_cpc = Cpc_a([bids[pulled_arm]]).item()
        return [rew_ndc, rew_cpc]
