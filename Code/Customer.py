import numpy as np
from DataGenerativeFunctions import Mr
from DataGenerativeFunctions import RETURNS

class Customer:

    # Initialization
    def __init__(self, ID, price_proposed, feature, accepted):
        self.mapping = dict({'[0, 0]':'C1','[0, 1]':'C2','[1, 0]':'C3','[1, 1]':'C3'})
        self.ID = ID                                         # unique number identifying customer
        self.price_proposed = price_proposed                 # price accpeted at first purchase
        self.timer = 0                                       # keep track of customer history
        self.feature = feature                               # format: [0,0]
        self.currentReturns = 0
        self.accepted = accepted
        self.available = accepted
        self.monthly_returns = np.zeros(30)

        # Simulate Returns within a month if customer accepted the deal/a month is not passed since first purchase
        if self.available:
          self.monthly_returns[0]=1
          times = RETURNS[self.mapping[str(self.feature)]]()
          if not times == 0:
            idx = 30 // (times + 1)
            for i in range(1, times + 1):
                self.monthly_returns[idx * i] = 1


    # Methods
    def dailyReward(self):
        if self.available:
           return self.monthly_returns[self.timer]*Mr(self.price_proposed)
        else:
           return 0.0

    def reward(self):
        return self.accepted*Mr(self.price_proposed)

    def update(self):
        self.timer += 1
        if self.timer >= 30:
           self.available = False
        else:
            if self.monthly_returns[self.timer]:
               self.currentReturns += 1






