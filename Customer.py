import numpy as np
from Data_Exp import RETURNS

class Customer( ID ):
    def __init__(self, ID, price_accepted, feature):
        self.mapping = dict({'[0,0]':'C1','[0,1]':'C2','[1,0]':'C3','[1,1]':'C3'})
        self.ID = ID                                         # unique number identifying customer
        self.price_accepted = price_accepted                 # price accpetedd at first purchase
        self. monthly_returns = np.zeros(30)                 # simulated returns within 30 days
        self.monthly_returns[0]=1
        times = RETURNS[self.mapping[str(self.feature)]](1)
        if not times == 0:
            idx = 30 // (times + 1)
            for i in range(1, times + 1):
                self.monthly_returns[idx * i] = 1
        self.timer = 0                                       # keep track of customer history
        self.feature = feature                               # format: [0,0]
        self.currentReturns = 0

    def check(self):
        if self.monthly_returns[(self.timer % 30)]:
            self.currentReturns += 1
        return self.monthly_returns[(self.timer % 30)]              # check if today customer replenish

    def ID(self):
        return self.ID

    def timer(self):
        return self.timer

    def feature(self):
        return self.feature

    def reward(self):
        return self.check()*self.price_accepted

    def currentReturns(self):
        return self.currentReturns

    def update(self):
        self.timer += 1
        if self.timer % 30 == 0:
           self.currentReturns = 0
           self.monthly_returns = np.zeros(30)
           self.monthly_returns[0] = 1
           times = RETURNS[self.mapping[str(self.feature)]](1)
           if not times == 0:
               idx = 30//(times+1)
               for i in range(1,times+1):
                   self.monthly_returns[idx*i]=1
