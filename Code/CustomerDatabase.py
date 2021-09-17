# Import Packages
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# Import Feature, Context and Customer tools
from FeatureGenerator import *
from Customer import *
from Context import *
from ContextGenerators import *
# Import Functions
from DataGenerativeFunctions import Cpc_a, Ndc_a, prices, bids

#parameters
delta = 0.05

# Database definition
class CustomerDatabase:
    def __init__(self,context = Context(Naive)):
        self.Database = []
        self.DailySpecifics = []
        self.currentDay = 0
        self.currentContext = context
        self.maxReturns = 0
        self.currentID = 0
        self.contextHistory = [context]

    # Methods to manage Daily Routine
    def addDay(self, priceProposed, bidProposed,CostPerClickObtained):
        self.Database.append([])
        self.DailySpecifics.append([priceProposed,bidProposed, CostPerClickObtained])


    def addCustomer(self, feature, accepted):
        self.Database[self.currentDay].append(Customer(self.currentID,self.DailySpecifics[self.currentDay][0][self.currentContext.evaluate(feature)],feature,accepted))
        self.currentID += 1

    def dailyReward(self):
        today_reward = 0.0
        today_returns = 0.0
        for day in range(0,self.currentDay + 1):
            for customer in self.Database[day]:
                 if customer.timer==0:
                    today_reward += customer.dailyReward() - self.DailySpecifics[day][2]
                 else:
                    today_returns += customer.dailyReward()
        return [float(today_reward+today_returns), float(today_reward), float(today_returns)]

    def dailyUpdate(self):
        for day in range(0,self.currentDay + 1):
            for customer in self.Database[day]:
                customer.update()
                self.maxReturns = max(self.maxReturns, customer.currentReturns)
        self.currentDay += 1

    # Methods for Learning
    def getMeanReturnsEstimate(self, currContext ):
        est = [[0 for _ in range(self.maxReturns + 2)] for _ in range(currContext.structure)]
        counts = [0]*currContext.structure
        means = [0.0]*currContext.structure
        for day in self.Database:
            for customer in day:
                if customer.accepted :
                   counts[currContext.evaluate(customer.feature)] += 1
                   est[currContext.evaluate(customer.feature)][customer.currentReturns] += 1
        for i in range(0,len(est)):
            if not counts[i]==0:
               for j in range(0,len(est[i])):
                     means[i] += float(j)*(est[i][j]/counts[i])
        return means

    def updateCurrentContext(self, context):
        self.currentContext = context
        self.contextHistory.append(self.currentContext)

    def printContextHistory(self):
        ret = []
        for v in self.contextHistory:
            ret.append(v.getRuleName())
        print(ret)

    # Methods for Context Generation
    def retrieveSplitValue(self, currContext):
        # Initialization
        ret = 0.0
        partials = [[[] for _ in range(len(prices))] for _ in range(currContext.structure)]
        values = [0.0] * currContext.structure
        probabilities = [0.0]*currContext.structure
        total = 0.0

        for day in range(0,len(self.Database)):
           for p in self.DailySpecifics[day][0]:
               for i in range(currContext.structure):
                   partials[i][prices.index(p)].append(0.0)
           for customer in self.Database[day]:
               probabilities[currContext.evaluate(customer.feature)] += 1
               total += 1
               j = prices.index(self.DailySpecifics[day][0][self.contextHistory[day // 7].evaluate(customer.feature)])
               partials[currContext.evaluate(customer.feature)][j][-1] += float(customer.reward() - self.DailySpecifics[day][2])
        for i in range(0,currContext.structure):
            s = [0.0] * len(partials[i])
            for j in range(0, len(s)):
                if not partials[i][j]==[]:
                   s[j] = np.mean(partials[i][j])
            ind = np.argmax(s)
            m = s[ind]*(1.0 + float(self.getMeanReturnsEstimate(currContext)[i]))
            n = float(len(partials[i][ind]))
            s = np.std(partials[i][ind])
            values[i] = m - stats.norm.ppf(1-delta)*np.sqrt(s/n)
        for i in range(0, len(partials)):
            ret += values[i]
        return ret

########################################################################################################################
if __name__ == "__main__":
  if True:
    # Feature
    features = FeatureGenerator(10)
    print(features)

    # Rules
    r1 = Naive
    r2 = F2
    print(r1(features[0]))
    print(r2(features[0]))

    # Context
    c1 = Context(r1)
    c2 = Context(r2)
    print([c1.structure, c1.evaluate(features[0]), c1.getRule() == Naive])
    print([c2.structure, c2.evaluate(features[0]), c2.getRule() == F1])
    print([Context(Disagg).structure, Context(Disagg).evaluate(features[0])])

    # Customer
    cT = Customer(0, 22.5, features[0], True)
    cF = Customer(1, 22.5, features[0], False)
    print([cT.ID, cT.timer, cT.currentReturns, cT.monthly_returns])
    print([cF.ID, cF.timer, cF.currentReturns, cF.monthly_returns])
    print([cT.dailyReward(), cT.reward()])
    print([cF.dailyReward(), cF.reward()])
    cT.update()
    cF.update()
    print([cT.dailyReward(), cT.reward()])
    print([cF.dailyReward(), cF.reward()])
    for i in range(0, 30):
        cT.update()
        print([cT.dailyReward(), cT.reward()])
    print(cT.available)

    # Customer Database
    D = CustomerDatabase()
    D.addDay(prices[0], bids[0], Cpc_a(bids[0]))
    D.addCustomer([0, 0], True)
    D.addCustomer([0, 0], True)
    D.addCustomer([0, 1], True)
    D.addCustomer([0, 1], True)
    D.addCustomer([0, 1], True)
    D.addCustomer([1, 1], True)
    D.addCustomer([0, 0], True)
    D.addCustomer([0, 1], True)
    D.addCustomer([1, 1], True)
    D.addCustomer([0, 1], True)
    print(D.dailyReward())
    D.dailyUpdate()
    print(D.getMeanReturnsEstimate(D.currentContext))
    print(D.retrieveSplitValue(D.currentContext))
    for el in D.Database:
        for e in el:
            print(e.ID)

    v = D.retrieveSplitValue(Context(Naive))
    print(v)
    v = D.retrieveSplitValue(Context(F1))
    print(v)
    v = D.retrieveSplitValue(Context(F2))
    print(v)
    v = D.retrieveSplitValue(Context(C11))
    print(v)
    v = D.retrieveSplitValue(Context(C12))
    print(v)
    v = D.retrieveSplitValue(Context(C21))
    print(v)
    v = D.retrieveSplitValue(Context(C22))
    print(v)
    v = D.retrieveSplitValue(Context(Disagg))
    print(v)

    # Context Generators
    CB=BruteForceContexGenerator(D)
    CG=GreedyContextGenerator(D)
    print(CB.getRuleName())
    print(CG.getRuleName())

    del D

  if True:
    # Final Check: Simulate a month
    D = CustomerDatabase()
    dailyRewards = []
    dailyRevenue = []
    dailyReturns = []
    ChosenContextsB = ['Naive']
    ChosenContextsG = ['Naive']
    for day in range(0,30):                     # Simulate day
        if day%3==0 and day!=0:
            ChosenContextsB.append(BruteForceContexGenerator(D).getRuleName())
            ChosenContextsG.append(GreedyContextGenerator(D).getRuleName())
        p = np.random.choice(prices,1)
        b = np.random.choice(bids,1)
        c = Cpc_a([b])
        d = Ndc_a([b])
        D.addDay(p,b,c)
        for customer in range(0,d.item()):
            f = FeatureGenerator(1)
            acc = np.random.binomial(1,0.4)
            D.addCustomer(f,acc)
        dailyRewards.append(D.dailyReward()[0])
        dailyRevenue.append(D.dailyReward()[1])
        dailyReturns.append(D.dailyReward()[2])
        D.dailyUpdate()
    print(ChosenContextsB)
    print(ChosenContextsG)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(dailyRewards, 'b', alpha=0.1)
    ax1.set_xlabel('t')
    ax1.set_title('Effective Daily Rewards')
    ax2.plot(dailyRevenue, 'b', alpha=0.1)
    ax2.set_xlabel('t')
    ax2.set_title('Due to new customers')
    ax3.plot(dailyReturns,'b', alpha=0.1)
    ax3.set_xlabel('t')
    ax3.set_title('Due to old customers')
    plt.show()





