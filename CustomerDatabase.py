import numpy as np
from Customer import *
from ContextGenerator import *

delta = 0.05
def normalize(my_dict):
    sum_p = sum(my_dict.values())
    for i in my_dict:
        my_dict[i] = float(my_dict[i] / sum_p)
    return my_dict


class CustomerDatabase:
    def __init__(self):
        self.Database = []
        self.allCustomers = []
        self.currentContextStructure = NaiveContextGenerator    #given any feature, return 0
        self.currentMeanReturnsEstimates = dict({})                   #given the current context structure, update estimates as data are collected
        for el in self.currentContextStructure.structure():
             self.currentMeanReturnsEstimates[str(el)].append(dict({}))

        # self.observedReturns = dict({})
        # self.probabilities = [1.0/3.0,1.0/3.0,1.0/3.0]
        # self.count = dict({'C1':1, 'C2':1, 'C3':1 })
        # self.returnsDistr = dict({'C1':[1.0,0.0,0.0,0.0],'C2':[1.0,0.0,0.0,0.0],'C3':[1.0,0.0,0.0,0.0]})
        # self.returnsCount = dict({'C1':[1,0,0,0],'C2':[1,0,0,0],'C3':[1,0,0,0]})


    def addCustomer(self, ID, price, feature, purchase):
        if purchase:
            self.Database.append(Customer(ID,price,feature))
            self.allCustomers.append([ID,feature])
        else:
            self.allCustomers.append([ID, feature])

        # self.observedReturns[ID]=0
        # self.count[group] = self.count[group] + 1
        # self.probabilities = [x / sum(self.count.values()) for x in self.count.values()]
        # self.returnsCount[group][self.actualReturns[ID]] += 1


    def dailyReward(self):
        today_reward = 0.0
        for el in self.Database:
            today_reward += el.reward()
        return today_reward

    def dailyUpdate(self):                          # can be included in dailyReward?
        for el in self.Database:
            el.update()

    def dailyMeanReturnsUpdate(self):
        count = dict({})
        for el in self.Database:
            self.currentMeanReturnsEstimates[self.currentContextStructure.evaluate(el.feature())][el.currentReturns()] += 1
        for c in self.currentMeanReturnsEstimates.values():
            normalize(c)

    def retrieveMeanReturnsEstimates(self):
        return self.currentMeanReturnsEstimates

    def retrieveProbabilitiesBounds(self, context_structure):
        probabilities = dict({})
        for el in context_structure.structure():
            probabilities[str(el)].append(0)
        for el in self.allCustomers:
            probabilities[context_structure.evaluate(el[1])] += 1
        normalize(probabilities)
        lower_prob = np.sqrt(np.log(delta)/(2*sum(probabilities.values())))
        for i in probabilities:
            probabilities[i] -= lower_prob
        normalize(probabilities)
        return probabilities

    def retrieveRewardsBounds(self, context_structure):




        return




