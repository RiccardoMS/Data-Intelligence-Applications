import numpy as np
from Context import *


# Note: considered binary features only
# mapping of all possible rules
rules = []

# Naive
def Naive(feature):
    return 0

# First Feature
def F1(feature):
    if feature[0]:                           # F1 == 1
        return 1
    else:                                    # F1 == 0
        return 0

# Second Feature
def F2(feature):                             # F2 == 1
    if feature[1]:
        return 1
    else:                                    # F2 == 0
        return 0

# Combination 11
def C11(feature):
    if feature[0]:                           # F1 == 1
        if feature[1]:                       # F1 == 1 && F2 == 1
            return 0
        else:                                # F1 == 1 && F2 == 0
            return 1
    else:                                    # F1 == 0
        return 2

# Combination 12
def C12(feature):
    if feature[0]:                           # F1 == 1
        return 0
    else:                                    # F1 == 0
        if feature[1]:                       # F1 == 0 && F2 == 1
            return 1
        else:                                # F1 == 0 && F2 == 0
            return 2

# Combination 21
def C21(feature):
    if feature[1]:                           # F2 == 1
        if feature[0]:                       # F2 == 1 && F1 == 1
            return 0
        else:                                # F2 == 1 && F1 == 0
            return 1
    else:                                    # F2 == 0
        return 2

# Combination 22
def C22(feature):
    if feature[1]:                           # F2 == 1
        return 0
    else:                                    # F2 == 0
        if feature[0]:                       # F2 == 0 && F1 == 1
            return 1
        else:                                # F2 == 0 && F1 == 0
            return 2

# Disaggregate
def Disagg(feature):
    if feature[0]:
        if feature[1]:
            return 0
        else:
            return 1
    else:
        if feature[1]:
            return 2
        else:
            return 3


rules.append(Naive)
rules.append(F1)
rules.append(F2)
rules.append(C11)
rules.append(C12)
rules.append(C21)
rules.append(C22)
rules.append(Disagg)


## Brute Force Context Generator
def BruteForceContextGenerator(custDatabase):
    v = []
    for rule in rules:
        v.append(custDatabase.retrieveSplitValue(Context(rule)))
    #print(v)
    return Context(rules[np.argmax(v)])

## Greedy Context Generator
def getNodeChildren( rule ):
    if rule == Naive :
        return [F1, F2]
    if rule == F1:
        return [C11,C12]
    if rule == F2:
        return [C21,C22]
    if rule in [C11,C12,C21,C22]:
        return [Disagg]
    if rule is Disagg:
        return None


def GreedyContextGenerator(custDatabase):
    choice = np.random.binomial(1,0.5)

    if not choice:                                                                            # Follow the previous path
       C = custDatabase.currentContext.getRule()
       v = custDatabase.retrieveSplitValue(custDatabase.currentContext)
       Children = getNodeChildren(C)
       print("Previous Path {}".format(Children))
       go = 0
       if Children is not None:
           go=1
       while (go):
           values = []
           for i in range(0,len(Children)):
              values.append(custDatabase.retrieveSplitValue(Context(Children[i])))
           if v > np.max(values):
               return Context(C)
           else:
              C = Children[np.argmax(values)]
           Children = getNodeChildren(C)
           if Children is None:
                  go = 0
       return Context(C)
                                                                                          # Scan whole tree from root
    else:
        C = Naive
        v = custDatabase.retrieveSplitValue(Context(Naive))
        Children = getNodeChildren(C)
        print("wholeSCan {}".format(Children))
        go = 1
        while (go):
            values = []
            for i in range(0, len(Children)):
                values.append(custDatabase.retrieveSplitValue(Context(Children[i])))
            if v > np.max(values):
                return Context(C)
            else:
                C = Children[np.argmax(values)]
            Children = getNodeChildren(C)
            if Children is None:
                go = 0
        return Context(C)

def NaiveContextGenerator(custDatabase):
        return Context(Naive)