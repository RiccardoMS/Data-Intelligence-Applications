import numpy as np
import itertools

class Context:
    def __init__(self, rule):
        self.rule = rule                 # rule is a function which classifies features to a discrete space

        lst = [list(i) for i in itertools.product([0, 1], repeat=2)]
        un = set()
        for el in lst:
            un.add(self.rule(el))
        un = list(un)
        self.structure = len(un)

    def evaluate(self, feature):
        return self.rule(feature)

    def getRule(self):
        return self.rule

    def getRuleName(self):
        return self.rule.__name__
