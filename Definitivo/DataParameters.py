import numpy as np

# CANDIDATES
prices = [15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5 ]
bids = [0.65, 0.80, 0.95, 1.05, 1.20, 1.35, 1.50, 1.65, 1.8, 1.95]

# MARGIN FUNCTION CANDIDATES
margin = [x - 5 - 0.3*x for x in prices ]
margin1 = [x - 5 - 5 for x in prices ]
margin2 = [x - 5 -(1.5*x/100)*x  for x in prices]

# CUSTOMER CLASSES PARAMETERS
# Proportions
weights = [0.3, 0.5, 0.2]

# Each underlying class has specific:
# 1) Diet Profile Feature: 0 for low, 1 for high
# 2) Physical Activity Profile Feature: 0 for low, 1 for high
# 3) Conversion Rate Function ( deterministic wrt to price)
# 4) Probability Distribution over the number of returns in the 30 days following first purchase
# 5) Cost per Click Stochastic Function parameters
# 6) Number of new Daily clicks Stochastic Function parameters

C1 = dict({
    'Diet Profile': 0,
    'Physical Activity Profile': 0,
    'conversion rate': [0.5, 0.8, 0.7, 0.4, 0.25, 0.1, 0.05, 0.1, 0.1, 0.1],
    'return probs': [[0,1,2,3],[0.5,0.5,0.0,0.0]],
    'CPC': [[0.01, 0.3],[0.0, 2.0],[0.1]],
    'NNC': [[0.01],[100,2],[10]]
})

C2 = dict({
    'Diet Profile': 1,
    'Physical Activity Profile': 0,
    'conversion rate': [0.15, 0.25, 0.25, 0.5, 0.55, 0.65, 0.7, 0.65, 0.5, 0.2],
    'return probs': [[0,1,2,3],[0.0,0.5,0.5,0.0]],
    'CPC': [[0.15, 0.15],[0.0, 2.4],[0.07]],
    'NNC': [[0.15],[120, 2],[10]]
})

C3 = dict({
    'Diet Profile': [0,1],
    'Physical Activity Profile': 1,
    'conversion rate': [0.1, 0.1, 0.1, 0.1, 0.15, 0.55, 0.6, 0.75, 0.7, 0.6],
    'return probs': [[0,1,2,3], [0.0,0.0,0.5,0.5]],
    'CPC': [[0.25, 0.1],[0.0, 2.8],[0.045]],
    'NNC': [[0.25],[130,2],[10]]
})

# AGGREGATE DATA
# Due to stochasticity and mixing approach used to define aggregate functions, CPC and NNC parameters are retrieved via MC simulation
aggregate = dict({
    'conversion rate': np.average([C1['conversion rate'], C2['conversion rate'], C3['conversion rate']], axis=0,weights= weights),
    'return probs': [[0,1,2,3],(weights[0]*np.array(C1['return probs'][1]) + weights[1]*np.array(C2['return probs'][1])
                                + weights[2]*np.array(C3['return probs'][1])).tolist()],
    'CPC': None,
    'NNC': None
})

