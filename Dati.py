import numpy as np


# ENVIRONMENT DATA
prices = [15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5 ]
bids = [0.65, 0.80, 0.95, 1.05, 1.20, 1.35, 1.50, 1.65, 1.8, 1.95]

# COMMON DATA
# Idea: fixed costs accounted for manufacturing and materials. Consider fixed amount or fixed percentage.

margin = [x - 0.3*x for x in prices]
margin1 = [x - 5 for x in prices]
margin2 = [x - (1.5*x/100)*x  for x in prices]

# CUSTOMERS CLASSES

weights = [0.3, 0.5, 0.2]

C1 = dict({
    'Diet Profile': 0,
    'Physical Activity Profile': 0,
    'conversion rate': [0.5, 0.8, 0.7, 0.4, 0.25, 0.1, 0.05, 0.1, 0.1, 0.1],
    'return probs': [[0,1,2,3],[0.5,0.5,0.0,0.0]],
    'CPC': [[0.01, 0.3],[0.0, 2.0],[0.1]],
    'NNC': [[0.01],[100,2],[5]]
})

C2 = dict({
    'Diet Profile': 1,
    'Physical Activity Profile': 0,
    'conversion rate': [0.15, 0.25, 0.25, 0.5, 0.55, 0.65, 0.7, 0.65, 0.5, 0.2],
    'return probs': [[0,1,2,3],[0.0,0.5,0.5,0.0]],
    'CPC': [[0.15, 0.15],[0.0, 2.4],[0.07]],
    'NNC': [[0.15],[120, 2],[5]]
})

C3 = dict({
    'Diet Profile': [0,1],
    'Physical Activity Profile': 1,
    'conversion rate': [0.1, 0.1, 0.1, 0.1, 0.15, 0.55, 0.6, 0.75, 0.7, 0.6],
    'return probs': [[0,1,2,3], [0.0,0.0,0.5,0.5]],
    'CPC': [[0.25, 0.1],[0.0, 2.8],[0.045]],
    'NNC': [[0.25],[130,2],[5]]
})

aggregate = dict({
    'conversion rate': np.average([C1['conversion rate'], C2['conversion rate'], C3['conversion rate']], axis=0,weights= weights),
    'return probs': [[0,1,2,3],(weights[0]*np.array(C1['return probs'][1]) + weights[1]*np.array(C2['return probs'][1])
                                + weights[2]*np.array(C3['return probs'][1])).tolist()],
    'CPC': None,
    'NNC': None
})

