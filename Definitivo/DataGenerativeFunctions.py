import sys
import numpy as np

from scipy.interpolate import interp1d
from scipy import stats

from DataParameters import *

####################################   MARGINAL REVENUE FUNCTIONS  #####################################################
def Mr(price):
    return price - 5 - (1.5*price/100)*price

###################################### CONVERSION RATES FUNCTIONS ######################################################
# Smooth interpolation via Cubic splines
x = np.array(prices)
xx = np.linspace(x.min()-0.5, x.max()+0.5, 2000)
y1 = C1['conversion rate']
y2 = C2['conversion rate']
y3 = C3['conversion rate']
ya = aggregate['conversion rate']

# Define interpolating functions
it_1 = interp1d(x, y1, kind='cubic', fill_value='extrapolate')
it_2 = interp1d(x, y2, kind='cubic', fill_value='extrapolate')
it_3 = interp1d(x, y3, kind='cubic', fill_value='extrapolate')

# Define Functions to export
def Cr_1( price ):
    return np.around(it_1(price), 3)
def Cr_2( price ):
    return np.around(it_2(price), 3)
def Cr_3( price ):
    return np.around(it_3(price), 3)
def Cr_a( price ):
    return 0.3*Cr_1(price) + 0.5*Cr_2(price) + 0.2*Cr_3(price)

###################################### RETURN PROBS FUNCTIONS #########################################################

r_1 = stats.rv_discrete(values=(C1['return probs'][0], C1['return probs'][1]))
r_2 = stats.rv_discrete(values=(C2['return probs'][0], C2['return probs'][1]))
r_3 = stats.rv_discrete(values=(C3['return probs'][0], C3['return probs'][1]))
r_a = stats.rv_discrete(values=(aggregate['return probs'][0], aggregate['return probs'][1]))

# Return probability Random variable generation
def R_1():
      r=r_1.rvs()
      return r
def R_2():
      r = r_2.rvs()
      return r
def R_3():
      r = r_3.rvs()
      return r
def R_a():
      return r_a.rvs()

RETURNS = dict({
    'C1' : R_1,
    'C2' : R_2,
    'C3' : R_3,
    'AGG': R_a
})

#################################### COST PER CLICK STOCHASTIC FUNCTIONS ###############################################
b_cpc = np.array(bids)
bb_cpc = np.linspace(0.5, 2, 2000)

b1 = [b_cpc.min(), b_cpc.min() + C1['CPC'][0][0], b_cpc.max() - C1['CPC'][0][1],b_cpc.max()]
k1 = np.repeat(C1['CPC'][1],2)
cpc_1 = interp1d(b1,k1,kind='linear', fill_value='extrapolate')

b2 = [b_cpc.min(), b_cpc.min() + C2['CPC'][0][0], b_cpc.max() - C2['CPC'][0][1],b_cpc.max()]
k2 = np.repeat(C2['CPC'][1],2)
cpc_2 = interp1d(b2,k2,kind='linear', fill_value='extrapolate')

b3 = [b_cpc.min(), b_cpc.min() + C3['CPC'][0][0], b_cpc.max() - C3['CPC'][0][1],b_cpc.max()]
k3 = np.repeat(C3['CPC'][1],2)
cpc_3 = interp1d(b3,k3,kind='linear', fill_value='extrapolate')

cpc_a = lambda bid: weights[0]*cpc_1(bid) + weights[1]*cpc_2(bid) + weights[2]*cpc_3(bid)

# Cost per click Functions to export
def Cpc_1(bid):
      if  isinstance(bid, float) or isinstance(bid, int):
          l=1
      else:
          l=len(bid)
      res = np.around(cpc_1(bid) + np.random.normal(0.0, C1['CPC'][2], l), 3)
      res[res < 0.0] = 0.0
      return res
def Cpc_2(bid):
    if  isinstance(bid, float) or isinstance(bid, int):
        l = 1
    else:
        l = len(bid)
    res = np.around(cpc_2(bid) + np.random.normal(0.0, C2['CPC'][2], l), 3)
    res[res < 0.0] = 0.0
    return res
def Cpc_3(bid):
    if  isinstance(bid, float) or isinstance(bid, int):
        l = 1
    else:
        l = len(bid)
    res = np.around(cpc_3(bid) + np.random.normal(0.0, C3['CPC'][2], l), 3)
    res[res < 0.0] = 0.0
    return res
def Cpc_a(bid):
      return weights[0]*Cpc_1(bid) + weights[1]*Cpc_2(bid) + weights[2]*Cpc_3(bid)

############################### NUMBER OF DAILY CLICKS FUNCTIONS ################################################
b_ndc = np.array(bids)
bb_ndc = np.linspace(0, 2, 2000)

def ndc_1(bid):
    if isinstance(bid, float):
        bid=[bid]
    if not isinstance(bid, list) :
        bid = list(bid)
    return np.piecewise(bid, [bid < 0.5 + np.array(C1['NNC'][0]),bid >= 0.5 + np.array(C1['NNC'][0])],
                        [lambda bid: 0.0*bid, lambda bid: np.array(C1['NNC'][1][0])*(1.0-np.exp(-np.array(C1['NNC'][1][1])*(bid-(np.array(C1['NNC'][0])+0.5))))])
def ndc_2(bid):
    if isinstance(bid, float):
        bid=[bid]
    if not isinstance(bid, list):
        bid = list(bid)
    return  np.piecewise(bid, [bid < 0.5 + np.array(C2['NNC'][0]),bid >= 0.5 + np.array(C2['NNC'][0])],
                        [lambda bid: 0.0*bid, lambda bid: np.array(C2['NNC'][1][0])*(1.0-np.exp(-np.array(C2['NNC'][1][1])*(bid-(np.array(C2['NNC'][0])+0.5))))])
def ndc_3(bid):
    if isinstance(bid, float):
        bid=[bid]
    if not isinstance(bid, list):
        bid = list(bid)
    return  np.piecewise(bid, [bid < 0.5 + np.array(C3['NNC'][0]),bid >= 0.5 + np.array(C3['NNC'][0])],
                        [lambda bid: 0.0*bid, lambda bid: np.array(C3['NNC'][1][0])*(1.0-np.exp(-np.array(C3['NNC'][1][1])*(bid-(np.array(C3['NNC'][0])+0.5))))])

def ndc_a(bid):
    return weights[0]*ndc_1(bid) + weights[1]*ndc_2(bid) + weights[2]*ndc_3(bid)

def Ndc_1 ( bid ):
    res = np.around(ndc_1(bid) + np.random.normal(0.0, C1['NNC'][2], len(bid)), 3)
    res[res < 0.0] = 0.0
    return res.astype('int32')
def Ndc_2 ( bid ):
    res = np.around(ndc_2(bid) + np.random.normal(0.0, C2['NNC'][2], len(bid)), 3)
    res[res < 0.0] = 0.0
    return res.astype('int32')
def Ndc_3 ( bid ):
    res = np.around(ndc_3(bid) + np.random.normal(0.0, C3['NNC'][2], len(bid)), 3)
    res[res < 0.0] = 0.0
    return res.astype('int32')
def Ndc_a (bid):
    res = 0.3*Ndc_1(bid) + 0.5*Ndc_2(bid) + 0.2*Ndc_3(bid)
    return res.astype('int32')


########################################################################################################################



