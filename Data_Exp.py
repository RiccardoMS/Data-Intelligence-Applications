from Dati import *
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import savgol_filter

# MARGINAL REVENUES PLOT
def Mr(price):
    return price - (1.5*price/100)*price

if __name__ == "__main__":
  plt.figure(0)
  plt.xlim([14.5, 25.5])
  plt.plot(prices, margin, 'ro')
  plt.plot(prices, margin1, 'go')
  plt.plot(prices, margin2, 'bo')
  plt.legend(['Candidate: Fixed cost percentage', 'Candidate: Fixed cost','Chosen margin: Scaling margin'])
  plt.plot(np.linspace(15, 25, num=1000), [x - 0.3*x for x in np.linspace(15, 25, num=1000)],'r-')
  plt.plot(np.linspace(15, 25, num=1000), [x - 5 for x in np.linspace(15, 25, num=1000)],'g-')
  plt.plot(np.linspace(15, 25, num=1000), [x - (1.5*x/100)*x for x in np.linspace(15, 25, num=1000)],'b-')
  plt.xlabel('Price')
  plt.ylabel('Marginal Revenue')
  plt.show()


###################################### CONVERSION RATES FUNCTIONS ######################################################

# Plot: linear interpolation
if __name__ == "__main__":
  plt.figure(1)
  plt.plot(prices, C1['conversion rate'], 'ro-')
  plt.plot(prices, C2['conversion rate'], 'go-')
  plt.plot(prices, C3['conversion rate'], 'bo-')
  plt.plot(prices, aggregate['conversion rate'], 'mo-')
  plt.legend(['C1 : Low Needs', 'C2 : High Needs',
            'C3 : Very High Needs', 'Aggregate'])
  plt.xlabel('Price')
  plt.ylabel('Conversion Rate')
  plt.ylim([0, 1])
  plt.show()

# Function design: smooth interpolation via Cubic splines
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

# Plot obtained smooth conversion rate functions
if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(x, y1, 'ro')
    ax[0, 0].plot(xx, Cr_1(xx), 'r-')
    ax[0, 0].legend(['Class 1'])
    ax[0, 0].set_xlim([15,25])
    ax[0, 0].set_ylim([0, 1])
    ax[0, 0].set_xlabel('Price')
    ax[0, 0].set_ylabel('COnversion Rate')
    ax[0, 1].plot(x, y2, 'go')
    ax[0, 1].plot(xx, Cr_2(xx), 'g-')
    ax[0, 1].legend(['Class 2'])
    ax[0, 1].set_xlim([15, 25])
    ax[0, 1].set_ylim([0, 1])
    ax[0, 1].set_xlabel('Price')
    ax[0, 1].set_ylabel('COnversion Rate')
    ax[1, 0].plot(x, y3, 'bo')
    ax[1, 0].plot(xx, Cr_3(xx), 'b-')
    ax[1, 0].legend(['Class 3'])
    ax[1, 0].set_xlim([15, 25])
    ax[1, 0].set_ylim([0, 1])
    ax[1, 0].set_xlabel('Price')
    ax[1, 0].set_ylabel('COnversion Rate')
    ax[1, 1].plot(x, ya, 'mo')
    ax[1, 1].plot(xx, Cr_a(xx), 'm-')
    ax[1, 1].plot(xx, Cr_1(xx), 'r-',alpha=0.5)
    ax[1, 1].plot(xx, Cr_2(xx), 'g-',alpha=0.5)
    ax[1, 1].plot(xx, Cr_3(xx), 'b-',alpha=0.5)
    ax[1, 1].legend(['Aggregate'])
    ax[1, 1].set_xlim([15, 25])
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].set_xlabel('Price')
    ax[1, 1].set_ylabel('Conversion Rate')
    plt.suptitle('Conversion Rate Functions ')
    plt.show()


###################################### RETURN PROBS FUNCTIONS #########################################################

r_1 = stats.rv_discrete(values=(C1['return probs'][0], C1['return probs'][1]))
r_2 = stats.rv_discrete(values=(C2['return probs'][0], C2['return probs'][1]))
r_3 = stats.rv_discrete(values=(C3['return probs'][0], C3['return probs'][1]))
r_a = stats.rv_discrete(values=(aggregate['return probs'][0], aggregate['return probs'][1]))

if __name__ == "__main__":
  times = np.array([0,1,2,3])
  fig, ax = plt.subplots(2, 2)
  ax[0, 0].plot(times, r_1.pmf(times), 'ro', ms=5, mec='r')
  ax[0, 0].set_ylim(0,1)
  ax[0, 0].legend(['Class 1'])
  ax[0, 0].set_xlabel('Times')
  ax[0, 0].set_ylabel('Probability')
  ax[0, 1].plot(times, r_2.pmf(times), 'go', ms=5, mec='g')
  ax[0, 1].set_ylim(0,1)
  ax[0, 1].legend(['Class 2'])
  ax[0, 1].set_xlabel(['Times'])
  ax[0, 1].set_ylabel(['Probability'])
  ax[1, 0].plot(times, r_3.pmf(times), 'bo', ms=5, mec='b')
  ax[1, 0].set_ylim(0,1)
  ax[1, 0].legend(['Class 3'])
  ax[1, 0].set_xlabel('Times')
  ax[1, 0].set_ylabel('Probability')
  ax[1, 1].plot(times, r_a.pmf(times), 'mo', ms=5, mec='m')
  ax[1, 1].set_ylim(0,1)
  ax[1, 1].legend(['Aggregate'])
  ax[1, 1].set_xlabel('Times')
  ax[1, 1].set_ylabel('Probability')
  fig.suptitle("Customer Loyalty")
  plt.show()


# Return probability Random variable generation
def R_1(size):
      return r_1.rvs(size)
def R_2(size):
      return r_2.rvs(size)
def R_3(size):
      return r_3.rvs(size)
def R_a(size):
      return r_a.rvs(size)

RETURNS = dict({
    'C1' : R_1(),
    'C2' : R_2(),
    'C3' : R_3(),
    'AGG': R_a()
})

#################################### COST PER CLICK STOCHASTIC FUNCTIONS ###############################################

# -> Class' different mean levels are considered (non-decreasing, piecewise linear)
# -> Saturation may happen both at small and high values of bid
# -> Stochastic component (gaussian zero-mean error with class specific variance is introduced to account for the
#    not accounted advertising auction mechanism

b = np.array(bids)
bb = np.linspace(0.5, 2, 2000)

b1 = [b.min(), b.min() + C1['CPC'][0][0], b.max() - C1['CPC'][0][1],b.max()]
k1 = np.repeat(C1['CPC'][1],2)
cpc_1 = interp1d(b1,k1,kind='linear', fill_value='extrapolate')

b2 = [b.min(), b.min() + C2['CPC'][0][0], b.max() - C2['CPC'][0][1],b.max()]
k2 = np.repeat(C2['CPC'][1],2)
cpc_2 = interp1d(b2,k2,kind='linear', fill_value='extrapolate')

b3 = [b.min(), b.min() + C3['CPC'][0][0], b.max() - C3['CPC'][0][1],b.max()]
k3 = np.repeat(C3['CPC'][1],2)
cpc_3 = interp1d(b3,k3,kind='linear', fill_value='extrapolate')

cpc_a = lambda bid: 0.3*cpc_1(bid) + 0.5*cpc_2(bid) + 0.2*cpc_3(bid)

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
      return 0.3*Cpc_1(bid) + 0.5*Cpc_2(bid) + 0.2*Cpc_3(bid)

if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(bb, cpc_1(bb), 'r')
    ax[0, 0].set_ylim(-0.02, 3)
    ax[0, 0].legend(['Class 1'])
    ax[0, 0].fill_between(bb,cpc_1(bb) - C1['CPC'][2] - C1['CPC'][2],
                          cpc_1(bb) + C1['CPC'][2] + C1['CPC'][2],color ='r', alpha=0.2)
    ax[0, 0].fill_between(bb, np.repeat([-0.01], len(bb)),np.repeat([-0.02], len(bb)), color='w', alpha=1)
    ax[0, 0].plot(b, Cpc_1(b), 'ro')
    ax[0, 0].set_xlabel('Bid')
    ax[0, 0].set_ylabel('Cost per Click')
    ax[0, 1].plot(bb, cpc_2(bb), 'g')
    ax[0, 1].set_ylim(-0.02, 3)
    ax[0, 1].legend(['Class 2'])
    ax[0, 1].fill_between(bb, cpc_2(bb) - C2['CPC'][2] - C2['CPC'][2],
                          cpc_2(bb) + C2['CPC'][2] + C2['CPC'][2], color='g', alpha=0.2)
    ax[0, 1].fill_between(bb, np.repeat([-0.01], len(bb)), np.repeat([-0.02], len(bb)), color='w', alpha=1)
    ax[0, 1].plot(b, Cpc_2(b), 'go')
    ax[0, 1].set_xlabel('Bid')
    ax[0, 1].set_ylabel('Cost per Click')
    ax[1, 0].plot(bb, cpc_3(bb), 'b')
    ax[1, 0].set_ylim(-0.02, 3)
    ax[1, 0].legend(['Class 3'])
    ax[1, 0].fill_between(bb, cpc_3(bb) - C3['CPC'][2] - C3['CPC'][2],
                          cpc_3(bb) + C3['CPC'][2] + C3['CPC'][2], color='b', alpha=0.2)
    ax[1, 0].fill_between(bb, np.repeat([-0.01], len(bb)), np.repeat([-0.02], len(bb)), color='w', alpha=1)
    ax[1, 0].plot(b, Cpc_3(b), 'bo')
    ax[1, 0].set_xlabel('Bid')
    ax[1, 0].set_ylabel('Cost per Click')
    ax[1, 1].plot(bb, cpc_a(bb), 'm')
    ax[1, 1].set_ylim(-0.02, 3)
    ax[1, 1].legend(['Aggregate'])
    ax[1, 1].plot(b, Cpc_a(b), 'mo')
    ax[1, 1].set_xlabel('Bid')
    ax[1, 1].set_ylabel('Cost per Click')
    fig.suptitle('Cost per Click Stochastic Functions')
    plt.show()

############################### NUMBER OF DAILY CLICKS FUNCTIONS ################################################

# -> Clicks obtained thtrough auction mechanism: our only input is the bid value
# -> Consider increasing function of the bid: higher bids permit to obtain better advs
# -> Upper saturation
# -> Class' specific

b = np.array(bids)
bb = np.linspace(0, 2, 2000)

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
    return 0.3*ndc_1(bid) + 0.5*ndc_2(bid) + 0.2*ndc_3(bid)

def Ndc_1 ( bid ):
    res = np.around(ndc_1(bid) + np.random.normal(0.0, C1['NNC'][2], len(bid)), 3)
    res[res < 0.0] = 0.0
    return res
def Ndc_2 ( bid ):
    res = np.around(ndc_2(bid) + np.random.normal(0.0, C2['NNC'][2], len(bid)), 3)
    res[res < 0.0] = 0.0
    return res
def Ndc_3 ( bid ):
    res = np.around(ndc_3(bid) + np.random.normal(0.0, C3['NNC'][2], len(bid)), 3)
    res[res < 0.0] = 0.0
    return res
def Ndc_a (bid):
    return 0.3*Ndc_1(bid) + 0.5*Ndc_2(bid) + 0.2*Ndc_3(bid)


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(bb, ndc_1(bb), 'r')
    ax[0, 0].set_ylim(-0.1, 200)
    ax[0, 0].legend(['Class 1'])
    ax[0, 0].fill_between(bb,ndc_1(bb) - C1['NNC'][2] - C1['NNC'][2],
                          ndc_1(bb) + C1['NNC'][2] + C1['NNC'][2],color ='r', alpha=0.2)
    #ax[0, 0].fill_between(bb, np.repeat([-0.01], len(bb)),np.repeat([-0.02], len(bb)), color='w', alpha=1)
    ax[0, 0].plot(b, Ndc_1(b), 'ro')
    ax[0, 0].set_xlabel('Bid')
    ax[0, 0].set_ylabel('Daily Clicks of New Users')
    ax[0, 1].plot(bb, ndc_2(bb), 'g')
    ax[0, 1].set_ylim(-0.1, 200)
    ax[0, 1].legend(['Class 2'])
    ax[0, 1].fill_between(bb, ndc_2(bb) - C2['NNC'][2] - C2['NNC'][2],
                          ndc_2(bb) + C2['NNC'][2] + C2['NNC'][2], color='g', alpha=0.2)
    #ax[0, 1].fill_between(bb, np.repeat([-0.01], len(bb)), np.repeat([-0.02], len(bb)), color='w', alpha=1)
    ax[0, 1].plot(b, Ndc_2(b), 'go')
    ax[0, 1].set_xlabel('Bid')
    ax[0, 1].set_ylabel('Daily Clicks of New Users')
    ax[1, 0].plot(bb, ndc_3(bb), 'b')
    ax[1, 0].set_ylim(-0.1, 200)
    ax[1, 0].legend(['Class 3'])
    ax[1, 0].fill_between(bb, ndc_3(bb) - C3['NNC'][2] - C3['NNC'][2],
                          ndc_3(bb) + C3['NNC'][2] + C3['NNC'][2], color='b', alpha=0.2)
    #ax[1, 0].fill_between(bb, np.repeat([-0.01], len(bb)), np.repeat([-0.02], len(bb)), color='w', alpha=1)
    ax[1, 0].plot(b, Ndc_3(b), 'bo')
    ax[1, 0].set_xlabel('Bid')
    ax[1, 0].set_ylabel('Daily Clicks of New Users')
    ax[1, 1].plot(bb, ndc_a(bb), 'm')
    ax[1, 1].set_ylim(-0.1, 200)
    ax[1, 1].legend(['Aggregate'])
    ax[1, 1].plot(b, Ndc_a(b), 'mo')
    ax[1, 1].set_xlabel('Bid')
    ax[1, 1].set_ylabel('Daily Clicks of New Users')
    fig.suptitle('Daily Clicks of New Users Stochastic Functions')
    plt.show()

########################################################################################################################

plt.close()



