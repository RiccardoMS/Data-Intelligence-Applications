from DataParameters import *
from DataGenerativeFunctions import *
import numpy as np

# Mean and Variance Function Estimation for Ndc and Cpc in the Aggregate Case via Monte Carlo Sampling
b = np.linspace(0.5,2,1000)
Training = False
if Training:
  n_exp = 1000
  ndc_samples = [[] for _ in range(1000)]
  cpc_samples = [[] for _ in range(1000)]

  for i in range(len(b)):
    for exp in range(n_exp):
        ndc_samples[i].append(Ndc_a([b[i]]).item())
        cpc_samples[i].append(Cpc_a([b[i]]).item())

  ndc_agg = np.mean(ndc_samples, axis = 1)
  cpc_agg = np.mean(cpc_samples, axis = 1)
  ndc_agg_std = np.std(ndc_samples, axis=1)
  cpc_agg_std = np.std(cpc_samples, axis = 1)

  np.save('ndc_agg.npy',ndc_agg)
  np.save('cpc_agg.npy',cpc_agg)
  np.save('ndc_agg_std.npy', ndc_agg_std)
  np.save('cpc_agg_std.npy', cpc_agg_std)

ndc_agg = np.load('ndc_agg.npy')
cpc_agg = np.load('cpc_agg.npy')
ndc_agg_std = np.load('ndc_agg_std.npy')
cpc_agg_std = np.load('cpc_agg_std.npy')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def ndc_a_MC(bid):
    bid_idx = find_nearest(b,bid)
    return ndc_agg[bid_idx]
def cpc_a_MC(bid):
    bid_idx = find_nearest(b,bid)
    return cpc_agg[bid_idx]
def ndc_a_std_MC(bid):
    bid_idx = find_nearest(b,bid)
    return ndc_agg_std[bid_idx]
def cpc_a_std_MC(bid):
    bid_idx = find_nearest(b,bid)
    return cpc_agg_std[bid_idx]

# Testing
if __name__ == "__main__":
    for el in bids:
      print("Bid : {}".format(el))
      print(ndc_a_MC(el), ndc_a(el).item())
      print(cpc_a_MC(el), cpc_a(el))