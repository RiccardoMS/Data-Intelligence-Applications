# Import packages
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
# Import Environment and Learner
from JointEnvironment import *
from JointLearner import *
# Import Parameters and Functions
from DataParameters import *
from DataGenerativeFunctions import *
# Import tools for handling Contextual Pricing
from FeatureGenerator import *
from CustomerDatabase import *

# Set ContextualPricing to True or False to control diversification in pricing phase.
# Optimal Partitioning found at step 4 is considered

ContextualPricing = False
if ContextualPricing:
    ndc_agg_opt = np.load('ndc_agg_opt.npy', allow_pickle=True)
    cpc_agg_opt = np.load('cpc_agg_opt.npy',allow_pickle=True)
    def Revenue(b,priceC1, priceC2, priceC3_1,priceC3_2):
        return ndc_agg_opt[bids.index(b)]*(weights[0]*Cr_1(priceC1)*Mr(priceC1)*(1 + (1/30)*np.dot(np.array(C1['return probs'][0]),np.array(C1['return probs'][1]))) +
                               weights[1]*Cr_2(priceC2)*Mr(priceC2)*(1 + (1/30)*np.dot(np.array(C2['return probs'][0]),np.array(C2['return probs'][1]))) +
                               0.5*weights[2]*Cr_3(priceC3_1)*Mr(priceC3_1)*(1 + (1/30)*np.dot(np.array(C3['return probs'][0]), np.array(C3['return probs'][1]))) +
                               0.5*weights[2]*Cr_3(priceC3_2)*Mr(priceC3_2)*(1 + (1/30)*np.dot(np.array(C3['return probs'][0]),np.array(C3['return probs'][1])))) - ndc_agg_opt[bids.index(b)]*cpc_agg_opt[bids.index(b)]
    def Maximizer(b,prices1,prices2,prices3_1,prices3_2):
        grid = list(product(b,prices1, prices2, prices3_1, prices3_2))
        evals = [Revenue(item[1][0], item[1][1],item[1][2],item[1][3],item[1][4]) for item in enumerate(grid)]
        val = np.max(evals)
        idx = np.argmax(evals)
        return [grid[idx], val]

    solJoint = Maximizer(bids,prices,prices,prices,prices)
    print(solJoint)
else:
    from OptimizationProblem import sol
    solJoint = sol
    print(solJoint)


########################################################################################################################
Training = False
# 10 possible prices
n_prices = 10
n_bids = 10
price_values = prices
bid_values = bids

# time horizon of 1 year
T = 365

# number of experiments
n_experiments = 100

# daily_rev
def daily_rev(cr, pulled_price, mean_ret,ndc,cpc):
     return ndc*cr*Mr(prices[pulled_price])*(1 + (1/30)*mean_ret)-ndc*cpc

if Training:
 # data structures
 jointBandit_revenue_per_experiment = []

 for e in range(0, n_experiments):
   # Customer Database
   if ContextualPricing:
       database = CustomerDatabase(context=Context(Disagg))
       context = Context(Disagg)
   else:
       database = CustomerDatabase(context=Context(Naive))
       context = Context(Naive)

   env = JointEnvironment(n_prices=n_prices, n_bids = n_bids)
   joint_learner = JointLearner(n_prices=n_prices,prices=prices,n_bids=n_bids,bids=bids, context = context, rev=daily_rev)

   for day in range(0,T):
      print("Experiment {}, Day {}".format(e,day))
      # Bid selection
      pulled_bid = joint_learner.pull_bid()
      sampledNdc, sampledCpc = env.BidRewards(pulled_bid)

      # Price selection
      actualPrices = joint_learner.pull_prices(pulled_bid)

      # Daily Simulation
      database.addDay(actualPrices,bids[pulled_bid],sampledCpc)
      n_cust = sampledNdc
      n_cust_per_context = [0 for _ in range(0,joint_learner.context.structure)]

      acceptances = [0 for _ in range(0,joint_learner.context.structure)]

      for customer in range(0,n_cust):
            # Create customer
            f = FeatureGenerator(1)
            # Update n_cust_per_context
            n_cust_per_context[joint_learner.context.evaluate(f)] += 1
            # Select arm
            pulled_arm = prices.index(actualPrices[joint_learner.context.evaluate(f)])
            # Observe reward
            acc = env.sampledAcceptance(f,pulled_arm)
            # Save customer
            database.addCustomer(f,acc)
            # Update Rewards
            acceptances[joint_learner.context.evaluate(f)] += acc

      # Update Databases
      database.dailyUpdate()

      # Update Learners
      pulled_prices = [prices.index(x) for x in actualPrices]
      #if day == 364:
      #    print(databaseTS.getMeanReturnsEstimate(ts_learner.context),databaseUCB1.getMeanReturnsEstimate(ucb1_learner.context))
      joint_learner.updateMeanReturns(database.getMeanReturnsEstimate(joint_learner.context))
      joint_learner.update(pulled_prices,pulled_bid,acceptances,n_cust_per_context,sampledCpc)

   # Save results
   jointBandit_revenue_per_experiment.append(joint_learner.collected_revenues)

 ## End Simulations

 # Save Results
 if ContextualPricing:
     np.save('JointContextualResults.npy',jointBandit_revenue_per_experiment)
 else:
     np.save('JointResults.npy', jointBandit_revenue_per_experiment)

# Plot results
if ContextualPricing:
  jointBandit_revenue_per_experiment = np.load('JointContextualResults.npy')
else:
  jointBandit_revenue_per_experiment = np.load('JointResults.npy')

# Cumulative regret
plt.figure(1)
plt.xlabel("t")
plt.ylabel("")
plt.plot(np.cumsum(np.mean(solJoint[1]+0.1 - np.array(jointBandit_revenue_per_experiment), axis=0)), color='#069af3', linewidth=2.5)
plt.title("Cumulative Regret")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()

# Daily Expected reward
plt.figure(2)
for el in jointBandit_revenue_per_experiment:
    plt.plot(el, color='#069af3', alpha=0.01, label='_nolegend_')
plt.plot(np.mean(jointBandit_revenue_per_experiment, axis=0), color='b', label='Mean Colllected Revenue')
plt.plot(np.repeat([solJoint[1]], T), 'crimson', label='Optimal value')
plt.ylim([0, 700])
plt.xlabel('t')
plt.title('Expected Daily Revenue')
plt.legend(loc='best')
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()

#plt.figure(1)
#plt.xlabel("t")
#plt.ylabel("")
#plt.plot(np.cumsum(np.mean(solJoint[1] - np.array(jointBandit_revenue_per_experiment), axis=0)), 'r')
#plt.title("Cumulative Regret")
#plt.show()

# Daily Expected reward
#plt.figure(2)
#for el in jointBandit_revenue_per_experiment:
#   plt.plot(el, 'b', alpha=0.1,label='_nolegend_')
#plt.plot(np.mean(jointBandit_revenue_per_experiment, axis=0), 'k',label='Mean Collected Revenue')
#plt.plot(np.repeat([solJoint[1]], T), 'g-',label='Optimale value')
#plt.ylim([0, 1000])
#plt.xlabel('t')
#plt.title('Expected Daily Revenue')
#plt.legend(loc='best')
#plt.show()
