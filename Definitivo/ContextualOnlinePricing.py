# Import packages
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
# Import Learners
from Contextual_UCB1_learner import *
from Contextual_TS_learner import *
# Import Environment
from ContextualEnvironment import *
# Import Parameters and Functions
from DataParameters import *
from DataGenerativeFunctions import *
from MC_Estimation_Stochastic_Aggregate_Distributions import ndc_a_MC, cpc_a_MC
# Import tools for handling contexts
from FeatureGenerator import *
from CustomerDatabase import *
from ContextGenerators import *
# Import Aggregate Case Solution
from OnlinePricing import sol

# Define New revenue Function to find best solution when Bid is fixed at optimal aggregate solution and we can provide
# each different feature subspace with a different price

opt_bid = sol[0][1]

def Revenue(b,priceC1, priceC2, priceC3_1,priceC3_2):
    return ndc_a_MC(b)*(weights[0]*Cr_1(priceC1)*Mr(priceC1)*(1 + (1/30)*np.dot(np.array(C1['return probs'][0]),np.array(C1['return probs'][1]))) +
                               weights[1]*Cr_2(priceC2)*Mr(priceC2)*(1 + (1/30)*np.dot(np.array(C2['return probs'][0]),np.array(C2['return probs'][1]))) +
                               0.5*weights[2]*Cr_3(priceC3_1)*Mr(priceC3_1)*(1 + (1/30)*np.dot(np.array(C3['return probs'][0]), np.array(C3['return probs'][1]))) +
                               0.5*weights[2]*Cr_3(priceC3_2)*Mr(priceC3_2)*(1 + (1/30)*np.dot(np.array(C3['return probs'][0]),np.array(C3['return probs'][1])))) - ndc_a_MC(b)*cpc_a_MC(opt_bid)

def Maximizer(prices1,prices2,prices3_1,prices3_2):
    grid = list(product(prices1, prices2, prices3_1, prices3_2))
    evals = [Revenue(opt_bid,item[1][0], item[1][1],item[1][2],item[1][3]) for item in enumerate(grid)]
    val = np.max(evals)
    idx = np.argmax(evals)
    return [grid[idx], val]

solNew = Maximizer(prices,prices,prices,prices)
print(solNew)

########################################################################################################################
# 10 possible prices
n_arms = 10
arms_value = prices

# time horizon of 1 year
T = 365

# number of experiments
n_experiments = 100

# daily_rev
opt_ndc = ndc_a_MC(opt_bid)
opt_cpc = cpc_a_MC(opt_bid)
def daily_rev(cr, pulled_arm, mean_ret,ndc = opt_ndc,cpc = opt_cpc):
     return ndc*cr*Mr(prices[pulled_arm])*(1 + (1/30)*mean_ret)-ndc*cpc

Training = False
if Training:
 # data structures
 ts_revenue_per_experiment = []
 ucb1_revenue_per_experiment = []
 ts_contexts_per_experiment =[]
 ucb1_contexts_per_experiment = []

 for e in range(0, n_experiments):
   print("Experiment {}".format(e))

   # Customer Database
   databaseTS = CustomerDatabase(context=Context(Naive))
   databaseUCB1 = CustomerDatabase(context=Context(Naive))

   # Initialize Context With Naive Context (Aggregated data)
   contextTS = Context(Naive)
   contextUCB1 = Context(Naive)

   # Initialize Environment & Learners
   env = ContextualEnvironment(n_arms=n_arms)
   ts_learner = Contextual_TS_learner(n_arms=n_arms, context = contextTS, rev=daily_rev)
   ucb1_learner = Contextual_UCB1_learner(n_arms=n_arms, context = contextUCB1, rev=daily_rev)

   # Simulation
   for day in range(0,T):
      # Context Generator if new week
      if day % 7 == 0 and day !=0:
         # pick best context
         contextTS = BruteForceContextGenerator(databaseTS)           # NaiveContextGenerator to perform Aggregated Pricing Optimization
         contextUCB1 = BruteForceContextGenerator(databaseUCB1)       # Also available: GreedyContextGenerator
         databaseTS.updateCurrentContext(contextTS)
         databaseUCB1.updateCurrentContext(contextUCB1)
         # if new context is different from current one,  update and perform offline training . Otherwise, keep learning.
         if ucb1_learner.updateContext(contextUCB1):
            ucb1_learner.updateMeanReturns(databaseUCB1.getMeanReturnsEstimate(ucb1_learner.context))
            ucb1_learner.offlineTraining(databaseUCB1)
         if ts_learner.updateContext(contextTS):
            ts_learner.updateMeanReturns(databaseTS.getMeanReturnsEstimate(ts_learner.context))
            ts_learner.offlineTraining(databaseTS)

      # Daily Routine
      pTS = ts_learner.pull_arm()
      pUCB1 = ucb1_learner.pull_arm()

      databaseTS.addDay(pTS,opt_bid,opt_cpc)
      databaseUCB1.addDay(pUCB1, opt_bid, opt_cpc)

      n_cust = Ndc_a([opt_bid])
      n_cust_per_contextTS = [0 for _ in range(0,ts_learner.context.structure)]
      n_cust_per_contextUCB1 = [0 for _ in range(0,ucb1_learner.context.structure)]

      rewTS = [0 for _ in range(0,ts_learner.context.structure)]
      rewUCB1 = [0 for _ in range(0,ucb1_learner.context.structure)]


      for customer in range(0,n_cust.item()):
            # Create customer
            f = FeatureGenerator(1)
            # Update n_cust_per_context
            n_cust_per_contextTS[ts_learner.context.evaluate(f)] += 1
            n_cust_per_contextUCB1[ucb1_learner.context.evaluate(f)] += 1
            # Select arm
            pulled_armTS = prices.index(pTS[ts_learner.context.evaluate(f)])
            pulled_armUCB1 = prices.index(pUCB1[ucb1_learner.context.evaluate(f)])
            # Observe reward
            accTS = env.round(f,pulled_armTS)
            accUCB1 = env.round(f, pulled_armUCB1)
            # Save customer
            databaseTS.addCustomer(f,accTS)
            databaseUCB1.addCustomer(f,accUCB1)
            # Update Rewards
            rewTS[ts_learner.context.evaluate(f)] += accTS
            rewUCB1[ucb1_learner.context.evaluate(f)] += accUCB1

      # Update Databases
      databaseTS.dailyUpdate()
      databaseUCB1.dailyUpdate()

      # Update Learners
      pulled_armsTS = [prices.index(x) for x in pTS]
      pulled_armsUCB1 = [prices.index(x) for x in pUCB1]
      ts_learner.updateMeanReturns(databaseTS.getMeanReturnsEstimate(ts_learner.context))
      ucb1_learner.updateMeanReturns(databaseUCB1.getMeanReturnsEstimate(ucb1_learner.context))
      ts_learner.update(pulled_armsTS,rewTS,n_cust_per_contextTS)
      ucb1_learner.update(pulled_armsUCB1,rewUCB1,n_cust_per_contextUCB1)

      # Check Mean Returns estimates at convergence:
      # if day == 364:
      #    print("TS Mean Returns Estimates: {}".format(ts_learner.mean_returns_estimates))
      #    print("UCB1 Mean Returns Estimates: {}".format(ucb1_learner.mean_returns_estimates))

   # Save results
   ts_revenue_per_experiment.append(ts_learner.collected_revenues)
   ucb1_revenue_per_experiment.append(ucb1_learner.collected_revenues)
   ts_contexts_per_experiment.append(databaseTS.contextHistory)
   ucb1_contexts_per_experiment.append(databaseUCB1.contextHistory)

 ## End Simulations
 np.save('TSContextualPricingResults_Revenue.npy',ts_revenue_per_experiment)
 np.save('TSContextualPricingResults_Contexts.npy', ts_contexts_per_experiment)
 np.save('UCB1ContextualPricingResults_Revenue.npy', ucb1_revenue_per_experiment)
 np.save('UCB1ContextualPricingResults_Contexts.npy', ucb1_contexts_per_experiment)



# Plot results
ts_revenue_per_experiment = np.load('TSContextualPricingResults_Revenue.npy', allow_pickle=True)
ucb1_revenue_per_experiment = np.load('UCB1ContextualPricingResults_Revenue.npy',allow_pickle=True)
ts_contexts_per_experiment = np.load('TSContextualPricingResults_Contexts.npy',allow_pickle=True)
ucb1_contexts_per_experiment = np.load('UCB1ContextualPricingResults_Contexts.npy',allow_pickle=True)
# Cumulative regret
plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(solNew[1] - np.array(ts_revenue_per_experiment), axis=0)), 'r')
plt.plot(np.cumsum(np.mean(solNew[1] - np.array(ucb1_revenue_per_experiment), axis=0)), 'b')
plt.legend(["TS", "UCB1"])
plt.title("Cumulative Regret")
plt.show()

# Daily Expected reward
fig, (ax1, ax2) = plt.subplots(1, 2)
for el in ucb1_revenue_per_experiment:
   ax1.plot(el, 'b', alpha=0.1,label='_nolegend_')
ax1.plot(np.mean(ucb1_revenue_per_experiment, axis=0), 'k',label='Mean Revenue')
ax1.plot(np.repeat([solNew[1]], T), 'g-',label='Optimale value')
ax1.plot(np.repeat([sol[1]], T), 'r-', label='Optimal Aggregate')
ax1.set_ylim([0, 1200])
ax1.set_xlabel('t')
ax1.set_ylabel('Expected Daily Revenue')
ax1.set_title('UCB1')
ax1.legend(loc='best')
for el in ts_revenue_per_experiment:
    ax2.plot(el, 'b', alpha=0.1, label='_nolegend_')
ax2.plot(np.mean(ts_revenue_per_experiment, axis=0), 'k',label='Mean Revenue')
ax2.plot(np.repeat([solNew[1]], T), 'g-',label='Optimal value')
ax2.plot(np.repeat([sol[1]], T), 'r-', label='Optimal Aggregate')
ax2.set_ylim([0, 1200])
ax2.set_xlabel('t')
ax2.set_ylabel('Expected Daily Revenue')
ax2.set_title('TS')
ax2.legend(loc='best')
plt.show()

# Context History per experiment
d = dict({'Naive':0,'F1':1,'F2':2,'C11':3,'C12':4,'C21':5,'C22':6,'Disagg':7})
for i in range(0,len(ts_contexts_per_experiment)):
     for j in range(0, len(ts_contexts_per_experiment[i])):
         ts_contexts_per_experiment[i][j]=d[ts_contexts_per_experiment[i][j].getRuleName()]
for i in range(0,len(ucb1_contexts_per_experiment)):
     for j in range(0, len(ucb1_contexts_per_experiment[i])):
         ucb1_contexts_per_experiment[i][j]=d[ucb1_contexts_per_experiment[i][j].getRuleName()]
fig, (ax1, ax2) = plt.subplots(2, 1)
for el in ts_contexts_per_experiment:
     ax1.plot(el, 'b--', alpha=0.1, label='_nolegend_')
ax1.set_ylim([-0.5, 7.5])
ax1.set_xticks(range(52))
ax1.set_ylabel('')
ax1.set_yticks([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
ax1.set_yticklabels(['Naive', 'F1', 'F2', 'C11', 'C12', 'C21', 'C22', 'Disaggregate'])
ax1.set_title('TS')
for el in ucb1_contexts_per_experiment:
     ax2.plot(el, 'b--', alpha=0.1, label='_nolegend_')
ax2.set_ylim([-0.5, 7.5])
ax2.set_xticks(range(52))
ax2.set_ylabel('')
ax2.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
ax2.set_yticklabels(['Naive', 'F1', 'F2', 'C11', 'C12', 'C21', 'C22', 'Disaggregate'])
ax2.set_title('UCB1')
plt.show()







