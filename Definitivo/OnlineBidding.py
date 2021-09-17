# Import packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Import Bidding Environment and Learners
from BiddingEnvironment import *
from GTS_Learner import *
from GPTS_Learner import *
# Import Parameters and Functions
from DataParameters import bids
from DataGenerativeFunctions import Cr_a, Mr, Ndc_a,Cpc_a
from MC_Estimation_Stochastic_Aggregate_Distributions import ndc_a_MC, cpc_a_MC
from OptimizationProblem import sol
from OnlinePricing import opt_R

## Revenue Function Plots
comparison = False
if comparison:
    actual_rev = lambda bid: ndc_a_MC(bid)*Cr_a(sol[0][0])*Mr(sol[0][0])*opt_R - ndc_a_MC(bid)*cpc_a_MC(bid)
    bb = np.linspace(min(bids),max(bids),1000)
    xpos = []
    ypos = []
    xneg = []
    yneg = []
    for i in bb:
        if actual_rev(i)>=0:
          xpos.append(i)
          ypos.append(actual_rev(i))
        else:
          xneg.append(i)
          yneg.append(actual_rev(i))
    fig = plt.figure(figsize=(8, 8))
    plt.plot(xpos, ypos,color='b' )
    plt.plot(xneg, yneg, color='r')
    plt.plot(bb,np.repeat([0],len(bb)))
    plt.xlabel("Bid")
    plt.ylabel("Aggregated Revenue")
    plt.title("Optimal Price: 22.5")
    plt.show()

    fake_rev = lambda bid: ndc_a_MC(bid) * Cr_a(15.5) * Mr(15.5) * opt_R - ndc_a_MC(bid) * cpc_a_MC(bid)
    bb = np.linspace(min(bids), max(bids), 10000)
    y = []
    c = []
    for i in bb:
        y.append(fake_rev(i))
        if y[-1] >= 0:
            c.append(1)
        else:
            c.append(0)
    colors = ['red', 'blue']
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(bb, y, s=1, c=c, cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(bb, np.repeat([0], len(bb)))
    plt.xlabel("Bid")
    plt.ylabel("Aggregated Revenue")
    plt.title("Suboptimal Price: 15.5")
    plt.show()

# fixed price wrt optimal solution
opt_bid = sol[0][1]
opt_price = sol[0][0]
opt_value = sol[1]
opt_cr = Cr_a(opt_price)
print(opt_bid, opt_price, opt_value)

# Parameters
bids = np.array(bids)
n_arms = len(bids)
T = 365
n_experiments = 100

# daily revenue with fixed price, returns and cr
daily_rev = lambda ndc, cpc: ndc * opt_cr * Mr(opt_price) * opt_R - ndc * cpc

# Set to True in order to simulate model training
Training = False
if Training:
 # data structures
 gts_revenue_per_exp = []
 gpts_revenue_per_exp = []

 # plot or not the estimated revenue function at e==0
 Show_Estimated_Revenue = False

 for e in range(n_experiments):
    env = BiddingEnvironment(bids=bids)
    gts_learner = GTS_Learner(n_arms=n_arms, rev=daily_rev)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids, rev=daily_rev)

    for t in range(T):
        print("Experiment {}, Day {}".format(e,t))
        # GTS
        pulled_arm = gts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gts_learner.update(pulled_arm, reward)

        # GPTS
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        if t>359:
         print(reward)
        gpts_learner.update(pulled_arm, reward)

    if e == 1:
        gts_arms = gts_learner.collected_arms
        gpts_arms = gpts_learner.collected_arms

    if Show_Estimated_Revenue:
      if e == 0:
        plt.figure()
        actual_rev = lambda bid: ndc_a(bid) * Cr_a(sol[0][0]) * Mr(sol[0][0]) * opt_R - ndc_a_MC(bid) * cpc_a_MC(bid)
        bb = np.linspace(min(bids), max(bids), 1000)
        y = []
        for i in bb:
            y.append(actual_rev(i))
        plt.plot(bb, y, 'b')
        plt.scatter(bids, gts_learner.means_rev, c='g')
        plt.scatter(np.array(bids) + 0.01, gpts_learner.means_rev, c='r')
        for i in range(len(bids)):
            plt.vlines(bids[i], gts_learner.means_rev[i] - 2 * gts_learner.sigmas_rev[i],gts_learner.means_rev[i] + 2 * gts_learner.sigmas_rev[i], 'g')
            plt.vlines(bids[i] + 0.01, gpts_learner.means_rev[i] - 2 * gpts_learner.sigmas_rev[i],gpts_learner.means_rev[i] + 2 * gpts_learner.sigmas_rev[i], 'r')
        plt.xlabel('Bid')
        plt.ylabel('Estimated Revenue Function')
        plt.show()

    # Save collected revenues
    gts_revenue_per_exp.append(gts_learner.collected_revenues)
    gpts_revenue_per_exp.append(gpts_learner.collected_revenues)

 # Save Results
 np.save('GTSOnlineBidding_revenue.npy',gts_revenue_per_exp)
 np.save('GPTSOnlineBidding_revenue.npy',gpts_revenue_per_exp)

 ## End Simulations

# Load
gts_revenue_per_exp = np.load('GTSOnlineBidding_revenue.npy', allow_pickle=True)
gpts_revenue_per_exp = np.load('GPTSOnlineBidding_revenue.npy', allow_pickle=True)

# Plots
plt.figure()
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt_value - np.array(gpts_revenue_per_exp), axis=0)), color='darkorange', linewidth=2.5)
plt.plot(np.cumsum(np.mean(opt_value - np.array(gts_revenue_per_exp), axis=0)), color='#0f9b8e', linewidth=2.5)
plt.legend(["GPTS", "GTS"])
plt.title("Cumulative Regret")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
for el in gts_revenue_per_exp:
  ax1.plot(el, color='#0f9b8e', alpha=0.01, label='_nolegend_')
ax1.plot(np.mean(gts_revenue_per_exp, axis=0), color='#005249', label='Mean Revenue')
ax1.plot(np.repeat([opt_value ], T), 'k',  label='Optimal Aggregated')
ax1.set_ylim([0, 700])
ax1.set_xlabel('t')
ax1.set_ylabel('Expected Daily Revenue')
ax1.set_title('GTS')
ax1.legend(loc='best')
ax1.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
for el in gpts_revenue_per_exp:
  ax2.plot(el, color='darkorange', alpha=0.01, label='_nolegend_')
ax2.plot(np.mean(gpts_revenue_per_exp, axis=0),  color='crimson', label='Mean Revenue')
ax2.plot(np.repeat([opt_value], T), 'k',  label='Optimal Aggregated')
ax2.set_ylim([0, 700])
ax2.set_xlabel('t')
# ax2.set_ylabel('Daily Revenue')
ax2.set_title('GPTS')
ax2.legend(loc='best')
ax2.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.show()


