# Import packages
import numpy as np
import matplotlib.pyplot as plt
# Import Learners
from UCB1_learner import *
from TS_learner import *
# Import Environment
from Environment import *
# Import Parameters & Functions
from DataParameters import *
from DataGenerativeFunctions import *
from MC_Estimation_Stochastic_Aggregate_Distributions import ndc_a_MC, cpc_a_MC

# Optimal values
from OptimizationProblem import sol
opt_bid = sol[0][1]
opt_price = sol[0][0]
opt_value = sol[1]
p = aggregate['conversion rate']
opt_cr = Cr_a(opt_price)

# Cost per click and number of new daily customers are set to mean value at optimal bid
# Number of Returns is supposed to be known, as it does not influence the online problem when considering aggregated data;
# its value is set to the optimal one, but in principle we can estimate it ( see Contextual Online Pricing using Naive Context)

#opt_cpc = cpc_a_MC(opt_bid)
#opt_ndc = ndc_a_MC(opt_bid)
opt_cpc_vec = np.load('cpc_agg_opt.npy', allow_pickle=True)
opt_ndc_vec = np.load('ndc_agg_opt.npy',allow_pickle=True)
opt_cpc = opt_cpc_vec[bids.index(opt_bid)]
opt_ndc = opt_ndc_vec[bids.index(opt_bid)]
opt_R =1 + (1/30)*np.dot(np.array(aggregate['return probs'][0]),np.array(aggregate['return probs'][1]))

# Price candidates
n_arms = 10
arms_value = prices

# Time horizon
T = 365

# Expected Daily Revenue Function
daily_rev = lambda cr, pulled_arm : opt_ndc*cr*Mr(prices[pulled_arm])*opt_R-opt_ndc*opt_cpc

# Number of experiments
n_experiments = 100

# Data structures
ts_rewards_per_experiment = []
ucb1_rewards_per_experiment = []
ts_revenue_per_experiment = []
ucb1_revenue_per_experiment = []
ts_arms = []
ucb1_arms = []

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_learner(n_arms=n_arms, rev = daily_rev)
    ucb1_learner = UCB1_learner(n_arms=n_arms, rev = daily_rev)

    for t in range(0, T):
        n_cust = Ndc_a([opt_bid])
        # TS learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm,n_cust)
        ts_learner.update(pulled_arm, reward, n_cust)

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm,n_cust)
        ucb1_learner.update(pulled_arm,reward, n_cust)

    #ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    #ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    ts_revenue_per_experiment.append(ts_learner.collected_revenues)
    ucb1_revenue_per_experiment.append(ucb1_learner.collected_revenues)
    if e == 1:
        ts_arms =ts_learner.collected_arms
        ucb1_arms = ucb1_learner.collected_arms


if __name__=="__main__":
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(np.mean(opt_value - np.array(ts_revenue_per_experiment), axis=0)), color='darkorange',
             linewidth=2.5)
    plt.plot(np.cumsum(np.mean(opt_value - np.array(ucb1_revenue_per_experiment), axis=0)), color='#0f9b8e',
             linewidth=2.5)
    plt.legend(["TS", "UCB1"])
    plt.title("Cumulative Regret")
    plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[0, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[1, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[2, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[3, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[4, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[5, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[6, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[7, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[8, :]))
    ax1.plot(np.cumsum(np.array(ucb1_arms).reshape(10, T)[9, :]))
    ax1.set_xlabel('t')
    ax1.set_ylabel('Times')
    ax1.legend([str(it) for it in prices])
    ax1.set_title('UCB1')
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[0, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[1, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[2, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[3, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[4, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[5, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[6, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[7, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[8, :]))
    ax2.plot(np.cumsum(np.array(ts_arms).reshape(10, T)[9, :]))
    ax2.set_xlabel('t')
    # ax2.set_ylabel('Times')
    ax2.legend([str(it) for it in prices])
    ax2.set_title('TS')
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for el in ucb1_revenue_per_experiment:
        ax1.plot(el, color='#0f9b8e', alpha=0.01, label='_nolegend_')
    ax1.plot(np.mean(ucb1_revenue_per_experiment, axis=0), color='#005249', label='Mean Revenue')
    ax1.plot(np.repeat([opt_value], T), 'k', label='Optimal Aggregated')
    ax1.set_ylim([0, 800])
    ax1.set_xlabel('t')
    ax1.set_ylabel('Expected Daily Revenue')
    ax1.set_title('UCB1')
    ax1.legend(loc='best')
    ax1.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
    for el in ts_revenue_per_experiment:
        ax2.plot(el, color='darkorange', alpha=0.01, label='_nolegend_')
    ax2.plot(np.mean(ts_revenue_per_experiment, axis=0), color='crimson', label='Mean Revenue')
    ax2.plot(np.repeat([opt_value], T), 'k', label='Optimal Aggregated')
    ax2.set_ylim([0, 800])
    ax2.set_xlabel('t')
    # ax2.set_ylabel('Expected Daily Revenue')
    ax2.set_title('TS')
    ax2.legend(loc='best')
    ax2.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
    plt.show()

