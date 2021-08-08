import numpy as np
import matplotlib.pyplot as plt
from UCB1_learner import *
from Environment import *
from TS_learner import *
from Dati import *
from Data_Exp import *

#NOTE: fixed bid at his optimal value, found at step 1.
#IMPORTANT: regret refers to all money lost during learning phase! ---> Optimal Conversion rate corresponds to
#                                                                       Cr(Pstar)

# Optimal values
from STEP_1 import sol
opt_bid = sol[0][1]
opt_price = sol[0][0]
opt_value = sol[1]
p = aggregate['conversion rate']
opt_cr = Cr_a(opt_price)


if __name__=="__main__":
    print(opt_price)
    print(prices[np.argmax(np.array(p))])


# Cost per click and number of new daily customers are set to mean value at optimal bid, aswell as number of returns
from Data_Exp import cpc_a, ndc_a
opt_cpc = cpc_a(opt_bid)
opt_ndc = ndc_a(opt_bid)
opt_R =1 + (1/30)*np.dot(np.array(aggregate['return probs'][0]),np.array(aggregate['return probs'][1]))

# 10 possible prices
n_arms = 10
arms_value = prices

# time horizon of 1 year
T = 365

# Define Revenue function to be used as a criterionn during learning phase; notice that as we assume to know
# opt_bid, the corresponding values of Ndc and Cpc at that point and Mr(p) for every p, we only need to optimize the
# unknown conversion rates. Considering solely the maximum value of Cr would be misleading!

daily_rev = lambda cr, pulled_arm : opt_ndc*cr*Mr(prices[pulled_arm])*opt_R-opt_ndc*opt_cpc

Test_1 = False
if Test_1:
 # Comparison UCB1 vs TS: One customer per day
 n_experiments = 500
 ts_rewards_per_experiment = []
 ucb1_rewards_per_experiment = []
 ts_revenue_per_experiment = []
 ucb1_revenue_per_experiment = []

 for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_learner(n_arms=n_arms,samples_per_round=1, rev = daily_rev)
    ucb1_learner = UCB1_learner(n_arms=n_arms,samples_per_round = 1, rev = daily_rev)

    for t in range(0, T):
        # TS learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm,1)
        ts_learner.update(pulled_arm, reward)

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm,1)
        ucb1_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    ts_revenue_per_experiment.append(ts_learner.collected_revenues)
    ucb1_revenue_per_experiment.append(ucb1_learner.collected_revenues)


 if __name__=="__main__":
  plt.figure(0)
  plt.xlabel("t")
  plt.ylabel("Regret")
  plt.plot(np.cumsum(np.mean(opt_cr - ts_rewards_per_experiment, axis=0)), 'r')
  plt.plot(np.cumsum(np.mean(opt_cr - ucb1_rewards_per_experiment, axis=0)), 'b')
  plt.legend(["TS", "UCB1"])
  plt.show()

  fig, (ax1, ax2) = plt.subplots(1, 2)
  ax1.plot(ucb1_revenue_per_experiment[1], 'b', alpha=0.1)
  ax1.plot(np.mean(ucb1_revenue_per_experiment, axis=0), 'b')
  ax1.plot(np.repeat([opt_value], T), 'r')
  ax1.set_xlabel('t')
  ax1.set_ylabel('Daily revenue')
  ax1.set_title('UCB1')
  ax2.plot(ts_revenue_per_experiment[1], 'b', alpha=0.1)
  ax2.plot(np.mean(ts_revenue_per_experiment, axis=0), 'b')
  ax2.plot(np.repeat([opt_value], T), 'r')
  ax2.set_xlabel('t')
  ax2.set_ylabel('Daily Revenue')
  ax2.set_title('TS')
  plt.show()


# Comparison UCB1 vs TS: Ndc customers per day
n_experiments = 500
ts_rewards_per_experiment = []
ucb1_rewards_per_experiment = []
ts_revenue_per_experiment = []
ucb1_revenue_per_experiment = []
ts_arms = []
ucb1_arms = []

for e in range(0, n_experiments):
    env = Environment(n_arms=n_arms, probabilities=p)
    ts_learner = TS_learner(n_arms=n_arms,samples_per_round=int(opt_ndc), rev = daily_rev)
    ucb1_learner = UCB1_learner(n_arms=n_arms,samples_per_round = int(opt_ndc), rev = daily_rev)

    for t in range(0, T):
        # TS learner
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm,int(opt_ndc))
        ts_learner.update(pulled_arm, reward)

        # UCB1 learner
        pulled_arm = ucb1_learner.pull_arm()
        reward = env.round(pulled_arm,int(opt_ndc))
        ucb1_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiment.append(ucb1_learner.collected_rewards)
    ts_revenue_per_experiment.append(ts_learner.collected_revenues)
    ucb1_revenue_per_experiment.append(ucb1_learner.collected_revenues)
    if e == 1:
        ts_arms =ts_learner.collected_arms
        ucb1_arms = ucb1_learner.collected_arms


if __name__=="__main__":
 plt.figure(0)
 plt.xlabel("t")
 plt.ylabel("Regret")
 plt.plot(np.cumsum(np.mean(np.abs(opt_cr - ts_rewards_per_experiment), axis=0)), 'r')
 plt.plot(np.cumsum(np.mean(np.abs(opt_cr - ucb1_rewards_per_experiment), axis=0)), 'b')
 plt.legend(["TS", "UCB1"])
 plt.title("Regret wrt Conversion Rate")
 plt.show()

 plt.figure(1)
 plt.xlabel("t")
 plt.ylabel("Regret")
 plt.plot(np.cumsum(np.mean(np.abs(opt_value - ts_revenue_per_experiment), axis=0)), 'r')
 plt.plot(np.cumsum(np.mean(np.abs(opt_value - ucb1_revenue_per_experiment), axis=0)), 'b')
 plt.legend(["TS", "UCB1"])
 plt.title("Regret wrt Revenue Value")
 plt.show()


 fig, (ax1,ax2) = plt.subplots(1, 2)
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
 ax2.set_ylabel('Times')
 ax2.legend([str(it) for it in prices])
 ax2.set_title('TS')
 plt.show()

 fig, (ax1, ax2) = plt.subplots(1, 2)
 ax1.plot(ucb1_revenue_per_experiment[1],'b', alpha = 0.1)
 ax1.plot(np.mean(ucb1_revenue_per_experiment, axis=0),'b')
 ax1.plot(np.repeat([opt_value],T), 'r')
 ax1.set_ylim([0,800])
 ax1.set_xlabel('t')
 ax1.set_ylabel('Expected Daily Revenue')
 ax1.set_title('UCB1')
 ax2.plot(ts_revenue_per_experiment[1],'b', alpha = 0.1)
 ax2.plot(np.mean(ts_revenue_per_experiment, axis=0),'b')
 ax2.plot(np.repeat([opt_value], T), 'r')
 ax2.set_ylim([0, 800])
 ax2.set_xlabel('t')
 ax2.set_ylabel('Daily Revenue')
 ax2.set_title('TS')
 plt.show()



