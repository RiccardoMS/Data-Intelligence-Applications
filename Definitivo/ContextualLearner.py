import numpy as np

class ContextualLearner:

    def __init__(self, n_arms, context, rev):
        self.n_arms = n_arms
        self.t = 0
        self.rev = rev
        self.context = context
        self.mean_returns_estimates = [0.0 for _ in range(context.structure)]
        self.rev = rev
        self.collected_revenues = np.array([])

    def updateMeanReturns(self,newEstimates):
        self.mean_returns_estimates=newEstimates

    def update_observations(self, pulled_arms, rewards, n_cust):
        today_expected_revenue = 0.0
        for i in range(self.context.structure):
            if n_cust[i]!=0:
               today_expected_revenue += self.rev(cr=rewards[i]/n_cust[i],pulled_arm=pulled_arms[i],mean_ret=self.mean_returns_estimates[i],ndc=n_cust[i])
        self.collected_revenues = np.append(self.collected_revenues,today_expected_revenue)
