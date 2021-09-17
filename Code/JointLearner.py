import numpy as np
from DataParameters import prices,bids
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
import warnings


class JointLearner:
    def __init__(self, n_prices,prices,n_bids,bids,context, rev):
        # General
        self.n_prices = n_prices
        self.prices = prices
        self.n_bids = n_bids
        self.bids = bids
        self.t = 0
        self.rev = rev
        # Pricing
        self.context = context
        self.mean_returns_estimates = [0.0 for _ in range(context.structure)]
        self.beta_parameters = [np.ones([n_prices, 2]) for _ in range(self.context.structure)]
        # Bidding
        self.means_rev = np.zeros(self.n_bids)
        self.sigmas_rev = np.ones(self.n_bids) * 50
        self.ndc_actualEstimates = [[] for _ in range(self.n_bids)]
        self.cpc_actualEstimates = [[] for _ in range(self.n_bids)]
        self.weightsEstimates = np.zeros(self.context.structure)
        self.treshold = 0.2
        self.pulled_bids = []
        alpha = 0.75
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=.1, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-4))
        self.gp_rev = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,n_restarts_optimizer=9)
        # Collected
        self.collected_revenues = np.array([])

    def updateMeanReturns(self, newEstimates):
        self.mean_returns_estimates = newEstimates

    def pull_prices(self, chosenBid):
        if self.t < self.n_prices:
            return [prices[self.t] for _ in range(self.context.structure)]
        else:
            ret = []
            for c in range(self.context.structure):
                idx = np.zeros(10)
                for i, x in enumerate(self.beta_parameters[c]):
                    idx[i] = self.rev(np.random.beta(x[0], x[1]),i,self.mean_returns_estimates[c], np.mean(self.ndc_actualEstimates[chosenBid])*(self.weightsEstimates[c]/np.sum(self.weightsEstimates)), np.mean(self.cpc_actualEstimates[chosenBid]))
                ret.append(prices[np.argmax(idx)])
        return ret

    def pull_bid(self):
        if self.t < self.n_bids:
            return self.t
        else:
            candidates = self.eligible_bids()
            vals = []
            for i in candidates:
                vals.append(np.random.normal(self.means_rev[i], self.sigmas_rev[i]))
            return candidates[np.argmax(vals)]

    def eligible_bids(self):
        ret = []
        for i in range(0,self.n_bids):
            if norm.cdf(0,loc=self.means_rev[i],scale=self.sigmas_rev[i]) < self.treshold:
                    ret.append(i)
        return ret

    def update_observations(self, pulled_prices,pulled_bid, acceptances, n_cust, cost):
        today_expected_revenue = 0.0
        for i in range(self.context.structure):
            if n_cust[i] != 0:
                today_expected_revenue += self.rev(acceptances[i] / n_cust[i], pulled_prices[i],self.mean_returns_estimates[i],n_cust[i], cost)
        self.collected_revenues = np.append(self.collected_revenues, today_expected_revenue)
        self.pulled_bids.append(self.bids[pulled_bid])

    def update(self, pulled_prices, pulled_bid, acceptances, n_cust, cost):
        self.t += 1
        self.update_observations(pulled_prices, pulled_bid, acceptances, n_cust, cost)
        for c in range(0,self.context.structure):
            self.beta_parameters[c][pulled_prices[c], 0] += acceptances[c]
            self.beta_parameters[c][pulled_prices[c], 1] += n_cust[c] - acceptances[c]
        self.update_gp()
        self.ndc_actualEstimates[pulled_bid].append(np.sum(n_cust))
        self.cpc_actualEstimates[pulled_bid].append(cost)
        for i in range(len(n_cust)):
            self.weightsEstimates[i]+= n_cust[i]

    def update_gp(self):
        x = np.atleast_2d(self.pulled_bids).T
        y = self.collected_revenues
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)
        self.gp_rev.fit(x, y)
        self.means_rev, self.sigmas_rev = self.gp_rev.predict(np.atleast_2d(self.bids).T, return_std=True)
        self.sigmas_rev = np.maximum(self.sigmas_rev, 1e-2)


#   NOT USED

#   def updateContext(self, context):
#        ret = False
#        if self.context.getRuleName() != context.getRuleName():
#           ret = True
#           self.context = context
#        return ret
#
#    def offlineTraining(self, database):
#        self.beta_parameters = [np.ones([self.n_arms, 2]) for _ in range(self.context.structure)]
#        for day in range(0, len(database.Database)):
#            for customer in database.Database[day]:
#                self.beta_parameters[self.context.evaluate(customer.feature)][prices.index(database.DailySpecifics[day][0][database.contextHistory[day // 7].evaluate(customer.feature)]), 0] += customer.accepted
#                self.beta_parameters[self.context.evaluate(customer.feature)][prices.index(database.DailySpecifics[day][0][database.contextHistory[day // 7].evaluate(customer.feature)]), 1] += 1.0 - customer.accepted
