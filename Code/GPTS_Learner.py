import numpy as np
from BiddingLearner import *
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
import warnings


class GPTS_Learner(BiddingLearner):
    def __init__(self, n_arms, arms, rev):
        super().__init__(n_arms, rev)
        self.arms = arms
        self.means_rev = np.zeros(self.n_arms)
        self.sigmas_rev = np.ones(self.n_arms)* 50
        self.treshold = 0.2
        self.pulled_arms = []
        alpha = 0.75
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=.1, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-4))
        self.gp_rev = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=9)

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_revenues
        warnings.simplefilter(action='ignore', category=ConvergenceWarning)
        self.gp_rev.fit(x, y)
        self.means_rev, self.sigmas_rev = self.gp_rev.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas_rev = np.maximum(self.sigmas_rev, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def eligible_arms(self):
        ret = []
        for i in range(0,self.n_arms):
            if norm.cdf(0,loc=self.means_rev[i],scale=self.sigmas_rev[i]) < self.treshold:
                    ret.append(i)
        return ret

    def pull_arm(self):
        if self.t<10:
            return self.t
        else:
            candidates = self.eligible_arms()
            vals = []
            for i in candidates:
                vals.append(np.random.normal(self.means_rev[i], self.sigmas_rev[i]))
            return candidates[np.argmax(vals)]


