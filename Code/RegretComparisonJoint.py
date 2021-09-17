# Import packages
import numpy as np
import matplotlib.pyplot as plt
# Import Solution
from OnlineJointOptimization import solJoint
# Load Data
rev_joint = np.load('JointResults.npy', allow_pickle=True)
rev_cont_joint = np.load('JointContextualResults.npy', allow_pickle=True)

plt.figure()
plt.figure(1)
plt.xlabel("t")
plt.ylabel("")
plt.plot(np.cumsum(np.mean(solJoint[1] - np.array(rev_joint), axis=0)), color='#069af3', linewidth=2.5)
plt.plot(np.cumsum(np.mean(solJoint[1] - np.array(rev_cont_joint), axis=0)), color='darkorange', linewidth=2.5)
plt.title("Cumulative Regret")
plt.grid(linewidth=0.5, color='#9dbcd4', alpha=0.5)
plt.legend(['Aggregated', 'Contextual'])
plt.show()