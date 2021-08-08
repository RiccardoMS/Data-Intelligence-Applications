import numpy as np
import matplotlib.pyplot as plt
from UCB1_learner import *
from Environment import *
from TS_learner import *
from Dati import *
from Data_Exp import *
from Feature_generator import *
from STEP_1 import *
from STEP_3 import *


# Disaggregated Solutions
print("Optimal price and bid for class 1: {}",format(sol_class1))
print("Optimal price and bid for class 2: {}",format(sol_class2))
print("Optimal price and bid for class 3: {}",format(sol_class3))

# Revenue surfaces
if __name__ == "__main__":
    show_plots = False

    if show_plots :
       pp = np.linspace(15, 25, 100)
       fig, ax = plt.subplots(2, 2)
       # Aggregated
       b_opt = sol[0][1]
       zz = [Expected_Revenue_Aggregated(pp[i],b_opt ) for i in range(0, len(pp))]
       zz = np.array(zz)
       ax[0, 0].plot(pp,zz,'m-')
       ax[0, 0].scatter(sol[0][0], sol[1],s=100,marker="*", color='m')
       ax[0, 0].legend([str(sol[0][1]) + " " + str(np.around(sol[1],2))])
       ax[0, 0].set_ylim([-550, 4000])
       ax[0, 0].set_xlabel("Price")
       ax[0, 0].set_ylabel("Expected Revenue")
       # Class1
       b_opt = sol_class1[0][1]
       zz = [Expected_Revenue_Class1(pp[i], b_opt) for i in range(0, len(pp))]
       zz = np.array(zz)
       ax[0, 1].plot(pp, zz, 'r-')
       ax[0, 1].scatter(sol_class1[0][0], sol_class1[1], s=100, marker="*", color='r')
       ax[0, 1].legend([str(sol_class1[0][1]) + " " + str(np.around(sol_class1[1],2))])
       ax[0, 1].set_ylim([-550, 4000])
       ax[0, 1].set_xlabel("Price")
       ax[0, 1].set_ylabel("Expected Revenue")
       # Class2
       b_opt = sol_class2[0][1]
       zz = [Expected_Revenue_Class2(pp[i], b_opt) for i in range(0, len(pp))]
       zz = np.array(zz)
       ax[1, 0].plot(pp, zz, 'g-')
       ax[1, 0].scatter(sol_class2[0][0], sol_class2[1], s=100, marker="*", color='g')
       ax[1, 0].legend([str(sol_class2[0][1]) + " " + str(np.around(sol_class2[1],2))])
       ax[1, 0].set_ylim([-550, 4000])
       ax[1, 0].set_xlabel("Price")
       ax[1, 0].set_ylabel("Expected Revenue")
       # Class 3
       b_opt = sol_class3[0][1]
       zz = [Expected_Revenue_Class3(pp[i], b_opt) for i in range(0, len(pp))]
       zz = np.array(zz)
       ax[1, 1].plot(pp, zz, 'b-')
       ax[1, 1].scatter(sol_class3[0][0], sol_class3[1], s=100, marker="*", color='b')
       ax[1, 1].legend([str(sol_class3[0][1]) + " " + str(np.around(sol_class3[1],2))])
       ax[1, 1].set_ylim([-550, 4000])
       ax[1, 1].set_xlabel("Price")
       ax[1, 1].set_ylabel("Expected Revenue")
       val = weights[0]*sol_class1[1] + weights[1]*sol_class2[1] + weights[2]*sol_class3[1]
       plt.suptitle("Aggregate: " + str(np.around(sol[1],2))+ "    " + "Disaggregate: " + str(np.around(val,2)) )
       plt.show()


# Context generation every two weeks -> Data collection, online learning
