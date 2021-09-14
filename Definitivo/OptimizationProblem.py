# Import packages
import matplotlib.pyplot as plt
import numpy as np
# Import Margin
from DataGenerativeFunctions import Mr
# Import Conversion Rate; Return probs; Click per day; Cost per click;
from DataGenerativeFunctions import Cr_1,Cr_2,Cr_3,Cr_a
from DataGenerativeFunctions import R_1,R_2,R_3,R_a
from DataGenerativeFunctions import Cpc_1,Cpc_2,Cpc_3,Cpc_a
from DataGenerativeFunctions import Ndc_1,Ndc_2,Ndc_3,Ndc_a
# Import Functions representing mean of stochastic functions
from DataGenerativeFunctions import cpc_1,cpc_2, cpc_3
from DataGenerativeFunctions import ndc_1, ndc_2, ndc_3
from MC_Estimation_Stochastic_Aggregate_Distributions import ndc_a_MC, cpc_a_MC
# Import Candidates
from DataParameters import *

# Check Generative Functions
__check__ = False
if __check__ == True:
   print("Generating Data Functions Check\n")

   bid_test=0.8
   price_test=23
   print("Example price_test is {}\n".format(price_test))
   print("Example bid_test is {}\n".format(bid_test))

   print("Conversion Rate: \n")
   print("CR1 is {}\n".format(Cr_1(price_test)))
   print("CR2 is {}\n".format(Cr_2(price_test)))
   print("CR3 is {}\n".format(Cr_3(price_test)))
   print("Agrregate CR is {}\n".format(Cr_a(price_test)))

   print("Customers Replenish Profile: \n")
   print("Sample from R1 is {}\n".format(R_1(1)))
   print("Sample from R1 is {}\n".format(R_2(1)))
   print("Sample from R1 {}\n".format(R_3(1)))
   print("Sample from R1 {}\n".format(R_a(1)))

   print("Cost per Click Functions: \n")
   print("CPC1 is {}\n".format(Cpc_a(bid_test)))
   print("CPC2 is {}\n".format(Cpc_a(bid_test)))
   print("CPC3 is {}\n".format(Cpc_a(bid_test)))
   print("Agrregate CPC is {}\n".format(Cpc_a(bid_test)))

   print("Daily Nmber of CLicks: \n")
   print("N1 is {}\n".format(Ndc_a(bid_test)))
   print("N2 is {}\n".format(Ndc_a(bid_test)))
   print("N3 is {}\n".format(Ndc_a(bid_test)))
   print("Agrregate N is {}\n".format(Ndc_a(bid_test)))


################################################ BRUTE FORCE SOLUTION ####################################################
## Define Expected Revenue Functions

def Expected_Revenue_Class1(price, bid):
      return (ndc_1(bid)*Cr_1(price)*Mr(price)*(1 + (1/30)*np.dot(np.array(C1['return probs'][0]),np.array(C1['return probs'][1]))) \
               - ndc_1(bid)*cpc_1(bid))

def Expected_Revenue_Class2(price, bid):
   return (ndc_2(bid) * Cr_2(price) * Mr(price) * (1 + (1/30)*np.dot(np.array(C2['return probs'][0]),np.array(C2['return probs'][1]))) \
          - ndc_2(bid) * cpc_2(bid))

def Expected_Revenue_Class3(price, bid):
      return (ndc_3(bid) * Cr_3(price) * Mr(price) * (
                 1 + (1/30)*np.dot(np.array(C3['return probs'][0]),np.array(C3['return probs'][1]))) \
             - ndc_3(bid) * cpc_3(bid))

def Expected_Revenue_Aggregated(price, bid):
   return ndc_a_MC(bid) * Cr_a(price) * Mr(price) * (1 + (1/30)*np.dot(np.array(aggregate['return probs'][0]),np.array(aggregate['return probs'][1]))) \
          - ndc_a_MC(bid) * cpc_a_MC(bid)


# Define Brute Force Maximizer
from itertools import product

def Brute_Force_Maximizer(prices, bids, rev):
   grid = list(product(prices, bids))
   evals = [rev(item[1][0], item[1][1]) for item in enumerate(grid)]
   val = np.max(evals)
   idx = np.argmax(evals)
   return [[prices[idx//10],bids[idx%10]], val]

# Brute Force Solutions
sol = (Brute_Force_Maximizer(prices,bids, Expected_Revenue_Aggregated))
sol_class1 = (Brute_Force_Maximizer(prices,bids, Expected_Revenue_Class1))
sol_class2 = (Brute_Force_Maximizer(prices,bids, Expected_Revenue_Class2))
sol_class3 = (Brute_Force_Maximizer(prices,bids, Expected_Revenue_Class3))

if __name__ == "__main__":
    print(" Aggregated: Maximum expected revenue of {} in {}".format(np.around(sol[1],2),sol[0]))
    print(" Class1: Maximum expected revenue of {} in {}".format(np.around(sol_class1[1],2),sol_class1[0]))
    print(" Class2: Maximum expected revenue of {} in {}".format(np.around(sol_class2[1],2),sol_class2[0]))
    print(" Class3: Maximum expected revenue of {} in {}".format(np.around(sol_class3[1],2),sol_class3[0]))

# Revenue surfaces
if __name__ == "__main__":
    show_plots = True

    if show_plots :
       from mpl_toolkits.mplot3d import Axes3D
       from matplotlib import cm
       from matplotlib.ticker import LinearLocator, FormatStrFormatter
       import matplotlib.pyplot as plt
       import numpy as np
       from array import *

       #   Aggregated
       pp = np.linspace(15,25,100)
       bb = np.linspace(0.5, 2, 100)
       PP, BB = np.meshgrid(pp, bb)
       Z = [[Expected_Revenue_Aggregated(PP[i][j], BB[i][j]) for i in range(0, len(PP))] for j in range(0, len(PP[0]))]
       Z = np.array(Z).reshape(len(PP),len(PP))
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.scatter(sol[0][0], sol[0][1], sol[1],s=100,marker="*", color='g')
       ax.legend(["Optimum"])
       surf = ax.plot_surface(PP, BB, Z.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                         linewidth=0, antialiased=False, alpha=.4)
       ax.set_xlabel("Price")
       ax.set_ylabel("Bid")
       ax.set_zlabel("Revenue")
       ax.set_title("Revenue Surface")
       fig.colorbar(surf, shrink=0.5, aspect=5)
       plt.show()

       #   Class1
       pp = np.linspace(15, 25, 100)
       bb = np.linspace(0.5, 2, 100)
       PP, BB = np.meshgrid(pp, bb)
       Z = [[Expected_Revenue_Class1(PP[i][j], BB[i][j]) for i in range(0, len(PP))] for j in range(0, len(PP[0]))]
       Z = np.array(Z).reshape(len(PP), len(PP))
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.scatter(sol_class1[0][0], sol_class1[0][1], sol_class1[1], s=100, marker="*", color='g')
       ax.legend(["Optimum"])
       surf = ax.plot_surface(PP, BB, Z.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False, alpha=.4)
       ax.set_xlabel("Price")
       ax.set_ylabel("Bid")
       ax.set_zlabel("Revenue")
       ax.set_title("Revenue Surface")
       fig.colorbar(surf, shrink=0.5, aspect=5)
       plt.show()

       #   Class2
       pp = np.linspace(15, 25, 100)
       bb = np.linspace(0.5, 2, 100)
       PP, BB = np.meshgrid(pp, bb)
       Z = [[Expected_Revenue_Class2(PP[i][j], BB[i][j]) for i in range(0, len(PP))] for j in range(0, len(PP[0]))]
       Z = np.array(Z).reshape(len(PP), len(PP))
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.scatter(sol_class2[0][0], sol_class2[0][1], sol_class2[1], s=100, marker="*", color='g')
       ax.legend(["Optimum"])
       surf = ax.plot_surface(PP, BB, Z.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False, alpha=.4)
       ax.set_xlabel("Price")
       ax.set_ylabel("Bid")
       ax.set_zlabel("Revenue")
       ax.set_title("Revenue Surface")
       fig.colorbar(surf, shrink=0.5, aspect=5)
       plt.show()

       #   Class3
       pp = np.linspace(15, 25, 100)
       bb = np.linspace(0.5, 2, 100)
       PP, BB = np.meshgrid(pp, bb)
       Z = [[Expected_Revenue_Class3(PP[i][j], BB[i][j]) for i in range(0, len(PP))] for j in range(0, len(PP[0]))]
       Z = np.array(Z).reshape(len(PP), len(PP))
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.scatter(sol_class3[0][0], sol_class3[0][1], sol_class3[1], s=100, marker="*", color='g')
       ax.legend(["Optimum"])
       surf = ax.plot_surface(PP, BB, Z.T, rstride=1, cstride=1, cmap=cm.coolwarm,
                              linewidth=0, antialiased=False, alpha=.4)
       ax.set_xlabel("Price")
       ax.set_ylabel("Bid")
       ax.set_zlabel("Revenue")
       ax.set_title("Revenue Surface")
       fig.colorbar(surf, shrink=0.5, aspect=5)
       plt.show()


