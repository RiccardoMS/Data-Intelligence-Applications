import numpy as np

# Two Binary Features --> Four possible contexts
# Where: first feature refers to Physical Activity Profile, second feature refers to Diet Profile
c1 = [[0,0]]           #0.3
c2 = [[0,1]]           #0.5
c3 = [[1,0],[1,1]]   #0.2

# Generator
def FeatureGenerator(n_customers):
    elements = [c1,c2,c3[0],c3[1]]
    probabilities = [0.3, 0.5, 0.1,0.1]
    idx = np.random.choice([0,1,2,3], n_customers, p=probabilities)
    ret = []
    for it in idx:
        ret.append(elements[it])
    return ret

# Test Generator
if __name__ == "__main__":
  print(FeatureGenerator(107))
  context = FeatureGenerator(1)
  print(context)
  print([0,1] in context)


# Brute Force Context Structure Generator
def Brute_Force_Generator():
    # load data: each datum is composed by [pulled arm, reward(number of purchase






