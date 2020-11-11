import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cosangel(vec1, vec2):
    product = np.dot(vec1.T, vec2)
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)
    angle = (product/(vec1_norm*vec2_norm))
    degree = (np.arccos(angle)/np.pi)*180
    return degree

def get_counts(sequence, n):
    counts={}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    for i in counts:
        counts[i] = counts[i]/n
    return counts

def PMF(n, d):
    pairs = np.random.choice([-1, 1], (n, 2, d))
    cos_angle = []
    for i in range(n):
        angle = cosangel(pairs[i][0], pairs[i][1])
        cos_angle.append(angle)
    dict1 = get_counts(cos_angle, n)
    plt.figure(figsize=(12, 5))
    plt.subplots(1, 1)
    sns.distplot(cos_angle, rug=True)
    print('The minimum degree is {}'.format(min(cos_angle)))
    print('The maximum degree is {}'.format(max(cos_angle)))
    print('The value range is {}'.format(max(cos_angle) - min(cos_angle)))
    print('The mean is {}'.format(np.mean(cos_angle)))
    print('The variance is {}'.format(np.var(cos_angle)))
    plt.title('PMF of degrees between two diagonals in high dimension ' + str(d), size=10)
    plt.xlabel('Degree of two half-diagnoal', size=15)
    plt.ylabel('Probability of degrees', size=15)
    plt.show()


PMF(100000, 10)
PMF(100000, 100)
PMF(100000, 1000)

"""As dimension d become infinty, the angle between two half diagonals that randomly draw from a 
d-dimensional hypercube will become 90 degree, which is orthogonal in other words.
The PMF do conform to this trend. First of all, the mean of the PMF is nearly 90 degrees 
and the variance of the variable, which is our degree, will become smaller as the dimension d goes up. 
I think it will become 0 as d goes up to infinity. And the distribution of degrees 
looks like a normal distribution which mean equals to 90 degrees and variance keep shrinking as d goes up."""
