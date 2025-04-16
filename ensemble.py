import random
import numpy as np
ndim = 2
L_walkers = 5

def stretch(L_walkers, ndim):
    X_matrix = np.zeros((L_walkers, ndim))
    X_matrix[,0] = np.random.uniform(0, 1, (L_walkers, ndim))
    for i in range(L_walkers):
        complimentary_ensemble_indices = [j for j in range(L_walkers) if j != i]
        j = random.choice(complimentary_ensemble_indices)
        Y = X_0[j] + 

X_matrix = np.zeros