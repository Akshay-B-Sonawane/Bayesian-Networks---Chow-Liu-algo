import sys
import numpy as np
import pandas as pd


#dataset=np.loadtxt("accidents.ts.data",delimiter=',')
#test_data=np.loadtxt("accidents.test.data",delimiter=',')



#thetas_T = (dataset.sum(axis = 0) + 1)/(dataset.shape[0] + 2)
#theta_F = 1 - thetas_T


#Pr = np.ones(len(test_data))
#row = dataset[0]


def probabilities(row, t_T, t_F):
    res = 0.0
    for i in range(row.shape[0]):
        if row[i] == 1:
            res += np.log2(t_T[i])
        else:
            res += np.log2(t_F[i])
    return res


#pr = np.apply_along_axis(probabilities, 1, test_data, t_T = thetas_T, t_F = theta_F)
#
#print(pr.sum()/len(test_data))





if __name__ == "__main__":
    file = sys.argv[1]
    
    dataset=np.loadtxt(file+".ts.data",delimiter=',')
    test_data=np.loadtxt(file+".test.data",delimiter=',')
    
    thetas_T = (dataset.sum(axis = 0) + 1)/(dataset.shape[0] + 2)
    theta_F = 1 - thetas_T
    
    Pr = np.ones(len(test_data))
    row = dataset[0]
    
    pr = np.apply_along_axis(probabilities, 1, test_data, t_T = thetas_T, t_F = theta_F)

    print(pr.sum()/len(test_data))
    