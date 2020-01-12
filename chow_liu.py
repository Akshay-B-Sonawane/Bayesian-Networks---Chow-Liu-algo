# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 03:55:16 2019

@author: aks18596
"""
import sys
import pandas as pd
import numpy as np




#prob_x_1 =(dataset[dataset == 1].count(axis = 0)+2)/(len(dataset)+4)
#prob_x_0 = 1-prob_x_1
#
#
#
#M_info = np.zeros((len(dataset.columns),len(dataset.columns)))
#
#from sklearn.metrics.cluster import mutual_info_score
#for i in dataset.columns:
#    print(i)
#    for j in dataset.columns:
#        
#        M_info[i][j] = mutual_info_score(dataset[i].values, dataset[j].values)
#        
#        
#from scipy.sparse import csr_matrix, find
#from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
#
#X = csr_matrix(M_info)
#Tcsr = -minimum_spanning_tree(-X)
#Array1 = Tcsr.toarray().astype(float)
#
#
##Y = csr_matrix(A)
#Tcsr_depth = depth_first_tree(Array1, 1, directed = False)
#Array2 = Tcsr_depth.toarray().astype(float)
#
#
#really = np.column_stack(((find(Array2))[0], (find(Array2))[1]))
#




#row = dataset.iloc[:,[really[0][0], really[0][1]]].header(None)




def check(X,i,j):
    count = 0
    if(X[0]==i and X[1]==j):
        count+=1
    return count

def forRoot(row,p_1,p_0,root):
    ans = 0
    if row[root] == 1:
        ans += np.log2(p_1[root])
    else:
        ans += np.log2(p_0[root])
    return ans



def gettingRow(X, dataset, p_1, p_0, test, prediction):
    table = dataset.iloc[:,[X[0],X[1]]]
    
    CPD = np.zeros((2,2))
    
    CPD[0][0] = (np.apply_along_axis(check, 1, table, 0, 0).sum()+2)
    CPD[0][0] = CPD[0][0]/(len(dataset)+4)
    CPD[0][1] = (np.apply_along_axis(check, 1, table, 0, 1).sum()+2)
    CPD[0][1] = CPD[0][1]/(len(dataset)+4)
    CPD[1][0] = (np.apply_along_axis(check, 1, table, 1, 0).sum()+2)
    CPD[1][0] = CPD[1][0]/(len(dataset)+4)
    CPD[1][1] = (np.apply_along_axis(check, 1, table, 1, 1).sum()+2)
    CPD[1][1] = CPD[1][1]/(len(dataset)+4)
    
    
    for i in range(len(test)):
        if(test.iloc[i,X[0]] == 0 and test.iloc[i,X[1]] == 0):
            prediction[i] += np.log2(CPD[0][0]/(p_0[X[0]]))
        elif(test.iloc[i,X[0]] == 0 and test.iloc[i,X[1]] == 1):
            prediction[i] += np.log2(CPD[0][1]/(p_0[X[0]]))
        elif(test.iloc[i,X[0]] == 1 and test.iloc[i,X[1]] == 0):
            prediction[i] += np.log2(CPD[1][0]/(p_1[X[0]]))
        elif(test.iloc[i,X[0]] == 1 and test.iloc[i,X[1]] == 1):
            prediction[i] += np.log2(CPD[1][1]/(p_1[X[0]]))

    


#    print()
#    print("CPD for("+str(X[0])+" "+str(X[1])+")is:")
#    print()
#    print(CPD)
    return CPD


#pred = np.zeros(len(test_data))
#CPD = np.apply_along_axis(gettingRow, 1, really, dataset = dataset, p_1 = prob_x_1, p_0 = prob_x_0, test = test_data, prediction = pred)
#
#root = really[0][0]
##data = test_data.iloc[:,root]
#
#Temp = np.apply_along_axis(forRoot, 1,test_data, p_1 = prob_x_1, p_0 = prob_x_0, root = root)
#
#pred = pred + Temp
#
#
#print(pred.sum()/len(test_data



if __name__ == "__main__":
    file = sys.argv[1]
        
    dataset = pd.read_csv(file+".ts.data", header= None)
    test_data = pd.read_csv(file+".test.data", header= None)
       
    prob_x_1 =(dataset[dataset == 1].count(axis = 0)+2)/(len(dataset)+4)
    prob_x_0 = 1-prob_x_1
    
    M_info = np.zeros((len(dataset.columns),len(dataset.columns)))

    from sklearn.metrics.cluster import mutual_info_score
    for i in dataset.columns:
        print(i)
        for j in dataset.columns:
            
            M_info[i][j] = mutual_info_score(dataset[i].values, dataset[j].values)
            
            
    from scipy.sparse import csr_matrix, find
    from scipy.sparse.csgraph import minimum_spanning_tree, depth_first_tree
    
    X = csr_matrix(M_info)
    Tcsr = -minimum_spanning_tree(-X)
    print(Tcsr)
    Array1 = Tcsr.toarray().astype(float)
    
    
    #Y = csr_matrix(A)
    Tcsr_depth = depth_first_tree(Array1, 1, directed = False)
    print(Tcsr_depth)
    Array2 = Tcsr_depth.toarray().astype(float)
    
    
    really = np.column_stack(((find(Array2))[0], (find(Array2))[1]))
    
    pred = np.zeros(len(test_data))
    CPD = np.apply_along_axis(gettingRow, 1, really, dataset = dataset, p_1 = prob_x_1, p_0 = prob_x_0, test = test_data, prediction = pred)
    
    root = really[0][0]
    #data = test_data.iloc[:,root]
    
    Temp = np.apply_along_axis(forRoot, 1,test_data, p_1 = prob_x_1, p_0 = prob_x_0, root = root)
    
    pred = pred + Temp
    
    
    print(pred.sum()/len(test_data))
       
    