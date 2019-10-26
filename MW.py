import pandas as pd
import math
import timeit

import numpy as np

from utils import input_check

# The general function to be used as an oracle to MW looks like below
#   def oracle_to_MW(k, k_obj, constraints, weight):
#        '''
#        Given k objectives Obj_1,...,Obj_k and weight as w_1,...,w_k, the oracle will solve the optimization where the objective is 
#        sum_{i=1}^k w_i * Obj_i. The optimization is under the constraints given.
#        '''
    
#however, for specific case of PCA, we can do much better with a specific structure



def weightedPCA(n, k, d, B, weight=None, alpha=None, beta=None, calculate_objective=False):
    '''
    Arguments:
        n: dimension of matrix B
        k: number of B_i
        d: target dimension of the PCA
        weight: vector of k numbers as numpy list specifying weight of PCA to each group to combine. This is all 1/k by default. 
        alpha: additional weight to multiply to <B_i,X>, if any. This is all 1 by default. 
        beta: constant adding to the objective: alpha_i (<B_i,X>) + beta_i.
        
    Task:
    Given the objective 
    sum_{i=1}^k w_i * (alpha_i (<B_i,X>) + beta_i).
    where B_i and X are n-by-n matrices, solve to get the rank-d solution X. This is simply standard PCA on the weighted data
    sum_{i=1}^k w_i alpha_i B_i
    
    Note:
    - The solution is independent of beta. Beta only affects the objective function.
    - It seems redundant to have weight and alpha, as both are just weights multiplied together. 
      The only reason for separating them is to calculate the objective value alpha_i (<B_i,X>) + beta_i 
      which is independent of weight but dependent on alpha.
    
    Output: 
    [X = P P^T , P , Obj_list = [Obj_1, ..., Obj_k]] or [X = P P^T , P]
    - solution X which is n x n matrix of rank d. 
    - P is the n x d matrix of d principle eigenvectors as columns, sorted from ones with biggest eigenvalue first to lower.
    - objective value Obj_i = alpha_i (<B_i,X>) + beta_i
    
    to save time, calculate_objective can be set to false when calculating Obj_i is not needed. This is the default.
    
    '''
    if (input_check(n, k, d, B, function_name='weightedPCA') > 0):
        return -1
    
    #I have to define default value in the function, not at the declaration of funciton, as the defaul value is dependent on k
    if (weight is None):
        weight = np.full(k,1/k)
    
    if (alpha is None):
        alpha = np.full(k,1)
        
    if (beta is None):
        beta = np.zeros(k)
    
    if ( weight.shape != (k,) or alpha.shape != (k,) or beta.shape != (k,) ):
        print("Error: weightedPCA is called with wrong weight or alpha or beta coefficient size. They should be numpy vectors of length k")
        return -1

    #normalization
    weight = weight/weight.sum()
    #alpha_normalized = alpha/alpha.sum() #no need for this
    #Note that I define new alpha because I don't want to edit alpha when I calculate objective value later, which has alpha in the expression
    
    W = np.zeros((n,n))
    for i in range(k):
        W = W + (weight[i] * alpha[i]) * B[i]
    
    [eigenValues,eigenVectors] = np.linalg.eig(W)
    #note: sometimes the numerical problems makes small complex parts, Put np.real to avoid them, as we know W should be PSD.
    eigenValues = eigenValues.real
    eigenVectors= eigenVectors.real

    #sort eigenvalues and eigenvectors in decending orders
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    #take the first d vectors. Obtained the solution
    P = eigenVectors[:,:d]
    X = P @ P.T
    
    if (calculate_objective == False):
        return [X,P]
    else:
        Obj_list = [alpha[i] * np.multiply(B[i],X).sum() + beta[i] for i in range(k)]
        return [X,P,Obj_list]
    
def MW_for_PCA(n,m,d,B, weight=None, alpha=None, beta=None, eta=1, T=10, verbose = False):
    '''
    Arguments:
        n: size of matrices in B
        m: number of objectives
        d: target dimension
        B: list of m matrices
        weight: any initial weight, if user wants to specify. By default, this is set to all 1/m.
        alpha: list of m numbers, used in the objectives By default, this is set to all 1.
        beta: list of m numbers, used in the objectives. By default, this is set to all 0.
        eta: learning rate of MW.
        T: number of iteration
        verbose: will print the objective value in each single iteration. 
                 This may not be needed as the function will output pandas dataframe of all statistics already.
    Note: in theory, for eps < 1 approximation to optimization problem with bounded objective in [0,1],
        eta = eps/8
        T = 32log(m)/eps^2
    analyzed in "The Price of Fair PCA: One Extra Dimension."
        
    Given m objectives to maximize simultanously
        alpha_1 <B_1,X> + beta_1
        alpha_2 <B_2,X> + beta_2
        ...
        alpha_m <B_m,X> + beta_m
    subject to
        tr(X) <= d
        0 << X << I (matrix inequality)
    the function uses MW to maximize the minimum of m objectives.
    
    Output:
    [X_last,X_avg,runstats]
        X_last: n x n matrix X from the last iterate. 
        X_avg: averge over T matrices of n x n matrices X from each of T iterates. 
        runstats: value of weights and objectives in each iterate
    '''
    #input check
    if (input_check(n, m, d, B, function_name='MW_for_PCA') > 0):
        return -1
    
    if (weight is None):
        weight = np.full(m,1/m)
    weight = weight/weight.sum() #Without loss of generality, make sure the sum of weight is 1
        
    if (alpha is None):
        alpha = np.full(m,1)
        
    if (beta is None):
        beta = np.zeros(m)
        
    if ( weight.shape != (m,) or alpha.shape != (m,) or beta.shape != (m,) ):
        print("Error: MW_for_PCA is called with wrong weight or alpha or beta coefficient size. They should be numpy vectors of length m")
        return -1
    
    if ( (eta>0) == False ):
        print("Error: MW_for_PCA is called eta not a positive real number.")
        return -1
    
    run_stats = pd.DataFrame()
    X_avg = np.zeros((n,n))
    
    for t in range(T):
        
        [X,_,Obj] = weightedPCA(n, m, d, B, weight=weight, alpha=alpha, beta=beta, calculate_objective=True)
        #update the average solution of X. In MW, the average is guaranteed to converge, not the last iterate, at least in theory
        X_avg = t*(X_avg/(t+1)) + X/(t+1)
        
        #this stats below keeps the weight and objective value of this iterate
        this_t_stats = {'iteration':t}
        this_t_stats.update(dict(('weight'+str(i),weight[i]) for i in range(m)))
        this_t_stats.update({'minimum of m objective, that iterate':min(Obj)})
        this_t_stats.update({'minimum of m objective, avg iterate':min([alpha[i] * np.multiply(B[i],X_avg).sum() + beta[i] for i in range(m)])})
        this_t_stats.update(dict(('Obj'+str(i),Obj[i]) for i in range(m)))
        if (verbose):
            print("stats at iteration " + str(t) + " is :")
            print(this_t_stats)
        run_stats = run_stats.append(pd.DataFrame(this_t_stats,index=[t]))
        
        #now the update of the weight
        Loss = np.multiply(-1,Obj)
        for i in range(m):
            weight[i] = math.exp(eta * Loss[i]) * weight[i]
        weight = weight/weight.sum()
        
    return [X, X_avg, run_stats]

def fairDimReduction_MW(n,k,d,B,Obj='MM_Loss',eta=1, T=10, verbose = False, timed = True):
    '''
    Arguments:
        n: size of matrices in B
        k: number of objectives
        d: target dimension
        B: list of k matrices
        Obj:the objective to optimize. Must be MM_Var (maximize the minimum variance) or MM_Lose (default) (minimize the maximum loss, output the negative number variance - best)
        eta: learning rate of MW.
        T: number of iteration
        verbose: will print the objective value in each single iteration of MW. 
                 This may not be needed as the function will output pandas dataframe of all statistics already. 
        timed: will print amount of time used by this method in total, in seconds.
    Output:
    [X_last,X_avg,runstats]
        X_last: n x n matrix X from the last iterate. 
        X_avg: averge over T matrices of n x n matrices X from each of T iterates. 
        runstats: value of weights and objectives in each iterate
    '''
    
    #we just take Obj and convert that into alpha-beta notation for MW for PCA
    if (Obj=='MM_Loss'):
        #best possible PCA projection for each group is easy to calculate: take d best eigenvalues
        best = [np.sum(np.sort(np.linalg.eigvalsh(B[i]))[-d:]) for i in range(k)]
        
        beta=np.multiply(-1,best)
    elif (Obj=='MM_Var'):
        beta=np.zeros(k)
    else:
        print("Error:fairDimReduction_MW is called with invalid input objective.")
        return -1;
    
    start = timeit.default_timer()
    [X_last, X_avg, Obj] = MW_for_PCA(n,k,d,B, weight=None, alpha=None, beta=beta, eta=eta, T=T, verbose = verbose)
    stop = timeit.default_timer()
    if (timed):
        print("fairDimReduction_MW is called. Total time used is: ", stop-start, " seconds.")
    return [X_last, X_avg, Obj]