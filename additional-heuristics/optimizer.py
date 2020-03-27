import pandas as pd
import math
import timeit

import numpy as np

from utils import input_check
from standard_PCA import std_PCA

# solve the continuous optimization problem
# g(f_1(X),...,f_k(X))
# for functions g=min,f=var or marginal; or g=product,f=var. 

# Two heuristics:
    # MW: works for maxmin var and marginal
    # FW (Frank-Wolfe) works for all.
#        '''
    
# The main idea is that the linear oracle is just SVD. This allows quick mirror descent on dual (MW) and FW on primal.



def weightedPCA(n, k, d, B, weight=None, alpha=None, beta=None, calculate_objective=False):
    '''
    Arguments:
        n: dimension of (symmetric real) matrix B
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
    - Assume matrices B[i] are all symmetric reals. If not, change 'np.linalg.eigh(W)' to 'np.linalg.eig(W)
    
    Output: 
    [X = P P^T , P , Obj_list = [Obj_1, ..., Obj_k] ],
    or [X = P P^T , P] if calculate_objective = False
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
    
    [eigenValues,eigenVectors] = np.linalg.eigh(W)
    #note: sometimes the numerical problems makes small complex parts, Put np.real to avoid them, as we know W should be PSD. We put linalg.eigh instead of linalg.eig for this, or alternatively do what follow(s):
    #eigenValues = eigenValues.real
    #eigenVectors= eigenVectors.real

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
    
def MW_for_PCA(n,m,d,B, weight=None, alpha=None, beta=None, eta=1, T=10, verbose = False, report_all_obj=False, n_X_last = 0, dual_function = None,stopping_gap=1e-6, primal_function=None, NSW_update=False):
    '''
    Arguments:
        n: size of matrices in B
        m: number of objectives
        d: target dimension
        B: list of m matrices
        weight: any initial weight, if user wants to specify. By default, this is set to all 1/m.
        alpha: list of m numbers, used in the objectives By default, this is set to all 1.
        beta: list of m numbers, used in the objectives. By default, this is set to all 0.
        eta: learning rate of MW or an array of length T. Changing learning rate can be done by specifying the learning rate for each of T iterations as an array of length T
        T: number of iteration
        verbose: will print the objective value in each single iteration. 
                 This may not be needed as the function will output pandas dataframe of all statistics already when report_all_obj=True.
        report_all_obj: if objective and weight on each group will be included in the output dataframe statistics 
        n_X_last = number of X_last I will keep. It will keep n_X_last last iterates' solutions (iterates T-n_X_last up to T) rather than just the last one, if specified. Note that it does not keep any if MW terminates early due to close duality gap - in that case, the last or average iterate would be the choice to use, not the last few iterates.
        dual_function(w,B,X): given weight vector w=[w_1,...,w_m], B=[B_1,...,B_m], specify the dual objective function to calculate. This function can be obtained after knowing the social utility welfare objective (see 'Obj' in fairDimReduction_MW method). By default, it is None, so no dual will be calculated.
        Optionally, to speedup runtime, the dual will also receive the optimum solution to weighted PCA of that iteration. (Can be ignored when specifying the function definition)
        
        stopping_gap: if not None and positive, and if calculate_dual is true, the MW will stop automatically when primal and dual is no more than the gap specified. By default, this is set to 1e-6.
        
        primal_function(B,X): the primal by default is specfied as minimum of alpha_i <B_i,X> + beta_i. One can also specify others, such as for NSW, here. This can be used for comparing the MW on other objectives. 
        
        NSW_update: a different update rule is applied by the calculation of the gradient of weight in dual objective function 
        *** Depreciated. Do not recommend NSW_update due to bad performance on dual space that is not simplex ***
        
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
    or [when n_X_last > 0]
        X_last: list of n_X_last n x n matrices X from the last n_X_last iterates (ordered from (T - n_X_last +1)th iterates till the very last iterate).
    
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
    
    if isinstance(eta,list): #given a list
        if len(eta) < T:
            print('Error: MW_for_PCA is called with list of eta having less than T numbers')
            return -1
       
    elif (eta>0):
        #This is good case. Make eta an array for simplicity of code
        eta = [eta for i in range(T)]
    
    else:
        print("Error: MW_for_PCA is called eta not a positive real number nor a list.")
        return -1
    
    run_stats = pd.DataFrame()
    X_avg = np.zeros((n,n))
    
    if (n_X_last > 0): # I want to keep a few last iterates, not just the last one
        list_X = []
        
    for t in range(T):
        
        [X,_,Obj] = weightedPCA(n, m, d, B, weight=weight, alpha=alpha, beta=beta, calculate_objective=True)
        
        if (n_X_last > 0): # I want to keep this if it is closer to the end
            if (t + n_X_last >= T):
                list_X.append(X)
        
        #update the average solution of X. In MW, the average is guaranteed to converge, not the last iterate, at least in theory
        X_avg = t*(X_avg/(t+1)) + X/(t+1)
        
        #this stats below keeps the weight and objective value of this iterate
        this_t_stats = {'iteration':t}
        if report_all_obj: this_t_stats.update(dict(('weight'+str(i),weight[i]) for i in range(m)))
        this_t_stats.update({'minimum of m objective, that iterate':min(Obj)})
        avg_Obj = min([alpha[i] * np.multiply(B[i],X_avg).sum() + beta[i] for i in range(m)])
        this_t_stats.update({'minimum of m objective, avg iterate':avg_Obj})
        
        #add the primal objective, if specified
        if (primal_function is not None):
            this_t_stats.update({'primal objective, that iterate':primal_function(B,X)})
            this_t_stats.update({'primal objective, avg iterate':primal_function(B,X_avg)})
        
        #add the dual objective
        if (dual_function is not None):
            dual_val = dual_function(weight,B,X)
            
            #dual bound is the best we see so far
            if (t>0): dual_val = min([dual_val,dual_val_previous])
            this_t_stats.update({'dual objective':dual_val})
            
            dual_val_previous = dual_val
        
        #update with the objective
        if report_all_obj: this_t_stats.update(dict(('Obj'+str(i),Obj[i]) for i in range(m)))
        if (verbose):
            print("stats at iteration " + str(t) + " is :")
            print(this_t_stats)
        run_stats = run_stats.append(pd.DataFrame(this_t_stats,index=[t]))
        
        #now the update of the weight
        Loss = np.multiply(-1,Obj)
        
        for i in range(m):
            #gradient of dual will also have -1/w_i term
            if (NSW_update): Loss[i] -= 1/(weight[i])
                
            weight[i] = math.exp(eta[t] * Loss[i]) * weight[i]
            
            #boudn away from 0 to bound gradient norm 1/(weight[i])
            if (NSW_update): weight[i] = min([1e-4,weight[i]])
        if (NSW_update==False):  #normal MW, else no need to do this
            weight = weight/weight.sum()
        
        if ( (dual_function is not None) and (stopping_gap is not None) and (stopping_gap > 0) ): 
            #we have to check if we need to stop 
            # min(Obj) is the minimum of utility of all groups, which is the social welfare in MM_Loss and MM_Var case
            if (abs(dual_val-min(Obj)) < stopping_gap):
                print("MW terminated at T=",t," iterations: current iterate solution achieved primal-dual gap of",stopping_gap)
                break
            
            elif (abs(dual_val-avg_Obj) < stopping_gap):
                print("MW terminated at T=",t," iterations: average iterate solution achieved primal-dual gap of",stopping_gap)
                break
            
       
    if ((n_X_last > 0) and (len(list_X) > 0) ): #return the whole list of last few X's if not empty. Else, this happens when gap is reached earlier than T iterations
        return [list_X, X_avg, run_stats]
    else:
        return [X, X_avg, run_stats]
    
def fairDimReduction_MW(n,k,d,B,Obj='MM_Loss',eta=1, T=10, verbose = False, timed = True, n_X_last = 0, return_time_only = False, calculate_dual = False, eps=1e-9, stopping_gap = 1e-6):
    '''
    Arguments:
        n: size of matrices in B
        k: number of objectives
        d: target dimension
        B: list of k matrices
        Obj:the objective to optimize. Must be MM_Var (maximize the minimum variance) or MM_Lose (default) (minimize the maximum loss, output the negative number variance - best)
        
        *** Obj can also be NSW (Nash social welfare, which is the sum of log of variances across groups), but due to bad performance of MW when the dual space is not simplex, we do not recommend solving NSW by MW. Use Frank-Wolfe for NSW instead ***
        
        eta: learning rate of MW. See changing learning rate by putting eta as an array of T numbers in MW_for_PCA
        T: number of iteration
        verbose: will print the objective value in each single iteration of MW. 
                 This may not be needed as the function will output pandas dataframe of all statistics already. 
        timed: will print amount of time used by this method in total, in seconds.
        n_X_last: the number of X_last I will keep. It will keep n_X_last last iterates' solutions.
        return_time_only: put this to true if one wants to measure the runtime only. It will return only the time in seconds of this method.
        calculate_dual: if this is true, for each weight vector w during the multiplcative weight update, the method calculates the dual objective by w: 
        D(w) := max_{n-by-n matrix X: tr(X)<=d, 0<=X<=I} {sum_{group i} w_i*<B,X>} - f_*(w). 
        Here, f is the objective function (for example, with MM_Var, f(z_1,...,z_k)=min_i {z_i}), and f_* is the concave conjugate of function f_*.
        
    Note on the dual formulation: 1) strong duality holds for concave f. 2) We maximize f, and try to minimize the dual objective D(w) over reals w. 3) Some of the concave congulate functions (denote z_i=<B_i,X>):
        f = MM_Var = min_i {z_i} 
            --> f_*(z) = 0 if w>=0 and w_1+...+w_k = 1; = -\infty otherwise.
        f = MM_Loss = min_i {z_i + beta_i} for beta_i the best variance of group i 
            --> f_*(z) = -sum_i w_i*beta_i if w>=0 and w_1+...+w_k = 1; = -\infty otherwise.
        f = NSW = sum_i {log z_i}
            --> f_*(z) = sum_i {1 + log w_i} for w > 0; -\infty otherwise
            
        eps: numerical error threshold for checking if w satisfies sum w_i = 1 and w>=0
        stopping_gap: if not None and positive, and if calculate_dual is true, the MW will stop automatically when primal and dual is no more than the gap specified. By default, this is set to 1e-6.
    Output:
    [X_last,X_avg,runstats]
        X_last: n x n matrix X from the last iterate. 
        X_avg: averge over T matrices of n x n matrices X from each of T iterates. 
        runstats: value of weights and objectives in each iterate
    if n_X_last > 0:
        X_last: list of n_X_last n x n matrices X from the last n_X_last iterates (ordered from (T - n_X_last +1)th iterates till the very last iterate).
        
    OR
    
    Output:
    runtime
        runtime: time in seconds it takes to run all iterations. (Helpful for checking the running time)
    '''
    
    #input check
    if (input_check(n, k, d, B, function_name='fairDimReduction_MW') > 0):
        return -1
    
    #we just take Obj and convert that into alpha-beta notation for MW for PCA
    if (Obj=='MM_Loss'):
        #best possible PCA projection for each group is easy to calculate: take d best eigenvalues
        best = [np.sum(np.sort(np.linalg.eigvalsh(B[i]))[-d:]) for i in range(k)]
        
        #shift the objective by the best possible PCA for that single group
        beta=np.multiply(-1,best)
        
        #no need to modify anything in MW method
        primal_function = None
        
    elif (Obj=='MM_Var'):
        beta=np.zeros(k)
        
        #no need to modify anything in MW method
        primal_function = None
    
    elif (Obj=='NSW'):
        beta=np.zeros(k)
        
        #modify the objective, since NSW is not the min max form covered by MW method as it is
        def primal_function(B,X):
            utility = 0
            for i in range(len(B)):
                dot_product = np.multiply(B[i],X).sum()
                if (dot_product < 0):
                    print("Warning: the dot product <B[",i,"],X> is not positive. The value is", dot_product)
                    print("Eigenvalues of X is", np.linalg.eig(X)[0])
                    print("Eigenvalues of B[i] is", np.linalg.eig(B[i])[0])
                utility += math.log(dot_product)
            return utility
    else:
        
        print("Error:fairDimReduction_MW is called with an invalid input objective.")
        return -1;
    
    #specify the dual objective, if need to
    dual_function = None
    if (calculate_dual):
        if (Obj=='MM_Var'):
            def dual_function(w,B,X):
                if (abs(np.sum(w)-1) > 1e-9): print("Warning: dual is infeasible with w not summing to 1")
                if (np.amin(w) < 0): print("Warning: dual is infeasible with some w_i < 0")
                    
                weighted_matrix = np.full_like(B[0],0) #create a matrix of same size as first B initially as 0 matrix
                for i in range(len(B)):
                    weighted_matrix += w[i]*B[i]
                return np.sum(np.multiply(weighted_matrix,X)) #dot product <sum {w_i B_i}, X> - f_*(w)
        elif (Obj=='MM_Loss'):
            def dual_function(w,B,X):
                if (abs(np.sum(w)-1) > 1e-9): print("Warning: dual is infeasible with w not summing to 1")
                if (np.amin(w) < 0): print("Warning: dual is infeasible with some w_i < 0")
                    
                weighted_matrix = np.full_like(B[0],0) #create a matrix of same size as first B initially as 0 matrix
                for i in range(len(B)):
                    weighted_matrix += w[i]*B[i]
                return np.sum(np.multiply(weighted_matrix,X)) + np.sum(np.multiply(w,beta)) #dot product <sum {w_i B_i}, X> - f_*(w) and f_*(z) = -sum_i w_i*beta_i             
            
        elif (Obj=='NSW'):
            def dual_function(w,B,X):
                if (np.amin(w) < 0): print("Warning: dual is infeasible with some w_i < 0. The minimum found is",np.amin(w))
                elif (np.amin(w) == 0): return float("inf") #log(0) is -infinity
                
                dual = 0
                for i in range(len(B)):
                    dual += np.multiply(B[i],X).sum() - 1 - math.log(w[i])
                return dual
            
    if (stopping_gap <= 0): stopping_gap=None
    
    start = timeit.default_timer()
    [X_last, X_avg, Obj] = MW_for_PCA(n,k,d,B, weight=None, alpha=None, beta=beta, eta=eta, T=T, verbose = verbose, n_X_last = n_X_last, dual_function=dual_function,stopping_gap=stopping_gap,primal_function=primal_function,NSW_update=(Obj=='NSW'))
    
    stop = timeit.default_timer()
    if (timed):
        print("fairDimReduction_MW is called. Total time used is: ", stop-start, " seconds.")
        
    best_obj = max([Obj['minimum of m objective, that iterate'].max(),Obj['minimum of m objective, avg iterate'].max()])
    best_dual = Obj['dual objective'].min()
    print('The best solution found from avg and single iterate acheieves primal',best_obj,'. The dual is',best_dual,'. Gap is', best_dual-best_obj,'which is',abs((best_dual-best_obj)/best_dual)*100,'%.')
        
    if (return_time_only):
        return stop-start
    
    return [X_last, X_avg, Obj]

def FrankWolfe(init_X, grad_function, linear_oracle, update_rule='1/t', duality_gap=1e-4, num_iterations=None, function=None, print_primal=True):
    """
    Perform Frank-Wolfe algorithms as follows:
    1) Initialize X_0 = init_X
    2) Compute gradient at X_t by grad_function(X_t)
    3) Use linear oracle to obtain S_t = max_{S in feasible region} { <grad_function(X_t),S> }
    4) Update X_{t+1}=(1-eta_t) X_t + eta_t S_t, where eta_i is specified by the update rule (see argument below)
    5) Compute duality gap g_t = <grad_function(X_t),S_t-X_t>
    6) Continue until duality gap is reached. 
        If num_iterations is specifed, then the algorithm will also stop at that number of iterations.
    
    The algorithm returns the last iterates of Frank-Wolfe
    
    Arguments:
    - init_X: starting point X_0
    - grad_function: implement gradient evaluation at point X. Should have the method declaration grad_function(X).
    - linear_oracle: returns S_t = max_{S in feasible region} { <G,S> }. Should have the method declaration linear_oracle(G).
    - update_rule: a string specfying the update rule in each step.
        update_rule = '1/t' --> eta_t=1/(t+2)
        update_rile = '1/2' --> eta_t=1/2
        update_rule = 'line search' --> [not yet implemented] 
    - duality_gap: the algorithm will stop when duality_gap is reached. Specify this gap threshold
    - num_iterations: Put a hard stop at the number of iterations the algorithm can run. If none, the algorithm only stops when duality gap is reach.
    - function: the objective function itself is added for line search update rule
    - print primal: if true, it will also print primal and dual values. Require function not to be none, otherwise it will not print.
    """
    
    X = dict() #for storing X_0, X_1, ...
    #can be memory inefficient, but can modify this later. For now it allows for debugging potentially needed
    X[0] = init_X
    
    t=0
    start_time = timeit.default_timer()
    
    while (True):
        t += 1 #iteration round starts at 1,2,...
        
        #compute gradient
        G = grad_function(X[t-1])
        
        #linear pracle for S_t
        S = linear_oracle(G)
        
        #update rule
        if (update_rule == '1/t'):
            eta = 1.0/(t+1)
        elif (update_rule == '1/2'):
            eta = 1/2
        elif (update_rule == '1/2_every_10'):
            eta = 1/2 ** (math.ceil(t/10))
        elif (update_rule == '1/2_plus_every_10'):
            eta = 1/(1+math.ceil(t/10))
        elif (update_rule == 'line_search'):
            if (function is None):
                print('Error: FrankWolfe is called with update rule for line search, but the objective function is then needed. Return None.')
                return
            eta = line_search(X[t-1],S,function)
        else:
            print("Warning: update_rule specified for Frank-Wolfe not yet implemented. Using 1/t update rule.")
        
        #do the update
        X[t] = (1.0-eta)*X[t-1]+eta*S
        
        #compute duality gap
        dual_gap = np.multiply(G,S-X[t-1]).sum()
        
        #print the state of things
        time_now = timeit.default_timer()
        if print_primal:
            if function is None:
                print("Warning: print_primal requires a primal function to be given.")
                print(f"Iterations t={t}: gap is {dual_gap}. Time taken: {time_now-start_time} seconds.")   
            else:
                primal = function(X[t])
                dual = primal + dual_gap
                print(f"Iterations t={t}: primal is {primal}, dual is {dual}, and gap is {dual_gap} which is {abs((dual-primal)/dual)}%. Time taken: {time_now-start_time} seconds.")
        else:
            print(f"Iterations t={t}: gap is {dual_gap}. Time taken: {time_now-start_time} seconds.")
        
        if (dual_gap <= duality_gap):
            print(f"Duality gap is reached. Frank-Wolfe is terminated at {t} iterations.")
            break
            
        
        if (num_iterations is not None):
            if (t >= num_iterations):
                print(f"Number of iterations is reached. Frank-Wolfe is terminated at {t} iterations.")
                break
                
        #for memory efficient version
        del X[t-1]
                
    return X[t], dual_gap
            
    #version for NSW only
# def FW_NSW(n,k,d,B, delta=1e-4, start_solution='uniform', update_rule='1/t', duality_gap=1e-4, num_iterations=None):
#     """
#     Solve the fair PCA with social welfare objective
#         f(X) := sum_{i=1...k} log (<B_i,X> + delta)
        
#     The algorithm is by Frank-Wolf. Parameters update_rule, duality_gap, num_iterations are parameters of Frank-Wolfe algorithm.
    
#     It returns the last iterate of Frank-Wolfe.
#     Arguments:
#         n: original dimension of the data
#         k: number of groups
#         d: target dimenion
#         B: list of all k groups' data. Must be a list of k n-by-n matrices.
#         delta: buffer in the objective function for its stability and Lipschitzness. 
#         start_solution: options for the starting solution X_0.
#             'uniform' --> X_0 = d/n I_n
#             'standard_PCA' --> [not yet implemented]
#         update_rule, duality_gap, num_iterations: see documentation in FrankWolfe method.
#     """
    
#     #input check
#     if (input_check(n, k, d, B, function_name='FW_NSW') > 0):
#         return -1
    
#     #specify parameters to feed to FrankWolfe algorithm
#     if (start_solution=='uniform'):
#         init_X = d*np.eye(n)/n
#     else:
#         print("Warning: starting X_0 for FW_NSW not yet implemented. Using uniform as starting rule")
#         init_X = d*np.eye(n)/n
        
#     def grad_function(X):
#         #sum of 1/(<B_i,X>+delta)*B[i] is the gradient of f(X)
#         return sum([1/(np.multiply(B[i],X).sum() + delta)*B[i] for i in range(len(B))])
    
#     def linear_oracle(G):
#         P = std_PCA(G,d) #this is n x d matrix of d top sigular values of G
#         return P @ P.T #return an n x n matrix
    
#     #if using line_search, we also need to put the function definition into FrankWolfe as well
#     def NSW_obj(X):
#         #return sum of log of variances
#         return sum([np.log(np.sum(np.multiply(B[i],X))) for i in range(k)])
        
    
#     #perform FrankWolfe
#     final_X, dual_gap = FrankWolfe(init_X, grad_function=grad_function, linear_oracle=linear_oracle, update_rule=update_rule, duality_gap=duality_gap, num_iterations=num_iterations,function=NSW_obj)
    
#     NSW_primal = sum([np.log(np.sum(np.multiply(B[i],final_X))) for i in range(k)])
#     NSW_dual = NSW_primal + dual_gap
    
#     print('primal value (sum of log) is',NSW_primal,'. The (multiplicative) gap of product objective is ',(np.exp(dual_gap)-1)*100,'%.')
        
#     return final_X


def FW(n,k,d,B, Obj='NSW', delta=1e-4, start_solution='uniform', update_rule='1/t', duality_gap=1e-4, num_iterations=None):
    """
    Solve the fair PCA with social welfare objective specified by Obj
        
    The algorithm is by Frank-Wolf. Parameters update_rule, duality_gap, num_iterations are parameters of Frank-Wolfe algorithm.
    
    It returns the last iterate of Frank-Wolfe.
    Arguments:
        n: original dimension of the data
        k: number of groups
        d: target dimenion
        B: list of all k groups' data. Must be a list of k n-by-n matrices.
        delta: buffer in the objective function for its stability and Lipschitzness. 
        start_solution: options for the starting solution X_0.
            'uniform' --> X_0 = d/n I_n
            'standard_PCA' --> [not yet implemented]
        update_rule, duality_gap, num_iterations: see documentation in FrankWolfe method.
    """
    
    #input check
    if (input_check(n, k, d, B, function_name='FW_NSW') > 0):
        return -1
    
    #specify parameters to feed to FrankWolfe algorithm
    if (start_solution=='uniform'):
        init_X = d*np.eye(n)/n
    else:
        print("Warning: starting X_0 for FW_NSW not yet implemented. Using uniform as starting rule")
        init_X = d*np.eye(n)/n
    
    def linear_oracle(G):
        P = std_PCA(G,d) #this is n x d matrix of d top sigular values of G
        return P @ P.T #return an n x n matrix        
    
    #define functions based on objective function f
    if (Obj == 'NSW'):
        #if using line_search, we also need to put the function definition into FrankWolfe as well
        def primal(X):
        #return sum of log of variances
            return sum([np.log(np.sum(np.multiply(B[i],X))) for i in range(k)])
    
        def grad_function(X):
            #sum of 1/(<B_i,X>+delta)*B[i] is the gradient of f(X)
            return sum([1/(np.multiply(B[i],X).sum() + delta)*B[i] for i in range(len(B))])
        
    elif (Obj == 'MM_Var'):
        def primal(X):
            return np.min([np.sum(np.multiply(B[i],X)) for i in range(k)])
        
        def grad_function(X):
            #B[j] of group j with lowest objective
            return B[ np.argmin([np.sum(np.multiply(B[i],X)) for i in range(k)]) ]
        
    elif (Obj == 'MM_Loss'):
        #the best possible variance for each group. Constant independent of X
        best = [np.sum(np.sort(np.linalg.eigvalsh(B[i]))[-d:]) for i in range(k)]
        
        def primal(X):
            return np.min([np.sum(np.multiply(B[i],X)) - best[i] for i in range(k)])
            
        def grad_function(X):
            #B[j] of group j with lowest objective
            
            #return B[ np.argmin([np.sum(np.multiply(B[i],X)) - best[i] for i in range(k)]) ]
            
            #-------------------
            #try softmax
            lam = -50 #should be -log k / eps for error eps. Use negative for softmin. positive for softmax
            w = dict() #weights of groups
            is_0_w = dict() #is 1 if the weight is so small
            for i in range(k):
                #weight of group i
                w[i]=0
                #find the sum in denominator
                for j in range(k):
                    exponent = lam * ( np.sum(np.multiply(B[j]-B[i],X)) - best[j] + best[i] )
                    #if exponent is too low, ignore. If it is too high, this weight (after inverting) is pretty much 0
                    if exponent < -20:
                        continue
                    elif exponent > 20:
                        is_0_w = True
                        break #done with this w_i
                    else:
                        w[i]+=math.exp(exponent)
                if is_0_w:
                    w[i] = 0
                else:
                    w[i] = 1/w[i]
                    
            #for checking
            print('sanity check: sum of w is',sum([w[i] for i in range(k)]),'w is',[w[i] for i in range(k)])
            
            return sum([ w[i]*B[i] for i in range(k)])
    else:
        print('Error: objective for FW is invalid. Return None for FW method.')
        return None
    
    #perform FrankWolfe
    X_final, dual_gap = FrankWolfe(init_X, grad_function=grad_function, linear_oracle=linear_oracle, update_rule=update_rule, duality_gap=duality_gap, num_iterations=num_iterations,function=primal)
    
    primal = primal(X_final)
    dual = primal + dual_gap
    
    if (Obj == 'NSW'):
        print('NSW primal value (sum of log) is',primal,'. The (multiplicative) gap of product objective is ',(np.exp(dual_gap)-1)*100,'%.')
        
    elif (Obj == 'MM_Var') or (Obj == 'MM_Loss'):
        print(Obj,'primal value is',primal,'. Dual is',dual,'. The gap is',dual_gap,', which is',abs((dual-primal)/dual)*100,'%.')        
        
    return X_final
    
def line_search(start, end, f, final_t_error=1e-2):
    """
    Given a concave function f to maximize, use ternary search to find a point in the line segment between start and end that maximizes f along that segment. More generally, it also works for unimodel function f (see Wikipedia).
    Arguments:
    - start: one end of the segment, corresponding to t=0
    - end: another end of the segment, corresponding to t=1
    - f: the function that must accepts f(x) for any convex combination of start and end.
    - final_t_error: the threshold error for final t.
    
    Output:
    - t: a value in [0,1] such that (1-t)*start + t*end maximizes f.
    
    """
    #t at left and right, started at 0 and 1
    t_l = 0
    t_r = 1
    
    def function(t): return f( (1-t)*start + t*end )
    
    while(t_r-t_l > final_t_error):
        t_1 = 2/3*t_l + 1/3*t_r
        t_2 = 1/3*t_l + 2/3*t_r

        if(function(t_1) < function(t_2)):
            #search the right two thirds
            t_l = t_1
        else:
            #search the left two thirds
            t_r = t_2
            
    return (t_l+t_r)/2
    