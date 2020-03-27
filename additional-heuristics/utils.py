import numpy as np


#use log to avoid the case product can overflow the floating point representation
def geo_mean_through_log(numberList):
    #if some is 0, return 0.
    if (np.amin(numberList) <= 1.e-12):
        return 0;
    
    logNumberList = np.log(numberList)
    return np.exp(logNumberList.sum()/len(numberList))

#data preprocessing helper methods
def scaleVar(dataframe,colArray):
    '''
    Normalize columns "together", meaning that I take the S.D. and mean to be the combined of all those columns.
    This makes more sense if columns are similar in meaning. For example, amount paid for 6 months as 6 months. 
    I normalize all 6 with the same variance.
    Example of usage:
        scaleVar(df.columns.values[2:4]) # scale the 3,4,5th columns together by dividing with the same variance
    '''
    SD = dataframe[colArray].stack().std(); #compute overall S.D.
    if SD == 0: #all number are the same. No need to do anything
        return;
    dataframe[colArray] = dataframe[colArray]/SD
    
def scaleVarOneCol(dataframe,nameStr):
    '''
    Given the name of one column, scale that column so that the S.D. is 1. The mean remains the same. For example,
    
    df = pandas.read.csv("some_path")
    scaleVar(df,feature1)
    '''
    if dataframe[nameStr].std() == 0: #all number are the same. No need to do anything
        return;
    dataframe[nameStr] = dataframe[nameStr]/dataframe[nameStr].std()
    
#input check
def input_check(n,k,d,B,function_name='the function'):
    '''
    Check that B is a list of k matrices of size n x n, and that d <= n.
    Arguments:
        function_name: indicate where this check happens inside, so that it prints the error message referring to the right location. 
    '''
    if (isinstance(function_name, str) == False):
        print("Error: check_input is used with function name that is not string. Exit the check.")
        return 1
        
    if (k<1):
        print("Error: " + function_name + " is called with k<1.")
        return 2
        
    if (len(B) < k):
        print("Error: " + function_name + " is called with not enough matrices in B.")
        return 3
        
    #check that matrices are the same size as n
    for i in range(k):
        if (B[i].shape != (n,n)):
            print("Error: " + function_name + " is called with input matrix B_i not the correct size." + "Note: i=" + str(i) + " , starting indexing from 0 to k-1")
            return 4
        
    if (((d>0) and (d<=n)) == False):
        print("Error: " + function_name + " is called with invalid value of d, which should be a number between 1 and n inclusive.")
        return 5
              
    return 0 #no error case

def getObj(n,k,d,B,X):
    """
    Given k PSD n-by-n matrices B1,...,Bk, and a projection matrix X which is n-by-n, give variance and loss of each group i.
    Additionally, compute max min variance, min max loss, Nash Social Welfare, and total variance objective by this solution X. 
    The matrix B_i should be centered (mean 0), since the formula to calculate variance will be by the dot product of B_i and X
    Arguments:
        k: number of groups
        n: original number of features (size of all B_i's)
        B: list of PSD matrices, as numpy matrices. It must contain at least k matrices. If there are more than k matrices provided, the first k will be used as k groups.
        X: given solution. This method will still work even if not PSD or symmmetric or wrong rank as long as X has a correct dimension
        
    Return: a dictionary with keys 'Loss', 'Var', and 'Best' (for the best possible PCA in that group as if other group does not exist) to each group, 
    and three objectives MM_Var, MM_Loss, and NSW.
    """
    #input check
    if (input_check(n, k, d, B, function_name='getObj') > 0):
        return -1
    #rank check
    if (np.linalg.matrix_rank(X) != d):
        print("Warning: getObj is called with X having rank not equal to d.")
        
    obj = dict() 
    
    best = [np.sum(np.sort(np.linalg.eigvalsh(B[i]))[-d:]) for i in range(k)]
    loss = [np.sum(np.multiply(B[i],X)) - best[i] for i in range(k)]
    var = [np.sum(np.multiply(B[i],X)) for i in range(k)]
    
    #welfare objective
    obj.update({'MM_Var':np.amin(var),'MM_Loss':np.amin(loss),'NSW':geo_mean_through_log(var),'Total_Var':np.sum(var)})
    
    #Loss, Var, and Best to each group
    for i in range(k):
        obj.update({'Loss'+str(i):loss[i],'Var'+str(i):var[i],'Best'+str(i):best[i]})
        
    return obj  

