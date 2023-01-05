from sklearn.model_selection import train_test_split
import numpy as np
import math

def Dkl(p,q):
    """ Computes KL Divergence between p and q 
        p and q must be 1D arrays ar lists representing probability distributions
    """
    a = np.zeros(len(p))
    for i in range(len(p)):
        if p[i] != 0:
            a[i] = p[i]*math.log2(p[i]/q[i]) 
    
    return sum(a)

def IB(pxy, px, py_x, X, beta,N, epsilon=0.1, T = None):
    """ Solves the IB problem for a 2D feature space with binary labels

        INPUTS
            pxy is the joint distribution p(x,y) 
            py_x is the conditional distribution p(y|x) 
                the above distributons are represented as 3D arrays, with the 2 first dimentions for x and the third for y
            px is the marginal distribution p(x) as a 2D array
                pxy, py_x and px should be consistent i.e. p(x,y) = p(y|x)p(x)

            X is the feature space as a list of tupples of two integers, these tupples will be used as indices e.g., px[X[i]]
            beta is a Lagrange multiplier that determines the level of the tradeoff between compression of X and prediction of Y
            N is the maximum number of iterations
            epsilon is the treshold for convergence
            T is the list of representations as a 1D array or list of integers starting from 0 going up in increments of 1 e.g. T = [0,1,2].
                If no value is provided then T = Y = [0,1] 

        OUTPUTS  
            qt_x is the conditional distribution p(t|x) representing the IB optimal map between X and T
            Ixt is the mutual information I(T;X)
            Iyt is the mutual information I(T;Y)
            changes tracks the change in qt_x between two iterations
            n is the number of iterations needed for changes <= epsilon
    """

    Y = [0,1]
    if T == None:
        T = Y
    
    # initial qt_x (= p(t|x)) map generated randomly
    qt_x = np.zeros(py_x.shape)
    X_0, X_1 = train_test_split(X, test_size=1/2.0)
    for x in X_1:
        qt_x[x[0],x[1],1] = 1
    qt_x[:,:,0] = -qt_x[:,:,1]+1

    # initialise qt (= p(t)) and qy_t (= p(y|t)) based on qt_x
    qt = np.array([np.sum(px[:,:]*qt_x[:,:,t]) for t in T])
    qy_t = np.array([[np.sum(pxy[:,:,y]*qt_x[:,:,t])/qt[t] for y in Y] for t in T ])
    
    old_qt_x1 = np.zeros(px.shape)
    changes = np.zeros(N)

    n = 0
    while n < N:
        
        old_qt_x1 = qt_x[:,:,1]+0 # copy previous qt_x to compute changes and check convergence

        # update qt_x
        for x in X:
            for t in T:
                f = math.exp(-beta*Dkl(py_x[x], qy_t[t]))
                qt_x[x+(t,)] = qt[t]*f
            qt_x[x] = qt_x[x]/sum(qt_x[x]) # normalise qt_x

        qt = np.array([np.sum(px[:,:]*qt_x[:,:,t]) for t in T])# update qt
        qy_t = np.array([[np.sum(pxy[:,:,y]*qt_x[:,:,t])/qt[t] for y in Y] for t in T ])# update qt
        

        changes[n] = sum(sum(abs(q) for q in qt_x[:,:,1] -old_qt_x1))# track changes

        #check for convergence
        if changes[n] <= epsilon:
            break
        n += 1

    # compute mutual information of X and T (I(X;T))
    Ixt = 0.
    for t in T:
        if qt[t] > 0:
            for x in X:
                if qt_x[x+(t,)] > 0:
                    Ixt += qt_x[x+(t,)]*px[x]*math.log2(qt_x[x+(t,)]/qt[t])
        
    # compute mutual information of Y and T (I(Y;T))
    py = np.array([np.sum(pxy[:,:,y]) for y in Y])
    Iyt = 0.
    for y in Y:
        if py[y] > 0:
            for t in T:
                if qy_t[t,y] > 0:
                    Iyt += qy_t[t,y]*qt[t]*math.log2(qy_t[t,y]/py[y])

    
    return qt_x, Ixt, Iyt, changes, n