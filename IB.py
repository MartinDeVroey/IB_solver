from sklearn.model_selection import train_test_split
import numpy as np
import math

def Dkl(p,q):
    #print(p)
    #print(q)
    a = np.zeros(len(p))
    for i in range(len(p)):
        if p[i] != 0:
            a[i] = p[i]*math.log2(p[i]/q[i]) 
    
    #print(sum(a))
    return sum(a)

def IB(pxy, px, py_x, X,Y, beta,N, epsilon=0.1):
    T = Y
    
    # initial qt_x map generated randomly
    qt_x = np.zeros(py_x.shape)
    X_0, X_1 = train_test_split(X, test_size=1/2.0)
    for x in X_1:
        qt_x[x[0],x[1],1] = 1
    #plt.imshow(qt_x[:,:,1])
    #plt.show()
    qt_x[:,:,0] = -qt_x[:,:,1]+1

    qt = np.array([np.sum(px[:,:]*qt_x[:,:,t]) for t in T])
    qy_t = np.array([[np.sum(pxy[:,:,y]*qt_x[:,:,t])/qt[t] for y in Y] for t in T ])
    #plt.imshow(qt_x[:,:,1])
    old_qt_x1 = np.zeros(px.shape)
    changes = np.zeros(N)
    scores = np.zeros(N)

    n = 0
    while n < N:
        
        old_qt_x1 = qt_x[:,:,1]+0
        for x in X:
            for t in T:
                f = math.exp(-beta*Dkl(py_x[x], qy_t[t]))
                #print(f)
                qt_x[x+(t,)] = qt[t]*f
            qt_x[x] = qt_x[x]/sum(qt_x[x])

        qt = np.array([np.sum(px[:,:]*qt_x[:,:,t]) for t in T])
        qy_t = np.array([[np.sum(pxy[:,:,y]*qt_x[:,:,t])/qt[t] for y in Y] for t in T ])
        #print(qt_x[:,:,1])
        #plt.imshow(qt_x[:,:,0])
        #plt.show()
        changes[n] = sum(sum(abs(q) for q in qt_x[:,:,1] -old_qt_x1))
        scores[n] = sum(sum(abs(q) for q in qt_x[:,:,1] - py_x[:,:,1]))

        if changes[n] <= epsilon:
            break
        n += 1

    # compute mutual information of X and T
    Ixt = 0.
    for t in T:
        if qt[t] > 0:
            for x in X:
                if qt_x[x+(t,)] > 0:
                    Ixt += qt_x[x+(t,)]*px[x]*math.log2(qt_x[x+(t,)]/qt[t])
        
    # compute mutual information of Y and T
    py = np.array([np.sum(pxy[:,:,y]) for y in Y])
    Iyt = 0.
    for y in Y:
        if py[y] > 0:
            for t in T:
                if qy_t[t,y] > 0:
                    Iyt += qy_t[t,y]*qt[t]*math.log2(qy_t[t,y]/py[y])

    #plt.imshow(qt_x[:,:,1])
    return qt_x, Ixt, Iyt, changes, scores, n