# Sinkhorn algorithm
import numpy as np

def sinkhorn(K,Kt,p,q,delta):
    ''' Sinkhorn algorithm that compute an approximation of the Sinkhorn projection.
    Inputs
            K : function of signature K(v), compute K@v matrix-vector multiplication with the kernel.
            Kt : function of signature Kt(v), compute K^T@v matrix-vector multiplication.
            p,q : arrays of shape (n,1), the target marginals ( sum(P,axis=0,1) = p,q )
            delta : positive scalar,the tolerance.
    Outputs
            u,v : arrays of shape (n,1), the scaling vectors that define P = diag(u)Kdiag(v).
            W   : scalar, the associated wasserstein cost.
    '''
    (n,_) = np.shape(p)
    tau= delta/8
    u,v = np.ones(n,1), np.ones(n,1) # initialize
    # un = np.ones(n,1) # not used
    p,q = (1-tau)*p + tau/n, (1-tau)*q + tau/n # increase support
    k = 0
    while np.sum(np.abs(u*(K(v) - p))) + np.sum(np.abs(v*(Kt(u) - q))) >= delta/2:
        k=k+1
        if k%2 == 1:
            u = p / K(v)
        else:
            v = q / Kt(u)
    W = np.log(u).T @ (u*K(v)) + np.log(v).T @ (v*Kt(u))
    return u,v,W