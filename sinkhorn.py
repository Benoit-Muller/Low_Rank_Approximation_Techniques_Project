# The Sinkhorn algorithm
import numpy as np

def sinkhorn(K,Kt,p,q,delta,maxiter=10000):
    ''' Sinkhorn algorithm that compute an approximation of the Sinkhorn projection.
    Inputs
            K : function of signature K(v), matrix-vector multiplication with Gibbs kernel.
            Kt : function of signature Kt(v), matrix-vector multiplication with with transposed Gibbs kernel.
            p,q : arrays of shape (n,1), the target marginals ( sum(P,axis=0,1) = p,q )
            delta : positive scalar,the tolerance.
    Outputs
            u,v : arrays of shape (n,1), the scaling vectors that define P = diag(u)Kdiag(v).
            W   : scalar, the associated wasserstein cost.
    '''
    (n,_) = np.shape(p)
    tau= delta/8
    u,v = np.ones((n,1)), np.ones((n,1)) # initialize
    un = np.ones((n,1))
    p,q = (1-tau)*p + tau/n, (1-tau)*q + tau/n # increase support
    k = 0
    norm_u = [] #debuging
    norm_v = [] #debuging
    err = [] #debuging
    while np.sum(np.abs(u*K(v) - p)) + np.sum(np.abs(v*Kt(u) - q)) >= delta/2 and k<maxiter :
        k=k+1
        if k%2 == 1:
            u = p / K(v)
        else:
            v = q / Kt(u)
        norm_u.append(np.linalg.norm(u)) #debuging
        norm_v.append(np.linalg.norm(v)) #debuging
        err.append(np.sum(np.abs(u*K(v) - p)) + np.sum(np.abs(v*Kt(u) - q))) #debuging
        m = np.max(u) #idea
        u = u / m #idea
        v = v * m #idea
    W = np.log(u).T @ (u*K(v)) + np.log(v).T @ (v*Kt(u))
    return u,v,W,norm_u,norm_v,err