# The Sinkhorn algorithm
import numpy as np
import time
from sklearn.decomposition import TruncatedSVD
import warnings

def sinkhorn(K,Kt,p,q,delta,maxtime=60):
    ''' Sinkhorn algorithm that compute an approximation of the Sinkhorn projection.
    Inputs
            K : function of signature K(v), matrix-vector multiplication with Gibbs kernel.
            Kt : function of signature Kt(v), matrix-vector multiplication with with transposed Gibbs kernel.
            p,q : arrays of shape (n,1), the target marginals ( sum(P,axis=0,1) = p,q )
            delta : positive scalar,the tolerance.
    Outputs
            u,v : arrays of shape (n,1), the scaling vectors that define P = diag(u)Kdiag(v).
            W   : scalar, the associated wasserstein cost.
            err : array, the evolution of the marginal error (debugging purpose)
    '''
    t = time.time()
    (n,_) = np.shape(p)
    tau= delta/8
    u,v = np.ones((n,1)), np.ones((n,1)) # initialize
    un = np.ones((n,1))
    #p,q = (1-tau)*p + tau/n, (1-tau)*q + tau/n # increase support?
    k = 0
    err = [ np.sum(np.abs(u*K(v) - p)) + np.sum(np.abs(v*Kt(u) - q)) ] #debuging
    while err[-1] >= delta/2 and time.time()-t < maxtime :
        k=k+1
        if k%2 == 1:
            u = p / K(v)
        else:
            v = q / Kt(u)
        err.append(np.sum(np.abs(u*K(v) - p)) + np.sum(np.abs(v*Kt(u) - q))) #debuging
    if (time.time()-t)>=maxtime:
        warnings.warn("Maximum time of sinkhorn achieved.")
    # rescalling to avoid overflow
    u = u / np.min(u) # do we risk division by a "too small" number?
    v = v * np.min(u)
    if (np.min(u) < 1e-10) or (np.min(v) < 1e-10): #to see if keep or not
        warnings.warn("Overflow in sinkorn, arbitrary value assigned to W.") # put back lign number 24?
        W=-9999
    else:
        W = (np.log(u).T @ (u*K(v)) + np.log(v).T @ (v*Kt(u))) # /eta !...
    # ...erreur dans le paper, mais on a pas accès à eta ici, à voir si on le rajoute en paramètre
    # ou si on garde eta=1
    W=np.squeeze(W) # sinon donne [[W]]
    return u,v,W,err


def sinkhorn_debug(K,Kt,p,q,delta,maxtime=60):
    ''' !!! Old version of sinkhorn used as debuging purpose !!!
    Le calcul de norm_u, norm_v et err modifient la complexité de l'algo original.
    Sinkhorn algorithm that compute an approximation of the Sinkhorn projection.
    Inputs
            K : function of signature K(v), matrix-vector multiplication with Gibbs kernel.
            Kt : function of signature Kt(v), matrix-vector multiplication with with transposed Gibbs kernel.
            p,q : arrays of shape (n,1), the target marginals ( sum(P,axis=0,1) = p,q )
            delta : positive scalar,the tolerance.
    Outputs
            u,v : arrays of shape (n,1), the scaling vectors that define P = diag(u)Kdiag(v).
            W   : scalar, the associated wasserstein cost.
            norm_u,norm_v,err : some norms and error
    '''
    t = time.time()
    (n,_) = np.shape(p)
    tau= delta/8
    u,v = np.ones((n,1)), np.ones((n,1)) # initialize
    un = np.ones((n,1))
    # p,q = (1-tau)*p + tau/n, (1-tau)*q + tau/n # increase support?
    k = 0
    norm_u = [] #debuging
    norm_v = [] #debuging
    err = [] #debuging
    while np.sum(np.abs(u*K(v) - p)) + np.sum(np.abs(v*Kt(u) - q)) >= delta/2 and time.time()-t<maxtime :
        k=k+1
        if k%2 == 1:
            u = p / K(v)
        else:
            v = q / Kt(u)
        norm_u.append(np.linalg.norm(u)) #debuging
        norm_v.append(np.linalg.norm(v)) #debuging
        err.append(np.sum(np.abs(u*K(v) - p)) + np.sum(np.abs(v*Kt(u) - q))) #debuging
    W = (np.log(u).T @ (u*K(v)) + np.log(v).T @ (v*Kt(u))) # /eta !...
    # ...erreur dans le paper, mais on a pas accès à eta ici, à voir si on le rajoute en paramètre
    # ou si on garde eta=1
    W=np.squeeze(W) # sinon donne [[W]]
    return u,v,W,norm_u,norm_v,err

def low_rank_Sinkhorn(Kmat,k,p,q,delta,maxtime=60):
    ''' Sinkhorn algorithm where the matrix Kmat is approximated by a rank k matrix
    Inputs
        Kmat: kernel matrix to project
        k: scalar the rank of our approximation matrix
        p,q : arrays of shape (n,1), the target marginals ( sum(P,axis=0,1) = p,q )
        delta : positive scalar,the tolerance.
    Outputs
        u,v : arrays of shape (n,1), the scaling vectors that define P = diag(u)Kdiag(v).
            W   : scalar, the associated wasserstein cost.
            norm_u,norm_v,err : some norms and error
    '''
    svd=TruncatedSVD(k)
    US=svd.fit_transform(Kmat) 
    V=svd.components_
    def K(v):
        return US@(V@v)
    def Kt(v):
        return V.T@(US.T@v)
    S_time=time.time()
    [u,v,W,err]=sinkhorn(K,Kt,p,q,delta,maxtime=60)
    end_time=time.time()-S_time
    P=u*Kmat*v.T
    P=(u*US)@(V*v.T) #computing associated coupling P matrix
    return [u,v,W,err, P, end_time]






