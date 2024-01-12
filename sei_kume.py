import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize,Bounds
from scipy.special import factorial,gammaln
from scipy.linalg import expm
import math
import sys
import time


### Pfaffian for Fisher-Bingham distribution
def dG_fun_FB(alpha, G, fn_params=None):
    if(fn_params is None): fn_params={"ns":[1]*int(len(alpha)/2), "s":1}

    # s: scale
    d=len(alpha)
    p=int(d/2)
    r=len(G) # rはd+1と一致する
    th=alpha[0:p]
    xi=alpha[p:p*2]
    gam=[x**2/4 for x in xi]
    Gth=G[1:p+1]
    Gxi=G[p+1:1+p*2]
    dG=np.zeros((d,r))
    ns=fn_params["ns"]
    s=fn_params["s"]

    dG[:,0]=G[1:d+1]
    # derivative of dC/dth[i] with respect to th[j]
    for i in range(p):
        dG[i][i+1] = -s*Gth[i]
        for j in range(p):
            if(j!=i):
                a1 = -( ns[j]/2/(th[j]-th[i]) + xi[j]**2/4/(th[j]-th[i])**2 )
                a2 = -( ns[i]/2/(th[i]-th[j]) + xi[i]**2/4/(th[j]-th[i])**2 )
                a3 = -( ns[j]*xi[i]/4/(th[j]-th[i])**2 + xi[i]*xi[j]**2/4/(th[j]-th[i])**3 )
                a4 = -( ns[i]*xi[j]/4/(th[i]-th[j])**2 + xi[i]**2*xi[j]/4/(th[i]-th[j])**3 )
                dG[j][1+i] = a1*Gth[i] + a2*Gth[j] + a3*Gxi[i] + a4*Gxi[j]
                dG[i][1+i] -= dG[j][1+i]
    
    # derivative of dC/dxi[i] with respect to th[j]
    # derivative of dC/dth[j] with respect to xi[i]
    for i in range(p):
        dG[i][1+p+i] = -s*Gxi[i]
        for j in range(p):
            if(j!=i):
                b2 = xi[i]/2/(th[i]-th[j])
                b3 = - ( ns[j]/2/(th[j]-th[i]) + xi[j]**2/4/(th[j]-th[i])**2 )
                b4 = xi[i]*xi[j]/4/(th[i]-th[j])**2
                dG[j][1+p+i] = b2*Gth[j] + b3*Gxi[i] + b4*Gxi[j]
                dG[p+i][1+j] = dG[j][1+p+i]
                dG[i][1+p+i] -= dG[j][1+p+i]
        dG[p+i][1+i] = dG[i][1+p+i]

    # derivative of dC/dxi[i] with respect to xi[j]
    for i in range(p):
        for j in range(p):
            if(j!=i):
                c3 = xi[j]/2/(th[j]-th[i])
                c4 = -xi[i]/2/(th[j]-th[i])
                dG[p+j][1+p+i] = c3*Gxi[i] + c4*Gxi[j]
        if(ns[i] == 1 or abs(xi[i]) < 1e-10):  dG[p+i][1+p+i] = -Gth[i]  # cheat
        else:  dG[p+i][1+p+i] = -Gth[i] - (ns[i]-1)/xi[i]*Gxi[i]  # singular if xi[i] = 0
    
    return dG


### Initial value for Pfaffian system (by power series expansion)
# parametrization: exp(th*x^2 + xi*x)
def C_FB_power(alpha,v=None,d=None,Ctol=1e-6,alphatol=1e-10,Nmax=None):
    if(v is None): v=[0]*int(len(alpha))
    if(d is None): d=np.ones(int(len(alpha)/2))
    if(Nmax is None): Nmax = math.ceil(10**(10/len(alpha)))

    p = int(len(alpha)/2)
    d=np.array(d)
    alpha_sum = np.sum(np.abs(np.array(alpha[0:p]))) + np.sum(np.abs(np.array(alpha[p:p*2])))
    N0 = max(math.ceil(alpha_sum),1)
    N=N0
    while(N < Nmax+1):
        logep = N*np.log(alpha_sum) - np.log(math.factorial(N)) + np.log((N+1)/(N+1-alpha_sum))
        if(logep < np.log(Ctol)): break
        N += 1

    def f(k):
        kn = len(k)
        if(kn == 2*p):
            k1 = np.array(k[0:p])
            k2 = np.array(k[p:(2*p)])
            v1 = np.array(v[0:p])
            v2 = np.array(v[p:(2*p)])
            if(any((k2+v2) % 2 == 1)): return 0
            w = [i for i, x in enumerate(k) if x>0]
            a = 1
            for w in w: a *= alpha[w]**(k[w]) 
            b1 = sum( - np.log(factorial(k1)) - np.log(factorial(k2)) + gammaln(k1 + v1 + (k2 + v2)/2 + (d/2)) + gammaln((k2+v2)/2 + (1/2)) - gammaln((k2+v2)/2 + (d/2)))
            b2 = - math.lgamma(sum(k1 + v1 + (k2 + v2)/2 + (d/2)))
            b3 = math.lgamma(sum(d)/2) - p*math.lgamma(1/2)
            return( a * np.exp(b1+b2+b3) )
        else: # recursive part
            knxt = kn
            if(abs(alpha[knxt]) < alphatol): return( f(k+[0]) )  # speed-up
            a = 0
            imax = N - sum(k)
            for i in range(0,imax+1): a += f(k+[i])
            return a
    
    return f([])


def G_FB_power(alpha,ns=None,Ctol=1e-6,alphatol=1e-10):
    if(ns is None): ns=[1]*int(len(alpha)/2)

    # not restricted to Bingham
    p = int(len(alpha)/2)
    th = alpha[0:p]
    xi = alpha[p:(2*p)]
    thxi = list(-np.array(th)) + list(xi)
    C = C_FB_power(thxi, d=ns, Ctol=Ctol, alphatol=alphatol)  # note: parameterization
    dC = np.zeros(2*p)
    for i in range(p):
        e = [0]*(2*p)
        e[i] = 1
        dC[i] = -1 * C_FB_power(thxi, v=e, d=ns, Ctol=Ctol, alphatol=alphatol)  # note: parametrerization
        e = [0]*(2*p)
        e[p+i] = 1
        dC[p+i] = C_FB_power(thxi, v=e, d=ns, Ctol=Ctol, alphatol=alphatol)  # note: parametrerization
    return [C]+list(dC)


### Initical value for Pfaffian system (by Monte Carlo)
def rsphere(N,p):
    N,p = int(N),int(p)
    x = np.random.normal(size=N*p).reshape((N,p))
    r = np.sqrt(np.sum(x**2, axis=1))
    for i in range(len(r)): x[i,:] /= r[i]  # ndarrayの場合は[,]で範囲指定する
    return x

def G_FB_MC(alpha, ns=None, N=2*1e4, t=None):
    if(ns is None): ns=[1]*int(len(alpha)/2)
    if(t is None): t = rsphere(N, sum(ns))
    # N=1e6

    N=int(N)
    p = int(len(alpha)/2)
    th = np.array(alpha[0:p])
    xi = np.array(alpha[p:p*2])
    G = np.zeros(2*p+1)
    idx = [0] + list(np.cumsum(ns)) # 累積和
    t1,t2 = np.zeros((N, p)),np.zeros((N, p))
    for i in range(p):
        t2[:,i] = np.sum(t[:, idx[i]:idx[i+1]+1]**2, axis=1) # np.sumで行ベクトルになるからOK
        t1[:,i] = t[:,idx[i]]
    itrd = np.exp(t2 @ (-th.T) + t1 @ xi.T)  # integrand
    G[0] = np.mean(itrd)
    for i in range(p):
        G[1+i] = -np.mean(itrd * t2[:,i:i+1])
        G[1+p+i] = np.mean(itrd * t1[:,i:i+1])
    return list(G)


### Initial value (wrapper)
def G_FB(alpha, ns=None, method="power", withvol=True):
    if(ns is None): ns=[1]*int(len(alpha)/2)

    dsum = sum(ns)
    v0 = 2 * np.pi**(dsum/2)/math.gamma(dsum/2)
    v = v0 if withvol else 1

    if(method == "power"):
        return ( v * np.array(G_FB_power(alpha, ns=ns)) )
    if(method == "MC"):
        return ( v * np.array(G_FB_MC(alpha, ns=ns)) )
    sys.exit("method not found")

def my_ode_hg(t, G, th,dG_fun,v,fn_params):
    th =list( np.array(th) + t * np.array(v) )
    if(fn_params is None): dG = dG_fun(th, G)
    else: dG = dG_fun(th, G, fn_params)
    G_rhs = np.array(v) @ np.array(dG)
    return list(G_rhs)


### hg main
def hg(th0, G0, th1, dG_fun, t_span=[0,1], fn_params=None, show_trace=False):
    rk_res = solve_ivp(my_ode_hg, t_span, G0, method='RK45', args=(th0,dG_fun,np.array(th1)-np.array(th0),fn_params), t_eval=np.linspace(0, 1, 101))
    trace = rk_res.y if show_trace else None
    return {"G": rk_res.y[ :,-1 ], "trace": trace }


def my_ode_hg_mod(tau, G, th0,th1,dG_fun,fn_params):
    p = int(len(th0) / 2)
    th,v = np.zeros(2*p),np.zeros(2*p)
    for w in np.arange(0,p):
        th[w] = th0[w] + tau * (th1[w] - th0[w])
        v[w] = th1[w] - th0[w]
    for w in np.arange(p,2*p):
        th[w] = np.sign(th1[w]) * np.sqrt(th0[w]**2 + tau * (th1[w]**2 - th0[w]**2))
        if(th[w] != 0): v[w] = (th1[w]**2 - th0[w]**2)/2/th[w]

    dG = dG_fun(th, G, fn_params)
    G_rhs = v @ dG
    return list(G_rhs)



### hg main (for square-root transformation)
def hg_mod(th0, G0, th1, dG_fun, t_span=[0,1], fn_params=None, show_trace=False):
    rk_res = solve_ivp(my_ode_hg_mod, t_span, G0, method='RK45', args=(th0,th1,dG_fun,fn_params))
    trace = rk_res.y if show_trace else None

    return {"G": rk_res.y[:,-1], "trace": trace }


### evaluating FB normalising constant by HGM
def hgm_FB(alpha, ns=None, alpha0=None, G0=None, withvol=True):
    if(ns is None): ns=[1]*int(len(alpha)/2)

    p = int(len(alpha) / 2)
    r = sum(np.abs(np.array(alpha)))
    N = max(r, 1)**2 * 10
    if(alpha0 is None):  alpha0 = np.array(alpha[0:(2*p)]) / N
    if(G0 is None): G0 = G_FB(alpha0, ns=ns, method="power", withvol=withvol)
    fn_params = {"ns": ns, "s": 1}

    return list(hg(alpha0, G0, alpha, dG_fun_FB, fn_params=fn_params)["G"])


### evaluating FB normalising constant by HGM (via square-root transformation)
def hgm_FB_2(alpha, ns=None, withvol=True):
    if(ns is None): ns=[1]*int(len(alpha)/2)

    p = int(len(alpha) / 2)
    r = sum(np.abs(np.array(alpha)))
    N = max(r, 1)**2 * 10
    alpha0 = list(np.array(alpha[0:p]) / N) + list(np.array(alpha[p:(2*p)]) / np.sqrt(N))
    G0 = G_FB(alpha0, ns=ns, method="power", withvol=withvol)
    fn_params = {"ns": ns, "s": 1}

    return list(hg_mod(alpha0, G0, alpha, dG_fun_FB, fn_params=fn_params)["G"])


####SPA calculation
def saddleaprox_FB_revised(L,M=None,dub=3, order=3):
    if(M is None): M = np.array(L)*0
    #calculates the normalising constant of Fisher-Bingham distribution in pre-shape space there are three methods as described 
    #L is the vector of positive values
    #M is the vecor of mu<-i's
    #dub is the number at which each entry of L is doubled

    L = np.array(list(L)*dub)
    a = 1/np.sqrt(np.prod(L/np.pi))
    Y = 0
    M = np.array(M)
    
    KM = lambda t: sum(-1/2*np.log(1-t/L) + M**2/(1-t/L)/L)
    KM1 = lambda t: sum(1/2*1/(L-t) + M**2/(L-t)**2)
    KM2 = lambda t: sum(1/2*1/(L-t)**2 + 2*M**2/(L-t)**3)
    KM3 = lambda t: sum(1/(L-t)**3 + 6*M**2/(L-t)**4)
    KM4 = lambda t: sum(3/(L-t)**4 + 24*M**2/(L-t)**5)

    def sol(KM1,y):
        loc = lambda t: np.abs(KM1(t)-1)
        Y = minimize(loc, 0, tol = sys.float_info.epsilon**2)
        return Y.x[0]

    ##
    that = sol(KM1,1)
    Y = 2*a/np.sqrt(2*np.pi*KM2(that))*np.exp(KM(that)-that)
    if (order==3):
        rho3sq = KM3(that)**2/KM2(that)**3
        rho4 = KM4(that)/KM2(that)**(4/2)
        Rhat = 3/24*rho4 - 5/24*rho3sq
        Y = Y*np.exp(Rhat)
    elif (order==2):
        rho3sq = KM3(that)**2/KM2(that)**3
        rho4 = KM4(that)/KM2(that)**(4/2)
        Rhat = 3/24*rho4 - 5/24*rho3sq
        Y = Y*(1+Rhat)

    return Y


def SPA(alpha, ns=None, withvol=True):
    if(ns is None): ns=[1]*int(len(alpha)/2)

    p = int(len(alpha)/2)
    theta,mu = [],[]
    for i in range(p):
        theta += [ alpha[i]+1 ] *ns[i] 
        mu += [ alpha[i+p]/np.sqrt(ns[i])/2 ] *ns[i] 

    dsum = sum(ns)
    v0 = 2 * np.pi**(dsum/2)/math.gamma(dsum/2)
    coef = 1 if withvol else 1/v0
    
    res = saddleaprox_FB_revised(theta,mu,dub=1,order=3) 
    res = res*np.exp(1)/coef
    return res



####################################################################
####################################################################
# kume_sei/11222_2017_9765_MOESM2_ESM.r
####################################################################
####################################################################



def Loglikelihood(theta,gamma, A,B, O=None,n=1,method="hg",check=False):
    if(O is None): O = np.eye(len(gamma))  
    #Calculates the loglikelohood where A and B are observed second and first moments A=sum X_i #X_i^t and B=sum X_i.
    #O is the orthogonal component of the covariance matrix

    p = len(theta)
    ths = np.argsort(np.array(theta)) 
    theta,gamma = [theta[th] for th in ths], [gamma[th] for th in ths]
    alpha1 = theta + gamma
    alpha0 = theta + [0]*p
    Q = np.zeros((p,p))
    for i in range(p): Q[i][ths[i]]=1
    O = Q@O

    if(method=="MC"):
        hgout = G_FB(alpha1, method="MC", withvol=True)
    if(method=="hg"):
        hgout = hgm_FB_2(alpha1)
    if(method=="SPA"):
        hgout = saddleaprox_FB_revised(np.sort(alpha1[0:p])+1, M=np.array(alpha1[p:(2*p)])/2, dub=1, order=3)
        hgout = np.array(hgout) * np.exp(1)
    if(method=="power"):
        hgout = G_FB(alpha1, method="power", withvol=True)

    nc = hgout[0]
    lo = -n*np.log(nc) - np.sum(np.diag(A@O.T@np.diag(theta)@O - np.array([gamma]).T@B.T@O.T))

    grad = -n*np.array(hgout[1:])/nc + np.array(list(-np.diag(O@A@O.T))+list((O@B).T[0]))
    # grad[1]=0 #seems to be very important
    grad[:p] = grad[:p]@Q
    grad[p:] = grad[p:]@Q

    y={"log":lo,"grad":grad}	
    return y



def Grad_mle_update(th,gg,A,B,O=None,n=1,log_print=True):
    if(O is None): O=np.eye(len(gg))

    p = len(th)
    # 追加 
    # ths = np.argsort(np.array(th))
    # th,gg = [th[s] for s in ths] ,[gg[s] for s in ths]
    # Q = np.zeros((len(ths),len(ths)))
    # for i in range(len(ths)): Q[i][ths[i]]=1
    # O = Q@O

    current = Loglikelihood(th,gg,A=A,B=B,O=O,n=n)	
    delta = 1
    current["grad"][0] = 0
    newth = np.array(th) + delta*np.array(current["grad"][0:(p)])
    newth[0] = 0 # might need that for guaranteeing convergence
    newgg = np.array(gg) + delta*np.array(current["grad"][p:(2*p)])

    new = Loglikelihood(newth,newgg,A=A,B=B,n=n,O=O)
    while(new["log"]<current["log"] and delta>1e-3):
        delta = delta/2
        newth[1:p] = np.array(th[1:p]) + delta*np.array(current["grad"][1:(p)])
        newgg = np.array(gg) + delta*np.array(current["grad"][p:(2*p)])	
        new = Loglikelihood(newth,newgg,A=A,B=B,n=n,O=O)
        if(log_print): print(delta, new["log"] < current["log"])
        #readline()

    y={"th":newth,"gg":newgg,"mle":new,"delt":delta}
    return y



def Grad_mle_update_optim(th,gg,A,B,O=None,n=1):
    if(O is None): O=np.eye(len(gg))

    # 追加 
    # ths = np.argsort(np.array(th))
    # th,gg = [th[s] for s in ths] ,[gg[s] for s in ths]
    # Q = np.zeros((len(ths),len(ths)))
    # for i in range(len(ths)): Q[i][ths[i]]=1
    # O = Q@O

    p = len(th)
    current = Loglikelihood(th,gg,A=A,B=B,O=O,n=n)	
    delta = 1
    newth = np.array(th) + delta*np.array(current["grad"][0:(p)])
    newth[0] = 0  # might need that for guaranteeing convergence
    newgg = np.array(gg) + delta*np.array(current["grad"][p:(2*p)])

    def f(dl):
        newth[1:p] = np.array(th[1:p]) + dl*np.array(current["grad"][1:(p)])
        newgg = np.array(gg) + dl*np.array(current["grad"][p:(2*p)])
        x = -Loglikelihood(newth,newgg,A=A,B=B,n=n,O=O)["log"]
        return x

    dmax = np.amin( [2] + list(np.abs( 1/ np.array(current["grad"][1:p])* np.array(th[1:p]) )) )

    bounds = Bounds(0, dmax)
    delta = minimize(f,dmax/2,bounds=bounds).x[0]
    newth[1:p] = np.array(th[1:p]) + delta*np.array(current["grad"][1:(p)])
    newgg = np.array(gg) + delta*np.array(current["grad"][p:(2*p)])
    new = Loglikelihood(newth,newgg,A=A,B=B,n=n,O=O)
    y = {"th":newth,"gg":newgg,"mle":new,"delt":delta}
    return y


def Grad_mle_update_optim_sqrt(th,gg,A,B,O=None,n=1,dmax=1):
    #since th>0 and gg>0 we can write th=x^2 and gg=y^2 and optimize wrt x and y
    if(O is None): O=np.eye(len(gg))

    p = len(th)
    current = Loglikelihood(th,gg,A=A,B=B,O=O,n=n)	
    x = np.sqrt(th)
    y = np.sqrt(gg)
    delta = 10
    newx = x * (1 + delta*np.array(current["grad"][0:(p)]))
    newy = y * (1 + delta*np.array(current["grad"][p:(2*p)]))
    newx[0] = 0
    current["grad"][0] = 0 # might need that for guaranteeing convergence

    def f(dl):
        x = np.sqrt(th)
        y = np.sqrt(gg)
        newx = x*(1 + dl*current["grad"][0:(p)])
        newy = y*(1 + dl*current["grad"][p:(2*p)])
        
        z = -Loglikelihood(newx**2,newy**2,A=A,B=B,n=n,O=O)["log"]
        return z

    bounds = Bounds(0, dmax)
    delta = minimize(f,dmax/2,bounds=bounds).x[0]
    newx = x*(1 + delta*np.array(current["grad"][0:(p)]))
    newy = y*(1 + delta*np.array(current["grad"][p:(2*p)]))

    newth = newx**2
    newgg = newy**2
    new = Loglikelihood(newth,newgg,A=A,B=B,n=n,O=O)
    y = {"th":newth,"gg":newgg,"mle":new,"delt":delta}
    return y



def Grad_mle_orth_update(th,gg,A,B,O=None,n=1,tol=1e-3,log_print=True,AA_plus=False):
    if(O is None): O=np.eye(len(gg))

    if(AA_plus): AA = np.diag(th)@O@A@O.T - O@A@O.T@np.diag(th) + np.array([gg]).T@B.T@O.T
    else: AA = np.diag(th)@O@A@O.T - O@A@O.T@np.diag(th) - np.array([gg]).T@B.T@O.T  # ここもマイナス
    vhat = AA - AA.T
    # cc = -np.sum(np.diag(A@O.T@np.diag(th)@O - np.array([gg]).T@B.T@O.T))
    current = Loglikelihood(th,gg,A=A,B=B,n=n,O=O)
    delta = 1*np.sign(np.sum(np.diag(AA@vhat)))
    newO = (expm(vhat*delta))@O
    # newO = O   # これで最悪現状維持ができる
    # newcc = -np.sum(np.diag(A@newO.T@np.diag(th)@newO - np.array([gg]).T@B.T@newO.T))
    new = Loglikelihood(th,gg,A=A,B=B,n=n,O=newO)

    def f0(t0):
        return -Loglikelihood(th,gg,A=A,B=B,n=n,O=(expm(vhat*t0))@O)["log"]	

    bounds = Bounds(-100, 100)
    out0 = minimize(f0,0,bounds=bounds).x[0]
    # out0 = minimize(f0,1).x[0]
    newO1 = (expm(vhat*out0))@O
    if(log_print): print(out0,np.sum(vhat**2))

    new1 = Loglikelihood(th,gg,A=A,B=B,n=n,O=newO1)
    if(new["log"]<new1["log"]): newO = newO1.copy()  # 大小の符号が逆になってた
    elif(log_print): print("too small vhat?")

    y = {"O":newO,"mle":new1,"AA":AA}
    return y



def optimal_orth(th,gg,A,B,O=None,n=1,tol=1e-2):
    #provides the  optimal orthogonal matrix using the gradient method as in the paper. 
    if(O is None): O=np.eye(len(gg))

    GG = Grad_mle_orth_update(th,gg,A=A,B=B,O=O,n=n)
    newO = GG["O"]
    a = np.linalg.norm(O-newO,ord=1)
    k = 1
    while(a > tol):
        Oold = newO.copy()
        GG = Grad_mle_orth_update(th,gg,A=A,B=B,O=Oold,n=n)
        newO = GG["O"]
        a = np.linalg.norm(Oold-newO,ord=1)
        # a = np.linalg.norm(GG["AA"] - (GG["AA"]).T, ord=1) 
        k = k+1

    return {"O":GG["O"].copy(), "AA":GG["AA"].copy()}



def optimisation(th,gg,A,B,n=1,O=None,orth="no",tol=1e-3,iterss=200,log_print=True,AA_plus=False):
    #This optimises the likelihood for a fixed orthogonal matrix O
    if(O is None): O=np.eye(len(gg))
    start = time.time()

    # 追加 
    ths = np.argsort(np.array(th))
    th,gg = [th[s] for s in ths] ,[gg[s] for s in ths]
    Q = np.zeros((len(ths),len(ths)))
    for i in range(len(ths)): Q[i][ths[i]]=1
    O = Q@O

    # print(th,gg)
    ll = Grad_mle_update(th,gg,A=A,B=B,O=O,n=n,log_print=log_print)
    # print(ll["th"],ll["gg"])
    llnew = Grad_mle_update(ll["th"],ll["gg"],A=A,B=B,n=n,log_print=log_print)
    # print(llnew["th"],llnew["gg"])
    a = np.linalg.norm(ll["th"] - llnew["th"], ord=1)
    b = np.linalg.norm(ll["gg"] - llnew["gg"], ord=1)
    if(log_print): print(a, b, llnew["mle"]["log"])
    newO = O
    iters = 0
    c = np.linalg.norm(ll["mle"]["grad"] - llnew["mle"]["grad"], ord=1)

    while(max(c,a+b) > tol and iters < iterss):
        a = np.linalg.norm(ll["th"] - llnew["th"], ord=1)
        b = np.linalg.norm(ll["gg"] - llnew["gg"], ord=1)
        c = np.linalg.norm(llnew["mle"]["grad"], ord=1)
        ll = llnew.copy()
        if(orth=="yes"):
            newO = Grad_mle_orth_update(ll["th"],ll["gg"],A=A,B=B,O=newO,n=n,log_print=log_print,AA_plus=AA_plus)["O"]
            llnew = Grad_mle_update_optim_sqrt(ll["th"],ll["gg"],A=A,B=B,O=newO,n=n)
        else:
            llnew = Grad_mle_update_optim_sqrt(ll["th"],ll["gg"],A=A,B=B,O=newO,n=n)
        
        if(log_print): 
            print("iteration")
            print(iters+1,a,b,c,llnew["mle"]["log"],llnew["mle"]["log"] - ll["mle"]["log"])
        iters = iters+1
        if((llnew["mle"]["log"] - ll["mle"]["log"])<1e-10): break

    llnew["O"] = newO
    end= time.time()
    llnew["worked time"] = {"time":end-start,"iters":iters}
    return llnew



