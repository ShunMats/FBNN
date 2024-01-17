import numpy as np
import random
from scipy.stats import ortho_group

from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Activation,LeakyReLU


# for 20points
def f_Sigmu(t,Sigma,mu):
    si=np.sin(t)
    co=np.cos(t)
    num = -(Sigma[0][0]*co**2)/2 - (Sigma[1][1]*si**2)/2 - Sigma[0][1]*si*co
    num +=  (mu[0]*Sigma[0][0] + mu[1]*Sigma[0][1])*co + (mu[0]*Sigma[0][1] + mu[1]*Sigma[1][1])*si
    return np.exp(num)

# def hist_gene(Sigma,mu,multi=100):
#     hist=[f_Sigmu(rad*0.1*np.pi,Sigma,mu) for rad in range(0, 20)]
#     myhist=[int(hist[i]*multi) for i in range(len(hist))]
#     return myhist

def hist_gene_int(Sigma,mu,multi=100):
    hist=[f_Sigmu(rad*0.1*np.pi,Sigma,mu) for rad in range(0, 20)]
    hist2=[int(hist[i]*multi) for i in range(len(hist))]
    N=sum(hist2)
    if(N!=0): N=1/N
    myhist=[hist2[i]*N for i in range(len(hist2))]
    return myhist

def hist_gene_true(Sigma,mu,multi=100):
    hist=[f_Sigmu(rad*0.1*np.pi,Sigma,mu) for rad in range(0, 20)]
    N=sum(hist)
    if(N!=0): N=1/N
    myhist=[hist[i]*N for i in range(len(hist))]
    return myhist


def ave(hist):
    N=sum(hist)
    if(N!=0): N=1/N
    return [hist[i]*N for i in range(len(hist))]

def hist_gene(Sigma,mu,multi=100):
    hist=[f_Sigmu(rad*0.1*np.pi,Sigma,mu) for rad in range(0, 20)]
    if(multi>0):
        hist = ave(hist)
        hist=[int(hist[i]*multi) for i in range(len(hist))]
    return ave(hist)

def gene_data(gene_func,Sig_lim=20,mu_lim=20,gene_size=500,multi=100):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        eig=[random.uniform(0, Sig_lim),random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(2)
        Sig = S@np.diag(eig)@S.T
        mu=[random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]  
        # mu=[random.uniform(0, mu_lim),random.uniform(0, mu_lim)]
        myhist = gene_func(Sig,mu,multi)
        x_gene.append(myhist)
        y_gene.append(list(Sig[0])+list(Sig[1][1:2])+list(mu))

    return (np.array(x_gene),np.array(y_gene))

def hist_Normalize(hist,Sig,mu,center=0):
    M = int(center/np.pi*10) + hist.index(max(hist))
    if(M>19): M-=20
    # center/np.pi*10 
    rad = center - M*0.1*np.pi
    Orad = np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
    Orev = np.array([[np.cos(-rad),-np.sin(-rad)],[np.sin(-rad),np.cos(-rad)]])
    newhist = hist[M:20]+hist[0:M]
    # print(len(newhist))
    newSig = Orev.T@Sig@Orev
    newmu = Orad@mu
    return newhist,newSig,newmu,rad

def gene_data_Normalize(gene_func,Sig_lim=20,mu_lim=20,gene_size=500,multi=100,center=0,rad_option=False):
    x_gene=[]
    y_gene=[]
    rads=[]
    for _ in range(gene_size):
        eig=[random.uniform(0, Sig_lim),random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(2)
        Sig = S@np.diag(eig)@S.T
        mu=[random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]  
        # mu=[random.uniform(0, mu_lim),random.uniform(0, mu_lim)]
        myhist = gene_func(Sig,mu,multi)
        # 正規化する部分
        myhist,Sig,mu,rad = hist_Normalize(myhist,Sig,mu,center)
        x_gene.append(myhist)
        y_gene.append(list(Sig[0])+list(Sig[1][1:2])+list(mu))
        rads.append(rad)
    x_gene=np.array(x_gene)
    y_gene=np.array(y_gene)

    if(rad_option): return (x_gene,y_gene),rads
    else: return (x_gene,y_gene)


def set_model(n_hidden=100,n_layer=1,drop_rate=0.2,print_summary=True):
    model = Sequential()
    model.add(Dense(n_hidden, activation='relu', input_shape=(20,)))
    model.add(Dropout(drop_rate))
    for _ in range(n_layer-1):
        model.add(Dense(n_hidden, activation='relu'))
        model.add(Dropout(drop_rate))
    model.add(Dense(5, activation='linear'))

    if(print_summary): model.summary()

    model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['mae'])
    # mse（平均二乗誤差）
    # mae（平均絶対誤差）
    
    return model

def set_model_BN(n_hidden=100,n_layer=1,print_summary=True):
    model = Sequential()
    model.add(Dense(n_hidden, input_shape=(20,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for _ in range(n_layer-1):
        model.add(Dense(n_hidden))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dense(5, activation='linear'))
    # BatchNormalization(synchronized=True)

    if(print_summary): model.summary()

    model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['mae'])
    # mse（平均二乗誤差）
    # mae（平均絶対誤差）
    
    return model

def set_model_LeakyReLU(optimizer='adam',n_hidden=100,n_layer=1,print_summary=True,leaky_alpha=0.01):
    model = Sequential()
    model.add(Dense(n_hidden, input_shape=(20,)))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(0.2))
    for _ in range(n_layer-1):
        model.add(Dense(n_hidden))
        model.add(LeakyReLU(alpha=leaky_alpha))
        model.add(Dropout(0.2))
    model.add(Dense(5, activation='linear'))

    if(print_summary): model.summary()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    # mse（平均二乗誤差）; mae（平均絶対誤差）
    
    return model

def AB_gene(myhist):
    A,B = np.zeros((2,2)),np.zeros((2,1))
    n = sum(myhist)
    for i in range(len(myhist)):
        rad = i*0.1*np.pi
        xi = np.array([[np.cos(rad),np.sin(rad)]]).T
        A += np.dot(xi,xi.T)*myhist[i]
        B += xi*myhist[i]
    A,B = A/n,B/n
    # print(n,"\n",A,"\n",B)
    return A,B,n



###############################################
### th[0]=0の前提の下でNNを構成したい
###############################################

def gene_data_zero(gene_func,Sig_lim=20,mu_lim=20,gene_size=500,multi=100,center=0):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        eig=[0, random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(2)
        Sig = S@np.diag(eig)@S.T
        mu=[random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]  
        # mu=[random.uniform(0, mu_lim),random.uniform(0, mu_lim)]
        myhist = gene_func(Sig,mu,multi)
        # 正規化する部分
        if(center!=None): myhist,Sig,mu,rad = hist_Normalize(myhist,Sig,mu,center)
        x_gene.append(myhist)
        y_gene.append(list(Sig[0])+list(Sig[1][1:2])+list(mu))

    return (np.array(x_gene),np.array(y_gene))

def set_model_zero_LeakyReLU(optimizer='adam',output_shape=3,n_hidden=100,n_layer=1,print_summary=True,leaky_alpha=0.01):
    model = Sequential()
    model.add(Dense(n_hidden, input_shape=(20,)))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(0.2))
    for _ in range(n_layer-1):
        model.add(Dense(n_hidden))
        model.add(LeakyReLU(alpha=leaky_alpha))
        model.add(Dropout(0.2))
    model.add(Dense(output_shape, activation='linear'))

    if(print_summary): model.summary()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    # mse（平均二乗誤差）; mae（平均絶対誤差）
    
    return model

def so2rad(so):
    # 2次元ならarccos,arcsinの奴で求めた方が早い
    acco = np.arccos(so[0][0])
    acsi = np.arcsin(so[1][0])
    alpha = acco if acsi>1e-5 else 2*np.pi-acco
    alpha *= np.linalg.det(so)
    if(np.linalg.norm(rad2so(alpha)-so)>1e-5): print("not correct\n",so)
    return alpha

def rad2so(alpha):
    i = 0 if alpha>0 else 1
    rad = abs(alpha)
    co = np.cos(rad)
    si = np.sin(rad)
    so = np.array([[co,si],[si,co]])
    so[i][1] *= -1
    return so

# def gene_data_zero_thga(gene_func,Sig_lim=20,mu_lim=20,gene_size=500,multi=100,center=None):
#     x_gene=[]
#     y_gene=[]
#     for _ in range(gene_size):
#         eig=[0,random.uniform(0, Sig_lim)]
#         S = ortho_group.rvs(2)
#         Sig = S@np.diag(eig)@S.T
#         mu=[random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]
#         myhist = gene_func(Sig,mu,multi)
#         # 正規化する部分
#         if(center!=None): myhist,Sig,mu,rad = hist_Normalize(myhist,Sig,mu,center)
#         # th,ga
#         O = np.linalg.eig(Sig)[1].T
#         O = np.diag(np.sign(O@Sig@mu+1e-2))@O
#         O = np.array([[0,1],[1,0]])@O if abs((O@Sig@O.T)[0][0])>1e-10 else O
#         th = np.diag(O@Sig@O.T)/2
#         ga = O@Sig@mu
#         if(abs(th[0])>1e-8): print("th[0] is too large:%g"% th[0])
#         alpha = so2rad(O) 
#         x_gene.append(myhist)
#         y_gene.append(list(th[1:2])+list(ga[1:2])+[alpha])
#         # y_gene.append(list(th[0:2])+list(ga[0:2])+[alpha])

#     return (np.array(x_gene),np.array(y_gene))

def so2rad_2(so):
    so0=so.copy()
    # 2次元ならarccos,arcsinの奴で求めた方が早い
    sign = np.linalg.det(so)
    if(sign==0): print("error")
    if(sign<0): so = np.array([[0,1],[1,0]])@so
    acco = np.arccos(so[0][0])
    acsi = np.arcsin(so[1][0])
    alpha = acco if acsi>1e-5 else 2*np.pi-acco
    alpha *= sign
    if(np.linalg.norm(rad2so_2(alpha)-so0)>1e-5): print("not correct\n",rad2so_2(alpha),"\n",so)
    return alpha

def rad2so_2(alpha):
    sign = np.sign(alpha)
    rad = abs(alpha)
    co = np.cos(rad)
    si = np.sin(rad)
    so = np.array([[co,-si],[si,co]])
    if(sign<0): so = np.array([[0,1],[1,0]])@so
    return so

def gene_data_zero_thga2(gene_func,Sig_lim=20,mu_lim=20,gene_size=500,multi=100,center=None):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        eig=[0,random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(2)
        Sig = S@np.diag(eig)@S.T
        mu=[random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]
        myhist = gene_func(Sig,mu,multi)
        # 正規化する部分
        if(center!=None): myhist,Sig,mu,rad = hist_Normalize(myhist,Sig,mu,center)
        # th,ga
        O = np.linalg.eig(Sig)[1].T
        O = np.diag(np.sign(O@Sig@mu+1e-2))@O
        O = np.array([[0,1],[1,0]])@O if abs((O@Sig@O.T)[0][0])>1e-10 else O
        th = np.diag(O@Sig@O.T)/2
        ga = O@Sig@mu
        if(abs(th[0])>1e-8): print("th[0] is too large:%g"% th[0])
        alpha = so2rad_2(O) # ==>ここの表現を変更
        x_gene.append(myhist)
        y_gene.append(list(th[1:2])+list(ga[1:2])+[alpha])

    return (np.array(x_gene),np.array(y_gene))


def gene_data_zero_thga3(gene_func,Sig_lim=20,mu_lim=20,gene_size=500,multi=100):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        thgaO = [random.uniform(0, Sig_lim),random.uniform(0, Sig_lim),random.uniform(0, Sig_lim),random.uniform(-2*np.pi, 2*np.pi)]
        th = [0,thgaO[0]]
        ga = thgaO[1:3]
        O = rad2so_2(thgaO[3])
        myhist = gene_func(th,ga,O=O,multi=multi)
        x_gene.append(myhist)
        y_gene.append(thgaO)

    return (np.array(x_gene),np.array(y_gene))


def f_thga(t,theta,gamma,O=None):
    if(O is None): O = np.eye(int(len(gamma)))
    si=np.sin(t)
    co=np.cos(t)
    x = np.array([[co,si]]).T
    ox = O@x
    num = - ox.T@np.diag(theta)@ox + ox.T@gamma
    if(num.shape!=(1,1)): print("shape errer:",num.shape)
    res = np.exp(num[0,0])
    if(res==np.nan): print("res is nan")
    return res

def hist_gene_thga(theta,gamma,O=None,multi=100):
    if(O is None): O = np.eye(int(len(gamma)))
    hist=[f_thga(rad*0.1*np.pi,theta,gamma,O) for rad in range(0, 20)]
    if(multi>0): hist=[int(hist[i]*multi) for i in range(len(hist))]
    N=sum(hist)
    if(N!=0): N=1/N
    myhist=[hist[i]*N for i in range(len(hist))]
    return myhist

def hist_gene_thga2(theta,gamma,O=None,multi=100):
    if(O is None): O = np.eye(int(len(gamma)))
    hist=[f_thga(rad*0.1*np.pi,theta,gamma,O) for rad in range(0, 20)]
    def ave(hist):
        N=sum(hist)
        if(N!=0): N=1/N
        return [hist[i]*N for i in range(len(hist))]
    hist=ave(hist)
    if(multi>0): hist=[int(hist[i]*multi) for i in range(len(hist))]
    hist=ave(hist)
    return hist




##########################################################
### for noise

# def hist_gene_noise(Sigma,mu,multi=100,noise=1):
#     hist=[f_Sigmu(rad*0.1*np.pi,Sigma,mu) + random.uniform(-noise, noise) for rad in range(0, 20)]
#     myhist=[int(hist[i]*multi) for i in range(len(hist))]
#     return myhist


def hist_gene_noise(Sigma,mu,multi=100):
    hist=[f_Sigmu(rad*0.1*np.pi,Sigma,mu) for rad in range(0, 20)]
    hist=ave(hist)
    rng = np.random.default_rng()
    if(multi>0): 
        hist=[int(hist[i]*multi)+int(abs(rng.standard_normal())) for i in range(len(hist))]
    else: 
        hist=[hist[i]+abs(rng.standard_normal()) for i in range(len(hist))]
    # N=max(hist)
    hist=ave(hist)
    return hist

def rot(rad):
    return np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])

def gene_data_Normalize_noise(Sig_lim=20,mu_lim=20,gene_size=500,multi=100,center=0,noise=1):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        eig=[random.uniform(0, Sig_lim),random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(2)
        Sig = S@np.diag(eig)@S.T
        mu=[random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]  
        # mu=[random.uniform(0, mu_lim),random.uniform(0, mu_lim)]
        test_hist = hist_gene_noise(Sig,mu,multi,noise)
        N=sum(test_hist)
        if(N!=0): N=1/N
        test_hist_N = [test_hist[i]*N for i in range(len(test_hist))]
        # 正規化する部分
        myhist,Sig,mu,rad = hist_Normalize(test_hist_N,Sig,mu,center)
        x_gene.append(myhist)
        y_gene.append(list(Sig[0])+list(Sig[1][1:2])+list(mu))

    return (x_gene,y_gene)



#####################################################
from sei_kume import Loglikelihood

def comp_logL_Sigmu(model,x_test,y_test):
    y_preds = model.predict(x_test, verbose=0)

    res=[]
    for i,y_pred in enumerate(y_preds):
        Sig0 = [y_pred[0:2],y_pred[1:3]]
        mu0 = y_pred[3:5]
        O0 = np.linalg.eig(Sig0)[1].T
        O0 = np.diag(np.sign(O0@Sig0@mu0)+1e-2)@O0
        th0 = np.diag(O0@Sig0@O0.T)/2
        ga0 = O0@Sig0@mu0

        Sig_T=[y_test[i][0:2],y_test[i][1:3]]
        mu_T=y_test[i][3:5]
        O_T = np.linalg.eig(Sig_T)[1].T
        O_T = np.diag(np.sign(O_T@Sig_T@mu_T))@O_T
        th_T = np.diag(O_T@Sig_T@O_T.T)/2
        ga_T = O_T@Sig_T@mu_T

        hist = x_test[i]
        A,B,n = AB_gene(hist)

        log_0 = Loglikelihood(th0,ga0,A,B,O=O0,n=1,method="hg")
        log_T = Loglikelihood(th_T,ga_T,A,B,O=O_T,n=1,method="hg")
        # print("%2d:log_T, log_0: %f, %f"%(i+1,log_T["log"],log_0["log"]))
        res.append(abs(log_T["log"]-log_0["log"]))
    return np.mean(res), res