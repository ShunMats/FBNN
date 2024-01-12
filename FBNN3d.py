import numpy as np
import random
from scipy.stats import ortho_group

from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Activation,Input


def f_3d_uv(u,v,Sigma,mu):
    th=np.arccos(-2*u+1)
    phi=2*np.pi*v
    x = np.array([[np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)]]).T
    num = - (x.T@Sigma@x)/2 + x.T@Sigma@mu
    res = np.exp(num)
    if(res==np.nan): print("res is nan")
    return res

def hist_gene_3d(Sigma,mu,multi=100,normalize=True):
    eps=0.05
    hist_3d=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if(multi<1): hist_3d[i][j] = f_3d_uv(i*0.1+eps, j*0.1, Sigma, mu)
            else: hist_3d[i][j] = int(multi * f_3d_uv(i*0.1+eps, j*0.1, Sigma, mu))
    # N = np.sum(hist_3d)
    # if(N!=0): N=1/(N)
    # if(N!=0): N=1/np.sqrt(N)
    # myhist = hist_3d*N
    myhist = hist_3d/np.max(hist_3d) if normalize else hist_3d
    return myhist.reshape(-1)
    # return hist_3d/np.max(hist_3d)

def gene_data_3d(Sig_lim=20,mu_lim=20,gene_size=500,multi=100):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        eig=[random.uniform(0, Sig_lim),random.uniform(0, Sig_lim),random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(3)
        Sig = S@np.diag(eig)@S.T
        mu = [random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]  
        # mu=[random.uniform(0, mu_lim),random.uniform(0, mu_lim)]
        myhist = hist_gene_3d(Sig,mu,multi)
        x_gene.append(myhist)
        y_gene.append(list(Sig[0])+list(Sig[1][1:3])+list(Sig[2][2:3])+list(mu))

    return (np.array(x_gene),np.array(y_gene))

def set_model_3d(n_hidden=100,n_layer=1,drop_rate=0.2,print_summary=True):
    model = Sequential()
    # model.add(Dense(n_hidden, activation='relu', input_shape=(100,)))
    model.add(Input(shape=(100,)))
    model.add(Dense(n_hidden, activation='relu'))
    model.add(Dropout(drop_rate))
    for _ in range(n_layer-1):
        model.add(Dense(n_hidden, activation='relu'))
        model.add(Dropout(drop_rate))
    model.add(Dense(9, activation='linear'))

    if(print_summary): model.summary()
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])
    # mse（平均二乗誤差）; mae（平均絶対誤差）

    return model

def set_model_LeakyReLU_3d(n_hidden=100,n_layer=1,print_summary=True,leaky_alpha=0.01):
    model = Sequential()
    model.add(Input(shape=(100,)))
    model.add(Dense(n_hidden))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(0.2))
    for _ in range(n_layer-1):
        model.add(Dense(n_hidden))
        model.add(LeakyReLU(alpha=leaky_alpha))
        model.add(Dropout(0.2))
    model.add(Dense(5, activation='linear'))

    if(print_summary): model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # mse（平均二乗誤差）; mae（平均絶対誤差）
    
    return model

def AB_gene_3d(myhist):
    A,B = np.zeros((3,3)),np.zeros((3,1))
    n = sum(myhist)
    eps=0.05
    for i in range(10):
        for j in range(10):
            th=np.arccos(-2*(i*0.1+eps)+1)
            phi=2*np.pi*(j*0.1)
            # rad = i*0.1*np.pi
            xi = np.array([[np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)]]).T
            A += np.dot(xi,xi.T)*myhist[i*10+j]
            B += xi*myhist[i*10+j]
    A,B = A/n,B/n
    # print(n,"\n",A,"\n",B)
    return A,B,n
