import numpy as np
import random
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

from keras import optimizers,Model
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Activation,Input,Flatten
# from keras.models import Sequential
# from keras.layers import Dense,Dropout,BatchNormalization,Activation,Input
from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet18 import ResNet18
# from keras.applications.resnet18 import ResNet18Backbone


def f_3d_uv(u,v,Sigma,mu):
    th=np.arccos(-2*u+1)
    phi=2*np.pi*v
    x = np.array([[np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)]]).T
    num = - (x.T@Sigma@x)/2 + x.T@Sigma@mu
    res = np.exp(num)
    if(res==np.nan): print("res is nan")
    return res

def hist_rot_3d(hist_3d,rad):
    hist = hist_3d.reshape((10,10))
    slide = int(rad*5/np.pi)
    endp = 10-slide
    if(endp>9): endp-=10
    hist = np.concatenate([hist[:,endp:], hist[:,:endp]], 1)
    if(hist_3d.shape!=(10,10)): hist = hist.reshape(hist_3d.shape)
    return hist, (slide)*0.2*np.pi

def z_Normalize_3d(hist_3d):
    hist = hist_3d.reshape((10,10))
    idx = np.unravel_index(np.argmax(hist), hist.shape)
    endp = 5+idx[1]
    if(endp>9): endp-=10
    hist = np.concatenate([hist[:,endp:], hist[:,:endp]], 1)
    if(hist_3d.shape!=(10,10)): hist = hist.reshape(hist_3d.shape)
    return hist, (5-idx[1])*0.2*np.pi

def hist_gene_3d(Sigma,mu,multi=100,normalize=False,out_shape=-1):
    eps=0.05
    hist_3d=np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            hist_3d[i][j] = f_3d_uv(i*0.1+eps, j*0.1, Sigma, mu)
    if(multi>=0): 
        hist_3d = hist_3d/np.max(hist_3d)
        hist_3d = (multi * hist_3d).astype(int) 
    hist_3d = hist_3d/np.max(hist_3d)
    rad=None
    if(normalize is True): 
        hist_3d,rad = z_Normalize_3d(hist_3d)
    return hist_3d.reshape(out_shape),rad


def z_rot(rad):
    z_rot = [
        [np.cos(rad),-np.sin(rad),0],
        [np.sin(rad),np.cos(rad),0],
        [0,0,1]
    ]
    return np.array(z_rot)

def gene_data_3d(Sig_lim=20,mu_lim=20,gene_size=500,multi=100,normalize=False,eig0=None,out_shape=-1):
    x_gene=[]
    y_gene=[]
    for _ in range(gene_size):
        if(eig0 is None): 
            eig=[random.uniform(0, Sig_lim),random.uniform(0, Sig_lim),random.uniform(0, Sig_lim)]
        else:
            eig=[eig0,random.uniform(0, Sig_lim),random.uniform(0, Sig_lim)]
        S = ortho_group.rvs(3)
        Sig = S@np.diag(eig)@S.T
        mu = [random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim),random.uniform(-mu_lim, mu_lim)]  
        # mu=[random.uniform(0, mu_lim),random.uniform(0, mu_lim)]
        myhist,rad = hist_gene_3d(Sig,mu,multi,normalize,out_shape=out_shape)
        if(rad!=None):
            Sig = z_rot(-rad).T@Sig@z_rot(-rad)
            mu  = z_rot(rad)@mu
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
    model.add(Dense(9, activation='linear'))

    if(print_summary): model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # mse（平均二乗誤差）; mae（平均絶対誤差）
    
    return model

def set_model_resnet50_3d(N=10,input_batch=1,n_hidden=100,n_layer=1,print_summary=False):
    # input_tensor = Input(shape=(N,N,input_batch))
    input_tensor = Input(shape=(N,N,1,))
    resNet50 = ResNet50(include_top=False, weights=None ,input_tensor=input_tensor)
    
    top_model = Sequential()
    top_model.add(Flatten(input_shape=resNet50.output_shape[1:]))
    top_model.add(Dense(9, activation='linear'))
    # model = Model(input=resNet50.input, output=top_model(resNet50.output))
    model = Model(resNet50.input, top_model(resNet50.output))

    if(print_summary): model.summary()
    model.compile(loss='mse',optimizer='adam',metrics=['mae'])

    return model

def AB_gene_3d(hist):
    myhist=hist.reshape((10,10))
    A,B = np.zeros((3,3)),np.zeros((3,1))
    n = np.sum(myhist)
    eps=0.05
    for i in range(10):
        for j in range(10):
            th=np.arccos(-2*(i*0.1+eps)+1)
            phi=2*np.pi*(j*0.1)
            # rad = i*0.1*np.pi
            xi = np.array([[np.sin(th)*np.cos(phi), np.sin(th)*np.sin(phi), np.cos(th)]]).T
            A += np.dot(xi,xi.T)*myhist[i,j]
            B += xi*myhist[i,j]
    A,B = A/n,B/n
    # print(n,"\n",A,"\n",B)
    return A,B,n


#####################################################
from sei_kume import Loglikelihood

def comp_logL_Sigmu_3d(model,x_test,y_test):
    y_preds = model.predict(x_test, verbose=0)

    res=[]
    for i,y_pred in enumerate(y_preds):
        if(len(y_pred)==9):
            pred=list(y_pred)
            Sig0 = np.array([pred[0:3],pred[1:2]+pred[3:5],pred[2:3]+pred[4:5]+pred[5:6]])
            mu0 = np.array(pred[6:])
            O0 = np.linalg.eig(Sig0)[1].T
            O0 = np.diag(np.sign(O0@Sig0@mu0)+1e-2)@O0
            th0 = np.diag(O0@Sig0@O0.T)/2
            ga0 = O0@Sig0@mu0

            y=list(y_test[i])
            Sig_T = np.array([y[0:3],y[1:2]+y[3:5],y[2:3]+y[4:5]+y[5:6]])
            mu_T = np.array(y[6:])
            O_T = np.linalg.eig(Sig_T)[1].T
            O_T = np.diag(np.sign(O_T@Sig_T@mu_T))@O_T
            th_T = np.diag(O_T@Sig_T@O_T.T)/2
            ga_T = O_T@Sig_T@mu_T

        hist = x_test[i]
        A,B,n = AB_gene_3d(hist)

        log_0 = Loglikelihood(th0,ga0,A,B,O=O0,n=1,method="hg")
        log_T = Loglikelihood(th_T,ga_T,A,B,O=O_T,n=1,method="hg")
        # print("%2d:log_T, log_0: %f, %f"%(i+1,log_T["log"],log_0["log"]))
        res.append(abs(log_T["log"]-log_0["log"]))
    return np.mean(res), res



def plot_3dFB(hist_3d, fig=None,figsize=(10,10),axes=(1,1,1),title=None,lim=1.3,multi=10):
    myhist=hist_3d.reshape(10,10)

    if(fig is None):
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(axes[0],axes[1],axes[2],projection='3d')
    ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim)
    if(title!=None): ax.set_title(title, fontsize=20)
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    ax.set_zlabel("Z", fontsize=15)

    eps=0.05
    for i in range(10):
        for j in range(10):
            th=np.arccos(-2*(i*0.1+eps)+1)
            phi=2*np.pi*(j*0.1)
            X=np.sin(th)*np.cos(phi)
            Y=np.sin(th)*np.sin(phi)
            Z=np.cos(th)
            ms=myhist[i,j]*multi
            ax.plot(X,Y,Z,marker="o",markersize=ms,linestyle='None',color="r")

    th1,th2 = np.mgrid[0:2*np.pi:40j,0:2*np.pi:40j]
    x = np.cos(th1)*np.sin(th2)
    y = np.sin(th1)*np.sin(th2)
    z = np.cos(th2)
    ax.plot_wireframe(x, y, z, color="skyblue", linewidth=0.5)

    # plt.show()
    return fig