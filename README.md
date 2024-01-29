# FBNN
NN for Fisher-Bingham distribution

$$
    f(x) =
        \frac{1}{\mathcal{C} \left( \frac{\Sigma^{-1}}{2} , \Sigma^{-1} \mu \right)}
        e^{-\frac{x^T \Sigma^{-1} x}{2} + x^T \Sigma^{-1} \mu}
        \, d_{S^{p-1}} (x)
$$


<!--  -->
## テストファイルの中身
### test-1.ipynb
$R^{2}$上の1次元Fisher-Bingham分布

1. 最適なハイパーパラメータの選択

2. データの正規化

3. LeakyReLU

4. optimizerの選択


### test2-ipynb
$R^{2}$上の1次元Fisher-Bingham分布

1. th[0]=eig[0]=0に固定したモデルの構築

mymodel_0start1.h5：固有値を制限しない通常のモデル

mymodel_0start2.h5：eig[0]=0

mymodel_0startN2.h5：eig[0]=0+Normalize

mymodel_0start3.h5：th[0]=ga[0]=0

mymodel_0start4.h5：th[0]=0

mymodel_0start5.h5：Resnet18+thga

mymodel_0start6.h5：Resnet18+thga+LeakyReLU

mymodel_0start7.h5：Resnet18+Sig

2. ResNetを用いたモデルの構築


### test-3.ipynb
$R^{3}$上の2次元Fisher-Bingham分布

1. 2次元Fisher-Bingham分布のハイパーパラメータの選択
mymodel_3d1.h5
<!-- ：最適な選択でのモデル -->

2. z軸周りでの正規化を行った結果の比較
mymodel_3d2.h5

3. 固有値の一つを2もしくは0に固定した学習データ
mymodel_3d3.h5：eig[0]=2,non Normalize

mymodel_3d4.h5：eig[0]=2,Normalize

4. ResNetを用いたモデルの構築


### test-4.ipynb
1. 少ないデータに対する予測精度

2. ノイズありのデータに対する予測精度

3. ノイズありの訓練データでのモデルの学習
mymodel_0start8.h5：noise+LReLU

mymodel_0start9.h5：noise+ReLU+Resnet

mymodel_0start10.h5：noise+LReLU+Resnet



### test-5.ipynb
th[0]=1で最尤推定および、モデルの構築をおこなう試み

mymodel_1start1.h5:eig[0]=1+LeakyReLU

mymodel_1start2.h5:eig[0]=1+ReLU

mymodel_1start3.h5:eig[0]=2+LReLU

mymodel_1start4.h5:eig[0]=2+LReLU+Normalize


### test-fig.ipynb
1次元及び2次元Fisher-Bingham分布における初期値決定から最尤推定までの過程を図表


<!--  -->
## ソースファイルの種類
### FBNN2d.py
$R^{2}$上の1次元Fisher-Bingham分布の初期値決定のためのニューラルネット構築にかかわる関数の管理

### FBNN3d.py
$R^{3}$上の2次元Fisher-Bingham分布の初期値決定のためのニューラルネット構築にかかわる関数の管理

### sei_kume.py
Sei-Kumeの論文にある最尤推定アルゴリズムをPythonで書き換えたのちに、エラー等を修正したもの

### myResNet.py
ResNetから畳み込みやプーリング処理を除いたモデルをクラス化したもの