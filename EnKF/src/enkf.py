import numpy as np
from random import gauss
import matplotlib.pyplot as plt
from lorenz96 import Lorenz96, bulk_Lorenz96

def Model1(x:np.ndarray) -> np.ndarray:
    """
    理想的なモデル．各状態変数は前の状態の1.003倍になる
    """
    F = np.array([[1.003,0,0],
                 [0,1.003,0],
                 [0,0,1.003]])
    return F@x

def operator1(m: np.ndarray): # 解析データto観測データの変換
    H = np.array([[1/2, 1/2, 0],
                  [0, 1/2, 1/2]])
    return H@m

def get_error(m: np.ndarray, N: int)->np.ndarray: # 誤差行列を求める
    return m @ (np.eye(N)-np.full((N, N), 1/N))

def get_cov(m: np.ndarray, N: int)->np.ndarray: # 誤差分散共分散行列を求める
    m_ = get_error(m, N)
    return m_@m_.T/(N-1)



def main():
    # 観測データ
    obs = np.loadtxt('out/u_obs.csv', 
                     delimiter=',', dtype='float64')
    # (仮の)真値
    n_true = np.loadtxt('out/u_true.csv', 
                     delimiter=',', dtype='float64') 
    
    # 設定
    Model:function = Lorenz96
    bulk_model:function = bulk_Lorenz96
    operator:function = operator1
    N:int = 128  # メンバ数
    time_step = obs.shape[1]  # タイムステップ
    R = np.eye(2)
    each_member = np.zeros((N, time_step))
    open_list = [] 
    
    # 初期メンバ生成
    A_ini = np.zeros((3, N))
    for member in range(N):
        A_ini[:, member:member+1] = n_true[:, 0:1]+np.array([[gauss()], [gauss()], [gauss()]])
    
    A_ini_open = n_true[:, 0:1]
    open_list.append(A_ini_open[0,0])
    each_member[:, 0:1] = A_ini[0:1, :].T
    
    # main loop
    for time in range(1, time_step):
        A_f = bulk_model(A_ini)
        Z = get_error(A_f, N)/((N-1)**0.5)
        Y = get_error(operator(A_f), N)/((N-1)**0.5)
        drand = np.random.standard_normal((2, N))
        K = (Z @ Y.T @ np.linalg.inv(Y@Y.T+R))
        if time==1:
            print(K)
        A_ini = A_f + K@(np.tile(obs[:,time:time+1], (1, N)) + drand - (operator(A_f)))\
                + np.random.standard_normal((3, N))
        #NOTE:分散が収束するのを防ぐためにad hocにランダムな行列を追加している
        #多分分かってないだけなのでサーベイが必要
         
        each_member[:, time:time+1] = A_ini[0:1, :].T
        A_ini_open = Model(A_ini_open)
        open_list.append(A_ini_open[0,0])
              
    print("loop done!")  
    print(each_member.shape)
    
    # (必要であれば)可視化    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for member in range(N):
        ax.plot(each_member[member], color='#bce2e8')
        
    mean_of_member = np.mean(each_member, axis=0)
    ax.plot(mean_of_member, color='red')
    ax.plot(n_true[0], color='blue')
    ax.plot(open_list, color='green')
    plt.savefig('out/line_plot.png')
    plt.close()
    
if __name__ == '__main__':
    main()
    
    
