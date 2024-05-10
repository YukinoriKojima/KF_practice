import numpy as np
from random import gauss
import matplotlib.pyplot as plt 

def Model(x:np.ndarray) -> np.ndarray:
    """
    理想的なモデル．各状態変数は前の状態の2倍になる
    """
    F = np.array([[1.003,0,0],
                 [0,1.003,0],
                 [0,0,1.003]])
    return F@x

def get_cov(m: np.ndarray, N: int)->np.ndarray:
    m_ = m @ (np.eye(N)-np.full((N, N), 1/N))
    return m_@m_.T/(N-1)
    

def main():
    
    # 各タイムステップでのメンバを格納するリスト    
    
    # 設定
    N:int = 128  # メンバ数
    time_step = 400  # タイムステップ
    R = np.eye(2)
    H = np.array([[1/2, 1/2, 0],
                  [0, 1/2, 1/2]])
    each_member = np.zeros((N, time_step))
    open_list = []
    
    # 観測データ
    obs = np.loadtxt('out/u_obs.csv', 
                     delimiter=',', dtype='float64')
    n_true = np.loadtxt('out/u_true.csv', 
                     delimiter=',', dtype='float64')
    
    # 初期メンバ
    A_ini = np.zeros((3, N))
    for member in range(N):
        A_ini[0, member] = 0+gauss()
        A_ini[1, member] = 2+gauss()
        A_ini[2, member] = 4+gauss()
    
    A_ini_open = np.array([[0+gauss()],[2+gauss()],[4+gauss()]])
    open_list.append(A_ini_open[0,0])
        
    each_member[:, 0:1] = A_ini[0:1, :].T
    
    
    for time in range(1, time_step):
        A_f = Model(A_ini)
        P_f = get_cov(A_f, N)
        drand = np.random.standard_normal((2, N))
        A_ini = A_f + P_f@H.T@np.linalg.inv((H@P_f@H.T+R))@ \
                (np.tile(obs[:,time:time+1], (1, N)) + drand - H@A_f)
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
        pass
        
    mean_of_member = np.mean(each_member, axis=0)
    ax.plot(mean_of_member, color='red')
    ax.plot(n_true[0], color='blue')
    ax.plot(open_list, color='green')
    plt.show()


if __name__ == '__main__':
    main()
    
    
