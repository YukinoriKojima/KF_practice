from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from lorenz96 import Lorenz96, bulk_Lorenz96

def Model_example(x:np.ndarray) -> np.ndarray:
    """
    理想的なモデル．各状態変数は前の状態の2倍になる
    """
    F = np.array([[1.003,0,0],
                 [0,1.003,0],
                 [0,0,1.003]])
    return F@x

def H(x:np.ndarray) -> np.ndarray:
    h = np.array([[1/2, 1/2, 0],
                  [0, 1/2, 1/2]])
    return h@x

def main():
    # タイムステップは400まで
    # code上は0~399
    time_step = 400
    Model:function = Lorenz96
    
    # 真値の作成
    # u_t+1 = 1.003*u_t+誤差項(~N(0,1))    
    u_true = np.zeros((3, 400))
    init = [[10+gauss()], [20+gauss()], [30+gauss()]]
    print(init)
    u_true[:, 0:1] = np.array(init)
    
    for time in range(1, time_step):
        u_true[:, time:time+1] = Model(u_true[:, time-1:time])+  \
                                np.array([[gauss()],[gauss()],[gauss()]])
                                        
    plt.plot(u_true[0])
    plt.show()
    
    # 観測値の作成
    # README読んでください
    u_obs = np.zeros((2, 400))
    for time in range(0, time_step):
        u_obs[:, time:time+1] = H(u_true[:, time:time+1]) + np.array([[gauss()], [gauss()]])
    
    # (必要であれば)可視化    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(u_true[0])
    ax.plot(u_true[1])
    ax.plot(u_true[2])
    ax.plot(u_obs[0])
    ax.plot(u_obs[1])
    ax.legend(['true_1', 'true_2', 'true_3', 'obs_1', 'obs_2'])
    plt.show()

    # 真値と観測値をcsvに保存
    # 既に作ったので一旦コメントアウト
    np.savetxt('out/u_obs.csv', u_obs, delimiter=',')
    np.savetxt('out/u_true.csv', u_true, delimiter=',')
    
    
    
                                
    
    
                                
        
    
    
    
if __name__ == "__main__":
    main()