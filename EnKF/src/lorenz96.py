import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Runge_Kutta(u:np.ndarray, dt:float, dif_func):
    """1階の微分方程式に対して4次のルンゲクッタ法で逐次解を与える"""
    du = u
    s1 = dif_func(u/1.0)
    s2 = dif_func(u + s1 * dt / 2.0)
    s3 = dif_func(u + s2 * dt / 2.0)
    s4 = dif_func(u + s3 * dt/ 1.0)
    du += (s1 + s2*2 + s3*2 + s4) * (dt / 6)
    return du

def Lorenz96(u_ini:np.ndarray=np.ones((40,1)), F:float=0.6)->np.ndarray:
    """
    input: u_t(state matrix at time t, dtype:numpy.ndarray(N, 1)),
    F(ext. force term in L96)
    output: u_{t+1}
    """
    
    u_ini = u_ini.astype(np.float64)
    N:int = len(u_ini)
    
    def L96_dif_func(u):
        f = np.zeros((N,1))
        for i in range(0, N):
            f[i,0] = (u[(i+1)%N, 0]-u[(i-2)%N, 0])*u[(i-1)%N]-u[i]+F
        return f
    
    return Runge_Kutta(u_ini, 0.05, L96_dif_func)    

def bulk_Lorenz96(E:np.ndarray):
    for i in range(int(E.shape[1])):
        E[:, i:i+1] = Lorenz96(E[:, i:i+1])
    return E
        
    

if __name__ == '__main__':
    u1_list, u2_list, u3_list = [], [], []
    u_ini = np.array([[2],[3],[5],[0],[8],[5],[6]])
    for time in range(0,1000):
        u1_list.append(u_ini[0, 0])
        u2_list.append(u_ini[1,0])
        u3_list.append(u_ini[2,0])
        u_ini = Lorenz96(u_ini, 6)
        
    
    # ax = plt.figure().add_subplot(projection='3d')
    
    # ax.plot(u1_list, u2_list, u3_list)
    plt.plot(u1_list)
    plt.show()
    
    
        