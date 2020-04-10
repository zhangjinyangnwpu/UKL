import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
times = 1
class_num = 9
path = 'result'

for i in range(times):
    info_path = os.path.join(path,str(i),'result_list.mat')
    info = sio.loadmat(info_path)
    matrix = info['matrix']
    oa = info['oa'][0]
    aa = info['aa'][0]
    kappa = info['kappa'][0]
    plt.plot(oa)
    plt.title('max value: %f'%np.max(oa))
    plt.savefig(os.path.join(path,str(i),'oa.png'))
    plt.close()
    plt.plot(aa)
    plt.title('max value: %f'%np.max(aa))
    plt.savefig(os.path.join(path, str(i), 'aa.png'))
    plt.close()
    plt.plot(kappa)
    plt.title('max value: %f'%np.max(kappa))
    plt.savefig(os.path.join(path, str(i), 'kappa.png'))
    plt.close()






