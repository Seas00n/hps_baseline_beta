import numpy as np
import matplotlib.pyplot as plt
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)
# from hps_baseline.greggdataset_loader import load_data, v_key, s_key
from hps_baseline.opensource_loader import load_data, v_key, s_key
from scipy.ndimage import gaussian_filter1d

data = load_data(vk="1.0", sk="0")

q_k = gaussian_filter1d(data[0],2)
t_k = gaussian_filter1d(data[1],2)
q_a = gaussian_filter1d(data[2],2)
t_a = gaussian_filter1d(data[3],2)
idx = np.arange(q_k.shape[0])

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


ax1.plot(idx, q_k)
ax1.grid(True)


ax2.plot(idx, t_k)

ax3.plot(idx, q_a)


ax4.plot(idx, t_a)

plt.show()
