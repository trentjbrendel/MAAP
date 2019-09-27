import pickle

import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

with open('x_orig', 'rb') as fp:
    x_orig = pickle.load(fp)
with open('y_orig', 'rb') as fp:
    y_orig = pickle.load(fp)
with open('z_orig', 'rb') as fp:
    z_orig = pickle.load(fp)

mav = 2.24435943773655

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# index_i = 0
# for i in range(len(x_orig)):
    # for j in range(len(x_orig[i])):
for i in range(2):
    for j in range(2):
        index = np.random.choice(np.arange(0, int(len(x_orig[i][j]))), 10000, replace=False)
        x_inaccuracy = np.random.random()*0.06 - 0.03 # 10 um accuracy
        y_inaccuracy = np.random.random()*0.06 - 0.03 # 10 um accuracy
        z_inaccuracy = np.random.random()*0.06 - 0.03 # 10 um accuracy
        ax.scatter(x_orig[i][j][index]+x_inaccuracy, y_orig[i][j][index]+y_inaccuracy, z_orig[i][j][index]+z_inaccuracy, s=1)
        # ax.scatter(x_orig[i][j][index], y_orig[i][j][index], z_orig[i][j][index], s=1)
    # index_i += int(num_data[i])
plt.tight_layout()
plt.show()