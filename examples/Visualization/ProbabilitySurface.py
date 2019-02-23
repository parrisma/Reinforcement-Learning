import matplotlib.pyplot as plt
import numpy as np

# generate 2 2d grids for the x & y bounds
n = 60 * 3

cell_11 = (1, 1)
z = np.zeros((n, n))
nv = .1
sv = .1
ev = .8
wv = .0
cx = 20
cy = 20
for a in range(1, n, 3):
    for b in range(1, n, 3):
        i = a - 1
        j = b - 1
        z[i + 0, j + 1] = np.sqrt((cx - (i + 0)) ** 2 + (cy - (j + 1)) ** 2)  # N
        z[i + 1, j + 2] = np.sqrt((cx - (i + 1)) ** 2 + (cy - (j + 2)) ** 2)  # E
        z[i + 2, j + 1] = np.sqrt((cx - (i + 2)) ** 2 + (cy - (j + 1)) ** 2)  # S
        z[i + 1, j + 0] = np.sqrt((cx - (i + 1)) ** 2 + (cy - (j + 0)) ** 2)  # W
        z[i + 1, j + 1] = z[i:i + 3, j:j + 3].sum() / 4.0
        z[i + 0, j + 0] = (z[i + 0, j + 1] + z[i + 1, j + 0] + z[i + 1, j + 1]) / 3.0
        z[i + 2, j + 0] = (z[i + 1, j + 0] + z[i + 2, j + 1] + z[i + 1, j + 1]) / 3.0
        z[i + 2, j + 2] = (z[i + 2, j + 1] + z[i + 1, j + 2] + z[i + 1, j + 1]) / 3.0
        z[i + 0, j + 2] = (z[i + 0, j + 1] + z[i + 1, j + 2] + z[i + 1, j + 1]) / 3.0
        z[i:i + 3, j:j + 3] -= np.max(z[i:i + 3, j:j + 3])
        z[i:i + 3, j:j + 3] = np.fabs(z[i:i + 3, j:j + 3])
        # z[i:i + 3, j:j + 3] /= np.exp(z[i:i + 3, j:j + 3] * -1)
        slz = z[i:i + 3, j:j + 3]
        slz /= slz.sum()

cmap = plt.get_cmap('terrain')

fig, (ax1) = plt.subplots(nrows=1)

cf = ax1.imshow(z, interpolation='nearest', cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()
