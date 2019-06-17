
"""
Convert to contours with a kernel density estimator.
Write contours to xml 
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import os

def prob2contour(prob, levels=[0.5, 0.7, 0.85], fineness=50):
  prob = cv2.resize(prob,fx=0.5,fy=0.5,dsize=(0,0))

  pmax = np.argmax(prob, axis=-1)
  pmax[np.sum(prob, axis=-1) < 1e-2] = 4

  null_prob = 1 / (np.prod(pmax.shape))

  plt.figure()
  ax = plt.gca()
  for c in [3,2,1,0]:
    color = ListedColormap(colors[c])
    msk = pmax == c
    if msk.sum() < np.prod(msk.shape) * 0.01:
      print('Skip {}'.format(c))
      continue

    x,y = np.where(msk)

    xmin = 0
    xmax = msk.shape[0]
    ymin = 0
    ymax = msk.shape[1]
    xx, yy = np.mgrid[xmin:xmax:fineness*1j, ymin:ymax:fineness*1j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x,y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions), xx.shape, order='C')[::-1,:].T

    plt_levels = np.quantile(f[f>null_prob], levels)
    ax.contour(xx, yy, f, [plt_levels], cmap=color)

  dst = os.path.splitext(pth)[0] + '.kde.png'
  print(dst)
  plt.savefig(dst, bbox_inches='tight') 
