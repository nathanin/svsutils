"""
Tools for working with outputs 

Convert to contours with a kernel density estimator.
write contours to xml 

Colors generated with  http://vrl.cs.brown.edu/color
"""
import numpy as np
import cv2
import os

def color_mask(mask, colors):
  uq = np.unique(mask)
  r = np.zeros(shape=mask.shape, dtype=np.uint8)
  g = np.copy(r)
  b = np.copy(r)
  for u in uq:
    r[mask==u] = colors[u,0]
    g[mask==u] = colors[u,1]
    b[mask==u] = colors[u,2]
  newmask = np.dstack((b,g,r))
  return newmask

def overlay_img(base, pred, colors, mixture):
  img = cv2.imread(base)
  ishape = img.shape[:2][::-1]

  ext = os.path.splitext(pred)[-1] 
  if ext == '.npy':
    y = np.load(pred)
    y = cv2.resize(y, fx=0, fy=0, dsize=ishape, 
        interpolation=cv2.INTER_LINEAR)
    ymax = np.argmax(y, axis=-1)
    # Find unprocessed space
    ymax[np.sum(y, axis=-1) < 1e-2] = 4 # white
  elif ext == '.png':
    y = cv2.imread(pred, -1)
    y = cv2.resize(y, fx=0, fy=0, dsize=ishape, 
        interpolation=cv2.INTER_NEAREST)
    ymax = y
    ymax[y == 255] = 4

  # Find pure black and white in the img
  gray = np.mean(img, axis=-1)
  img_w = gray > 220
  img_b = gray < 10

  ymax = color_mask(ymax, colors)
  img = np.add(img*mixture[0], ymax*mixture[1])
  channels = np.split(img, 3, axis=-1)
  for c in channels:
    c[img_w] = 255
    c[img_b] = 255
  img = np.dstack(channels)
  return cv2.convertScaleAbs(img)
