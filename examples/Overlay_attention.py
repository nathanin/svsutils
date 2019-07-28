#!/usr/bin/env python

"""

"""
from svsutils import repext

import argparse
import cv2
import glob
import os

from scipy.special import softmax
import numpy as np
import seaborn as sns


def color_mask(mask, colors):
  uq = np.unique(mask)
  r = np.zeros(shape=mask.shape, dtype=np.uint8)
  g = np.copy(r)
  b = np.copy(r)
  for u in uq:
    u_m = mask == u
    c = colors[u]
    # n_u = u_m.sum()
    # print('u: {} = {} {}'.format(u , n_u, c))
    r[mask==u] = c[0] * 255
    g[mask==u] = c[1] * 255
    b[mask==u] = c[2] * 255

  # Use white for 1
  u_m = mask == 1
  r[u_m] = 255
  g[u_m] = 255
  b[u_m] = 255

  newmask = np.dstack((b,g,r))
  return newmask





from matplotlib import pyplot as plt
def savehist(ydig, colors, dst):
  plt.clf()
  hist = repext(dst, '.hist.png')
  print('hist --> {}'.format(hist))
  N, bins, patches = plt.hist(ydig.ravel(), bins=50, log=False, density=True)  
  for k in range(50):
    patches[k].set_facecolor(colors[k])
  plt.savefig(hist, bbox_inches='tight')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
def overlay_img(base, pred, mixture, colors, dst):
  img = cv2.imread(base)
  ishape = img.shape[:2][::-1]

  # Find pure black and white in the img
  gray = np.mean(img, axis=-1)
  img_w = gray > 220
  img_b = gray < 10

  y = np.load(pred)
  ymin, ymax = y.min(), y.max()
  if y.shape[-1] == 1: 
    y = np.squeeze(y)

  # Using a foreground mask :
  yshape = y.shape[::-1]
  gray_s = cv2.resize(gray, dsize=yshape, interpolation=cv2.INTER_LINEAR)
  img_w_s = gray_s > 230
  img_b_s = gray_s < 20
  background = (img_w_s + img_b_s).astype(np.bool)
  foreground = np.logical_not(background)
  print(y.shape, y.dtype)
  print(background.shape, background.dtype)
  print(foreground.shape, foreground.dtype)
  y_fg = y[foreground]
  y_fg = softmax(y_fg)
  y[background] = 0.
  y[foreground] = y_fg
  print(y.min(), y.max())
  print('nnz(y) =', (y != 0).sum(), '/', np.prod(y.shape))

  bins = np.linspace(0., y.max(), args.n_colors-1)
  ydig = np.digitize(y, bins)
  # Emphasize sparse positive points
  ydig = cv2.dilate(ydig.astype(np.uint8), kernel=kernel, iterations=1)
  savehist(ydig, colors, dst)
  print('ydig', np.unique(ydig))
  ydig_dst = repext(dst, '.ydig.png')
  print(ydig.shape, '-->', ydig_dst)
  cv2.imwrite(ydig_dst, ydig * (255./args.n_colors))

  ydig = cv2.resize(ydig, fx=0, fy=0, dsize=ishape, interpolation=cv2.INTER_NEAREST)

  # Find unprocessed space
  # ymax[np.sum(y, axis=-1) < 1e-2] = 4 # white

  ycolor = color_mask(ydig, colors)
  img = np.add(img*mixture[0], ycolor*mixture[1])
  # Whiten the background
  # channels = np.split(img, 3, axis=-1)
  # for c in channels:
  #   c[img_w] = 255
  #   c[img_b] = 255
  # img = np.dstack(channels)
  return cv2.convertScaleAbs(img)


def main(args):
  # baseimgs = sorted(glob.glob('{}/*img.jpg'.format(args.s)))
  # predictions = sorted(glob.glob('{}/*{}'.format(args.p, args.r)))
  # baseimgs, predictions = matching_basenames(baseimgs, predictions)

  if args.color == 'cubehelix':
    colors = sns.cubehelix_palette(args.n_colors)
  else:
    try:
      colors = sns.color_palette(args.color, n_colors=args.n_colors)
    except Exception as e:
      traceback.print_tb(e.__traceback__)
      print('{} not a valid seaborn colormap name'.format(args.color))
      print('Defaulting to "RdBu_r"')
      colors = sns.color_palette("RdBu_r", n_colors=args.n_colors)

  mixture = [0.3, 0.7]

  with open(args.s, 'r') as f:
    baseimgs = [x.strip() for x in f]

  with open(args.p, 'r') as f:
    predictions = [x.strip() for x in f]

  for bi, pr in zip(baseimgs, predictions):
    dst = repext(pr, args.suffix)
    if os.path.exists(dst):
      print('Exists {}'.format(dst))
      continue

    combo = overlay_img(bi, pr, mixture, colors, dst)
    print('{} --> {}'.format(combo.shape, dst))
    cv2.imwrite(dst, combo)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('s', default=None, type=str)
  parser.add_argument('p', default=None, type=str)
  parser.add_argument('--suffix', default='.attcol.jpg', type=str)

  parser.add_argument('--color', default='cubehelix', type=str)
  parser.add_argument('--n_colors', default=50, type=int)


  args = parser.parse_args()
  main(args)
