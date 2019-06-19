#!/usr/env/bin python
from svsutils import repext, Slide, PythonIterator

import tensorflow as tf
import numpy as np

from milk.eager import ClassifierEager
from milk.encoder_config import get_encoder_args

import argparse
import os

def main(args):
  def compute_fn(slide, args, model=None):
    print('Slide with {}'.format(len(slide.tile_list)))
    it_factory = PythonIterator(slide, args)
    for k, (img, idx) in enumerate(it_factory.yield_one()):
      img = np.expand_dims(img, 0)
      img = tf.constant((img/255.).astype(np.float32))
      prob = model(img)
      if k+1 % 50 == 0:
        print('Img {} {} prob {} {}'.format(k, idx, img.shape, prob.shape))
      slide.place(prob, idx, 'prob', mode='tile')
    ret = slide.output_imgs['prob']
    return ret

  # Set up the model first
  encoder_args = get_encoder_args(args.encoder)
  model = ClassifierEager(encoder_args, n_classes=5)
  x = tf.zeros((1, args.process_size,
                args.process_size, 3))
  yhat = model(x, verbose=True, training=True)
  model.load_weights(args.snapshot)

  # keras Model subclass
  model.summary()

  # Load inputs
  with open(args.slides, 'r') as f:
    slides = [x.strip() for x in f]

  # Loop over slides
  for src in slides:
    dst = repext(src, '.prob.npy')
    print(dst)
    slide = Slide(src, args)
    slide.initialize_output('prob', 5, mode='tile',
      compute_fn=compute_fn)
    ret = slide.compute('prob', args, model=model)
    print('{}'.format(ret.shape))

    np.save(dst, ret)



if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this program
  p.add_argument('slides') 
  p.add_argument('snapshot') 
  p.add_argument('encoder') 

  # common arguments with defaults
  p.add_argument('-b', dest='batchsize', default=1, type=int)
  p.add_argument('-r', dest='ramdisk', default='./', type=str)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=128, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.1, type=float)

  args = p.parse_args()

  tf.enable_eager_execution()
  main(args)