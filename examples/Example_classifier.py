#!/usr/env/bin python
"""
This example deploys a classifier to a list of SVS slides

Utilities demonstrated here:

cpramdisk      - manages and copies data between slow and fast media
Slide          - core object for managing slide data read/write
PythonIterator - hooks for creating generators out of a Slide
xx
TensorflowIterator - A wrapped PythonIterator with multithreading
                     and direct integration with TensorFlow graphs

This script takes advantage of model constructors defined in 
https://github.com/nathanin/milk


Usage
-----
```
python Example_classifier.py [slides.txt] [model/snapshot.h5] [encoder type] [options]
```

June 2019
"""
from svsutils import repext
from svsutils import cpramdisk
from svsutils import Slide
from svsutils import reinhard
from svsutils import PythonIterator
from svsutils import TensorflowIterator

import tensorflow as tf
import numpy as np
import traceback

from milk.eager import ClassifierEager
from milk.encoder_config import get_encoder_args

import argparse
import os

def main(args):
  # Define a compute_fn that should do three things:
  # 1. define an iterator over the slide's tiles
  # 2. compute an output with given model parameter
  # 3. 

  if args.iter_type == 'python':
    def compute_fn(slide, args, model=None):
      print('Slide with {}'.format(len(slide.tile_list)))
      it_factory = PythonIterator(slide, args)
      for k, (img, idx) in enumerate(it_factory.yield_batch()):
        prob = model(img)
        if k % 50 == 0:
          print('Batch #{:04d} idx:{} img:{} prob:{}'.format(k, idx.shape, img.shape, prob.shape))
        slide.place_batch(prob, idx, 'prob', mode='tile')
      ret = slide.output_imgs['prob']
      return ret

  # Tensorflow multithreaded queue-based iterator (in eager mode)
  elif args.iter_type == 'tf':
    def compute_fn(slide, args, model=None):
      assert tf.executing_eagerly()
      print('Slide with {}'.format(len(slide.tile_list)))

      # In eager mode, we return a tf.contrib.eager.Iterator
      eager_iterator = TensorflowIterator(slide, args).make_iterator()

      # The iterator can be used directly. Ququeing and multithreading
      # are handled in the backend by the tf.data.Dataset ops
      for k, (img, idx) in enumerate(eager_iterator):
        # Batches are already returned with the proper 4D shape
        prob = model(img, training=False)
        # Now everything is a tf.EagerTensor so we should call the numpy() method
        prob, img, idx = prob.numpy(), img.numpy(), idx.numpy()
        if k % 50 == 0:
          print('Batch #{:04d} idx:{} img:{:3.3f} ({}) prob:{}'.format(k, idx.shape, img.max(), img.shape, prob.shape))
          # print(prob)
        slide.place_batch(prob, idx, 'prob', mode='tile')
      ret = slide.output_imgs['prob']
      return ret

  # Set up the model first
  encoder_args = get_encoder_args(args.encoder)
  model = ClassifierEager(encoder_args=encoder_args, n_classes=args.n_classes)
  x = tf.zeros((1, args.process_size,
                args.process_size, 3))
  _ = model(x, verbose=True, training=True)
  model.load_weights(args.snapshot)

  # keras Model subclass
  model.summary()

  # Read list of inputs
  with open(args.slides, 'r') as f:
    slides = [x.strip() for x in f]

  # Loop over slides
  for src in slides:
    # Dirty substitution of the file extension give us the
    # destination. Do this first so we can just skip the slide
    # if this destination already exists.
    # Set the --suffix option to reflect the model / type of processed output
    dst = repext(src, args.suffix)

    # Loading data from ramdisk incurs a one-time copy cost
    rdsrc = cpramdisk(src, args.ramdisk)
    print('File:', rdsrc)

    # Wrapped inside of a try-except-finally.
    # We want to make sure the slide gets cleaned from 
    # memory in case there's an error or stop signal in the 
    # middle of processing.
    try:
      # Initialze the side from our temporary path, with 
      # the arguments passed in from command-line.
      # This returns an svsutils.Slide object
      slide = Slide(rdsrc, args)

      # This step will eventually be included in slide creation
      # with some default compute_fn's provided by svsutils
      # For now, do it case-by-case, and use the compute_fn
      # that we defined just above.
      slide.initialize_output('prob', args.n_classes, mode='tile',
        compute_fn=compute_fn)

      # Call the compute function to compute this output.
      # Again, this may change to something like...
      #     slide.compute_all
      # which would loop over all the defined output types.
      ret = slide.compute('prob', args, model=model)
      print('{} --> {}'.format(ret.shape, dst))
      np.save(dst, ret[:,:,::-1])
    except Exception as e:
      print(e)
      traceback.print_tb(e.__traceback__)
    finally:
      print('Removing {}'.format(rdsrc))
      os.remove(rdsrc)

if __name__ == '__main__':
  """
  standard __name__ == __main__ ?? 
  how to make this nicer
  """
  p = argparse.ArgumentParser()
  # positional arguments for this program
  # last call
  # python Example_classifier.py
  #
  p.add_argument('slides') 
  p.add_argument('snapshot') 
  p.add_argument('encoder') 
  p.add_argument('--iter_type', default='tf', type=str) 
  p.add_argument('--suffix', default='.prob.npy', type=str) 

  # common arguments with defaults
  p.add_argument('-b', dest='batchsize', default=64, type=int)
  p.add_argument('-r', dest='ramdisk', default='/dev/shm', type=str)
  p.add_argument('-j', dest='workers', default=8, type=int)
  p.add_argument('-c', dest='n_classes', default=4, type=int)

  # Slide options
  p.add_argument('--mag',   dest='process_mag', default=5, type=int)
  p.add_argument('--chunk', dest='process_size', default=96, type=int)
  p.add_argument('--bg',    dest='background_speed', default='all', type=str)
  p.add_argument('--ovr',   dest='oversample_factor', default=1.1, type=float)
  p.add_argument('--verbose', dest='verbose', default=False, action='store_true')

  args = p.parse_args()

  # Functionals for later:
  args.__dict__['preprocess_fn'] = lambda x: (reinhard(x) / 255.).astype(np.float32)

  tf.enable_eager_execution()
  main(args)