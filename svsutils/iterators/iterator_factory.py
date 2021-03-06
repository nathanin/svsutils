"""

June 2019


NOTE:
preprocessing is handled by the call to Slide._read_region() which
obfuscates online control of the preprocessing function. 

For example, maybe we want to apply several models to the slide, 
all requiring different preprocessing. 

Maybe we want to ablate the ppr_fn itself.

A theme where the ppr_fn is passed into the 
Iterator constructor via the args might be the way to go here.

We'd track it:
'''
self.ppr_fn = args.preprocess_fn
'''
and add a call to it immediately after reading a tile.

This would go inside the wrapped_fn for the TensorFlow iteraror, to use
multithreading in the backend.

Also, turn off the autmoatic preprocessing inside slide._read_tile()

"""

import numpy as np
from svsutils.svs_reader import Slide

try:
  import tensorflow as tf
except:
  print('Failed to import tensorflow')


class PythonIterator():
  """
  A class for building an iterator for use like:

  it_factory = PythonIterator(..)

  for img, idx in it_factory.yield_one(...):
    ...

  By default PythonIterator produces one image at a time,
  yielding (img, index) pairs where the batch dimension
  of img is 1.

  Parameters:
  src (svsutils.Slide): Path to slide (required)
  img_idx (bool): Whether to yield a tuple (img, index). (True)
  batchsize (int): Size of batches to yield. 
                   Batches will be (batchsize, h, w, c). (1)
  """
  def __init__(self, slide, args, **kwargs):
    self.arg_defaults = {
      'batch':1,
      'img_idx':True
    }
    # Check identity of slide
    # assert 

    self.slide = slide
    self.extract_args(args)
    assert self.batch > 0

    # Do not create an iterator 

  def compute_output(self, outname, **kwargs):
    """
    Run the output function
    """
    fn = self.compute_fns[outname]
    ret = fn(**kwargs)
    return ret

  def extract_args(self, aparse_space):
    # Extract parts of aparse_space that are 
    # 1:1 in the dictionary
    input_keys = list(aparse_space.__dict__.keys())
    for key, val in self.arg_defaults.items():
      if key in input_keys:
        setattr(self, key, aparse_space.__dict__[key])
      else:
        setattr(self, key, val)
    

  def yield_one(self, shuffle=True, also_yield=None):
    """
    for img, idx in slideobj.yield_one():
      ...

    also_yield is a list-like or constant that should be yielded next to the slide contents
    """

    if also_yield is not None:
      for idx in self.slide.generate_index(shuffle=shuffle):
        coords = self.slide.tile_list[idx]
        img = self.slide._read_tile(coords)
        yield img, idx, also_yield

    else:
      for idx in self.slide.generate_index(shuffle=shuffle):
        coords = self.slide.tile_list[idx]
        img = self.slide._read_tile(coords)
        yield img, idx


  def yield_batch(self, shuffle=True):
    """
    it's faster to call yield_one() if batchsize=1

    Note because of array_split the batchsize may be off
    """
    indices = np.arange(len(self.slide.tile_list))
    n_batches = len(indices) // self.batch
    batches = np.array_split(indices, n_batches)
    for bidx in batches:
      bimg = []
      for idx in bidx:
        coords = self.slide.tile_list[idx]
        img = self.slide._read_tile(coords)
        bimg.append(img)

      bimg = np.stack(bimg, axis=0)
      yield bimg, bidx


# How to check tensorflow is imported ? 
# Maybe we'll just let the errors flow naturally
class TensorflowIterator(PythonIterator):
  """
  A class for building an iterator for use like:

  my_iterator = 

  Parameters:
  src (svsutils.Slide): Path to slide (required)
  img_idx (bool): Whether to yield a tuple (img, index). (True)
  batchsize (int): Size of batches to yield. 
                   Batches will be (batchsize, h, w, c). (1)
  """
  def __init__(self, slide, args, dtypes=[tf.float32, tf.int64], **kwargs):
    super(TensorflowIterator, self).__init__(slide, args)

    self.arg_defaults = {
      'batch':1,
      'img_idx':True,
      'prefetch': 512,
      'workers': 4
    }

    self.slide = slide
    self.extract_args(args)
    assert self.batch > 0

    self.dtypes = dtypes

  def read_region_at_index(self, idx):
    def wrapped_fn(idx):
      coords = self.slide.tile_list[idx]
      img = self.slide._read_tile(coords)
      return img, idx

    return tf.py_function(func = wrapped_fn,
                      inp  = [idx],
                      Tout = self.dtypes)

  def make_dataset(self):
    ds = tf.data.Dataset.from_generator(generator=self.slide.generate_index,
        output_types=tf.int64)
    ds = ds.map(self.read_region_at_index, num_parallel_calls=self.workers)
    ds = ds.prefetch(self.prefetch)
    ds = ds.batch(self.batch)
    return ds

  def make_iterator(self):
    # TODO allow sequential reading ?
    # ds = tf.data.Dataset.from_generator(generator=self.slide.generate_index,
    #     output_types=tf.int64)
    # ds = ds.map(self.read_region_at_index, num_parallel_calls=self.workers)
    # ds = ds.prefetch(self.prefetch)
    # ds = ds.batch(self.batch)
    ds = self.make_dataset()

    if tf.executing_eagerly():
      return tf.contrib.eager.Iterator(ds)
    else:
      self.iterator = ds.make_one_shot_iterator()
      # self.img_op, self.idx_op = next(self.iterator)
      # return self.img_op, self.idx_op
      return self.iterator

    
